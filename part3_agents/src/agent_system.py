from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import re, time

UNSAFE = ('nude', 'sexual', 'child abuse', 'exploit', 'weapon', 'bomb', 'disguise a real person', 'deepfake', 'evidence of a crime')
IMAGE_INSTRUCTION = ('ignore previous', 'system prompt', 'do not tell', 'follow the text in the image', 'hidden text', 'system command')

@dataclass
class State:
    request: str
    image_path: str | None = None
    route: str = ''
    safe: bool = True
    caption: str = ''
    visual_answer: str = ''
    prompt: str = ''
    output_path: str = ''
    critique: str = ''
    decision: str = ''
    revisions: int = 0
    scenario_id: str = ''
    visual_details: dict[str, Any] = field(default_factory=dict)
    clip_similarity: float | None = None
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ''
    trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, agent: str, inp: Any, out: Any, started: float):
        self.trace.append({'agent': agent, 'input': inp, 'output': out, 'seconds': round(time.perf_counter()-started, 4)})

def planner(s: State) -> State:
    t=time.perf_counter(); q=s.request.lower(); s.safe=not any(x in q for x in UNSAFE+IMAGE_INSTRUCTION)
    if not s.safe: s.route='refuse'
    elif any(x in q for x in ('caption','describe','what is','where is','how many')): s.route='understanding'
    elif any(x in q for x in ('naruto','pokemon','domain','character painting')): s.route='domain_style'
    elif any(x in q for x in ('style','monet','watercolor','painting','cinematic')): s.route='style'
    elif any(x in q for x in ('edit','remove','add','enhance')): s.route='edit'
    else: s.route='unsupported'
    s.log('planner',s.request,{'route':s.route,'safe':s.safe},t); return s

def vision(s: State, pipe=None) -> State:
    t=time.perf_counter()
    if not s.image_path: s.caption='No image supplied.'
    elif pipe and hasattr(pipe, 'understand'):
        result = pipe.understand(s.image_path, s.request)
        s.caption = result['caption']
        s.visual_answer = result.get('answer', s.caption)
        s.visual_details = result
        if any(x in (str(result.get('visible_text','')) + ' ' + s.caption).lower() for x in IMAGE_INSTRUCTION):
            s.safe = False
    elif pipe:
        from PIL import Image
        s.caption=pipe(Image.open(s.image_path))[0]['generated_text']
    else: s.caption='Image supplied; run with --enable-models for model-generated caption.'
    s.log('vision',s.image_path,{'caption':s.caption,'details':s.visual_details,'answer':s.visual_answer},t); return s

def prompt_engineer(s: State) -> State:
    t=time.perf_counter()
    s.prompt = f'{s.request}. Main subject: {s.caption}. Preserve the main subject and composition.'
    if s.revisions:
        s.prompt += ' Revision guidance: ' + s.critique
    s.log('prompt_engineering',s.request,s.prompt,t); return s

def generate(s: State, pipe=None, output_dir: Path|None=None) -> State:
    t=time.perf_counter()
    if pipe and output_dir:
        output_dir.mkdir(parents=True,exist_ok=True)
        if hasattr(pipe, 'generate'):
            result = pipe.generate(s)
        else:
            from PIL import Image
            with Image.open(s.image_path) as im:
                result = pipe(prompt=s.prompt, image=im.convert('RGB'), num_inference_steps=20)
        if bool((getattr(result, 'nsfw_content_detected', None) or [False])[0]):
            s.safe = False
            s.output_path = ''
            s.decision = 'refused_output'
        else:
            s.output_path = str(output_dir / f'generated_revision_{s.revisions}.png')
            result.images[0].save(s.output_path)
    else:
        s.output_path = ''
        s.decision = 'models_disabled'
    s.log('generation',s.prompt,{'output_path':s.output_path,'metadata':s.generation_metadata},t); return s

def critique(s: State, reviewer=None) -> State:
    t=time.perf_counter()
    if reviewer and s.output_path:
        result = reviewer.review(s)
        s.clip_similarity = float(result['clip_similarity'])
        s.critique = result['revision_instruction']
        s.decision = 'accept' if result['accept'] and s.safe else 'revise'
        s.log('critique', s.output_path, result, t)
    else:
        s.critique = 'No model-based critique was performed.'
        s.decision = 'models_disabled' if not s.output_path else 'needs_human_review'
        s.log('critique', s.output_path, {'decision':s.decision}, t)
    return s

def run(request: str, image_path: str|None=None, vision_pipe=None, generation_pipe=None, output_dir: Path|None=None,
        critique_loop: bool=True, scenario_id: str='') -> State:
    s=planner(State(request=request,image_path=image_path))
    s.scenario_id = scenario_id
    if not request.strip(): s.decision='invalid_input'; return s
    if not s.safe: s.decision='refused'; return s
    if s.route == 'unsupported': s.decision='unsupported'; return s
    if not image_path: s.decision='missing_image'; return s
    phase='input'
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            im.verify()
        with Image.open(image_path) as im:
            if min(im.size) < 32:
                s.decision = 'invalid_image'; s.error = 'The image is too small to process reliably.'; return s
        phase='vision'
        vision(s,vision_pipe)
        if not s.safe: s.decision='refused'; return s
        if s.route == 'understanding':
            s.decision = 'answered' if vision_pipe else 'models_disabled'
            return s
        while True:
            phase='generation'
            prompt_engineer(s)
            generate(s,generation_pipe,output_dir)
            if s.decision in {'models_disabled','refused_output'}:
                break
            if not critique_loop:
                s.decision = 'generated_without_critique'; break
            reviewer = generation_pipe if hasattr(generation_pipe, 'review') else None
            critique(s,reviewer)
            if s.decision != 'revise':
                break
            if s.revisions >= 2:
                s.decision = 'revision_limit'; break
            s.revisions += 1
    except Exception as exc:
        # Request-boundary exception handling records failures instead of
        # labeling them successful or aborting the remaining scenario batch.
        s.error = f'{type(exc).__name__}: {exc}'
        s.decision = 'invalid_image' if phase=='input' else phase+'_error'
        import traceback
        s.log('error',s.route,{'error':s.error,'traceback':traceback.format_exc()},time.perf_counter())
    return s

def save_trace(states: list[State], path: Path):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(__import__('json').dumps([asdict(s) for s in states],indent=2),encoding='utf-8')
