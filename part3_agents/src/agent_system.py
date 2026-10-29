from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import re, time

UNSAFE = ('nude', 'sexual', 'child abuse', 'exploit', 'weapon', 'bomb', 'disguise a real person', 'deepfake')
IMAGE_INSTRUCTION = ('ignore previous', 'system prompt', 'do not tell', 'follow the text in the image')

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
    trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, agent: str, inp: Any, out: Any, started: float):
        self.trace.append({'agent': agent, 'input': inp, 'output': out, 'seconds': round(time.perf_counter()-started, 4)})

def planner(s: State) -> State:
    t=time.perf_counter(); q=s.request.lower(); s.safe=not any(x in q for x in UNSAFE+IMAGE_INSTRUCTION)
    if not s.safe: s.route='refuse'
    elif any(x in q for x in ('caption','describe','what is','where is','how many')): s.route='understanding'
    elif any(x in q for x in ('edit','remove','add','enhance')): s.route='edit'
    elif any(x in q for x in ('style','monet','naruto','pokemon','domain')): s.route='style'
    else: s.route='unsupported'
    s.log('planner',s.request,{'route':s.route,'safe':s.safe},t); return s

def vision(s: State, pipe=None) -> State:
    t=time.perf_counter()
    if not s.image_path: s.caption='No image supplied.'
    elif pipe:
        from PIL import Image
        s.caption=pipe(Image.open(s.image_path))[0]['generated_text']
    else: s.caption='Image supplied; run with --enable-models for model-generated caption.'
    s.log('vision',s.image_path,s.caption,t); return s

def prompt_engineer(s: State) -> State:
    t=time.perf_counter(); s.prompt=s.request if s.route=='style' else f'{s.request}. Preserve the main subject and composition.'
    s.log('prompt_engineering',s.request,s.prompt,t); return s

def generate(s: State, pipe=None, output_dir: Path|None=None) -> State:
    t=time.perf_counter()
    if pipe and output_dir:
        output_dir.mkdir(parents=True,exist_ok=True); image=pipe(s.prompt,num_inference_steps=20).images[0]; s.output_path=str(output_dir/'generated.png'); image.save(s.output_path)
    else: s.output_path='generation skipped (models disabled)'
    s.log('generation',s.prompt,s.output_path,t); return s

def critique(s: State) -> State:
    t=time.perf_counter(); s.critique='accept' if s.safe and s.output_path else 'revise'; s.decision=s.critique; s.log('critique',s.output_path,{'decision':s.decision},t); return s

def run(request: str, image_path: str|None=None, vision_pipe=None, generation_pipe=None, output_dir: Path|None=None) -> State:
    s=planner(State(request=request,image_path=image_path))
    if not s.safe: s.decision='refused'; return s
    if s.route in ('understanding','edit','style'): vision(s,vision_pipe)
    if s.route in ('edit','style'): prompt_engineer(s); generate(s,generation_pipe,output_dir); critique(s)
    elif s.route=='understanding': s.decision='answered'
    else: s.decision='unsupported'
    return s

def save_trace(states: list[State], path: Path):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(__import__('json').dumps([asdict(s) for s in states],indent=2),encoding='utf-8')
