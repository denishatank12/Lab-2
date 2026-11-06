from __future__ import annotations
import argparse, json
from pathlib import Path
from agent_system import run, save_trace

SCENARIOS=[
 ('caption_1','caption this image'),('caption_2','what objects are visible?'),('caption_3','where is the subject?'),
 ('style_1','change this image to an impressionist style'),('style_2','make this look like a watercolor painting'),('style_3','apply a cinematic style'),
 ('edit_1','remove the background'),('edit_2','enhance the lighting'),('edit_3','add a blue sky'),
 ('domain_1','make this image look like a Naruto illustration'),('domain_2','apply the domain style'),('domain_3','turn this into a character painting'),
 ('ambiguous_1','do something interesting'),('broken_1','caption the missing image'),('broken_2','')]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='outputs/scenario_traces.json')
    ap.add_argument('--enable-models',action='store_true')
    ap.add_argument('--image',default=None)
    ap.add_argument('--lora-path',default=None)
    ap.add_argument('--no-critique',action='store_true')
    ap.add_argument('--inputs', help='Prepared inputs.json: real held-out images and actual broken-file fixtures.')
    a=ap.parse_args()
    runtime = None
    if a.enable_models:
        from model_runtime import ModelRuntime
        provenance_path = Path(a.lora_path or '') / 'hub_provenance.json'
        if not provenance_path.is_file():
            raise RuntimeError('Run prepare_evaluation.py --download-adapter first; Hub revision provenance is required.')
        runtime = ModelRuntime(a.lora_path, adapter_provenance=json.loads(provenance_path.read_text()))
    root=Path(__file__).resolve().parents[1]
    scenarios=json.loads((root/'eval/scenarios.json').read_text())
    inputs=json.loads(Path(a.inputs).read_text()) if a.inputs else {}
    states=[]
    for scenario in scenarios:
        image=inputs.get(scenario.get('image_key'),{}).get('path',a.image) if scenario['type'] != 'broken' else None
        state=run(scenario['request'], image_path=image, vision_pipe=runtime, generation_pipe=runtime,
                  output_dir=Path(a.output).parent/'generated'/Path(a.output).stem/scenario['id'],
                  critique_loop=not a.no_critique,scenario_id=scenario['id'])
        states.append(state)
        save_trace(states,Path(a.output))
        print(scenario['id'],state.decision,state.error,flush=True)
    metadata = dict(schema_version=2, models_enabled=a.enable_models, critique_loop=not a.no_critique,
        inputs=inputs, adapter_provenance=runtime.adapter_provenance if runtime else None,
        scope='Actual model-backed scenarios' if runtime else 'Control-flow smoke only, not evaluation results')
    Path(a.output).with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'scenarios':len(states),'models_enabled':a.enable_models,
        'critique_loop':not a.no_critique,'safe_refusals':sum(x.decision=='refused' for x in states),
        'generated':sum(bool(x.output_path and x.output_path.endswith('.png')) for x in states),
        'note':'Task success and critic agreement require the two members to judge the saved traces.',
        'output':a.output},indent=2))
if __name__=='__main__': main()
