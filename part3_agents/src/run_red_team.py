from __future__ import annotations
import argparse, json
from pathlib import Path
from agent_system import run

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='outputs/red_team_results.json')
    ap.add_argument('--inputs')
    ap.add_argument('--enable-models',action='store_true')
    ap.add_argument('--no-critique',action='store_true')
    a=ap.parse_args()
    if not a.inputs and a.enable_models:
        raise ValueError('Actual image-based red-team evaluation needs --inputs; text-only tests are not sufficient.')
    inputs=json.loads(Path(a.inputs).read_text()) if a.inputs else {}
    runtime=None
    if a.enable_models:
        from model_runtime import ModelRuntime
        runtime=ModelRuntime()
    cases=json.loads((Path(__file__).resolve().parents[1]/'eval'/'red_team_cases.json').read_text())
    results=[]
    for case in cases:
        image=inputs.get(case['image_key'],{}).get('path')
        state=run(case['request'],image_path=image,vision_pipe=runtime,generation_pipe=runtime,
                  output_dir=Path(a.output).parent/'red_team_generated'/Path(a.output).stem/case['id'],
                  critique_loop=not a.no_critique)
        passed=state.decision in case['expected']
        results.append({'id':case['id'],'request':case['request'],'image_key':case['image_key'],
            'decision':state.decision,'safe':state.safe,'pass':passed,'attack_success':not passed,
            'trace':state.trace,'error':state.error,'seconds':state.seconds})
    if runtime:
        runtime.vlm.unload()
    result=dict(cases=results,models_enabled=a.enable_models,image_based=bool(a.inputs),critique_loop=not a.no_critique,
        attack_success_rate=sum(r['attack_success'] for r in results)/len(results) if a.enable_models else None,
        definition='Conservative security-test failure rate: outcomes outside the predefined safe decision set / 10. Not a claim that secrets were exfiltrated.',
        scope='model-backed image tests' if a.enable_models else 'control-flow tests only')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'cases':len(results),'passed':sum(x['pass'] for x in results),
                      'attack_success_rate':result['attack_success_rate'],'scope':result['scope']},indent=2))
if __name__=='__main__': main()
