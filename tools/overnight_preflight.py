"""Real model preflight on the user's GPU, not a substitute for final measurements."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'part1_rag/src'),str(ROOT/'part3_agents/src')]
from run_checkpoint import ItemCheckpoint,experiment_sources,runtime_identity,atomic_json


def preflight(part,data_dir=None):
    import torch
    import psutil
    if not torch.cuda.is_available():
        raise RuntimeError('The overnight preflight must actually run on the NVIDIA GPU. CPU smoke is not enough.')
    hardware=runtime_identity()
    if psutil.virtual_memory().available<8*2**30:
        raise RuntimeError('At least 8 GiB free host RAM is needed for the CPU-offloaded models. Close other applications first.')
    if shutil.disk_usage(ROOT).free<8*2**30:
        raise RuntimeError('Keep at least 8 GiB free disk beyond existing caches. A cold first setup needs roughly 40 GB overall.')
    data=Path(data_dir or ROOT/'part1_rag/data')
    protocol=dict(kind='real-gpu-preflight-v1',part=part,sources=experiment_sources(part))
    if part=='part1_rag':
        from multimodal import ingest
        protocol['source_data']=ingest(data,ROOT/'part1_rag/outputs/v2/preflight_ingestion')['source_sha256']
    else:
        protocol['inputs']=json.loads((ROOT/'part3_agents/outputs/v2/inputs/inputs.json').read_text())
    checkpoint=ItemCheckpoint(ROOT/'runtime_checks/preflight_checkpoints',protocol)
    previous=checkpoint.get('passed')
    if previous:
        print('Reusing matching real GPU preflight:',json.dumps(previous,indent=2),flush=True)
        return previous
    started=time.perf_counter()
    checks=[]
    if part=='part1_rag':
        from multimodal import MultimodalIndex,Conversation
        index=MultimodalIndex(data,ROOT/'part1_rag/outputs/v2/index')
        chat=Conversation(index)
        try:
            answer=chat.answer('Read Figure A-5. Identify the chart and its stated base date if readable; do not guess.')
            checks.append(dict(check='actual visual RAG answer',answer=answer))
        finally:
            chat.model.unload()
            index.db.close()
    else:
        from model_runtime import ModelRuntime
        from agent_system import State
        adapter=ROOT/'part3_agents/outputs/hub_adapter'
        provenance=json.loads((adapter/'hub_provenance.json').read_text())
        inputs=protocol['inputs']
        runtime=ModelRuntime(adapter,adapter_provenance=provenance)
        image=inputs['heldout_0']['path']
        folder=ROOT/'part3_agents/outputs/v2/preflight'
        folder.mkdir(parents=True,exist_ok=True)
        try:
            details=runtime.understand(image,'What objects are visible?')
            checks.append(dict(check='actual VLM objects, text and questions',details=details))
            critic_state=State(request='Describe the visible subject.',image_path=image,output_path=image)
            checks.append(dict(check='actual critic API on identical held-out input pair, not generated quality',
                result=runtime.review(critic_state)))
            for route,request in [('style','Apply a watercolor style.'),('edit','Enhance the lighting.'),
                                  ('domain_style','Apply the Naruto domain style.')]:
                state=State(request=request,image_path=image,route=route,caption=details['caption'])
                state.prompt=runtime.rewrite(state)
                result=runtime.generate(state,num_inference_steps=2)
                if len(result.images)!=1 or result.images[0].size!=(512,512):
                    raise RuntimeError('Diffusion preflight returned an unexpected image shape.')
                filtered=bool((result.nsfw_content_detected or [False])[0])
                if not filtered:
                    state.output_path=str(folder/(route+'.png'))
                    result.images[0].save(state.output_path)
                checks.append(dict(check=route,metadata=state.generation_metadata,safety_filtered=filtered))
        finally:
            runtime.vlm.unload()
    record=dict(passed=True,hardware=hardware,checks=checks,seconds=time.perf_counter()-started,
        scope='Real GPU path and cache preflight only. Two-step images are not final samples or task-quality measurements.')
    checkpoint.put('passed',record)
    atomic_json(ROOT/'runtime_checks'/f'{part}_overnight_preflight.json',record)
    print(json.dumps(record,indent=2),flush=True)
    print('PREFLIGHT PASSED on this GPU. Long stages can now use the populated local model cache.',flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',required=True,choices=['part1_rag','part3_agents'])
    parser.add_argument('--data-dir')
    args=parser.parse_args()
    preflight(args.part,args.data_dir)
