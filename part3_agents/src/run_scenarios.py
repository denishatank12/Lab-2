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
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='outputs/scenario_traces.json'); ap.add_argument('--enable-models',action='store_true'); ap.add_argument('--image',default=None); ap.add_argument('--lora-path',default=None); a=ap.parse_args()
    vision_pipe = generation_pipe = None
    if a.enable_models:
        import torch
        from diffusers import StableDiffusionPipeline
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        generation_pipe=StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5',torch_dtype=dtype,safety_checker=None).to('cuda' if torch.cuda.is_available() else 'cpu')
        generation_pipe.enable_attention_slicing()
        if a.lora_path: generation_pipe.load_lora_weights(a.lora_path)
    states=[run(q,image_path=a.image,generation_pipe=generation_pipe,output_dir=Path(a.output).parent/'generated') for _,q in SCENARIOS]
    save_trace(states,Path(a.output)); print(json.dumps({'scenarios':len(states),'safe_refusals':sum(x.decision=='refused' for x in states),'generated':sum(bool(x.output_path and x.output_path.endswith('.png')) for x in states),'output':a.output},indent=2))
if __name__=='__main__': main()
