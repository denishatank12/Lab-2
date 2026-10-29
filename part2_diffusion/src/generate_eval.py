from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml

PROMPTS = [
    'a young ninja warrior in a forest', 'a small ninja village at sunset', 'a warrior standing beside a waterfall', 'a girl with dark hair and a red scarf', 'a masked character in a snowy mountain', 'a character reading beside a river', 'a ninja running through autumn leaves', 'a character looking at the moon', 'a warrior in a bamboo forest', 'a group of characters near a village gate',
    'a Victorian steam train in a city', 'a bowl of fruit on a kitchen table', 'a lighthouse beside the ocean', 'a robot in a laboratory', 'a cozy cabin in the snow',
    'two characters, one in red and one in blue, standing together', 'a small character holding a lantern beside a large dragon', 'three warriors crossing a bridge in rain', 'a character riding a bicycle through a flower field', 'two friends sharing tea under a tree'
]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); ap.add_argument('--steps',type=int,default=20); a=ap.parse_args(); root=Path(__file__).resolve().parents[1]; c=yaml.safe_load((root/a.config).read_text()); out=root/'outputs'/'generations'; out.mkdir(parents=True,exist_ok=True)
    from diffusers import StableDiffusionPipeline
    import torch
    device='cuda' if torch.cuda.is_available() else 'cpu'; dtype=torch.float16 if device=='cuda' else torch.float32
    models={'base':None,'run1':root/c['output_root']/'run1','run2':root/c['output_root']/'run2'}; manifest=[]
    for name, adapter in models.items():
        pipe=StableDiffusionPipeline.from_pretrained(c['base_model'], torch_dtype=dtype, safety_checker=None).to(device); pipe.enable_attention_slicing()
        if adapter and adapter.exists(): pipe.load_lora_weights(str(adapter))
        for pi,prompt in enumerate(PROMPTS):
            for seed in (0,1):
                gen=torch.Generator(device=device).manual_seed(seed); image=pipe(prompt,num_inference_steps=a.steps,generator=gen).images[0]; fn=f'{name}_{pi:02d}_{seed}.png'; image.save(out/fn); manifest.append({'model':name,'prompt':prompt,'seed':seed,'file':fn})
        del pipe; torch.cuda.empty_cache()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(f'generated {len(manifest)} images in {out}')
if __name__=='__main__': main()
