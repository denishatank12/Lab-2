"""Real tiny-UNet forward/backward, LoRA update, save and interrupted-run resume."""
from pathlib import Path
import argparse
import tempfile
import torch
import yaml
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPTextConfig, CLIPTextModel
from peft.utils import get_peft_model_state_dict
from prepare_data import split_rows
from train_lora import attach_lora, train_updates

ROOT = Path(__file__).resolve().parents[1]


def tiny_model(device, adapter=True):
    torch.manual_seed(266)
    model = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        layers_per_block=1, block_out_channels=(16, 32),
        down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),
        up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),
        cross_attention_dim=16, attention_head_dim=4, norm_num_groups=4)
    model.to(device=device, dtype=torch.float16 if device=='cuda' else torch.float32)
    if adapter:
        attach_lora(model, 2)
    return model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-cuda',action='store_true')
    args=parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA smoke test requested, but this interpreter has no working CUDA device.')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text())
    assert cfg['seed']==266 and cfg['resolution']==512 and cfg['train_batch_size']==1
    assert cfg['gradient_checkpointing'] is True and not cfg['use_8bit_adam']
    rows=list(range(100))
    a,b=split_rows(rows)
    assert len(a)==90 and len(b)==10 and not(set(a)&set(b))
    assert (a,b)==split_rows(rows) and (a,b)!=split_rows(rows,267)
    cfg.update(max_train_steps=2,checkpointing_steps=1,gradient_accumulation_steps=2)
    model=tiny_model(device)
    frozen=next(p for p in model.parameters() if not p.requires_grad)
    original=frozen.detach().clone()
    latents=torch.randn(4,4,8,8)
    hidden=torch.randn(4,8,16)
    scheduler=DDPMScheduler(num_train_timesteps=20)
    with tempfile.TemporaryDirectory(prefix='lab2-lora-smoke-') as temporary:
        first=train_updates(model,latents,hidden,scheduler,temporary,cfg,2,stop_after=1)
        assert first['steps']==1 and torch.equal(original,frozen)
        del model
        model=tiny_model(device)
        second=train_updates(model,latents,hidden,scheduler,temporary,cfg,2,resume=True)
        assert second['steps']==2 and torch.equal(original,next(p for p in model.parameters() if not p.requires_grad))
        assert (Path(temporary)/'pytorch_lora_weights.safetensors').exists()
        assert all(torch.isfinite(p).all() for p in get_peft_model_state_dict(model).values())
        expected={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        reference=tiny_model(device)
        train_updates(reference,latents,hidden,scheduler,Path(temporary)/'reference',cfg,2)
        reference_state=get_peft_model_state_dict(reference)
        assert all(torch.allclose(value,reference_state[key].float().cpu(),atol=1e-4,rtol=1e-3)
                   for key,value in expected.items()), 'Resumed adapter differs from uninterrupted training.'
        del reference
        # This random miniature is an integration fixture, not a public image model.
        vae=AutoencoderKL(in_channels=3,out_channels=3,latent_channels=4,norm_num_groups=4,
            block_out_channels=(16,32),down_block_types=('DownEncoderBlock2D','DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D','UpDecoderBlock2D'))
        text=CLIPTextModel(CLIPTextConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                                        num_hidden_layers=1,num_attention_heads=4))
        pipe=StableDiffusionPipeline(unet=tiny_model(device,adapter=False),vae=vae,text_encoder=text,
            tokenizer=None,scheduler=DDIMScheduler(num_train_timesteps=20,clip_sample=False,steps_offset=1),
            safety_checker=None,feature_extractor=None,requires_safety_checker=False)
        pipe.to(device=device,dtype=torch.float16 if device=='cuda' else torch.float32)
        pipe.load_lora_weights(temporary,adapter_name='roundtrip')
        pipe.disable_lora()
        pipe.enable_lora()
        loaded=get_peft_model_state_dict(pipe.unet,adapter_name='roundtrip')
        assert set(expected)==set(loaded)
        assert all(torch.allclose(value,loaded[key].float().cpu(),atol=1e-4) for key,value in expected.items())
        pipe.set_progress_bar_config(disable=True)
        image=pipe(prompt_embeds=hidden[:1].to(device=device,dtype=pipe.unet.dtype),
                   negative_prompt_embeds=torch.zeros_like(hidden[:1]).to(device=device,dtype=pipe.unet.dtype),
                   num_inference_steps=2,height=16,width=16,
                   generator=torch.Generator('cpu').manual_seed(266)).images[0]
        assert image.size==(16,16)
    print(f'PASS: {device} LoRA training, frozen base, save/resume, adapter round-trip and tiny diffusion generation')


if __name__=='__main__':
    main()
