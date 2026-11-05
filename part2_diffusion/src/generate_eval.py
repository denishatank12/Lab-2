from __future__ import annotations
import argparse, gc, hashlib, json
from pathlib import Path
import yaml

PROMPTS = [
    'a young ninja warrior in a forest', 'a small ninja village at sunset', 'a warrior standing beside a waterfall', 'a girl with dark hair and a red scarf', 'a masked character in a snowy mountain', 'a character reading beside a river', 'a ninja running through autumn leaves', 'a character looking at the moon', 'a warrior in a bamboo forest', 'a group of characters near a village gate',
    'a Victorian steam train in a city', 'a bowl of fruit on a kitchen table', 'a lighthouse beside the ocean', 'a robot in a laboratory', 'a cozy cabin in the snow',
    'two characters, one in red and one in blue, standing together', 'a small character holding a lantern beside a large dragon', 'three warriors crossing a bridge in rain', 'a character riding a bicycle through a flower field', 'two friends sharing tea under a tree'
]
ROOT = Path(__file__).resolve().parents[1]


def freeze_prompts():
    path = ROOT / "data/processed/eval_prompts.json"
    metadata = ROOT / "data/processed/test/metadata.jsonl"
    test = [json.loads(line) for line in metadata.read_text().splitlines()]
    if path.exists():
        rows = json.loads(path.read_text())
        assert len(rows) == 20
        if [r['prompt'] for r in rows[:10]] != [r['text'] for r in test[:10]]:
            raise RuntimeError('Frozen prompts refer to a different test split. Preserve the old experiment and use a fresh data folder.')
        return rows
    if len(test) < 10:
        raise RuntimeError("The held-out split needs at least ten captioned images.")
    rows = [{"prompt": r["text"], "category": "in_domain"} for r in test[:10]]
    rows += [{"prompt": p, "category": "out_of_domain"} for p in PROMPTS[10:15]]
    rows += [{"prompt": p, "category": "compositional"} for p in PROMPTS[15:]]
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--freeze-prompts", action="store_true")
    args = parser.parse_args()
    prompts = freeze_prompts()
    if args.freeze_prompts:
        print("Frozen 10 held-out captions, 5 out-of-domain and 5 compositional prompts.")
        return
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image, ImageDraw
    if not torch.cuda.is_available():
        raise RuntimeError("Full generation requires CUDA. The offline smoke test does not download SD 1.5.")
    config = yaml.safe_load((ROOT / args.config).read_text())
    out = ROOT / config["output_root"] / "generations"
    out.mkdir(parents=True, exist_ok=True)
    models = {"base": None, "run1": ROOT / config["output_root"] / "run1",
              "run2": ROOT / config["output_root"] / "run2"}
    for name, adapter in models.items():
        if adapter is not None and not (adapter / "pytorch_lora_weights.safetensors").is_file():
            raise FileNotFoundError(f"Missing {name} adapter. Base weights will NOT be mislabeled as a trained run.")
    adapter_hashes = {name: hashlib.sha256((folder/'pytorch_lora_weights.safetensors').read_bytes()).hexdigest()
                     for name,folder in models.items() if folder is not None}
    signature = hashlib.sha256(json.dumps({"prompts": prompts, "steps": args.steps,
                              "base": config["base_model"], 'revision':config['base_revision'],
                              'adapter_sha256':adapter_hashes}, sort_keys=True).encode()).hexdigest()
    record = out / "manifest.json"
    previous = json.loads(record.read_text()) if record.exists() else []
    if previous and any(r.get("signature") != signature for r in previous):
        raise RuntimeError("Generation settings changed. Use a fresh output_root rather than mix outputs.")
    manifest = {(r["model"], r["prompt_index"], r["seed"]): r for r in previous}
    for name, adapter in models.items():
        if all((name, i, seed) in manifest and (out / manifest[(name, i, seed)]["file"]).exists()
               for i in range(20) for seed in [0, 1]):
            print(f"{name}: all 40 images already complete.")
            continue
        pipe = StableDiffusionPipeline.from_pretrained(config["base_model"], torch_dtype=torch.float16,
                revision=config['base_revision'],variant=config['variant'])
        # Do not disable the checkpoint's NSFW checker. Offload components instead.
        if pipe.safety_checker is None:
            raise RuntimeError("A safety checker is required for this generation run.")
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=True)
        if adapter is not None:
            pipe.load_lora_weights(str(adapter))
        for index, item in enumerate(prompts):
            for seed in [0, 1]:
                key = (name, index, seed)
                if key in manifest and (out / manifest[key]["file"]).exists():
                    continue
                result = pipe(item["prompt"], num_inference_steps=args.steps,
                              generator=torch.Generator("cpu").manual_seed(seed))
                filename = f"{name}_{index:02d}_{seed}.png"
                result.images[0].save(out / filename)
                manifest[key] = {"model": name, "prompt_index": index, **item, "seed": seed,
                                 "file": filename, "steps": args.steps, "signature": signature,
                                 "nsfw_flagged": bool((result.nsfw_content_detected or [False])[0])}
                record.write_text(json.dumps(list(manifest.values()), indent=2), encoding="utf-8")
                print(f"Generated {name} prompt {index + 1}/20 seed {seed}", flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    for index in range(8):
        grid = Image.new("RGB", (512 * 3, 512 + 32), "white")
        draw = ImageDraw.Draw(grid)
        for column, name in enumerate(models):
            draw.text((column * 512 + 10, 8), name, fill="black")
            with Image.open(out / manifest[(name, index, 0)]["file"]) as image:
                grid.paste(image, (column * 512, 32))
        grid.save(out / f"comparison_{index:02d}.png")
    print("Completed 120 images and eight comparison grids.", flush=True)


if __name__ == "__main__":
    main()
