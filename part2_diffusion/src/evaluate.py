"""Compute real KID, CLIP alignment and LPIPS; never write placeholder scores."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor, resize
import yaml
from blinding import prepare_blind_pairs


def load_rgb(path, size=299):
    with Image.open(path) as im:
        return resize(pil_to_tensor(im.convert("RGB")), [size, size], antialias=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--final-run", choices=["run1", "run2"], default="run1")
    parser.add_argument("--human-only", action="store_true", help="Prepare a new blinding packet without recomputing metrics.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text())
    output = root / config["output_root"]
    generated = output / "generations"
    rows = json.loads((generated / "manifest.json").read_text())
    assert len(rows) == 120 and len({(r["model"], r["prompt_index"], r["seed"]) for r in rows}) == 120
    assert all((generated / r["file"]).exists() for r in rows)
    if args.human_only:
        print("Prepared blinded pairs:", prepare_blind_pairs(output, rows, args.final_run, config["seed"]))
        return
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from transformers import CLIPModel, CLIPProcessor
    data = root / "data/processed"
    test = [json.loads(line) for line in (data / "test/metadata.jsonl").read_text().splitlines()]
    train = [json.loads(line) for line in (data / "train/metadata.jsonl").read_text().splitlines()]
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    scores = []
    for name in ["base", "run1", "run2"]:
        torch.manual_seed(config["seed"])
        selected = [r for r in rows if r["model"] == name and r["category"] == "in_domain"]
        subset = min(20, len(selected), len(test))
        kid = KernelInceptionDistance(subset_size=subset, subsets=20, normalize=False).cpu()
        for offset in range(0, len(test), 4):
            kid.update(torch.stack([load_rgb(data / "test" / r["file_name"]) for r in test[offset:offset + 4]]), real=True)
        for offset in range(0, len(selected), 4):
            kid.update(torch.stack([load_rgb(generated / r["file"]) for r in selected[offset:offset + 4]]), real=False)
        mean, std = kid.compute()
        scores.append({"model": name, "kid_mean": mean.item(), "kid_std": std.item(),
                       "kid_generated_n": len(selected), "kid_test_n": len(test), "kid_subset_size": subset,
                       "nsfw_filtered": sum(r["nsfw_flagged"] for r in rows if r["model"] == name)})
        del kid
        gc.collect()
        print(f"Computed KID for {name}", flush=True)
    clip_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip = CLIPModel.from_pretrained(clip_name).cpu().eval()
    for parameter in clip.parameters():
        parameter.data = parameter.data.clone()
    image_features = {}
    per_image = []
    with torch.inference_mode():
        for model_score in scores:
            selected = [r for r in rows if r["model"] == model_score["model"]]
            alignments = []
            for row in selected:
                with Image.open(generated / row["file"]) as im:
                    inputs = processor(text=[row["prompt"]], images=im.convert("RGB"),
                                       return_tensors="pt", padding=True, truncation=True, max_length=77)
                result = clip(**inputs)
                cosine = (result.image_embeds * result.text_embeds).sum().item()
                alignments.append(max(0.0, cosine) * 100)
                per_image.append({**row, "clip_score": max(0.0, cosine) * 100})
                image_features[row["file"]] = result.image_embeds[0]
            model_score["clip_score_mean"] = float(np.mean(alignments))
        training_features = []
        for row in train:
            with Image.open(data / "train" / row["file_name"]) as im:
                values = processor(images=im.convert("RGB"), return_tensors="pt")
            features = clip.get_image_features(**values)
            training_features.append(F_normalize(features)[0])
        training_features = torch.stack(training_features)
    memorization = []
    for name in ["run1", "run2"]:
        selected = [r for r in rows if r["model"] == name and r["category"] == "in_domain" and r["seed"] == 0]
        for row in selected:
            similarities = training_features @ image_features[row["file"]]
            score, index = similarities.max(dim=0)
            memorization.append({"model": name, "generated_file": row["file"],
                                 "training_file": train[index.item()]["file_name"], "clip_cosine": score.item()})
        closest = sorted([r for r in memorization if r["model"] == name], key=lambda r: -r["clip_cosine"])[:3]
        for number, pair in enumerate(closest):
            grid = Image.new("RGB", (1024, 544), "white")
            ImageDraw.Draw(grid).text((5, 5), f"Generated | nearest training image, CLIP cosine {pair['clip_cosine']:.4f}", fill="black")
            for x, path in [(0, generated / pair["generated_file"]), (512, data / "train" / pair["training_file"])]:
                with Image.open(path) as im:
                    grid.paste(im.convert("RGB").resize((512, 512)), (x, 32))
            grid.save(output / f"memorization_{name}_{number}.png")
    del clip, processor, training_features
    gc.collect()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cpu().eval()
    for score in scores:
        selected = [r for r in rows if r["model"] == score["model"]]
        distances = []
        for index in range(20):
            pair = sorted([r for r in selected if r["prompt_index"] == index], key=lambda r: r["seed"])
            with torch.inference_mode():
                distance = lpips(load_rgb(generated / pair[0]["file"], 256).unsqueeze(0).float() / 255,
                                 load_rgb(generated / pair[1]["file"], 256).unsqueeze(0).float() / 255)
            distances.append(distance.item())
        score["lpips_seed_pair_mean_256px"] = float(np.mean(distances))
        for key, value in score.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise RuntimeError(f"Nonfinite {key}; evaluation is not complete.")
    human_dir = prepare_blind_pairs(output, rows, args.final_run, config["seed"])
    (output / "clip_scores_per_image.json").write_text(json.dumps(per_image, indent=2))
    candidates = sorted([r for r in per_image if r["model"] != "base" and r["category"] != "in_domain"],
                        key=lambda r: r["clip_score"])[:8]
    (output / "review_candidates.json").write_text(json.dumps({
        "note": "Low alignment is a candidate to inspect, not a confirmed failure. Add real observations to review.json.",
        "candidates": candidates}, indent=2))
    (output / "memorization.json").write_text(json.dumps(memorization, indent=2))
    result = {"generated_images": 120, "metrics": scores, "clip_model": clip_name,
              "sampling": rows[0].get("sampling", {"steps": rows[0]["steps"], "scheduler": "legacy checkpoint default"}),
              "human_evaluation": "Pending: collect 25 pair judgments from at least three real raters.",
              'human_evaluation_folder':str(human_dir),
              "kid_note": "Only 20 in-domain generations per model; KID is noisy and may be negative."}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


def F_normalize(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


if __name__ == "__main__":
    main()
