"""Audit measured files, aggregate real ratings, and explicitly publish safe artifacts."""
from __future__ import annotations
import argparse
import csv
import json
import math
import hashlib
from pathlib import Path
import re
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def human_results(folder):
    """Validate independent complete ballots. No judgments are guessed or imputed."""
    path = folder / "ratings.json"
    if not path.is_file():
        return {"complete": False, "reason": "ratings.json not supplied; at least three real raters are required."}
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("ratings.json must contain a list of individual judgments.")
    if any(not isinstance(row,dict) for row in rows):
        raise ValueError("Each rating must be an individual judgment object.")
    if any(row.get('sample_only') is True for row in rows):
        return {"complete":False,"reason":"Sample-only ratings cannot be used as human evaluation."}
    if any(not isinstance(row.get('rater_id'),str) or not row['rater_id'].strip()
           or any(row.get(key) is None for key in ['domain_style','prompt_match']) for row in rows):
        return {"complete":False,"reason":"The rating form is not filled in. Blank fields are not human judgments."}
    key = {r["pair_id"]: r["final_is_left"] for r in read_json(folder / "human_evaluation_key.json", [])}
    if set(key) != set(range(25)):
        raise ValueError("The original 25-pair blinding key is missing or incomplete.")
    roles, seen, counts = {}, set(), {}
    totals = {criterion: {"wins": 0, "losses": 0, "ties": 0} for criterion in ["domain_style", "prompt_match"]}
    for row in rows:
        rater, pair, role = row["rater_id"].strip(), row["pair_id"], row["role"]
        if not rater or type(pair) is not int or pair not in key or role not in ["team_member", "external"]:
            raise ValueError("Each ballot needs a rater ID, pair 0-24 and role team_member or external.")
        if (rater, pair) in seen or (rater in roles and roles[rater] != role):
            raise ValueError("Duplicate ballot or inconsistent rater role.")
        seen.add((rater, pair))
        roles[rater] = role
        counts[rater] = counts.get(rater, 0) + 1
        for criterion in totals:
            choice = row[criterion]
            if choice not in ["A", "B", "tie"]:
                raise ValueError("Judgments must be A, B or tie for each criterion.")
            final_choice = "A" if key[pair] else "B"
            outcome = "ties" if choice == "tie" else "wins" if choice == final_choice else "losses"
            totals[criterion][outcome] += 1
    complete = (len(roles) >= 3 and all(n == 25 for n in counts.values())
                and list(roles.values()).count("team_member") >= 2 and "external" in roles.values())
    if not complete:
        return {"complete": False, "reason": "Need 25 pairs from each of 3+ raters, including two team members and an external rater.",
                "ballots": len(rows), "rater_count": len(roles)}
    for result in totals.values():
        result["win_rate"] = result["wins"] / len(rows)
        result["tie_adjusted_win_rate"] = (result["wins"] + .5 * result["ties"]) / len(rows)
    return {"complete": True, "rater_count": len(roles), "pairs": 25, "ballots": len(rows),
            "definition": "Pooled final-run wins / all judgments; ties reported separately and as half wins in the secondary rate.",
            "criteria": totals}


def audit(root=ROOT):
    config = yaml.safe_load((root / "config.yaml").read_text())
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    summary = read_json(root / "data/processed/data_summary.json", {})
    prompts = read_json(root / "data/processed/eval_prompts.json", [])
    train, test = [], []
    for name, target in [("train", train), ("test", test)]:
        path = root / f"data/processed/{name}/metadata.jsonl"
        if path.exists():
            target.extend(json.loads(line) for line in path.read_text().splitlines())
    hashes = lambda rows: {r["sha256"] for r in rows}
    checks = {"500_plus_images_seeded_disjoint_split": bool(len(train) + len(test) >= 500
              and len(train) == int(.9 * (len(train) + len(test)))
              and len(hashes(train)) == len(train) and len(hashes(test)) == len(test)
              and not hashes(train) & hashes(test) and summary.get("seed") == 266
              and all((root / "data/processed" / name / row["file_name"]).is_file()
                      for name, rows in [("train", train), ("test", test)] for row in rows)),
              "20_fixed_prompts_from_heldout": bool(len(prompts) == 20 and len(test) >= 10
               and [r["prompt"] for r in prompts[:10]] == [r["text"] for r in test[:10]])}
    training = []
    for name in ["run1", "run2"]:
        metric = read_json(output / name / "training_metrics.json", {})
        cfg = read_json(output / name / "run_config.json", {})
        complete = (metric.get("steps") == config["max_train_steps"] and metric.get("gpu_hours") is not None
                    and cfg == {**config, "rank": 4 if name == "run1" else 8}
                    and (output / name / "pytorch_lora_weights.safetensors").is_file())
        checks[name + "_cuda_training"] = complete
        if metric:
            training.append({"model": name, **metric})
    (output / "training_table.json").write_text(json.dumps(training, indent=2))
    if training:
        columns = ["model", "rank", "trainable_parameters", "learning_rate", "steps", "batch_size",
                   "gradient_accumulation_steps", "peak_allocated_mib", "training_seconds", "gpu_hours", "final_loss"]
        with (output / "training_table.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(training)
    manifest = read_json(output / "generations/manifest.json", [])
    expected = {(m, i, seed) for m in ["base", "run1", "run2"] for i in range(20) for seed in [0, 1]}
    actual = {(r["model"], r["prompt_index"], r["seed"]) for r in manifest}
    checks["120_images_same_protocol"] = (len(manifest) == 120 and actual == expected
        and len({r["signature"] for r in manifest}) == 1
        and all((output / "generations" / r["file"]).is_file() for r in manifest))
    checks["8_comparison_grids"] = all((output / "generations" / f"comparison_{i:02d}.png").is_file() for i in range(8))
    evaluation = read_json(output / "evaluation.json", {})
    metrics = evaluation.get("metrics", [])
    checks["kid_clip_lpips_for_all_models"] = (len(metrics) == 3 and {r["model"] for r in metrics} == {"base", "run1", "run2"} and all(
        all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in ["kid_mean", "clip_score_mean", "lpips_seed_pair_mean_256px"])
        for r in metrics))
    review_template = output / "review_template.json"
    if not review_template.exists():
        review_template.write_text(json.dumps({"final_run": None, "selection_reason": "",
            "failure_modes": [], "instructions": "After viewing actual images, save a copy as review.json. Each of 2+ failure modes needs model (run1/run2), prompt_index (0-19), seed (0/1), observation and explanation. Do not call automatically selected low scores confirmed failures."}, indent=2))
    review = read_json(output / "review.json", {})
    final = review.get("final_run")
    checks["final_run_with_explanation"] = final in ["run1", "run2"] and len(review.get("selection_reason", "").strip()) >= 30
    failures = review.get("failure_modes", [])
    valid_failures = []
    for f in failures:
        key = (f.get("model"), f.get("prompt_index"), f.get("seed"))
        if (key in actual and key[0] in ["run1", "run2"]
            and len(f.get("observation", "").strip()) >= 20 and len(f.get("explanation", "").strip()) >= 20):
            valid_failures.append(f)
    checks["2_visually_reviewed_failure_modes"] = len({f["observation"].strip() for f in valid_failures}) >= 2
    human = human_results(output / f"human_{final}") if final in ["run1", "run2"] else {"complete": False, "reason": "Choose final run in review.json after inspecting results."}
    checks["25_blind_pairs_3_real_raters"] = human["complete"]
    memorization = read_json(output / "memorization.json", [])
    checks["memorization_10_images_and_3_pairs"] = bool(final in ["run1", "run2"]
        and len([r for r in memorization if r["model"] == final]) == 10
        and all((output / f"memorization_{final}_{i}.png").is_file() for i in range(3)))
    links = read_json(output / "hub_links.json", {})
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    checks["both_adapters_and_image_dataset_uploaded"] = bool(
        all(links.get(key) for key in ["run1", "run2", "images"])
        and all(links.get(name + "_adapter_sha256") == digest(output / name / "pytorch_lora_weights.safetensors")
                and links.get(name + "_adapter_sha256") for name in ["run1", "run2"])
        and links.get("images_manifest_sha256") == digest(output / "generations/manifest.json")
        and links.get("images_manifest_sha256"))
    checks["artifact_access_for_graders_verified"] = bool(
        checks["both_adapters_and_image_dataset_uploaded"]
        and (links.get("private") is False or links.get("grader_access_verified") is True))
    details = read_json(root / "submission_details.json", {})
    checks["team_details_recorded"] = bool(details.get("partner_name") and details.get("team_number"))
    # Acknowledging rights is a real prerequisite, not a made-up dataset license.
    checks["license_and_redistribution_reviewed"] = bool(links.get("rights_acknowledged"))
    result = {"ready": all(checks.values()), "checks": checks, "pending": [k for k, v in checks.items() if not v],
              "training": training, "data": summary, "evaluation": evaluation, "review": review,
              "human": human, "hub_links": links,
              "scope": "Part 2 only. Common team recordings, combined IEEE report, tag and Canvas zip still require final team verification."}
    (output / "submission_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("\nPart 2 training comparison")
    print("model | rank | trainable | LR | steps | batch | accumulation | peak MiB | seconds | GPU-hours | loss")
    for m in training:
        print(" | ".join(str(m.get(k, "not recorded")) for k in ["model", "rank", "trainable_parameters",
             "learning_rate", "steps", "batch_size", "gradient_accumulation_steps", "peak_allocated_mib", "training_seconds", "gpu_hours", "final_loss"]))
    for label, passed in checks.items():
        print(("PASS: " if passed else "PENDING: ") + label)
    return result


def publish(namespace, public=False, acknowledge=False):
    if not acknowledge:
        raise ValueError("Review redistribution rights and supply --acknowledge-data-rights before uploading.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,95}", namespace):
        raise ValueError("Use your verified Hugging Face username or organization, not a proposed name or URL.")
    result = audit()
    for check in ["run1_cuda_training", "run2_cuda_training", "120_images_same_protocol", "kid_clip_lpips_for_all_models"]:
        if not result["checks"][check]:
            raise RuntimeError("Upload stopped: actual training, images and metrics must be complete first.")
    from huggingface_hub import HfApi
    api = HfApi()
    identity = api.whoami()
    allowed = {identity["name"], *[o["name"] for o in identity.get("orgs", [])]}
    if namespace not in allowed:
        raise ValueError("That namespace does not belong to the logged-in account or its organizations.")
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    output = ROOT / config["output_root"]
    staging = output / "hub_export"
    links = {}
    final = result["review"].get("final_run")
    for name in ["run1", "run2"]:
        source, target = output / name, staging / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ["pytorch_lora_weights.safetensors", "run_config.json", "training_metrics.json"]:
            shutil.copy2(source / filename, target / filename)
        metrics = read_json(source / "training_metrics.json")
        card = f'''---
library_name: diffusers
base_model: {config['base_model']}
pipeline_tag: text-to-image
tags: [lora, stable-diffusion, education]
---
# Lab 2 {name}: Naruto-style LoRA

This is {'the selected final run' if final == name else 'a comparison run; selection is documented in the report'}.
Only attention adapters were trained, with rank {metrics['rank']}. The frozen base model is SD 1.5.
Data: lambdalabs/naruto-blip-captions, 90/10 split with seed 266; held-out captions are evaluation only.
Dataset revision: 1ed13e8ef280bd45e3bbac4cfa8bbd9d64ec9f89.
Base revision: {config['base_revision']}.

## Measured training
{json.dumps(metrics, indent=2)}

## Use
Load the pinned base using StableDiffusionPipeline.from_pretrained, then call
pipe.load_lora_weights on this repository. Preserve the base model safety checker.

## Limitations and rights
The source images are anime frames with automated captions, limited character and pose coverage,
and possible caption errors. The dataset card does not state a license; uploading is not a grant
of copyright or commercial-use rights. Comply with the base model's CreativeML Open RAIL-M
conditions and confirm data permissions. Do not use this adapter for deceptive or harmful images.
The NSFW checker can miss unsafe content. CLIP nearest-neighbor checks are not proof against memorization.

## Evaluation and selection
{result['review'].get('selection_reason') or 'Visual selection and human evaluation are still pending.'}
{json.dumps(result['evaluation'].get('metrics', []), indent=2)}
'''
        (target / "README.md").write_text(card, encoding="utf-8")
        repo_id = f"{namespace}/data266-lab2-p2-{name}"
        api.create_repo(repo_id, repo_type="model", private=not public, exist_ok=True)
        commit = api.upload_folder(repo_id=repo_id, folder_path=target, repo_type="model",
                                  allow_patterns=["README.md", "pytorch_lora_weights.safetensors", "run_config.json", "training_metrics.json"],
                                  commit_message=f"Save measured {name} LoRA experiment")
        links[name] = f"https://huggingface.co/{repo_id}"
        links[name + "_commit"] = commit.oid
        links[name + "_adapter_sha256"] = hashlib.sha256((source / "pytorch_lora_weights.safetensors").read_bytes()).hexdigest()
    target = staging / "images"
    target.mkdir(parents=True, exist_ok=True)
    rows = read_json(output / "generations/manifest.json")
    for row in rows:
        shutil.copy2(output / "generations" / row["file"], target / row["file"])
    (target / "metadata.jsonl").write_text("".join(json.dumps({"file_name": r["file"], **r}) + "\n" for r in rows))
    (target / "README.md").write_text("# Lab 2 controlled generations\n\n120 images: 20 fixed prompts, seeds 0 and 1, base and two LoRA runs.\n"
        "Metadata records model, prompt, category, seed, steps and safety-filter flags.\n"
        "No training images, rater identities or authentication tokens are uploaded.\n"
        "Images may reflect source anime copyrights and model restrictions; no additional rights are granted.\n")
    repo_id = f"{namespace}/data266-lab2-p2-images"
    api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
    commit = api.upload_folder(repo_id=repo_id, folder_path=target, repo_type="dataset",
                              allow_patterns=[*[r["file"] for r in rows], "metadata.jsonl", "README.md"],
                              commit_message="Add controlled image comparisons and generation metadata")
    links.update(images=f"https://huggingface.co/datasets/{repo_id}", images_commit=commit.oid,
                 rights_acknowledged=True, private=not public,
                 images_manifest_sha256=hashlib.sha256((output / "generations/manifest.json").read_bytes()).hexdigest())
    (output / "hub_links.json").write_text(json.dumps(links, indent=2))
    print(json.dumps(links, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--namespace")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--acknowledge-data-rights", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.publish:
        publish(args.namespace or "", args.public, args.acknowledge_data_rights)
    result = audit()
    if args.strict and not result["ready"]:
        raise SystemExit("Part 2 is not submission-ready. Resolve the listed pending items.")


if __name__ == "__main__":
    main()
