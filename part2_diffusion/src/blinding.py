"""Prepare blinded packets without loading models or recomputing expensive metrics."""
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw


def prepare_blind_pairs(output, rows, final_run, seed=266):
    output = Path(output)
    generated = output / "generations"
    folder = output / f"human_{final_run}"
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    candidates = [(i, sample_seed) for i in range(20) for sample_seed in [0, 1]]
    rng.shuffle(candidates)
    layout = [(i, prompt_index, sample_seed, bool(rng.getrandbits(1)))
              for i, (prompt_index, sample_seed) in enumerate(candidates[:25])]
    key = [{"pair_id": i, "final_is_left": left} for i, _, _, left in layout]
    key_path = folder / "human_evaluation_key.json"
    if key_path.exists() and json.loads(key_path.read_text()) != key:
        raise RuntimeError("Blinding key changed; preserve existing judgments and use a fresh output folder.")
    blinded = []
    for index, prompt_index, sample_seed, left_final in layout:
        left = f"{final_run if left_final else 'base'}_{prompt_index:02d}_{sample_seed}.png"
        right = f"{'base' if left_final else final_run}_{prompt_index:02d}_{sample_seed}.png"
        canvas = Image.new("RGB", (1024, 544), "white")
        ImageDraw.Draw(canvas).text((5, 5), "A (left) | B (right)", fill="black")
        for x, filename in [(0, left), (512, right)]:
            with Image.open(generated / filename) as image:
                canvas.paste(image.convert("RGB").resize((512, 512)), (x, 32))
        filename = f"blinded_pair_{index:02d}.png"
        canvas.save(folder / filename)
        prompt = next(r["prompt"] for r in rows if r["prompt_index"] == prompt_index)
        blinded.append({"pair_id": index, "image": filename, "prompt": prompt, "ratings": []})
    template = folder / "human_evaluation_template.json"
    if not template.exists():
        template.write_text(json.dumps(blinded, indent=2))
    key_path.write_text(json.dumps(key, indent=2))
    return folder
