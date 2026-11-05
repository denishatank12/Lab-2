"""Prepare a seeded split; metadata is the completion marker, not a folder's existence."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import tempfile

DATASET_REVISION = '1ed13e8ef280bd45e3bbac4cfa8bbd9d64ec9f89'


def split_rows(rows, seed=266, fraction=.1):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    split = int(len(rows) * (1 - fraction))
    return rows[:split], rows[split:]


def prepared(out):
    out = Path(out)
    if not (out / "data_summary.json").exists():
        return False
    summary = json.loads((out / "data_summary.json").read_text())
    if summary.get("schema_version") != 2 or summary.get("dataset_revision") != DATASET_REVISION:
        return False
    for split in ["train", "test"]:
        meta = out / split / "metadata.jsonl"
        if not meta.exists():
            return False
        rows = [json.loads(line) for line in meta.read_text().splitlines()]
        if len(rows) != summary[split] or not all((out / split / r["file_name"]).is_file() for r in rows):
            return False
    return True


def main():
    from datasets import load_dataset, Image as HFImage
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=266)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if prepared(out):
        print((out / "data_summary.json").read_text())
        return
    dataset = load_dataset("lambdalabs/naruto-blip-captions", revision=DATASET_REVISION,
                           split="train").cast_column("image", HFImage(decode=False))
    good, seen = [], set()
    removed = {"corrupt": 0, "duplicate": 0, "empty_caption": 0}
    with tempfile.TemporaryDirectory(prefix="preparing-", dir=out) as temporary:
        temporary = Path(temporary)
        for index, row in enumerate(dataset):
            try:
                value = row["image"]
                stream = io.BytesIO(value["bytes"]) if value.get("bytes") is not None else value["path"]
                with Image.open(stream) as im:
                    image = im.convert("RGB")
                    image.load()
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            except (OSError, ValueError, TypeError):
                removed["corrupt"] += 1
                continue
            if digest in seen:
                removed["duplicate"] += 1
                continue
            caption = str(row.get("text", "")).strip()
            if not caption:
                removed["empty_caption"] += 1
                continue
            seen.add(digest)
            filename = digest + ".png"
            cached = None
            for name in ['train','test']:
                candidate = out/name/filename
                if candidate.is_file():
                    try:
                        with Image.open(candidate) as saved:
                            if saved.size == (512,512) and saved.mode == 'RGB':
                                saved.verify()
                                cached = candidate
                                break
                    except OSError:
                        pass
            if cached is not None:
                shutil.copyfile(cached, temporary/filename)
            else:
                image.resize((512, 512), Image.Resampling.LANCZOS).save(temporary / filename)
            good.append({"file_name": filename, "text": caption, "source_index": index, "sha256": digest})
            if (index + 1) % 100 == 0:
                print(f'Checked image {index + 1}/{len(dataset)}', flush=True)
        if len(good) < 500:
            raise RuntimeError(f"Only {len(good)} valid unique captioned images; at least 500 are required.")
        train, test = split_rows(good, args.seed)
        for name, rows in [("train", train), ("test", test)]:
            folder = out / name
            folder.mkdir(exist_ok=True)
            for row in rows:
                shutil.copyfile(temporary / row["file_name"], folder / row["file_name"])
            (folder / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (out / "data_summary.json").write_text(json.dumps({"schema_version": 2,
            "dataset": "lambdalabs/naruto-blip-captions", "seed": args.seed,
            "dataset_revision": DATASET_REVISION,
            "train": len(train), "test": len(test), "removed": removed,
            "train_sha256": hashlib.sha256(json.dumps(train, sort_keys=True).encode()).hexdigest()}, indent=2))
    assert prepared(out)
    print((out / "data_summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
