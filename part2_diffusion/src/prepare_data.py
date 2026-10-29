from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from datasets import load_dataset
from PIL import Image

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output-dir', default='data/processed'); ap.add_argument('--seed', type=int, default=266)
    args = ap.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('lambdalabs/naruto-blip-captions', split='train')
    good, seen, removed_corrupt, removed_duplicate = [], set(), 0, 0
    for row in ds:
        try:
            image = row['image'].convert('RGB')
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen: removed_duplicate += 1; continue
            seen.add(digest); good.append({'image': image, 'text': str(row.get('text', '')).strip()})
        except Exception:
            removed_corrupt += 1
    if len(good) < 500: raise RuntimeError(f'Expected at least 500 valid images, found {len(good)}')
    split = int(len(good) * 0.90)
    train, test = good[:split], good[split:]
    for name, rows in [('train', train), ('test', test)]:
        folder = out / name; folder.mkdir(exist_ok=True)
        with (folder / 'metadata.jsonl').open('w', encoding='utf-8') as f:
            for i, row in enumerate(rows):
                filename = f'{i:06d}.png'; row['image'].resize((512, 512)).save(folder / filename)
                f.write(json.dumps({'file_name': filename, 'text': row['text']}) + '\n')
    (out / 'data_summary.json').write_text(json.dumps({'dataset': 'lambdalabs/naruto-blip-captions', 'seed': args.seed, 'train': len(train), 'test': len(test), 'removed_corrupt': removed_corrupt, 'removed_duplicate': removed_duplicate}, indent=2))
    print((out / 'data_summary.json').read_text())
if __name__ == '__main__': main()
