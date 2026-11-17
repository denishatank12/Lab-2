"""Ingest + describe the figures in a separate process (two-stage mode on GPU).

When this process exits, every byte of GPU memory used by the vision-language model is released, so the
answering model gets the whole GPU (an in-process unload did not reliably free bitsandbytes 4-bit weights).

    python -m rag.describe_cli <items_out.json>        (settings come from the same environment variables)
"""
import json
import sys
import time

from .config import Config
from .describe import describe_figures
from .ingest import ingest
from .llm import load_describer


def main(out_path: str) -> None:
    cfg = Config()
    t = time.time()
    items = ingest(cfg)
    ingest_s = time.time() - t
    t = time.time()
    if cfg.describe_figures:
        describe_figures(items, load_describer(cfg))
    describe_s = time.time() - t
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"items": [it.to_dict() for it in items], "ingest_s": ingest_s, "describe_s": describe_s}, fh)
    print(f"[describe_cli] {len(items)} items, ingest {ingest_s:.0f}s, describe {describe_s:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
