from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag_pipeline import build_index


def main() -> None:
    data = ROOT / "data"
    index = build_index(data)
    assert len(index.passages) > 20, "PDF ingestion returned too few passages"
    hits = index.search("What happened to unemployment after the crisis?", k=3)
    assert len(hits) == 3
    assert all("passage" in h and h["passage"]["doc_id"] in {"DOC1", "DOC2", "DOC3"} for h in hits)
    print(f"PASS: indexed {len(index.passages)} passages; retrieval returned 3 grounded hits")


if __name__ == "__main__":
    main()
