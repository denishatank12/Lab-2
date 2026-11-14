"""Measure retrieval alone on the unchanged candidate set, without generating answers.

Reference source IDs are read only after each retrieval call. No expected answer,
number, source ID or question ID is passed into the retrieval pipeline.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'part1_rag/src')]
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from multimodal import MultimodalIndex, retrieval_query
from run_checkpoint import atomic_json, experiment_sources


def run():
    items = json.loads((ROOT / 'part1_rag/eval/golden_candidate.json').read_text())
    index = MultimodalIndex(ROOT / 'part1_rag/data', ROOT / 'output/validation/evidence_review_retrieval')
    rows = []
    try:
        for name, dense_only in [('dense', True), ('hybrid', False)]:
            history = {}
            for item in items:
                turns = history.setdefault(item['conversation_id'], [])
                topic = [{'user': ' '.join(turn['user'] for turn in turns[-3:])}] if turns else []
                query = retrieval_query(item['question'], topic)
                started = time.perf_counter()
                hits = index.search(query, dense_only=dense_only)
                seconds = time.perf_counter() - started
                turns.append({'user': item['question']})
                actual = {r['citation'] for r in hits} | {f"{r['doc_id']}:p{r['page']}" for r in hits}
                wanted = item['sources']
                recall = sum(source in actual for source in wanted) / len(wanted) if wanted else None
                rows.append(dict(system=name, id=item['id'], question=item['question'],
                    source_recall_at_5=recall, expected_sources=wanted, retrieved=sorted(actual),
                    seconds=seconds))
                print(name, item['id'], recall, flush=True)
        summary = {}
        for name in ['dense', 'hybrid']:
            recalls = [r['source_recall_at_5'] for r in rows if r['system'] == name and r['source_recall_at_5'] is not None]
            summary[name] = dict(mean_source_recall_at_5=sum(recalls) / len(recalls),
                                 questions_with_reference_sources=len(recalls))
        result = dict(scope='Real cached CPU retrieval models only. No VLM answers or leaderboard-score claims.',
            sources=experiment_sources('part1_rag'), summary=summary, rows=rows)
        atomic_json(ROOT / 'validation/part1_evidence_review_retrieval.json', result)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        index.db.close()


if __name__ == '__main__':
    run()
