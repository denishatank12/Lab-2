"""Aggregate only supplied real ratings; empty templates never become scores."""
from pathlib import Path
import json
import statistics


def groundedness(path, expected_ids):
    path=Path(path)
    if not path.is_file():
        return dict(complete=False,mean_groundedness=None,agreement_within_one=None)
    rows=json.loads(path.read_text())
    if len(rows)!=len(expected_ids) or {r['id'] for r in rows}!=set(expected_ids):
        raise ValueError('Need exactly one rating row for every golden question.')
    for r in rows:
        if any(type(r.get(k)) is not int or not 1<=r[k]<=5 for k in ['member_a','member_b']):
            return dict(complete=False,mean_groundedness=None,agreement_within_one=None)
    return dict(complete=True,mean_groundedness=statistics.mean(r[k] for r in rows for k in ['member_a','member_b']),
        agreement_within_one=statistics.mean(abs(r['member_a']-r['member_b'])<=1 for r in rows))
