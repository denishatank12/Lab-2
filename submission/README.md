# Submission working record

This directory contains the requirement audit, artifact integrity checks, team
identity, cleanup record and Canvas package manifest. It is not a certificate
that the lab is complete.

The combined PDF and `Pair_Programming_Team_09.zip` are built under ignored `output/`
and published as GitHub review-release downloads rather than committed Git files.
The ZIP contains
the report, CONTRIBUTIONS.md, AI_USAGE.md and links.txt. It is a review package
while the audit still has open items.

Downloads: https://github.com/denishatank12/Lab-2/releases/tag/lab2-submission-review

Fictional format examples in `docs/rating_forms/EXAMPLES_ONLY.json` are excluded
from the Canvas ZIP and all measured human scores. Real people still need to
complete the required independent judgments.

To refresh without training or inference:

```bash
python tools/check_submission_evidence.py
python tools/final_submission_audit.py
python tools/build_combined_report.py
```

The original selected Part 1 ZIP is
`Lab2_Part1_Results_20261115T215956Z_866616449.zip`. It was supplied earlier but
is not in the present files. The notebook with the reused name is now v8.
Restore the original archive or executed notebook if available; do not transfer
v8 outputs into v6. The selected notebook remains frozen by SHA-256.

Real rating files belong in `submission/ratings/` with the run identifier and
original judgments retained. Blinded Part 2 ballots also need their private
blinding key for aggregation. Never fill missing ratings automatically.

The required three tags are intentionally not presented as final while the
missing evidence remains open. Repository name and visibility stay unchanged
at the user's explicit request. Prior files can be recovered from the backup
branch recorded in `cleanup.json`; the Git history has not been rewritten.
