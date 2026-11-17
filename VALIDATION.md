# Validation scope

The submission cleanup passed 20 Part 2 regression tests and 21 Part 3 baseline
tests. They cover safe generation recovery, real-rating validation, result ZIP
import, environment isolation, adapter access, bounded baseline routing,
fail-closed safety and explicit mock notebook execution. Fixtures are software
tests, not trained-model quality measurements.

The November 17 recheck passed the same 41 tests. Additional direct checks
confirmed that the fictional rating guide is rejected by the real ballot merge
and Part 1 groundedness helpers. The examples are excluded from the Canvas ZIP
and are not used in human success, agreement or win-rate measurements.

`python tools/check_submission_evidence.py` also checks:

* The selected v6 notebook and its 34 embedded files match their recorded hashes.
* Five canonical notebooks pass notebook-schema and IPython-aware syntax checks.
* The completed Part 2, Denisha baseline and partner notebooks have no saved cell errors.
* The imported Part 3 records match the original ZIP bytes, with 25 states and
  25 traces. The baseline uses no revision loop.
* No Hugging Face credential pattern appears in the imported evidence.

Denisha's **real** Part 3 baseline completed on a single Tesla T4. This is
separate from the CPU smoke tests. It had 15/15 automatic scenario checks,
0/10 red-team attacks and median request time 39.665 seconds. Automatic checks
do not replace human quality ratings. The original nonfatal host warning is
preserved in the log.

The original Part 2 measured training ran on RTX 4090. Its genuine results are
unchanged. The selected Part 1 v6 source is also unchanged, but the original
selected execution files still need restoring. The available later v8 upload
is not used as its experimental evidence.

No new full GPU run was performed during this cleanup on the Mac. Human
ratings, recording links, access permissions and external submissions are not
inferred from successful tests. The final requirement audit remains **not ready**;
see `SUBMISSION_STATUS.md` for the actual open items.
