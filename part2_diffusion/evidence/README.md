# Provenance of the Part 2 result review

`saved_gpu_run.json` records only evidence embedded in the supplied GPU notebook.
Its missing-artifact list describes what could not be recovered from that
notebook alone, not the later complete ZIP import.

`current_result_audit.json` is the later audit of the imported original files.
`result_bundle_import.json` records the original ZIP digest and import scope.
`visual_review.json` contains AI-assisted observations, not human-rating ballots.
`hub_links.json` records the three verified private uploads and exact commit IDs.

The canonical notebook preserves all original GPU cells and outputs and appends
a separately executed CPU review. No GPU experiment or human judgment was
invented. The complete ZIP stays in the user's Downloads directory. Adapter
weights and individual generated images are stored on Hugging Face, not copied
into Git history; the executed notebook includes the original comparison figures.

The report remains a draft until real ratings, team details, permitted sharing
and grader access are resolved. Common lab deliverables and the single-16-GB-GPU
target also need final verification. A 24-GB RTX 4090 run is not evidence of that
specific hardware requirement.
