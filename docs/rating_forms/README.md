# Rating guide

These are blank forms for real judgments. `sample_formats_DO_NOT_SUBMIT.json`
only shows the file format using invented examples. Do not submit its values as
measurements. Blank fields remain pending; the software does not guess them.

## Part 1: how well is the answer grounded?

Read the actual answer and its cited source. Each team member independently
gives a score from 1 to 5: 1 means unsupported, 2 mostly unsupported, 3 mixed,
4 mostly supported with a small issue, and 5 fully supported. A fluent answer
can still receive a low score if its sources do not support it. Briefly explain
any uncertain score. Do this for both the dense baseline and final system.

Save the completed forms as
`part1_rag/outputs/v2/evaluation/human_ratings_dense_baseline.json` and
`human_ratings_hybrid_multimodal.json` in the same folder. Do not change question
IDs. Both members must also review the proposed golden references and write or
revise their own questions as the assignment requires.

## Part 2: compare the two blinded images

Each of the two team members and one outside rater reviews all 25 pairs
independently. Use the pair's actual prompt. Choose `A`, `B`, or `tie` separately
for domain style and prompt match. These are different questions: an image can
look more like anime while following the prompt less accurately. Use a short
anonymous rater ID consistently. Do not show raters the blinding key or model
names. Do not interpret a black filtered panel as evidence of what the hidden
image contained.

After the three people finish, combine their real ballot lists into
`part2_diffusion/outputs/human_run1/ratings.json`. The merge helper accepts the
three completed files and refuses to overwrite an existing ratings file. Never
merge the sample-format object. The original blinding key stays in the outputs
folder and is not sent with the rater packet.

## Part 3: did the scenario meet its rubric?

Read the request, input, actual answer/output, and the saved rubric. Each member
independently marks `true` for pass or `false` for fail. An automatic `accept`
decision is not a reason to mark pass. Rate both baseline and critique systems.
For a disagreement, discuss the visible evidence, record `resolved_pass`, and
explain the resolution. Do not erase the two original judgments.

Save the completed form as `human_ratings.json` inside the actual Part 3 run
folder printed by the notebook. Then rerun the review/report cells, not image
generation. The report calculates the scores from these actual files.
