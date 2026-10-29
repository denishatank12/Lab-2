# Lab 2 Part 3 - Multimodal multi-agent system

The implementation uses plain typed Python functions so it is easy to audit:
Planner -> Vision -> Prompt -> Generation -> Critique, with at most two revisions.
Unsafe, deceptive, image-instruction, and broken-input cases are refused before
generation. The optional model calls use BLIP for vision and Diffusers for image
generation; the smoke test does not download models.

```bash
cd part3_agents
python -m pip install -r requirements.txt
python src/smoke_test.py
python src/run_scenarios.py --output outputs/scenario_traces.json
python src/app.py
```

Set `--enable-models` only on the GPU machine. Every request writes a JSON trace
with agent inputs, outputs, status, revisions, and timings.
