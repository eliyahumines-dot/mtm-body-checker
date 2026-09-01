# sam3d_mhr_clad_smoke

Smoke test for the pipeline:

```
image -> SAM 3D Body -> MHR params -> clad-body -> anthropometric measurements
```

This is an installability/interoperability/numerical-sanity test, not an
accuracy benchmark. Full findings:
`docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md`.

## What's here

- `adapter.py` — converts a SAM 3D Body `process_one_image()` output dict
  into the params dict clad-body's MHR loader expects. Our own code; not a
  reimplementation of either upstream project. See its docstring for the
  `scale_params` (28-dim PCA) vs. `mhr_model_params` (204-dim, already
  decoded) trap this exists to avoid.
- `rescale.py` — optional, explicit, deterministic uniform rescale of a
  recovered mesh to a known customer height. Off by default; only applied
  if `--known-height-cm` is passed to `run.py`.
- `mtm_mapping.py` — maps clad-body/ISO 8559-1 measurement keys to MTM
  tailoring terminology, with explicit per-measurement notes on where the
  definitions are known (or suspected, or unverified) to diverge.
- `_mhr_measure_worker.py` — runs clad-body's MHR load + measure in an
  isolated subprocess, so a native crash there (observed in this task's
  environment, see the findings doc) doesn't take down `run.py`.
- `run.py` — the CLI entry point.
- `tests/` — tests for the four files above only. No test touches SAM 3D
  Body or clad-body internals directly, and all tests run under a plain
  Python install with just `pytest`+`numpy` (no torch, no clad-body, no
  checkpoint download required).

## Running the tests

```bash
pip install pytest numpy
python3 -m pytest tests/ -v
```

## Running the CLI

Requires a Python >=3.12 environment with `clad-body[mhr]` installed (see
the findings doc for exact install steps and the native-crash blocker
encountered installing/running that path in this task's environment).

```bash
# Measure an existing SAM3D-style params JSON directly (e.g. one of
# clad-body's own bundled test fixtures) -- does not require SAM 3D Body
# or its gated checkpoint at all:
python3 run.py --mhr-params-json <path/to/mhr_params.json> \
    --known-height-cm 175 --output result.json

# Full pipeline from an image (requires the sam_3d_body package AND a
# manually-downloaded, license-gated checkpoint):
python3 run.py --image <path/to/photo.jpg> \
    --sam3d-checkpoint <path/to/model.ckpt> \
    --known-height-cm 175 --output result.json
```

The output JSON always contains a status field per pipeline stage
(`sam3d_inference_status`, `measurement_extraction_status`) so a blocked
stage is reported explicitly rather than crashing the CLI or silently
producing an empty result.
