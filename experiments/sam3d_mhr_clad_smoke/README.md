# sam3d_mhr_clad_smoke

Smoke test for the pipeline:

```
image -> SAM 3D Body -> MHR params -> clad-body -> anthropometric measurements
```

This is an installability/interoperability/numerical-sanity test, not an
accuracy benchmark. Findings from running the CLI-only path (no GPU, no
checkpoint access): `docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md`.
Findings from the real-GPU/real-checkpoint Colab path (Task 03) and the
resulting dependency-architecture fix (Task 03B, two fully isolated
environments exchanging one small interchange file):
`docs/experiments/TASK03_COLAB_END_TO_END_SMOKE_TEST.md`,
`docs/experiments/TASK03B_DEPENDENCY_RESOLUTION.md`, and
`notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` (that notebook reuses every
module in this directory — see below).

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
  isolated subprocess, so a native crash there (observed in Task 02's
  environment) doesn't take down `run.py`. Can be pointed at a *different*
  Python interpreter (see `run.py`'s `--clad-python-executable`), so it can
  run against a separately-configured venv (e.g. a CPU-torch venv matching
  `pymomentum-cpu`'s expected ABI, as Task 03's Colab notebook does)
  instead of always sharing the caller's own interpreter. Prints a
  `STAGE=<name>` marker to stderr before each of its two failure-prone
  calls, so a crash's stderr tail says which one failed
  (`parse_last_stage()`).
- `decision_gate.py` — deterministic classification of a pipeline run into
  one of Task 03's six decision-gate letters (A–F) from a
  `PipelineState` record of what happened at each stage, plus the eleven
  named failure categories. `phase_summary()` (Task 03B) additionally
  reports five independent PASS/FAIL/NOT_ATTEMPTED phase fields
  (`SAM3D_ENVIRONMENT`, `SAM3D_INFERENCE`, `MHR_CLAD_ENVIRONMENT`,
  `MHR_CLAD_EXTRACTION`, `END_TO_END`) plus the first failing boundary.
  Pure logic, no heavy dependency; reused identically by the Colab
  notebook's final cell.
- `interchange.py` (Task 03B) — the versioned `.npz` file contract between
  Environment A (SAM 3D Body GPU inference) and Environment B (MHR +
  clad-body measurement extraction), which run as two fully isolated
  processes/venvs in the Colab notebook. Plain numpy arrays and strings
  only, no pickle, loadable from either side regardless of which torch
  build is installed there. Builds on `adapter.py`'s validation, so it
  inherits the same `scale_params` exclusion guarantee.
- `_sam3d_inference_worker.py` (Task 03B) — Environment A's subprocess
  entry point: runs real SAM 3D Body inference and writes the interchange
  file. Mirrors `_mhr_measure_worker.py`'s pattern (always writes a JSON
  telemetry/status report, never lets a crash propagate as an opaque
  return code).
- `run.py` — the CLI entry point.
- `tests/` — tests for the files above only. No test touches SAM 3D Body
  or clad-body internals directly, and all tests run under a plain Python
  install with just `pytest`+`numpy` (no torch, no clad-body, no
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
