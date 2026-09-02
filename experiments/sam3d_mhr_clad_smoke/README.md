# sam3d_mhr_clad_smoke

Smoke test for the pipeline:

```
image -> SAM 3D Body -> MHR params -> clad-body -> anthropometric measurements
```

This is an installability/interoperability/numerical-sanity test, not an
accuracy benchmark. Findings from running the CLI-only path (no GPU, no
checkpoint access): `docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md`.
Findings from five rounds of the real-GPU/real-checkpoint Colab path —
Task 03 (first attempt), Task 03B (dual-environment fix, still failed on
real Colab), Task 03C (minimal core inference, no Detectron2, fixed
`chump`/`chumpy`, deterministic torch pin — got Environment A building for
the first time), Task 03D (fixed the `sam_3d_body` import path once
Environment A itself was confirmed working), Task 03E (the same
`ModuleNotFoundError: No module named 'sam_3d_body'` recurred on a second
real Colab run because Task 03D's fix partly relied on subprocess
`env=PYTHONPATH`; Task 03E removes that reliance everywhere in favor of a
standalone script that does its own `sys.path.insert()` from an explicit
CLI argument, verified against a real, non-mocked upstream clone, and adds
a further `SAM3D_MODEL_CODE_IMPORT` boundary distinct from both source
import and checkpoint loading):
`docs/experiments/TASK03_COLAB_END_TO_END_SMOKE_TEST.md`,
`docs/experiments/TASK03B_DEPENDENCY_RESOLUTION.md`,
`docs/experiments/TASK03C_MINIMAL_CORE_INFERENCE.md`,
`docs/experiments/TASK03D_SAM3D_IMPORT_PATH_FIX.md`,
`docs/experiments/TASK03E_SAM3D_SOURCE_IMPORT_HARDENING.md`, and
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
  `PipelineState` record of what happened at each stage, plus 19 named
  failure categories (11 original + 6 added in Task 03C + 1 in Task 03D +
  1 in Task 03E for precise Phase A diagnostics). `phase_summary()` reports
  eleven independent PASS/FAIL/NOT_ATTEMPTED phase fields — six for Phase A
  (`SAM3D_CORE_ENVIRONMENT`, `SAM3D_SOURCE_IMPORT` (Task 03D),
  `SAM3D_MODEL_CODE_IMPORT` (Task 03E), `SAM3D_MODEL_LOAD`,
  `SAM3D_CORE_INFERENCE`, `MHR_PARAMS_SERIALIZED`), two for Phase B
  (`MHR_CLAD_ENVIRONMENT`, `MHR_CLAD_EXTRACTION`), plus
  `PHASE_A_SUCCESSFUL` and `END_TO_END` — and the first failing boundary.
  Pure logic, no heavy dependency; reused identically by the Colab
  notebook's final cell.
- `sam3d_env_spec.py` (Task 03C) — single source of truth for Environment
  A's minimal dependency list and pinned torch/torchvision versions,
  imported directly by the notebook (so what it actually installs and
  what the tests check can never drift apart). Documents and guards the
  `chump`-vs-`chumpy` finding — the real Task 03B build failure was this
  agent's own transcription mistake, not an upstream change.
- `sam3d_source_path.py` (Task 03D) — validates the cloned
  `facebookresearch/sam-3d-body` repo root actually contains an
  importable `sam_3d_body` package, and deterministically builds the
  `PYTHONPATH` value used to expose it to Environment A's subprocess
  (preserving whatever was already set). Fixes a real
  `ModuleNotFoundError: No module named 'sam_3d_body'` observed on real
  Colab hardware — Environment A itself was confirmed fully working
  (torch/CUDA/GPU/pin all PASS) but nothing pointed the worker
  interpreter at the cloned repo.
- `_sam3d_source_import_check.py` (Task 03E) — standalone pre-flight
  script, independently runnable and independently testable: takes
  `sam3d_source_root` as an explicit CLI argument and does its own
  `sys.path.insert()` internally, with **no** subprocess `env=PYTHONPATH`
  reliance at all. Replaces Task 03D's notebook-inline `-c` preflight,
  which set `env=PYTHONPATH` — a second real Colab run showed that
  mechanism was not actually sufficient (the identical
  `ModuleNotFoundError` recurred), for a reason this project's own sandbox
  cannot reproduce or observe directly. Reports `SAM3D_SOURCE_IMPORT` and
  a further `SAM3D_MODEL_CODE_IMPORT` boundary (SAM 3D Body's own
  `build_models`/`sam_3d_body_estimator` submodules importing cleanly,
  distinct from `SAM3D_MODEL_LOAD` needing real checkpoint assets) as two
  independently-observable PASS/FAIL results, printed before any
  checkpoint download is attempted. Verified in this task against a real,
  non-mocked clone of `facebookresearch/sam-3d-body`, not only synthetic
  fixtures.
- `install_log.py` (Task 03C) — structured failure logging for every
  install command the notebook runs (exact command, return code, stderr
  tail, failure category). Fixes a real bug: Task 03B's notebook printed
  "Recorded failure categories: none" despite two observed build
  failures, because most failure paths only ever flipped a boolean.
- `interchange.py` (Task 03B) — the versioned `.npz` file contract between
  Environment A (SAM 3D Body GPU inference) and Environment B (MHR +
  clad-body measurement extraction), which run as two fully isolated
  processes/venvs in the Colab notebook. Plain numpy arrays and strings
  only, no pickle, loadable from either side regardless of which torch
  build is installed there. Builds on `adapter.py`'s validation, so it
  inherits the same `scale_params` exclusion guarantee.
- `_sam3d_inference_worker.py` (Task 03B, revised Task 03C/03D/03E) —
  Environment A's subprocess entry point: takes a validated
  `sam3d_source_root` and performs a two-stage pre-flight import check —
  `SAM3D_SOURCE_IMPORT` (bare `import sam_3d_body`) then
  `SAM3D_MODEL_CODE_IMPORT` (Task 03E: `build_models`/
  `sam_3d_body_estimator` submodule imports, split out as its own boundary,
  distinct from a model-load failure — see the module docstring), loads the
  model, builds and validates an explicit full-image bounding box (no
  learned detector — see Task 03C's doc for why `human_detector=None` is
  upstream's own first-class path, not a workaround), runs inference, and
  writes the interchange file. Reports each stage as an independently
  observable telemetry field, defaulting to `None` ("not attempted"), never
  `False` (Task 03E: fixes a real cascading-false-failure bug where a
  `False` default plus `bool()` coercion in the notebook made every
  downstream stage after an early failure misreport as FAIL instead of
  NOT_ATTEMPTED). Mirrors `_mhr_measure_worker.py`'s pattern (always writes
  a JSON telemetry/status report, never lets a crash propagate as an opaque
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
