# Task 03D — Fix SAM3D Repository Import Path Only

Continues from the real Colab execution of Task 03C's notebook. Like
Task 03/03B/03C, this agent's sandbox has no GPU, so this is a small,
narrowly-scoped notebook/code correction, authored and structurally
validated but **not executed** by this agent.

## Actual observed result (Task 03C's notebook, real Colab run)

| Phase | Result |
|---|---|
| SAM3D_CORE_ENVIRONMENT | **PASS** |
| SAM3D_MODEL_LOAD | **FAIL** |
| SAM3D_CORE_INFERENCE | FAIL |
| MHR_PARAMS_SERIALIZED | FAIL |
| First failing boundary | SAM3D_MODEL_LOAD |

Exact error: `ModuleNotFoundError: No module named 'sam_3d_body'`.

This confirms, for the first time in this project, that Python 3.11 +
pinned `torch==2.8.0`/`torchvision==0.23.0` (cu129) + CUDA + GPU + Hugging
Face auth + checkpoint download **all work** in real Colab. None of that
was touched in this task.

## Exact root cause

The worker subprocess (`/content/env_sam3d/bin/python3
_sam3d_inference_worker.py ...`) was launched with no `PYTHONPATH` and no
`sys.path` entry pointing at the cloned `facebookresearch/sam-3d-body`
repository. The repo was cloned to `/content/sam-3d-body` in the
notebook's "PHASE A3 — input image selection" cell (only so the bundled
sample image path could be constructed) — the clone happened, but nothing
ever told the Environment A interpreter where to find the `sam_3d_body`
Python package inside it. `import sam_3d_body` inside the worker had
nowhere to look.

## Exact upstream repo path used

`/content/sam-3d-body` (unchanged clone target from Task 03C), cloned via
`git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git`.
Verified directly for this task (fresh clone, commit `b5c765a` — same
commit Task 02/03C already established, confirming upstream hasn't
changed): the repo root contains `sam_3d_body/` with an `__init__.py`,
and has **no** `pyproject.toml`, `setup.py`, or `setup.cfg` anywhere.
`pip install -e .` is therefore not an option — per this task's explicit
instruction, no packaging metadata was invented for the third-party
repository. The repo root is used directly on the Python module search
path, exactly as running `demo.py` from that directory has always relied
on upstream.

## Import-path fix

Two deliberately redundant mechanisms:

1. **Preferred solution (Task 03D section 3): `PYTHONPATH`.** The notebook
   now launches both the pre-flight check and the real inference worker
   subprocess with `env={**os.environ, 'PYTHONPATH':
   build_pythonpath(validated_sam3d_root, os.environ.get('PYTHONPATH'))}`
   — `sam3d_source_path.build_pythonpath()` is a small, pure, tested
   function that prepends the validated repo root while preserving
   whatever `PYTHONPATH` was already set (never dropping it).
2. **Worker-side `sys.path.insert()`.** `_sam3d_inference_worker.py` now
   takes `sam3d_source_root` as a required positional argument, validates
   it with `sam3d_source_path.validate_sam3d_source_root()` (raises
   `Sam3dSourceRootError` if `<root>/sam_3d_body/__init__.py` doesn't
   exist — no invented/unverified path is ever accepted), and prepends it
   to `sys.path` itself before importing anything from `sam_3d_body`. This
   means the worker is correct even in isolation (e.g. if invoked
   directly without the notebook's `env=` wrapper), not dependent on the
   caller getting `PYTHONPATH` right.

Both mechanisms were implemented, per Task 03D section 3's explicit
preference for "the option that is more explicit/testable" — pure-Python
`validate_sam3d_source_root()`/`build_pythonpath()`/
`resolved_module_is_under_root()` (new module,
`experiments/sam3d_mhr_clad_smoke/sam3d_source_path.py`) are all directly
unit-tested with no GPU, torch, or real checkout required (synthetic
`tmp_path` fixtures build a valid/invalid layout).

## Pre-flight import check

Added exactly where Task 03D section 4 specifies — immediately after
Environment A setup (PHASE A0/A1) and the repository clone, before
checkpoint download or any model loading, using the same interpreter that
will run inference:

```python
import sam_3d_body
from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
print(sam_3d_body.__file__)
```

The notebook prints the resolved `sam_3d_body.__file__` and checks it via
`resolved_module_is_under_root()` against the validated repo root —
failing loudly (`SAM3D_SOURCE_IMPORT_FAILURE`, not proceeding) if it
resolves to some unrelated installed package rather than the intended
official Meta source tree, exactly per section 4's explicit requirement.
This is a new, clearly labeled `PHASE A1.5` cell — cheap and fast (no
checkpoint download needed yet), so a broken import is caught before
spending time/bandwidth on a multi-GB download.

The worker itself (`_sam3d_inference_worker.py`) repeats this exact check
internally as its own first stage, deliberately redundantly, so a broken
import is still caught even if something changed between the notebook's
pre-flight cell and the actual inference call — the notebook records a
fresh `SAM3D_SOURCE_IMPORT_FAILURE` if the two ever disagree, trusting the
worker's own result (it's the one that actually matters for inference).

## Worker error classification (Task 03D section 6)

`_sam3d_inference_worker.py` now separates two previously-conflated
failure modes into distinct stages/telemetry fields, checked in this
order:

- **A. Python cannot import `sam_3d_body`** — `telemetry["stage"] =
  "source_import"`, classified as `SAM3D_SOURCE_IMPORT_FAILURE`. This
  covers both an outright `ImportError`/`ModuleNotFoundError` and the
  "resolves to an unrelated installed package" case.
- **B. `load_sam_3d_body()` fails after a successful import** —
  `telemetry["stage"] = "model_load"`, classified as
  `SAM3D_MODEL_LOAD_FAILURE`, only ever reached once stage A has already
  succeeded (`telemetry["source_import_ok"] = True`).

`decision_gate.py` gained a matching new `FailureCategory`
(`SAM3D_SOURCE_IMPORT_FAILURE`, 18 total now, up from 17) and
`PipelineState` field (`sam3d_source_import_ok`), with `classify()`
checking it as its own precondition, positioned before the existing
`sam3d_model_load_ok` check — so a source-import failure is never reported
as a generic model-load or checkpoint failure. `phase_summary()` inserts
`SAM3D_SOURCE_IMPORT` as a fifth independent Phase A boundary, between
`SAM3D_CORE_ENVIRONMENT` and `SAM3D_MODEL_LOAD`.

## What was kept unchanged (Task 03D section 5)

Per explicit instruction, and because Task 03C's real Colab run proved
these already work: the `torch==2.8.0` pin, `torchvision==0.23.0` pin, the
`cu129` wheel index, checkpoint selection
(`facebook/sam-3d-body-dinov3`), `HF_TOKEN` retrieval/handling, the
explicit full-image bbox logic, Detectron2/MoGe exclusion, Environment B's
design (Pixi-preferred MHR/clad-body), and clad-body's measurement logic.
None of `run.py`, `adapter.py`, `rescale.py`, `mtm_mapping.py`,
`interchange.py`, `_mhr_measure_worker.py`, `sam3d_env_spec.py`, or
`install_log.py` were modified in this task.

## Tests

18 new tests, all passing, none requiring GPU:

- `tests/test_sam3d_source_path.py` (13) — source-root validation
  requires `sam_3d_body/__init__.py` (accepts a valid layout, rejects a
  missing directory, missing `__init__.py`, nonexistent path, empty
  string; resolves relative paths to absolute), `build_pythonpath()`
  determinism and existing-`PYTHONPATH` preservation (including that the
  new root is prepended first, so it takes precedence on a name
  collision), and `resolved_module_is_under_root()`'s true/false cases
  (the exact "unrelated installed package" scenario section 4 warns
  about).
- `tests/test_decision_gate.py` (+5) — `SAM3D_SOURCE_IMPORT_FAILURE`
  classification, its precedence over an unset model-load check, that a
  successful source import does not mask a subsequent model-load failure,
  that the two failure categories are never conflated on the same state,
  and `phase_summary()`'s new field/count/ordering (reproducing the exact
  real Colab result: `SAM3D_CORE_ENVIRONMENT` PASS, `SAM3D_SOURCE_IMPORT`
  FAIL, `SAM3D_MODEL_LOAD` NOT_ATTEMPTED).

**All 109 tests pass** (91 from Task 02/03B/03C + 18 new), all under plain
Python with `pytest`+`numpy` only — no torch, no GPU, no checkpoint, no
real `sam-3d-body` checkout (synthetic fixtures).

## Files changed

- `experiments/sam3d_mhr_clad_smoke/sam3d_source_path.py` (new)
- `experiments/sam3d_mhr_clad_smoke/tests/test_sam3d_source_path.py` (new)
- `experiments/sam3d_mhr_clad_smoke/_sam3d_inference_worker.py` (revised:
  takes `sam3d_source_root`, adds the source-import pre-flight stage,
  splits its failure classification into A/B per section 6)
- `experiments/sam3d_mhr_clad_smoke/decision_gate.py` (extended: one new
  `FailureCategory`, one new `PipelineState` field, one new `classify()`
  precondition, `phase_summary()` gains a fifth Phase A field)
- `experiments/sam3d_mhr_clad_smoke/tests/test_decision_gate.py` (extended)
- `notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` (new `PHASE A1.5` cell;
  worker invocation now passes `PYTHONPATH` + the source root; schema/
  decision-gate cells print the new boundary)
- `docs/experiments/TASK03D_SAM3D_IMPORT_PATH_FIX.md` (this document)

`sam3d_env_spec.py`, `install_log.py`, `run.py`, `adapter.py`,
`rescale.py`, `mtm_mapping.py`, `interchange.py`, `_mhr_measure_worker.py`:
unchanged.

## What human should run next

The same notebook, same setup (GPU runtime, `HF_TOKEN` Colab Secret),
`Runtime > Run all`. Watch for the new `PHASE A1.5` cell's
`sam_3d_body.__file__` printout — it should point into
`/content/sam-3d-body/sam_3d_body/...`. If `SAM3D_SOURCE_IMPORT: PASS`
there, the notebook proceeds to checkpoint download and the real
inference worker as before; if it still fails, the printed
`stderr_tail`/`install_log` entry will show the exact import error rather
than a generic model-load failure.

## Decision gate (this agent's own attempt)

Same as Task 03/03B/03C: **C. GPU_INSUFFICIENT** — no GPU in this sandbox.
Authored and structurally validated (valid nbformat, every cell's Python
syntax checked, full notebook pyflakes-clean, 109 tests passing) but not
executed. The real classification is whatever the human's next Colab run
reports.
