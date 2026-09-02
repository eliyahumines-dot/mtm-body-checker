# Task 03F — Fix Inherited Colab Matplotlib Backend

Continues from a real Colab run of Task 03E's notebook, which got past
source-root resolution entirely (Task 03E's fix held) and then failed
*inside* `import sam_3d_body` itself:

| Phase | Result |
|---|---|
| SAM3D_CORE_ENVIRONMENT | **PASS** |
| SAM3D_SOURCE_IMPORT | **FAIL** |
| SAM3D_MODEL_CODE_IMPORT | NOT_ATTEMPTED |
| SAM3D_MODEL_LOAD | NOT_ATTEMPTED |
| SAM3D_CORE_INFERENCE | NOT_ATTEMPTED |
| MHR_PARAMS_SERIALIZED | NOT_ATTEMPTED |
| First failing boundary | SAM3D_SOURCE_IMPORT |

Exact error:

```text
ValueError: Key backend: 'module://matplotlib_inline.backend_inline'
is not a valid value for backend
```

## Root cause (verified, not assumed)

Reproduced directly in this task against a real clone of
`facebookresearch/sam-3d-body` and a real dependency-complete venv:

```text
$ MPLBACKEND='module://matplotlib_inline.backend_inline' \
    /tmp/env_sam3d_test/bin/python3.11 -c "
import sys; sys.path.insert(0, '/tmp/sam3d_e2e_clone')
import sam_3d_body
"
  File ".../torchmetrics/utilities/plot.py", line 26, in <module>
    import matplotlib
  File ".../matplotlib/__init__.py", line 1289, in <module>
    rcParams['backend'] = os.environ.get('MPLBACKEND')
  ...
ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is
not a valid value for backend; supported values are [...]
```

Colab's interactive IPython kernel sets `MPLBACKEND` to
`module://matplotlib_inline.backend_inline` -- its own inline-plot-display
backend, registered only inside a live IPython/Jupyter kernel process. A
`subprocess.run(...)` launched from that kernel inherits the *same*
`os.environ` by default (Python does this unless an explicit `env=` is
given), including that `MPLBACKEND` value -- even though the child is a
plain, non-interactive Environment A worker process that never runs inside
any IPython kernel and has no `matplotlib_inline` package installed to
satisfy it. `torchmetrics` (imported transitively via `pytorch_lightning`,
itself part of `SAM3D_CORE_PIP_DEPENDENCIES`) imports `matplotlib.pyplot`
as a side effect during `import sam_3d_body`, and Matplotlib crashes trying
to activate the inherited, invalid-here backend name.

This is NOT an import-path failure: Task 03E's source-root resolution
worked correctly (the standalone subprocess found and inserted the right
`sys.path` entry) and `import sam_3d_body` was genuinely reached -- it just
crashed once actually executing, for an unrelated runtime reason.

## Fix

`experiments/sam3d_mhr_clad_smoke/sam3d_matplotlib_guard.py` (new) --
single source of truth for:

- `force_headless_matplotlib_backend()` -- unconditionally assigns (never
  `setdefault()`) `MPLBACKEND=Agg` in the *current* process's environment,
  returning whatever was inherited. Called at the very top of both
  `_sam3d_source_import_check.py` and `_sam3d_inference_worker.py`, before
  anything that might import matplotlib (`sam_3d_body`, `pyrender`,
  `torchmetrics` via `pytorch_lightning`, etc.).
- `sanitized_subprocess_env()` -- builds a `subprocess.run(..., env=...)`
  mapping with `MPLBACKEND` forced to `Agg`, preserving every other
  variable untouched. Used by the notebook when launching every standalone
  Environment A subprocess (the pre-flight check, the inference worker, and
  `verify_torch_pin()`'s GPU-tensor-op check) -- a first, redundant layer
  alongside each script's own in-process override, the same
  belt-and-suspenders pattern Task 03D/03E already established for the
  source-root mechanism.
- `effective_matplotlib_backend()` -- reports `matplotlib.get_backend()`
  if matplotlib has already been imported (by the process itself or as a
  side effect), without ever force-importing it -- so a script/test
  environment with no matplotlib installed at all is unaffected.
- `classify_import_exception()` -- distinguishes the specific,
  now-understood `SAM3D_MATPLOTLIB_BACKEND_FAILURE` from the general
  `SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE`, from the text of an exception
  raised during `import sam_3d_body`, retaining the full underlying
  exception text in every case.

**Not fixed by installing `matplotlib-inline`.** That would only mask this
one specific inherited value, leave a headless subprocess depending on an
interactive-kernel-only package for no functional reason, and would not
fix the underlying problem (a standalone worker inheriting whatever backend
its launching kernel happened to have configured). Verified directly:
`sam3d_env_spec.SAM3D_CORE_PIP_DEPENDENCIES` contains no `matplotlib-inline`
entry (`test_matplotlib_inline_was_not_added_as_a_dependency`).

## Failure semantics (Task 03F section 6)

`SAM3D_SOURCE_IMPORT_FAILURE` was too broad: it previously covered both "the
source root itself is invalid" and "the root is fine but the import
crashed." These are now distinct boundaries and failure categories:

- **`SAM3D_SOURCE_ROOT_VALIDATION`** (new phase, new
  `SAM3D_SOURCE_ROOT_VALIDATION_FAILURE` category) -- the given root exists
  and contains an importable `sam_3d_body/__init__.py`. Split out of what
  Task 03D/03E called `SAM3D_SOURCE_IMPORT`.
- **`SAM3D_SOURCE_IMPORT`** -- now strictly "the root validated, and
  `import sam_3d_body` was actually attempted." A failure here is
  classified via `classify_import_exception()`:
  - **`SAM3D_MATPLOTLIB_BACKEND_FAILURE`** (new) -- the specific case this
    task fixes.
  - **`SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE`** (new) -- any other
    post-root-validation import-time exception.
  - `SAM3D_SOURCE_IMPORT_FAILURE` is retained for the one case it always
    correctly described: `sam_3d_body` resolving to an unrelated installed
    package rather than the intended repo (a real path/search problem).

`decision_gate.py` gained all three new `FailureCategory` values (22
total, up from 19), one new `PipelineState` field
(`sam3d_source_root_validation_ok`), a new `classify()` precondition
positioned before the (reworded, no-longer-misleading)
`sam3d_source_import_ok` check, and `phase_summary()` inserts
`SAM3D_SOURCE_ROOT_VALIDATION` as a seventh independent Phase A boundary.

## Downstream NOT_ATTEMPTED semantics (Task 03E/03F section 7)

Every "not yet attempted" telemetry field in both scripts is `None`, never
`False`. A stage genuinely reached and failed sets its own field to
`False` explicitly (including via the worker's catch-all exception handler,
which maps whatever stage was active at crash time to its telemetry field).
Verified directly: a root-validation failure leaves `source_import_ok` and
`model_code_import_ok` as `None`; a source-import failure leaves
`model_code_import_ok` as `None`.

## Actual upstream import test performed (real, not mocked)

Reused this task's own `/tmp/sam3d_e2e_clone` (real, fresh
`git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git`)
and `/tmp/env_sam3d_test` (real venv, full
`SAM3D_CORE_PIP_DEPENDENCIES` installed):

```text
$ MPLBACKEND='module://matplotlib_inline.backend_inline' \
    /tmp/env_sam3d_test/bin/python3.11 _sam3d_source_import_check.py /tmp/sam3d_e2e_clone
Inherited MPLBACKEND: module://matplotlib_inline.backend_inline
Worker MPLBACKEND: Agg
SAM3D_SOURCE_ROOT: /tmp/sam3d_e2e_clone
SAM3D_MODULE_FILE: /tmp/sam3d_e2e_clone/sam_3d_body/__init__.py
SAM3D_SOURCE_ROOT_VALIDATION: PASS
SAM3D_SOURCE_IMPORT: PASS
SAM3D_MODEL_CODE_IMPORT: PASS
Effective matplotlib backend: Agg
$ echo $?
0
```

Confirmed the exact required result from Task 03F section 4, with the
precise invalid Colab value inherited into the subprocess's environment,
not a synthetic stand-in. Also confirmed the notebook's actual PHASE A1.5
and PHASE A4-A5 cell code (extracted and executed directly, not just
authored) run cleanly against this same real environment with no
exceptions, correctly reporting NOT_ATTEMPTED for every downstream stage
once the worker legitimately stops at the pre-existing, unrelated
`gpu_check` boundary (this sandbox has no GPU).

## Tests

29 new tests, all passing, none requiring GPU:

- `tests/test_sam3d_matplotlib_guard.py` (15) -- `force_headless_matplotlib_backend()`
  overriding an inherited invalid value (not `setdefault()` semantics),
  `sanitized_subprocess_env()` preserving unrelated variables while forcing
  `MPLBACKEND`, `effective_matplotlib_backend()` never force-importing
  matplotlib, and `classify_import_exception()`'s classification (including
  the exact observed error text).
- `tests/test_sam3d_inference_worker_matplotlib.py` (4) -- both scripts
  force `MPLBACKEND=Agg` at module-import time regardless of what was
  inherited (verified via a genuinely fresh subprocess, not a re-import
  into an already-running process), the worker module imports cleanly with
  no torch/cv2 available, and `matplotlib-inline` was not added as a
  dependency.
- `tests/test_sam3d_source_import_check.py` (+10 net, several rewritten for
  the 3-stage split) -- root-validation-vs-import-failure distinction,
  Matplotlib-backend vs. general-runtime-dependency classification,
  downstream NOT_ATTEMPTED semantics after each failure point, and a new
  real (non-mocked) test reproducing the exact invalid inherited
  `MPLBACKEND` end-to-end against the real upstream clone.
- `tests/test_decision_gate.py` (+7) -- the new failure categories, the new
  `classify()` precondition and its precedence, and `phase_summary()`'s new
  field/count/ordering.

**All 160 tests pass** (132 from Task 02-03E + 29 new, 1 legitimately
skipped -- `matplotlib` itself is not installed in this plain
pytest/numpy-only test environment, so the one test that needs a real
matplotlib import to check `effective_matplotlib_backend()`'s positive case
skips rather than fabricating a result).

## What was kept unchanged

Per this task's explicit "do not touch" list: Python 3.11 strategy,
`torch==2.8.0`, `torchvision==0.23.0`, the `cu129` wheel index, `HF_TOKEN`
handling, checkpoint access logic, the explicit full-image bbox logic,
Detectron2/MoGe exclusion, Task 03E's source-root validation *logic*
(`sam3d_source_path.py` itself is untouched -- only where its result is
reported changed), Environment B, and clad-body measurement logic. None of
`sam3d_source_path.py`, `run.py`, `adapter.py`, `rescale.py`,
`mtm_mapping.py`, `interchange.py`, `_mhr_measure_worker.py`,
`sam3d_env_spec.py`, or `install_log.py` were modified in this task.

## Colab/GPU execution

`COLAB_EXECUTION_UNAVAILABLE`. This agent's sandbox has no browser and no
GPU/Colab runtime access in any task in this project so far (Tasks 02
through 03F); reported only after every locally-reproducible boundary
above was made to genuinely pass. Authored and structurally validated
(valid nbformat, every cell's Python syntax checked, full notebook
pyflakes-clean, 160 tests passing) and additionally verified by extracting
and directly executing both PHASE A1.5 and PHASE A4-A5's actual cell code
against a real clone and a real dependency-complete venv -- not only
statically validated.

## Files changed

- `experiments/sam3d_mhr_clad_smoke/sam3d_matplotlib_guard.py` (new)
- `experiments/sam3d_mhr_clad_smoke/tests/test_sam3d_matplotlib_guard.py` (new)
- `experiments/sam3d_mhr_clad_smoke/tests/test_sam3d_inference_worker_matplotlib.py` (new)
- `experiments/sam3d_mhr_clad_smoke/_sam3d_source_import_check.py` (revised:
  3-stage split, matplotlib guard, refined failure classification)
- `experiments/sam3d_mhr_clad_smoke/_sam3d_inference_worker.py` (revised:
  same 3-stage split and guard, mirroring the standalone script)
- `experiments/sam3d_mhr_clad_smoke/decision_gate.py` (extended: three new
  `FailureCategory` values, one new `PipelineState` field, one new
  `classify()` precondition, `phase_summary()` gains a seventh Phase A
  field)
- `experiments/sam3d_mhr_clad_smoke/tests/test_decision_gate.py` (extended)
- `experiments/sam3d_mhr_clad_smoke/tests/test_sam3d_source_import_check.py`
  (extended/revised for the 3-stage split)
- `notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` (PHASE A1.5 and PHASE
  A4-A5 cells updated for the new boundary/telemetry fields and
  `sanitized_subprocess_env()`; decision-gate printout cell lists the new
  boundary)
- `docs/experiments/TASK03F_MATPLOTLIB_BACKEND_FIX.md` (this document)
- `experiments/sam3d_mhr_clad_smoke/README.md` (updated to describe the new
  module and Task 03F's fix)

`sam3d_source_path.py`, `sam3d_env_spec.py`, `install_log.py`, `run.py`,
`adapter.py`, `rescale.py`, `mtm_mapping.py`, `interchange.py`,
`_mhr_measure_worker.py`: unchanged.

## What human should run next

The same notebook, same setup (GPU runtime, `HF_TOKEN` Colab Secret),
`Runtime > Run all`. Watch PHASE A1.5's new `Inherited MPLBACKEND` /
`Worker MPLBACKEND` printout -- the first should show whatever Colab's
kernel had set (likely `module://matplotlib_inline.backend_inline`), the
second must always read `Agg`. `SAM3D_SOURCE_ROOT_VALIDATION`,
`SAM3D_SOURCE_IMPORT`, and `SAM3D_MODEL_CODE_IMPORT` should all print PASS
before any checkpoint download happens. If a failure still occurs at this
boundary, the printed `failure_category` and underlying exception text
(never discarded) will show exactly which of the three it is.

## Decision gate (this agent's own attempt)

Same as Task 03/03B/03C/03D/03E: **C. GPU_INSUFFICIENT** -- no GPU in this
sandbox. Authored, structurally validated, and -- beyond static validation
-- this task's actual notebook cell code was extracted and executed
directly against a real clone and real dependency-complete venv, with the
exact reported failure reproduced and then shown fixed. The real
classification is whatever the human's next Colab run reports.
