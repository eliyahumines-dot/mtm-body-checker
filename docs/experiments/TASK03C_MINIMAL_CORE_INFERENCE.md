# Task 03C — Minimal SAM 3D Body Core Inference

Continues from the second real Colab failure of Task 03B's notebook. Like
Task 03/03B, this agent's sandbox has no GPU, so this task is a
notebook/code correction, authored and structurally validated but **not
executed** by this agent. The next human Colab run is what actually tests
it.

**Update (Task 03D):** the real human Colab run of *this* document's
notebook made real progress — `SAM3D_CORE_ENVIRONMENT: PASS` (torch/CUDA/
GPU/pin all confirmed working for the first time) — but then failed with
`ModuleNotFoundError: No module named 'sam_3d_body'` at `SAM3D_MODEL_LOAD`.
The upstream repo was cloned but never put on the worker interpreter's
module search path. See `docs/experiments/TASK03D_SAM3D_IMPORT_PATH_FIX.md`
for the fix — a small, narrowly-scoped wiring correction; nothing this
document describes (the dependency list, the torch pin, Detectron2/MoGe
exclusion, Environment B) was changed.

## Actual observed failure (Task 03B's notebook, real Colab run)

| Field | Value |
|---|---|
| GPU available | Yes |
| Hugging Face auth | Yes |
| Checkpoint downloaded | Yes |
| Environment A | **FAIL** |
| SAM3D inference | Not attempted |

Observed install failures: a `chumpy` wheel build failure, a `detectron2`
wheel build failure (pinned commit `a1ce2f9`), and the environment drifted
to `torch 2.11.0+cu130` / `torchvision 0.26.0+cu130` — nothing this
project asked for. Task 03B's own decision-gate cell printed **"Recorded
failure categories: none"** despite these two clear, observed build
failures — a real bug in how failures were tracked, fixed in this task
(`install_log.py`).

## Root cause correction: `chump` vs. `chumpy`

**Verified directly, not assumed.** A fresh clone of
`facebookresearch/sam-3d-body` (commit `b5c765a`, re-cloned specifically
for this task, same commit as Task 02's original cache — confirming
upstream has not changed this) shows `INSTALL.md`'s pip list reads:

```
... pyrootutils webdataset chump networkx==3.2.1 roma ...
```

**`chump`**, not `chumpy`. `pip index versions` confirms both are real,
independent PyPI packages: `chump` (1.6.0, a small unrelated package) and
`chumpy` (0.70, the well-known SMPL-adjacent autodiff library many
body-mesh projects use — a reasonable guess, but wrong here). A full-repo
grep of `sam-3d-body`'s own Python source found **zero** occurrences of
either name as an `import` anywhere — neither is used directly by
`sam_3d_body`'s own code, so whichever one is genuinely needed is pulled
in transitively by something else in the list; this task does not attempt
to identify which, since it isn't necessary to fix the actual bug.

**This was this agent's own transcription mistake**, introduced when Task
03's notebook was first authored (Task 02's cached `INSTALL.md` already
had the correct `chump` text available at the time — it was simply
mistyped as the more famous `chumpy` when the dependency list was
written), not an upstream naming change. Grepping the repository before
this task confirmed exactly one place `chumpy` had been written: the
notebook's Environment A dependency list (present in both the Task 03 and
Task 03B versions). It is now fixed, and guarded going forward by
`sam3d_env_spec.validate_no_excluded_dependencies()` (a unit-tested
function, not just a one-time edit — see Tests below).

## Why Detectron2 was removed

Not because it can't ever work — because it isn't necessary to answer this
project's central question yet, and debugging its build is a distraction
from it (Task 03C section 9 explicitly forbids spending time compiling,
patching, or downgrading toolchains for it). Source reading of
`sam_3d_body/sam_3d_body_estimator.py::process_one_image` found that when
both `bboxes` and `self.detector` (i.e. `human_detector`) are `None`, the
method **already falls back to a full-image bounding box automatically**:

```python
else:
    boxes = np.array([0, 0, width, height]).reshape(1, 4)
    self.is_crop = False
```

This is a first-class, upstream-documented no-detector path, not a
workaround. `SAM3DBodyEstimator.__init__` already prints `"No human
detector is used..."` when `human_detector=None` — an expected message,
not a warning to silence. This task passes the box **explicitly** anyway
(`bboxes=np.array([[0, 0, width, height]])`) rather than relying on the
implicit fallback, for visibility and testability (Task 03C section 8) —
a full-repo grep confirmed `self.is_crop` (the one internal flag that
differs between the explicit and implicit paths) is assigned but never
read anywhere in the repository, so this choice has no behavioral effect,
only a documentation/telemetry one.

`MoGe`, `SAM2`, `SAM3` are removed for the same reason: `fov_estimator=None`
and `human_segmentor=None` are equally first-class, documented states
(`"No FOV estimator... Using the default FOV!"`, `"Mask-condition
inference is not supported..."`). None of these are required to answer
whether SAM 3D Body's core model can run and produce `mhr_model_params` —
which is the entire point of this task.

## Exact pinned Environment A

Not reproduced from Colab's ambient kernel anymore (Task 03B's strategy;
it drifted to `torch 2.11.0+cu130` in the real run). A dedicated venv
(`/content/env_sam3d`), Python 3.11 (installed via `apt-get` if not
already present, falling back to ambient `python3` only if that fails),
with:

```
torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
```

Verified after **every** dependency-installing step (`PHASE A0` right
after the pin, `PHASE A1` after the full core dependency list) via
`verify_torch_pin()`: exact `torch.__version__`/`torchvision.__version__`
string match, `torch.version.cuda`, `torch.cuda.is_available()`, and an
**actual GPU tensor operation** (`torch.randn(4,4,device='cuda') @
itself`, summed) — not just a version-string comparison. Any mismatch
stops Phase A there and records `TORCH_VERSION_DRIFT` or
`TORCHVISION_VERSION_DRIFT` explicitly; per Task 03C section 4, this is
never silently resolved by letting the drifted version stand.

## Minimal SAM 3D Body dependency set

Single source of truth: `experiments/sam3d_mhr_clad_smoke/sam3d_env_spec.py`
(`SAM3D_CORE_PIP_DEPENDENCIES`), imported by the notebook directly — the
pip command the notebook actually runs is built from this list, not a
separately hand-typed string that could drift from what's tested:

```
pytorch-lightning, pyrender, opencv-python, yacs, scikit-image, einops,
timm, dill, pandas, rich, hydra-core, hydra-submitit-launcher,
hydra-colorlog, pyrootutils, webdataset, chump, networkx==3.2.1, roma,
joblib, seaborn, wandb, appdirs, jsonlines, xtcocotools, loguru, optree,
fvcore, pycocotools, huggingface_hub, numpy
```

Upstream's own `INSTALL.md` list, minus macOS-only/demo-only extras not
needed for one inference call (`appnope`, `ffmpeg`, `cython`, `pytest`,
`black`, `tensorboard`) and minus everything this task explicitly excludes
(`detectron2`, `moge`, `sam2`/`sam-2`/`sam3`/`sam-3`) — enforced by
`validate_no_excluded_dependencies()`, called by the notebook itself
before installing anything, not only by tests.

## Notebook changes

`notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` restructured (35 cells).
Phase A is now:

- **PHASE A0** — deterministic Python 3.11 + pinned torch/torchvision venv,
  verified with a real GPU tensor op.
- **PHASE A1** — minimal core dependency install (`chump`, no
  Detectron2/MoGe/SAM2/SAM3), pin re-verified.
- **PHASE A2** — checkpoint auth + download (ambient kernel, unchanged —
  this has worked in both real Colab runs so far). Actual
  `load_sam_3d_body()` model-load PASS/FAIL is reported after PHASE A4-A5
  runs (one subprocess call handles model load, bbox validation,
  inference, and interchange write together, to avoid loading multi-GB
  checkpoint weights onto the GPU twice in one smoke-test run) — its
  status is still independently visible via its own telemetry field
  (`model_load_ok`), just observed at that point in execution rather than
  literally before image selection.
- **PHASE A3** — image selection (bundled public sample by default).
- **PHASE A4-A5** — one subprocess call to the rewritten
  `_sam3d_inference_worker.py`: model load, explicit full-image bbox
  construction + validation (dimensions checked, box logged), inference
  with `human_detector=None, human_segmentor=None, fov_estimator=None`,
  then `interchange.write_interchange()`. Reports
  `SAM3D_MODEL_LOAD`/`SAM3D_CORE_INFERENCE`/`MHR_PARAMS_SERIALIZED`
  independently from its telemetry's `model_load_ok`/`inference_ok`/
  `interchange_written` fields.
- **PHASE B** — unchanged in design from Task 03B (Pixi-preferred MHR/
  clad-body environment, pip fallback, four-step self-test against a
  bundled fixture) but now explicitly gated: `if not phase_a_successful:
  print('PHASE B skipped entirely...')` — Environment B is not built,
  investigated, or run at all unless all four Phase A boundaries passed,
  per Task 03C section 10.

`run.py`, `adapter.py`, `rescale.py`, `mtm_mapping.py`,
`_mhr_measure_worker.py`, and `interchange.py` are unchanged — reused
exactly as Task 02/03B left them.

## Error reporting improvements

`experiments/sam3d_mhr_clad_smoke/install_log.py` (new): every command the
notebook runs that can fail goes through `run_shell_logged()`, which
records the exact command, return code, stderr tail, and a specific
`FailureCategory` value into a shared `InstallLog` — not just a boolean.
The final decision-gate cell now does:

```python
for cat in install_log.categories():
    state.add_failure(FailureCategory(cat))
```

merging every logged failure into `state.failure_categories`, and prints
`install_log.summary()` (every failed command's full record) unconditionally
at the end — this is the direct fix for "Recorded failure categories:
none": that could only happen before because most failure paths flipped a
boolean without ever calling `state.add_failure()`.

Six new `FailureCategory` values (`decision_gate.py`):
`WRONG_DEPENDENCY_CHUMPY`, `TORCH_VERSION_DRIFT`,
`TORCHVISION_VERSION_DRIFT`, `SAM3D_CORE_DEPENDENCY_FAILURE`,
`SAM3D_MODEL_LOAD_FAILURE`, `SAM3D_CORE_INFERENCE_FAILURE` — 17 total, up
from 11. `phase_summary()` now reports Phase A as four independent
boundaries (`SAM3D_CORE_ENVIRONMENT`, `SAM3D_MODEL_LOAD`,
`SAM3D_CORE_INFERENCE`, `MHR_PARAMS_SERIALIZED`) plus a derived
`PHASE_A_SUCCESSFUL`, instead of Task 03B's two coarser fields
(`SAM3D_ENVIRONMENT`, `SAM3D_INFERENCE`) — a real, evidence-driven
correction (`classify()`'s own six-letter gate logic is unchanged, only
extended with one new precondition check for `sam3d_model_load_ok`).

## Tests

25 new tests, all passing, none requiring GPU:

- `tests/test_sam3d_env_spec.py` (11) — `chumpy` absence, `chump`
  presence, Detectron2/MoGe/SAM2/SAM3 absence, the exclusion-guard
  actually raising when triggered, the torch pin matching this task's
  exact specification, no unbounded `-U`/`--upgrade` flag.
- `tests/test_install_log.py` (7) — command recording, category-only-on-
  failure, stderr truncation, distinct-category ordering, and a test that
  directly reproduces Task 03B's exact bug scenario (two build failures)
  and proves `InstallLog` would have surfaced both.
- `tests/test_decision_gate.py` (+7) — the new `SAM3D_MODEL_LOAD_FAILURE`
  classification, its precedence over an unset inference check, the new
  `phase_summary()` field names/count, and a test reproducing Task 03C's
  own model-load-failure scenario.

**All 90 tests pass** (65 from Task 02/03B + 25 new), all under plain
Python with `pytest`+`numpy` only — no torch, no clad-body, no GPU, no
checkpoint download.

## What the human should run next

The same notebook (`notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb`), same
setup as before (GPU runtime, `HF_TOKEN` Colab Secret) — `Runtime > Run
all`. Watch for PHASE A0/A1's `verify_torch_pin()` output first; if that
doesn't show `torch_version_matches: true` and `gpu_tensor_op_ok: true`,
everything downstream is expected to be skipped and that is not a bug.

## Expected Phase A output (if this design is correct)

```
PHASE A0 — deterministic environment + torch pin: PASS
PHASE A1 — minimal core dependencies: PASS
PHASE A2 — checkpoint download: PASS
PHASE A3 — image selection: PASS
PHASE A2 (model load, reported here): PASS
PHASE A4 — explicit-bbox core inference: PASS   | person_detected: True
PHASE A5 — MHR params serialized to interchange file: PASS
mhr_model_params shape: [204]  (expected [204])
shape_params shape: [45]  (expected [45])
PHASE A OVERALL: PASS
```

Reaching this — genuinely, for the first time in this project — would be
the actual milestone. Per Task 03C section 14, do not report end-to-end
(clad-body measurements) success without this having actually happened on
real human GPU hardware first.

## Remaining technical risks

- The exact pin (`torch 2.8.0` / `torchvision 0.23.0` / `cu129`) is
  untested by this agent (no GPU). If PyPI/the CUDA 12.9 wheel index
  doesn't have this exact combination available, `PHASE A0` will fail
  cleanly and loudly (an `INSTALL_FAILURE`/`SAM3D_CORE_DEPENDENCY_FAILURE`
  from `run_shell_logged`), not silently.
- `chump` is confirmed as upstream's literal text, but this task did not
  determine which of the ~29 other listed packages actually needs it
  transitively (or whether it's itself an upstream typo for `chumpy`) —
  if `chump`'s absence/wrong-purpose ever surfaces as a real
  `ImportError` inside `sam_3d_body`'s own code at runtime, that would be
  a new, separately-diagnosable failure, not something this task
  preempts.
- `python3.11` may or may not be installable via `apt-get` on Colab's
  current base image — untested; the fallback to ambient `python3` is
  also untested.
- The explicit full-image bbox assumes a single, centered, full-body
  subject (this project's controlled MTM use case) — an off-center or
  partial-body photo is out of scope for this smoke test by design, not
  an oversight.

## Files changed

- `experiments/sam3d_mhr_clad_smoke/sam3d_env_spec.py` (new)
- `experiments/sam3d_mhr_clad_smoke/install_log.py` (new)
- `experiments/sam3d_mhr_clad_smoke/_sam3d_inference_worker.py` (rewritten:
  explicit bbox, separated model-load/inference/serialize stages)
- `experiments/sam3d_mhr_clad_smoke/decision_gate.py` (extended: 6 new
  `FailureCategory` values, `sam3d_model_load_ok` field, `classify()`
  precondition, restructured `phase_summary()`)
- `experiments/sam3d_mhr_clad_smoke/tests/test_sam3d_env_spec.py` (new)
- `experiments/sam3d_mhr_clad_smoke/tests/test_install_log.py` (new)
- `experiments/sam3d_mhr_clad_smoke/tests/test_decision_gate.py` (extended)
- `notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` (rewritten)
- `docs/experiments/TASK03C_MINIMAL_CORE_INFERENCE.md` (this document)

`run.py`, `adapter.py`, `rescale.py`, `mtm_mapping.py`,
`_mhr_measure_worker.py`, `interchange.py`: unchanged.

## Decision gate (this agent's own attempt)

Same as Task 03/03B: **C. GPU_INSUFFICIENT** — no GPU in this sandbox, so
the notebook was authored and structurally validated (valid nbformat,
every cell's Python syntax checked, full notebook source pyflakes-clean,
90 tests passing) but not executed. The real classification is whatever
the human's next Colab run reports in section 16's output.
