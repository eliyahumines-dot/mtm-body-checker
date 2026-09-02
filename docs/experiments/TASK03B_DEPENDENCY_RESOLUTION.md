# Task 03B — Dependency Architecture Fix: Isolated SAM3D and MHR/clad-body Environments

Continues directly from the human's real Colab run of Task 03's notebook,
which returned decision gate **D — DEPENDENCY_ENVIRONMENT_BLOCKED**,
reason `CUDA_PYTORCH_MISMATCH`, with GPU/HF-auth/checkpoint-download all
confirmed working (see the "Update (Task 03B)" section in
`TASK03_COLAB_END_TO_END_SMOKE_TEST.md` for the exact observed state).
This document covers only the dependency/environment fix — those three
confirmed-working areas were not re-investigated.

**Like Task 03, this redesign was authored and structurally validated by
this agent but not executed** — this sandbox still has no GPU (confirmed
identically in Task 02 and Task 03). Everything below is a reasoned,
evidence-driven design in response to the real failure the human reported,
not a confirmed fix. The next real Colab run is what confirms or refutes
it.

## Dependency root cause

Task 03's notebook installed SAM 3D Body's entire dependency list (a long
`pip install` of `pytorch-lightning`, `pyrender`, `yacs`, `timm`, and
about a dozen other packages, followed by `detectron2` built
`--no-build-isolation` from a pinned commit, followed by `MoGe`) directly
into Colab's **ambient kernel** — the same Python environment that already
had a `torch`/`torchvision` pair pre-installed and matched to that
runtime's specific GPU driver by Google.

Nothing in that install list pinned or protected `torch`/`torchvision`
from being silently changed. Any one of those packages (or `detectron2`'s
own build-time dependency resolution) declaring a `torch` version
constraint incompatible with Colab's pre-installed version would cause
`pip` to transitively upgrade or downgrade it — and once `torch` and
`torchvision` (or either of them and the host CUDA driver) fall out of
their originally-matched alignment, torchvision's compiled CUDA operators
(and `detectron2`'s, built against whatever torch happened to be active at
build time) stop resolving correctly. This is the most probable
explanation for `CUDA_PYTORCH_MISMATCH` fitting the observed evidence:
GPU, driver, and a working torch install were all present and functional
*before* section 4's install list ran (confirmed by `gpu_available: True`
and the notebook having gotten as far as checkpoint download), and the
failure appeared specifically at `dependencies_installed: False` /
`sam3d_inference_ok: False` — i.e., after that install list ran, before
inference.

This is a structural problem, not a specific-version problem: no single
hardcoded version pin fixes it, because the actual bug is the *absence of
a pin being enforced* against whatever Colab's current image happens to
ship. The fix is architectural.

## Why two fully isolated environments, not one

Task 03's earlier (also-unexecuted) design isolated only the
`clad-body`/`pymomentum` side into a second venv, reasoning from Task 02's
CPU-sandbox segfault. The real failure landed on the *other* side (SAM 3D
Body's own install, in the shared ambient kernel) instead. The general
lesson generalizes cleanly: **whichever side of this pipeline shares a
Python environment with "whatever Colab happens to have installed" is the
side at risk of an uncontrolled transitive dependency change.** The fix is
to remove that shared exposure entirely, for both sides, not just the one
that failed once.

## Environment A — SAM 3D Body GPU inference

Dedicated venv at `/content/env_sam3d`, built and used only via
subprocess (`_sam3d_inference_worker.py`) — the ambient kernel's own
packages are never modified by this notebook again.

1. **Python**: prefers `python3.11` (SAM 3D Body's documented target),
   installed via `apt-get install python3.11 python3.11-venv` if not
   already present on the Colab image; falls back to whatever `python3`
   is available rather than hard-failing on this alone (Colab's exact
   preinstalled Python version is outside this project's control and
   changes over time).
2. **torch / torchvision pin**: read once, from the **ambient kernel's own
   already-GPU-working versions** (`torch.__version__`,
   `torchvision.__version__`, captured in section 1 before anything else
   runs) — not a version this agent invented or guessed. Colab has
   already proven this exact pair works with this runtime's specific GPU
   and driver; Environment A reproduces it via `pip install
   torch==<ambient> torchvision==<ambient>` inside the fresh venv, which
   fetches the correct wheel for the venv's own Python version.
3. **SAM 3D Body's other dependencies**: the same pip list from
   `INSTALL.md`, plus `detectron2` (pinned commit `a1ce2f9`,
   `--no-build-isolation --no-deps`) and `MoGe` (`--no-deps`) — `--no-deps`
   on the two packages most likely to redeclare a `torch` constraint,
   since neither needs its own transitive dependencies beyond torch itself
   being present.
4. **Pin-survival check**: after every install step, `torch.__version__`
   and `torchvision.__version__` are read back from Environment A's venv
   and compared against the captured ambient values. Any mismatch sets
   `dependencies_installed = False` and records
   `FailureCategory.CUDA_PYTORCH_MISMATCH` explicitly — this is the
   concrete check that would have caught Task 03's actual failure *before*
   wasting a checkpoint download and inference attempt on a broken
   environment, by refusing to proceed rather than finding out from a
   downstream crash.
5. Only after this passes does the notebook attempt real SAM 3D Body
   inference (section 7), via `_sam3d_inference_worker.py`, which imports
   `torch` and `sam_3d_body` and nothing else heavy, runs
   `estimator.process_one_image()`, and on success calls
   `interchange.write_interchange()` to serialize the result.

This design is a direct, reasoned response to the observed failure. It has
**not been run** — Colab's actual current default torch/torchvision/CUDA
combination, and whether `detectron2`'s pinned commit still builds against
whatever version that turns out to be, are unknowns until the next real
run.

## Environment B — MHR + clad-body measurement extraction

CPU-only, no GPU/CUDA dependency, built and validated **independently**
of Environment A. Two paths, tried in order, so the human never has to
choose manually:

1. **Primary: Pixi.** MHR's own README calls this "the recommended
   installation method... most reliable environment setup," and
   explicitly warns that its plain-pip path (what Task 02/03 used) is
   "experimental... may not resolve correctly on all platforms" — which
   is exactly the class of failure Task 02 hit (a `pymomentum-cpu`
   native-extension segfault). The notebook installs Pixi
   (`curl -fsSL https://pixi.sh/install.sh | bash`, the officially
   documented installer), clones `facebookresearch/MHR`, runs
   `pixi install` against its own `pixi.toml` (a dependency-locked
   environment MHR's own maintainers curate and test), then
   `pixi run download-assets` (MHR's own asset-download task — this
   avoids Task 02's workaround of manually fetching the release archive,
   since the repo's own Pixi environment has the proper entry point,
   unlike the plain PyPI `mhr` package which was found to omit it).
   `clad-body` itself (not part of MHR's own environment) is then
   `pip install --no-deps`-ed directly into Pixi's resolved venv
   (`.pixi/envs/default/bin/pip`) — `--no-deps` so pip doesn't
   re-resolve `pymomentum-cpu`/`mhr` and potentially reintroduce a
   version Pixi didn't already validate.
2. **Automatic fallback: pip + CPU-only torch venv.** Only attempted if
   the Pixi path fails at any step. This is exactly Task 02/03's
   approach (a fresh venv, `torch` from
   `download.pytorch.org/whl/cpu` — reachable from real Colab, unlike
   Task 02's network-restricted sandbox — then `clad-body[mhr]`, then the
   MHR asset archive fetched directly from its public GitHub release,
   working around the same PyPI-package packaging gap Task 02 found).

**Validation order** (Task 03B section 4, both paths): before touching
any real SAM 3D Body output, the notebook (1) imports
`pymomentum.geometry`, (2) loads the MHR body model via
`MHR.from_files(device='cpu')` — this is the *exact* call that segfaulted
in Task 02's plain-pip sandbox; succeeding here directly tests whether
either installation path actually fixes that — (3) reconstructs a mesh
from one of clad-body's own bundled, known-good test fixtures, and
(4) runs `clad_body.measure.measure()` on it. Only if all four steps pass
does the notebook proceed to feed the real Phase A output through the same
call path (step 5). This is what makes "Environment B is broken" and
"SAM 3D Body's output doesn't match what clad-body expects" distinguishable
failures instead of one undifferentiated crash.

## Interchange contract

`experiments/sam3d_mhr_clad_smoke/interchange.py` — a single `.npz` file,
loadable with `numpy.load(path, allow_pickle=False)` from either
environment regardless of which (or whether any) torch build is present
on the reading side. No Python object, torch tensor, or pickle ever
crosses the process boundary.

| Field | Type | Required | Consumed by clad-body's current loader? |
|---|---|---|---|
| `schema_version` | string | yes | metadata only |
| `source_checkpoint` | string | yes | metadata only |
| `shape_params` | float32[45] | yes | **yes** |
| `mhr_model_params` | float32[204] | yes | **yes** — `[136:204]` slice is the decoded scale |
| `body_pose_params` | float32[133] | no (if present in source) | no — provenance only |
| `global_rot` | float32[3] | no | no — provenance only |
| `pred_cam_t` | float32[3] | no | no — provenance only |
| `focal_length` | float32 scalar | no | no — provenance only |

`write_interchange()` (Environment A side) builds on
`adapter.sam3d_output_to_clad_params()` unchanged from Task 02 — so the
same validated exclusion of SAM 3D Body's raw 28-dim `scale_params` (which
would silently corrupt clad-body's measurements if forwarded, per Task
02's finding) applies here by construction, not by re-implementing the
check. `read_interchange()` (Environment B side) raises
`InterchangeError` — loudly, before any MHR reconstruction is attempted —
on a missing or version-mismatched `schema_version`, or a missing required
field.

## Notebook changes

`notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` was substantially rewritten
(38 cells). Structure:

- Section 1-3: environment inspection (now also captures the ambient
  torch/torchvision pin reference), HF_TOKEN, repo checkout — mostly
  unchanged from Task 03, since GPU/HF-auth were confirmed working.
- **PHASE A** (sections 4-7): build Environment A, checkpoint
  auth+download (unchanged from Task 03, confirmed working), input image,
  real SAM 3D Body inference via subprocess + interchange write, actual
  output schema inspection. Prints `PHASE A — SAM3D GPU inference: PASS`
  or `FAIL` explicitly.
- **PHASE B** (sections 8-11): build Environment B (Pixi, falling back to
  pip), the four-step self-test against a bundled fixture, then (only if
  both Phase A and the self-test passed) step 5 — the real interchange
  data through MHR reconstruction and clad-body measurement, with optional
  known-height calibration. Prints `PHASE B — MHR / clad-body: PASS` or
  `FAIL` explicitly at each of its own sub-stages.
- Sections 12-15: calibration note, MTM terminology mapping (unchanged
  from Task 02/03), save results, compute/runtime summary, optional
  second-checkpoint comparison (off by default, reuses the same helper
  functions rather than duplicating the pipeline).
- Section 16: decision gate — prints the five Task 03B phase fields
  (`SAM3D_ENVIRONMENT`, `SAM3D_INFERENCE`, `MHR_CLAD_ENVIRONMENT`,
  `MHR_CLAD_EXTRACTION`, `END_TO_END`) each as PASS/FAIL/NOT_ATTEMPTED
  plus the first failing boundary, and the overall A-F letter gate.

Validated (not executed): valid JSON, valid `nbformat` 4 schema
(`nbformat.validate()`), every code cell's Python syntax checked
(`ast.parse()`, zero errors), and the full concatenated notebook source
checked with `pyflakes` (zero warnings — no undefined names across cell
boundaries, assuming linear top-to-bottom execution).

## Exact versions pinned

**Environment A**: Python 3.11 (or ambient `python3` if 3.11 is
unavailable/uninstallable — recorded either way in telemetry);
`torch`/`torchvision` pinned to whatever the ambient Colab kernel has
*at run time* (not hardcoded by this agent — this sandbox cannot observe
Colab's current default, so no specific version number is asserted here);
`detectron2` at pinned commit `a1ce2f9` (unchanged from upstream
`INSTALL.md`); the SAM 3D Body pip dependency list is the same one from
`INSTALL.md`, unpinned individually (their own version resolution is not
what caused the observed failure — the ambient torch/torchvision pair
was). Exact resolved versions are captured and printed by
`_sam3d_inference_worker.py`'s telemetry (`python_version`,
`torch_version`, `torch_cuda_version`) on the next real run.

**Environment B**: Pixi resolves its own versions from
`facebookresearch/MHR`'s `pixi.toml` (maintained upstream, not by this
project); the pip fallback pins nothing beyond `torch`'s CPU-wheel index
and `clad-body[mhr]`'s own declared constraints
(`pymomentum-cpu>=0.1.108`, confirmed in Task 02). Neither environment's
exact resolved version set is known until a real run captures it — this
document does not assert specific numbers it cannot verify.

**No unbounded `pip install -U` appears anywhere in the notebook.**

## Tests

12 new tests for `interchange.py` (round-trip, required-field validation,
`scale_params` exclusion, schema-version mismatch handling, no-pickle
guarantee) and 6 new tests for `decision_gate.phase_summary()` (each PASS/
FAIL/NOT_ATTEMPTED combination, first-failing-boundary detection,
including a test that reproduces the exact state the human's real Colab
run reported). All 47 of Task 02/03's original tests are unchanged and
still pass. **65 tests total, all passing, none requiring GPU, torch,
clad-body, Pixi, or a checkpoint download.**

## What the human must do next

Run the updated `notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` top-to-bottom
in Colab with a GPU runtime and the `HF_TOKEN` secret set, exactly as
before (see `TASK03_COLAB_END_TO_END_SMOKE_TEST.md`'s reproducibility
steps — unchanged). Section 16's output reports the real PASS/FAIL for
each of the five phases and the overall decision gate letter.

## Expected first meaningful output (if this design is correct)

`PHASE A — SAM3D GPU inference: PASS`, with a real observed `torch`/
`torchvision`/CUDA version triplet and actual `mhr_model_params` shape
(`[204]` expected) printed from section 7-8; `PHASE B` reaching at least
the four-step self-test (proving Environment B itself works, independent
of SAM 3D Body); and, if both phases pass, real numeric measurements in
`results/measurement_output.json` — which would be decision gate
**A — END_TO_END_MEASUREMENTS_PRODUCED**, the first time in this project
that actually happens.

## Remaining technical risks

- The ambient-torch-pin strategy assumes Colab's *current* default
  torch/torchvision pair is one `detectron2`'s pinned commit
  (`a1ce2f9`) can still build against. If Colab's default has moved far
  enough ahead, `detectron2`'s build itself could fail even inside the
  correctly-isolated Environment A — a different, but now much more
  clearly diagnosable, failure (`sam3d_environment_ok = False` at the
  `detectron2` install step specifically, not a downstream inference
  crash).
- The Pixi path for Environment B is unverified in this project — Task 02
  never tried it (only plain pip). It could hit its own, different
  installation issues in Colab specifically (e.g. Pixi's own install
  script assumptions about the host OS).
- `python3.11` may or may not be installable via `apt-get` on whatever
  Ubuntu base image Colab currently uses; the fallback to ambient
  `python3` is untested.
- None of this has been confirmed to fix the *original* reported
  `CUDA_PYTORCH_MISMATCH` — it is the best-evidenced hypothesis available
  from the reported state, not a verified fix.

## Files changed

- `experiments/sam3d_mhr_clad_smoke/interchange.py` (new)
- `experiments/sam3d_mhr_clad_smoke/_sam3d_inference_worker.py` (new)
- `experiments/sam3d_mhr_clad_smoke/decision_gate.py` (extended:
  `mhr_clad_environment_ok` field, `phase_summary()` function)
- `experiments/sam3d_mhr_clad_smoke/tests/test_interchange.py` (new)
- `experiments/sam3d_mhr_clad_smoke/tests/test_decision_gate.py` (extended)
- `notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` (rewritten)
- `docs/experiments/TASK03_COLAB_END_TO_END_SMOKE_TEST.md` (updated with
  the real human-run evidence)
- `docs/experiments/TASK03B_DEPENDENCY_RESOLUTION.md` (this document)

`run.py`, `adapter.py`, `rescale.py`, `mtm_mapping.py`, and
`_mhr_measure_worker.py` are unchanged from Task 02/03 — reused as-is, per
this task's instruction not to modify measurement logic.

## Decision gate (this agent's own attempt)

**C. GPU_INSUFFICIENT** — unchanged reason from Task 03: this sandbox has
no GPU, so the redesigned notebook was authored and structurally
validated (nbformat-valid, syntax-checked, pyflakes-clean, 65 tests
passing) but not executed. The real classification is whatever the
human's next Colab run of the updated notebook reports in its own
section 16 output.
