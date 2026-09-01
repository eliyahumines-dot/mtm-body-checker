# Task 02 — SAM 3D Body → MHR → clad-body Smoke Test

Goal: prove or disprove that `image → SAM 3D Body → MHR params → clad-body
→ measurements` runs end-to-end, and establish compute requirements,
output semantics, and blockers before any real-subject benchmarking. This
is a sanity/interoperability test, not an accuracy claim — see
"Measurement extraction is not body accuracy" below.

## Environment

See `TASK02_ENVIRONMENT.md` for the full probe. Summary: Ubuntu 24.04, 4
vCPU, 15 GiB RAM, **no GPU** (`nvidia-smi` absent, no `/proc/driver/nvidia`,
no VGA device in `lspci`), 30 GB disk available at start, system Python
3.11.15, network access to PyPI/GitHub confirmed working.

## Upstream versions/commits inspected

All three upstream repos were cloned directly (outside the project tree,
per repo hygiene) rather than relying on cached summaries:

| Repo | Commit | Date | Notes |
|---|---|---|---|
| `facebookresearch/sam-3d-body` | `b5c765a` | 2026-02-19 | "add arXiv" |
| `datar-psa/clad-body` | `a2140a7` | 2026-05-16 | tagged `0.6.1`, matches PyPI latest |
| `facebookresearch/MHR` | `e412e12` | 2026-08-23 | "Document facial expression mapping" |

Installed package versions (Python 3.12 venv, see below):
`clad-body 0.6.1`, `mhr 1.0.1`, `pymomentum-cpu 0.1.114.post0`,
`torch 2.8.0+cu128`, `numpy 2.5.2`, `scipy 1.18.1`, `trimesh 4.12.2`.

## A. SAM 3D Body ↔ clad-body compatibility (corrects Task 01's "UNKNOWN")

Task 01 left this as "untested, targets Anny/MHR not SMPL." Reading both
repos' source directly resolves it precisely — **not a drop-in
pass-through, but a documented, small, deterministic adapter closes the
gap.**

clad-body's README shows `load_mhr_from_params("path/to/sam3d_params.json")`
as a real, intended entry point. Its implementation
(`clad_body/load/mhr.py`) reads exactly two fields from that JSON:
`shape_params` (45-dim identity coefficients) and either `mhr_model_params`
(the full 204-dim `[pose(136) | scale(68)]` vector) or `scale_params`
(which it assumes is **already** the 68-dim decoded scale vector). Pose,
camera, hand, face, and mask fields are read from the file only for
provenance (`sam3d_params` metadata) — the mesh itself is always
reconstructed at a **zeroed rest pose**.

SAM 3D Body's own `SAM3DBodyEstimator.process_one_image()`
(`sam_3d_body/sam_3d_body_estimator.py`) returns a dict per detected person
with both a `scale_params` key and an `mhr_model_params` key — but its
`scale_params` is the **raw 28-dim PCA coefficient vector**
(`num_scale_comps = 28` in `mhr_head.py`), not the 68-dim decoded scale
clad-body's `scale_params` fallback branch assumes. `mhr_model_params`,
however, already contains the correctly-decoded 68-dim scale in its
`[136:204]` slice (`scales = scale_mean + scale_params @ scale_comps`,
computed inside SAM 3D Body's own head).

**The trap:** clad-body's loader does not validate vector length before
using it — passing SAM 3D Body's raw `scale_params` straight through would
silently fill only 28 of 68 expected slots and zero-pad the rest, with no
error raised, corrupting every downstream measurement invisibly. We
confirmed this isn't hypothetical: clad-body's own bundled test fixtures
(`clad_body/measure/testdata/mhr/*/mhr_params.json`, whose docstring says
they "come from SAM 3D Body ... pipelines") ship `scale_params` at length
**68**, not 28 — meaning whoever generated those fixtures already knew to
apply this exact correction, using unpublished conversion code not present
in either repo.

**Resolution — our adapter (`experiments/sam3d_mhr_clad_smoke/adapter.py`)**:
take only `shape_params` and `mhr_model_params` from SAM 3D Body's output,
verify their lengths (45 and 204), and never forward `scale_params`. This
is implemented, tested (11 unit tests, see below), and used by `run.py`.

No notebook, script, or documentation in either repo demonstrates this
conversion end-to-end — it had to be derived from source, cross-checked
against the bundled fixtures, and is recorded here so it doesn't have to be
re-derived.

## B. SMPL/SMPL-X licensing — reframed as an engineering/business decision

Per this task's explicit correction to Task 01: we do not draw a legal
conclusion about whether SMPL-derived numeric outputs are prohibited for
commercial use. The engineering/business decision, unchanged in substance
from Task 01 but stated without asserting a legal conclusion:

> SMPL/SMPL-X commercial-use terms carry enough licensing uncertainty and
> potential cost that they are **excluded from the initial commercial-
> oriented POC** unless and until explicit, suitable licensing is obtained.
> No SMPL-dependent model is used in this task's implemented path.

`LICENSE_AND_COMMERCIAL_USE.md` is updated to state this as a standing
project decision rather than a legal verdict (its underlying factual
findings — what the license text says — are unchanged and still recorded
for reference).

## C. NVIDIA `video_to_data` (Isaac) — SAM3D-Body/MHR multi-view module

Primary-source investigation (repo cloned and read directly, HEAD commit
2026-08-31): the relevant project is `nvidia-isaac/video_to_data`, and the
SAM3D-Body/MHR piece (`reconstruction/modules/v2d_sam3d_body`) is one
module inside a much larger robot-learning-data pipeline, not a standalone
body-measurement tool.

- **What it does**: per-camera SAM 3D Body inference per frame
  (`estimate_mhr_params.py`), then a genuine **joint multi-view
  optimization** (`mv_optimize_mhr_params.py`) that fits one shared MHR
  body per timestep across all cameras, minimizing a robust (Geman-McClure)
  multi-view 2D reprojection loss plus temporal smoothness, via Adam.
- **License**: repo code Apache-2.0, docs CC-BY-4.0 (verified in `LICENSE`).
  This covers V2D's own code only — SAM 3D Body/MHR weights are a separate,
  Hugging-Face-gated dependency, license unaffected by V2D's own license.
- **Shared/optimized parameters**: `global_trans`/`global_rot`/pose per
  frame; `shape_params` and `scale_params` are collapsed to **one shared
  row across the whole sequence** (one body, one shape, for all frames).
  Camera intrinsics/extrinsics are fixed inputs, not optimized.
- **Camera assumptions**: requires pre-solved, **calibrated intrinsics and
  extrinsics per camera**, from a separate chessboard-calibration module
  (`v2d_mv_calibration`). The optimizer hard-fails if per-camera frame
  counts differ, i.e. it assumes frame-aligned, synchronized multi-camera
  video streams — a fixed capture rig, not casual sequential phone photos.
- **Casual smartphone use case**: **not supported or documented anywhere in
  the repo.** No code path or doc mentions uncalibrated, sequential,
  single-camera capture. Reusing the core joint-optimization *principle*
  (one shared shape solved against multi-view reprojection error) for a
  guided front/side/back phone-photo flow is conceptually plausible but
  would require building our own calibration/pose-estimation front end —
  described here as a plausible **future R&D direction**, not something
  usable off the shelf.
- **Hardware**: Docker + NVIDIA Container Toolkit required; validated on
  RTX A6000/L40S-class GPUs. No VRAM figure documented for this specific
  module.

Added to `CANDIDATE_MATRIX.md` as a reference architecture: `RESEARCHED`,
not adopted, informative for a possible later multi-view enhancement
(Path C), not usable as-is for casual capture.

## Installation result

Working in an isolated Python **3.12** venv (clad-body's `pyproject.toml`
declares `requires-python = ">=3.12"`; this environment's default `python3`
is 3.11.15, but `python3.12` was already present via `apt`, so this was not
a real blocker, just a setup step):

1. `pip install -e .` (clad-body base, no extras): **14.5 s**, succeeded.
2. `pip install -e ".[mhr]"` (pulls `pymomentum-cpu`, `mhr`, `torch`):
   **5 min 11 s**, succeeded. Installed `torch==2.8.0+cu128` — PyPI's
   plain `torch` install defaults to the CUDA-tagged build even with no
   GPU present; the CPU-specific wheel index
   (`download.pytorch.org/whl/cpu`) is **not reachable** from this
   environment (network policy blocks that host; confirmed via the proxy's
   own status endpoint, which allowlists `pypi.org`/`files.pythonhosted.org`
   but not `download.pytorch.org`) — install/dependency blocker, recorded
   below.
3. MHR model assets: the **installed PyPI `mhr` package (1.0.1) does not
   ship the `mhr-download-assets` console script** that the GitHub repo's
   own source has (`mhr/download_assets.py` exists in the repo, not in the
   wheel) — a packaging gap between the repo and the PyPI release. Worked
   around by downloading `assets.zip` directly from the same public GitHub
   Releases URL the script would have used
   (`github.com/facebookresearch/MHR/releases/latest/download/assets.zip`,
   **no authentication required**), 199 MB, downloaded in ~2.2 s, unzipped
   to 4.5 GB under the venv's `site-packages/assets/`.
4. Total disk consumed: **~16 GB** (30 GB → 14 GB available), mostly torch
   (~6 GB incl. unused CUDA libraries) and the MHR asset bundle (4.5 GB).

`sam_3d_body` itself (the inference package) was **not installed** — its
own `INSTALL.md` requires compiling `detectron2` and other heavy
dependencies, and since checkpoint access is gated (below) and no GPU is
present, attempting that install would consume significant remaining
budget for zero additional evidence: we already have primary-source
confirmation (`INSTALL.md`'s own text) that checkpoint access requires
manual Hugging Face approval. This is recorded as a **deliberately
deferred** step, not a hidden failure.

## Checkpoint result — BLOCKED_BY_ACCESS

SAM 3D Body's `INSTALL.md` states plainly: checkpoints are hosted on
Hugging Face (`facebook/sam-3d-body-dinov3`, `facebook/sam-3d-body-vith`)
and require **manually requesting access and being approved** before
authenticated download is possible. No `HF_TOKEN` or other credential is
present in this environment (`env | grep -i hf` returned nothing). Per this
task's explicit instruction, **no credential was fabricated**; this is
recorded as a hard, unresolved access blocker for the image→MHR-params
half of the pipeline specifically. It does not block anything else in this
report — the MHR asset files (the body model geometry) are a **separate,
publicly-downloadable, Apache-2.0-licensed artifact** from the SAM 3D Body
*inference checkpoint* (which predicts params from an image); this
distinction was not clearly separated in Task 01 and is corrected here.

## Actual pipeline executed

Given the checkpoint blocker, the image→params half could not be run for
real. What was actually executed, in full, was:

```
existing MHR params JSON (clad-body's own bundled SAM3D-derived test fixture)
  → experiments/sam3d_mhr_clad_smoke/run.py --mhr-params-json ... --known-height-cm 180
  → _mhr_measure_worker.py (isolated subprocess)
  → clad_body.load.load_mhr_from_params(...)   <-- FAILS HERE (see below)
  → [never reached] clad_body.measure.measure(...)
```

This used clad-body's own repository-provided example data (test fixtures
under `clad_body/measure/testdata/mhr/`), not private customer data or
synthetic data of our own invention, per this task's sanity-test
constraints.

## Interoperability findings — MHR → clad-body blocker

`load_mhr_from_params` crashed the process with **SIGSEGV (signal 11)**
inside `pymomentum.geometry.Character.load_fbx`, called from
`mhr.mhr.MHR.from_files` (`mhr/mhr.py:155`), every time, reproducibly,
across two different bundled fixture files and both a direct call and a
call through our subprocess-isolated worker.

Diagnosis performed (each step below ruled something out):
- **Not a missing shared library**: `ldd` on every `pymomentum/*.so`
  resolved cleanly, no "not found" entries.
- **Not the FBX file itself**: `lod1.fbx` is a valid, correctly-sized
  Kaydara FBX v7700 file (downloaded from the same public release used
  above).
- **Not a numpy 2.x ABI issue**: downgrading to `numpy==1.26.4` did not
  change the outcome (crash persisted at the same line).
- **Not a CPU instruction-set gap**: this CPU supports AVX2, AVX-512F,
  AVX-512-VNNI, FMA — no missing ISA extension.
- **Most likely remaining explanation, not fully confirmed**:
  `pymomentum-cpu`'s own PyPI description states it is a "Torch C++
  extension package **linked against CPU PyTorch**." The only `torch`
  build obtainable in this environment via PyPI is the CUDA-tagged
  build (`+cu128`) — the CPU-specific wheel index needed to get a build
  ABI-matched to `pymomentum-cpu`'s expectations is network-blocked (see
  Installation result, step 2). A native-extension ABI mismatch between a
  CUDA-tagged and CPU-only torch build, despite an identical Python-facing
  version number, is a well-known class of PyTorch C++ extension crash —
  this fits the evidence but was not independently proven (we could not
  obtain the CPU wheel to test the fix directly).

This is recorded as an **install/dependency blocker specific to this
environment's network policy**, not a defect in clad-body's own code or
API design — the field-mapping and Python-level API worked exactly as
documented up to the point of the native crash, and clad-body's own
internal subprocess isolation of pymomentum correctly converted the crash
into a catchable `RuntimeError` rather than propagating a raw signal
(clad-body's `load_mhr_from_params` already shells out to a subprocess for
unrelated import-order reasons — that isolation incidentally also
contained this crash cleanly).

As a secondary, independent check, clad-body's **Anny** loader (no
pymomentum/native FBX dependency) was also tried, using clad-body's own
bundled Anny fixture — it did **not** segfault, but failed with a clean
Python `NotImplementedError: Pose parameterization root_relative_world not
implemented` inside the installed `anny` package, indicating a version
mismatch between the bundled fixture's expected pose parameterization and
the `anny` package version resolved by `pip`. This confirms the MHR-path
crash is specific to the native pymomentum/FBX loading step, not a general
problem with clad-body's Python-level measurement code — and surfaces a
second, independent (lower-severity) compatibility gap worth knowing about
before a real benchmark, though it is off this task's critical path (MHR,
not Anny, is what SAM 3D Body outputs).

**Our own orchestration code (`run.py`) did not crash in either case** — it
isolated both failures in a subprocess and reported a clean status and
full stderr in the machine-readable output. See "Sample output" below.

## Sample output

Actual JSON produced by `run.py` against a bundled fixture
(`clad_body/measure/testdata/mhr/female_average/mhr_params.json`,
`--known-height-cm 165`):

```json
{
  "subject_id": "female_average_fixture",
  "source_image": null,
  "sam3d_inference_status": "skipped_params_provided",
  "mhr_params_available": true,
  "mhr_params_source": "provided_json",
  "measurement_extraction_status": "error: worker exited with code 1",
  "raw_body_height_cm": null,
  "known_height_cm": 165.0,
  "rescale_applied": false,
  "rescale_factor": null,
  "measurements_cm": null,
  "runtime_seconds": { "measurement_extraction": 3.887, "total": 5.242 },
  "device": "cpu",
  "warnings": [],
  "failures": [
    "measurement_extraction_stderr_tail: ... RuntimeError: pymomentum subprocess failed:\nunknown error\n"
  ],
  "python_version": "3.12.3"
}
```

No successful `measurements_cm` was produced in this environment. This is
reported honestly rather than substituted with a fabricated or
placeholder result.

## Compute measurements (actually measured, not estimated)

| Metric | Value | How measured |
|---|---|---|
| GPU / VRAM | None present | `TASK02_ENVIRONMENT.md` probe |
| clad-body base install time | 14.5 s | `time pip install -e .` |
| clad-body[mhr] extras install time | 5 min 11 s | `time pip install -e ".[mhr]"` |
| MHR asset download | 199 MB in ~2.2 s | `time curl` against public GitHub release |
| MHR asset size on disk (unzipped) | 4.5 GB | `du -sh` |
| Total disk consumed by full setup | ~16 GB | `df -h` before/after |
| Peak RSS, `run.py` end-to-end (failed) run | 876,632 KB (~856 MB) | `/usr/bin/time -v` |
| Peak RSS, worker subprocess alone (failed) | 875,756 KB (~855 MB) | `/usr/bin/time -v` |
| Wall time, `run.py` end-to-end (failed) run | 5.90 s | `/usr/bin/time -v` |
| Model-load / inference time (SAM 3D Body) | Not measured | checkpoint access blocked |
| clad-body `measure()` time (successful case) | Not measured | never reached due to MHR load crash |

## Minimum practical GPU class (based on upstream evidence, not run here)

SAM 3D Body's largest published checkpoint is ~840M parameters (ViT-H
DINOv3); the smaller is ~631M. No official VRAM/runtime figure is stated
in the repo. Comparable-scale ViT-based HMR models in this space (per Task
01's research on Multi-HMR) were benchmarked on V100-32GB-class hardware.
A consumer 24 GB card (RTX 3090/4090-class) is very likely sufficient for
inference (not training) at this parameter count; this is an inference
from comparable model sizes, not confirmed by running SAM 3D Body itself.

## Cost estimate for the next small benchmark (not purchased)

Current spot/community GPU rental pricing (checked live, not from stale
knowledge): RTX 4090-class instances are available around **$0.29–$0.69/hr**
on Vast.ai and RunPod as of ~April 2026
([Vast.ai RTX 4090 pricing](https://www.synpixcloud.com/blog/vast-ai-vs-runpod-rtx-4090-pricing),
[RunPod RTX 4090](https://www.runpod.io/gpu-models/rtx-4090)). A first real
benchmark — environment setup, checkpoint download (once HF access is
approved), and inference over ~10 images plus clad-body measurement —
would plausibly take 2–4 hours of instance time, i.e. **roughly
$0.60–$2.80**, comfortably under the task's preferred $10 ceiling and the
project's $100 ceiling. This is an estimate for planning only; nothing was
purchased or provisioned.

## Metric-scale investigation

This was investigated precisely from source, not inferred:

1. **Does SAM 3D Body recover absolute metric body dimensions?**
   Conditionally, and only as well as its FOV estimator. The demo pipeline
   by default uses **MoGe2** (`tools/build_fov_estimator.py`) to estimate
   camera intrinsics (focal length) from the image itself, which feeds the
   model's camera-aware prediction. The estimator's own constructor prints
   `"No FOV estimator... Using the default FOV!"` when none is supplied —
   confirming that **without MoGe2 (or another FOV source), a fixed
   default FOV constant is used instead of anything derived from the
   actual photo**, which would not correspond to the real camera and would
   corrupt any absolute-scale claim. Per Task 01's own research, MoGe2's
   published accuracy is on general scene benchmarks, not human bodies —
   so even the "with FOV estimator" case carries UNKNOWN human-specific
   accuracy.

2. **Correction to a Task 01 assumption**: a field named `raw_height_m`
   appears in clad-body's bundled SAM3D-derived test fixtures (e.g.
   `1.657839 m`). We searched the *actual* `sam-3d-body` source for this
   field and **it does not exist anywhere in that repo** —
   `SAM3DBodyEstimator.process_one_image()` never produces a
   `raw_height_m` key. It is referenced only in a code *comment* inside
   clad-body's own `mhr.py` (listing it as one of several fields that
   *might* appear in a `sam3d_params` dict) and is **never read** by
   `load_mhr_from_params`'s actual logic. Its provenance in the shipped
   fixtures is therefore **UNKNOWN** — most plausibly computed by
   clad-body's own (unpublished) fixture-generation script from the
   resulting mesh, not a value SAM 3D Body itself outputs. **Do not treat
   `raw_height_m` as an authoritative SAM-3D-Body-native metric height.**

3. **Are MHR outputs normalized, camera-relative, or metrically
   meaningful?** Neither purely normalized nor camera-relative, as
   actually consumed by clad-body: `load_mhr_from_params` **ignores**
   `pred_cam_t` and `focal_length` entirely. It reconstructs a rest-pose
   mesh purely from `shape_params` and the scale slice of
   `mhr_model_params`, decoded through the MHR body model's own learned
   parameter space. Whatever real-world scale the resulting mesh has comes
   entirely from how well SAM 3D Body's regression network predicts those
   parameters against MHR's training-data-derived scale prior — **not**
   from any per-photo camera geometry correction inside this particular
   loading path. The camera-space fields SAM 3D Body also outputs
   (`pred_cam_t`, `focal_length`) are for reprojecting the mesh back onto
   the original image (e.g. for overlay visualization) and play **no role**
   in clad-body's measurement pipeline as currently used.

4. **Does clad-body assume the MHR body already has correct scale?**
   Yes, implicitly and entirely — there is no scale-correction step in
   `load_mhr_from_params` at all.

5. **Can known customer height be used as a deterministic scale
   constraint?** Yes, and this is exactly what we implemented — but
   **not blindly**. `experiments/sam3d_mhr_clad_smoke/rescale.py` applies
   a single **uniform isotropic scale factor** (`known_height_cm /
   computed_mesh_height_cm`) to all mesh vertices, off by default and only
   applied when `--known-height-cm` is passed. It is explicitly documented
   (module docstring) as correcting **overall scale only** — it does
   **not** correct camera-perspective distortion, posture effects, or
   non-uniform proportion errors (e.g. a torso predicted proportionally
   too long relative to the legs would remain wrong after a uniform
   rescale). This was validated with 8 unit tests against synthetic
   geometry (isotropy check, plausibility-range rejection, degenerate-mesh
   rejection, no-op-at-matching-height check) — it could not be validated
   against a real MHR mesh in this environment because the MHR loader
   itself does not run here (see blocker above).

## Measurement extraction is not body accuracy

Stated explicitly, per this task's instruction: if clad-body's `measure()`
had run successfully on a SAM-3D-Body-derived mesh, that would prove
clad-body can compute a chest/waist/etc. circumference **from the
predicted mesh** — it would **not** prove SAM 3D Body reconstructed the
photographed person's true body shape. In this task, we did not even reach
that first, weaker claim: the MHR loading step itself failed in this
environment, so **no measurement of any kind, on any mesh, was actually
produced** by the real clad-body MHR path. `run.py`'s output JSON carries
this caveat verbatim in its `note` field so it can't be silently dropped
in a later summary.

## Measurement-definition mapping (clad-body/ISO → MTM terminology)

Full table with per-measurement notes and confidence levels:
`experiments/sam3d_mhr_clad_smoke/mtm_mapping.py` (tested,
`tests/test_mtm_mapping.py`). Summary:

| MTM measurement | clad-body key | Confidence | Why |
|---|---|---|---|
| Height | `height_cm` | direct | Standing-height convention closely matches |
| Chest/bust circumference | `bust_cm` | review | ISO landmark vs. tailor ease-adjusted tape technique differ |
| Waist circumference | `waist_cm` | review | Exact landmark (natural waist vs. narrowest point) unverified vs. house convention |
| Seat/hip circumference | `hip_cm` | review | Convex-hull vs. raw-contour gap plausible on curvier bodies |
| Shoulder width | `shoulder_width_cm` | review | clad-body's own largest published internal calibration error (RMS 1.39 cm, max 5 cm); landmark placement varies by house |
| Sleeve/arm length | `sleeve_length_cm` | review | Straight sleeve-pattern length vs. tailor's bent-arm convention may differ |
| Upper-arm/bicep | `upperarm_cm` | direct | Tight internal calibration (≤1 cm), close concept match |
| Neck circumference | `neck_cm` | review | Landmark well-documented (ISO §5.3.2, below Adam's apple) but tailoring ease allowance unverified |
| Back length | none found | **gap** | No confirmed clad-body key |
| Front torso length | none found | **gap** | No confirmed clad-body key |
| Wrist circumference | `wrist_cm` | review | Key exists; calibration error not independently pulled |
| Inseam | `inseam_cm` | review | Tight internal calibration, but different measurement *method* (on-body vs. on-garment) than typical tailoring convention |
| Outseam | none found | **gap** | No confirmed clad-body key |
| Thigh circumference | `thigh_cm` | review | Tight internal calibration; landmark height not independently confirmed |

Three of thirteen target measurements (back length, front torso length,
outseam) have **no confirmed clad-body key at all** — a real coverage gap,
not previously stated this precisely in Task 01. All calibration-error
figures cited from clad-body's own README are against clad-body's **own
internal reference** (its differentiable vs. non-differentiable code
paths), not real human tape measurements — this was already flagged in
Task 01 and remains unresolved.

## Failures (by category, per this task's required breakdown)

- **Install blocker**: `download.pytorch.org` (the CPU-only PyTorch wheel
  index) is unreachable from this environment's network policy; only the
  CUDA-tagged `torch` build is obtainable via plain PyPI.
- **Checkpoint/access blocker**: SAM 3D Body checkpoints require manual
  Hugging Face access request/approval; no credential available, none
  fabricated.
- **GPU blocker**: no GPU present in this environment at all (separate
  from the checkpoint blocker — even with a checkpoint, inference here
  would be CPU-only and, per Task 01's model-size research, likely
  impractically slow for a large ViT).
- **Dependency blocker**: PyPI's `mhr==1.0.1` wheel omits the
  `mhr-download-assets` console script present in the GitHub source;
  worked around by downloading the same public release archive directly.
- **SAM3D→MHR format blocker**: resolved during this task (see section A)
  — not a remaining blocker, but was one until the source was read
  directly.
- **MHR→clad-body blocker**: `pymomentum`'s native FBX loader segfaults in
  this environment, most likely from a CUDA-vs-CPU-tagged torch build ABI
  mismatch that could not be corrected here due to the install blocker
  above. This is the pipeline's current hard stop.
- **Metric-scale blocker**: none found blocking implementation — the
  known-height rescale is implemented and unit-tested, though unverified
  against a real MHR mesh for the reason above.
- **Measurement-definition blocker**: three of thirteen MTM measurements
  (back length, front torso length, outseam) have no confirmed clad-body
  coverage at all.
- **Licensing blocker**: none newly found in this task. SMPL/SMPL-X
  remains excluded from the implemented path per the standing project
  decision (section B) — not used anywhere in this task's code.

## Licensing notes (new findings this task)

- **MHR model/asset files** (not just code) are **Apache-2.0**, confirmed
  by reading the `LICENSE.txt` bundled directly inside the public
  `assets.zip` release — this is a stronger, more specific finding than
  Task 01's general "MHR ecosystem, code Apache-2.0." The actual 3D body
  model geometry/rig, not just the Python wrapper, carries a fully
  commercial-friendly license, publicly downloadable with no gating.
- **SAM 3D Body's inference checkpoint** is under Meta's custom "SAM
  License" (unchanged from Task 01) and is separately Hugging-Face-gated —
  this is a distinct artifact from the MHR assets above and should not be
  conflated with them. Two different licenses, two different access paths,
  one shared "MHR" name.
- No other new licensing findings this task.

## NVIDIA multi-view — answers to the six required questions

1. **Does it optimize one body across multiple views?** Yes, genuinely
   jointly (shared parameters solved against a summed multi-view
   reprojection loss), not per-view-then-pick.
2. **Which body parameters are shared?** `shape_params` and `scale_params`,
   held as one row for the whole sequence. Pose/translation/rotation are
   per-frame, not shared. Camera intrinsics/extrinsics are fixed inputs,
   never optimized.
3. **What camera information does it require?** Pre-calibrated per-camera
   intrinsics and extrinsics, from a separate chessboard-calibration
   module.
4. **Does it assume simultaneous calibrated cameras?** Yes — frame-aligned,
   synchronized multi-camera streams; the code hard-fails on mismatched
   per-camera frame counts.
5. **Could the core principle be reused for guided smartphone
   front/side/back captures?** Conceptually plausible (the underlying idea
   — one shared shape, multi-view reprojection consistency — doesn't
   inherently require synchronization), but not supported today.
6. **Would that require substantial R&D?** Yes — we would need to build our
   own calibration/pose-estimation front end (the module has none of its
   own; it depends entirely on the separate rig-calibration module), which
   is a nontrivial, undocumented extension, not a configuration change.

## Conclusion

The `SAM 3D Body → MHR → clad-body` pipeline is **not currently runnable
end-to-end** in this environment. The blocker chain, from least to most
fundamental:

1. This environment cannot download SAM 3D Body's checkpoint (access
   gated, no credential) — blocks the image→params half entirely.
2. This environment has no GPU — would make that half impractically slow
   even with a checkpoint.
3. The MHR→clad-body half, which **can** be tested independently of both
   of the above using clad-body's own bundled example data, **also fails**
   here, on a native crash traced to a likely torch-build ABI mismatch
   this environment's network policy prevents fixing directly.

None of these are the SAM3D→MHR *field-mapping* problem Task 01 flagged as
unresolved — that is now resolved, documented, implemented, and tested.
What remains blocked is getting either half of the pipeline to actually
execute in *this specific* environment. The pipeline's design is sound and
its two halves are each independently well-specified; getting it running
is now an environment-provisioning problem (a GPU instance with unrestricted
PyPI/HF network access), not an unresolved integration-design problem.

## Tests

26 tests across `test_adapter.py` (11), `test_rescale.py` (8), and
`test_mtm_mapping.py` (7), all passing, all running under plain Python
3.11 with only `pytest`+`numpy` installed — no clad-body, no pymomentum, no
torch, no checkpoint download required, confirmed by running them outside
the venv used for the (failed) real pipeline attempt.

## Decision gate

**D. BLOCKED_BY_ACCESS**, compounded by a **C. BLOCKED_BY_COMPUTE**
(no GPU) finding for the image→params half, and an **E. BLOCKED_BY_
INTEGRATION** finding (native crash, likely fixable with unrestricted
network access to the correct PyTorch wheel index) for the MHR→clad-body
half. Access is listed as primary because it is the more fundamental,
harder-to-work-around blocker: even a GPU instance does not help until SAM
3D Body checkpoint access is approved, whereas the integration blocker has
a plausible fix already identified (a CPU-only torch build) that just
needs a network policy allowing `download.pytorch.org`, or an environment
where `pip install pymomentum-cpu` resolves a compatible torch
automatically. Neither blocker reflects a flaw in the pipeline's design
(section A resolved the one genuine design ambiguity Task 01 left open),
and neither is a licensing problem (section F does not apply — no
SMPL-dependent or otherwise commercially-blocked component is in this
task's implemented path). Do not reclassify this as
`NOT_TECHNICALLY_PROMISING` — nothing found here suggests the pipeline
itself is unsound, only that this particular sandboxed environment cannot
currently execute it.
