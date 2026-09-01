# Task 03 — Colab GPU End-to-End Smoke Test

## Important: this notebook was authored, validated, and NOT executed by this agent

This agent's execution environment is the same sandboxed CLI environment
used for Task 02: confirmed **no GPU** (`nvidia-smi` absent, no
`/proc/driver/nvidia`), no Google Colab runtime, and no `HF_TOKEN` (nor
should one be provided here — the token belongs in Colab Secrets, per this
task's own security requirement, not in this agent's shell environment).

Per the notebook's own section 1 logic — "if no GPU is available, stop with
a clear message rather than attempting full inference on CPU" — this agent
correctly stops rather than fake a run. What follows is what *was* done
(notebook authored, structurally validated, reused/extended Task 02 code,
all unit tests passing) and what remains for a human to actually execute in
real Google Colab, where a free GPU tier and unrestricted internet access
(neither available in this sandbox) make the pipeline plausible to run for
real. Nothing in this document should be read as reporting an actual
executed pipeline run or actual measurements — none were produced.

## Notebook setup

`notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` — 35 cells (18 markdown, 17
code), organized into the 15 sections the task specifies (some sections
share one cell, e.g. sections 6–11 and section 12's note, to avoid
unnecessary fragmentation). Validated:
- Valid JSON, valid `nbformat` 4 schema (`nbformat.validate()` passes).
- Every code cell parses as valid Python (`ast.parse()` on each cell's
  source, zero syntax errors).
- Not validated: actual execution, since that requires a GPU + Colab +
  approved HF access this sandbox does not have.

## How HF_TOKEN is provided securely

Read only via `google.colab.userdata.get('HF_TOKEN')` (Colab Secrets).
Never printed, logged, or written to a file. Specifically:
- No call to `huggingface_hub.login()` anywhere in the notebook — that
  function persists the token to `~/.huggingface/token` on disk, which
  would violate "do not save it." Every Hugging Face API call instead
  receives the token explicitly via a `token=` keyword argument.
- The one message that echoes anything token-adjacent
  (`access_status[repo_id] = f'BLOCKED: {type(exc).__name__}'`) stores only
  the *exception class name*, not its message text or the token — some HF
  client exceptions can echo request headers in their string
  representation, so only the type is kept.
- If `HF_TOKEN` is missing or empty, the notebook raises immediately with a
  clear message and stops (per this task's explicit requirement) — nothing
  downstream silently proceeds without it.

## Dependency versions — design and rationale (not yet confirmed by an actual install)

Task 02 found that `pymomentum-cpu` — described on PyPI as "linked against
CPU PyTorch" — segfaults when the *same process* also has a CUDA-tagged
`torch` build loaded (this sandbox's only obtainable `torch` build, since
the CPU-specific wheel index `download.pytorch.org` was network-blocked
here). Real Colab does **not** have that network restriction. The
notebook's dependency strategy (section 4) is built directly on this
diagnosis, reusing clad-body's own existing subprocess-isolation pattern
rather than patching any third-party source:

- **Main Colab kernel**: Colab's own pre-installed CUDA-enabled `torch` is
  left untouched. Only SAM 3D Body's own dependencies are added
  (`detectron2` at the commit pinned in upstream `INSTALL.md`, `MoGe`, and
  the plain-pip dependency list from that same file). This is what runs
  actual GPU inference.
- **A separate venv** (`/content/clad_env`) gets a **CPU-only** `torch`
  build from `download.pytorch.org/whl/cpu` (reachable from real Colab,
  unlike Task 02's sandbox), then `clad-body[mhr]` on top of it — giving
  `pymomentum-cpu` the exact ABI it expects, isolated from the GPU
  process's CUDA torch.
- `run.py`'s `measure_via_subprocess()` gained one new optional parameter,
  `python_executable`, so it can be pointed at this second venv's
  interpreter instead of always using its own (`sys.executable`, Task 02's
  behavior, still the default when the parameter is omitted — fully
  backward compatible).
- MHR's body-model/rig assets are fetched the same way Task 02 established
  works (direct download from the public, unauthenticated
  `facebookresearch/MHR` GitHub release archive — PyPI's `mhr` package
  omits the `mhr-download-assets` console script that would otherwise do
  this).

**This design is a reasoned prediction based on Task 02's diagnosis, not a
confirmed fix** — it could not be tested in this sandbox (no GPU to
validate the "does SAM 3D Body actually run" half, and this sandbox's
network policy would block the CPU-torch download even for the
`clad_env` half). The first real Colab run is exactly the experiment that
confirms or refutes it. Exact installed versions (torch, CUDA, pymomentum,
mhr, clad-body, detectron2, etc.) will only be known once that run
executes — the notebook prints all of them in section 1 and section 4.

## Actual checkpoint used

**Not yet determined by execution.** The notebook is designed to test
authenticated access to both `facebook/sam-3d-body-dinov3` and
`facebook/sam-3d-body-vith` (section 5), then download only
`facebook/sam-3d-body-dinov3` — the checkpoint upstream's own
`INSTALL.md`/README example command uses as its first/default example —
unless the optional section 15 comparison is explicitly enabled.

## Actual execution path

Not yet run. Designed path once a human runs it in Colab:

```
sample image (bundled in sam-3d-body repo, public) or user upload
  -> SAM 3D Body inference (GPU, main kernel)
  -> adapter.sam3d_output_to_clad_params() [Task 02 code, reused unchanged]
  -> _mhr_measure_worker.py, run in the separate CPU-torch venv via
     run.py's measure_via_subprocess(python_executable=clad_venv_python)
  -> clad_body.load.load_mhr_from_params() + clad_body.measure.measure()
  -> raw measurements, and (if KNOWN_HEIGHT_CM is set) a second,
     separately-computed calibrated result
  -> results/measurement_output.json (Colab runtime; also copied into the
     cloned repo's experiments/sam3d_mhr_clad_smoke/results/ only when the
     public sample image was used, never for a personal upload)
```

## Real output schema

Not yet captured — this requires an actual SAM 3D Body inference call.
Task 02's source-reading established the *expected* field names
(`shape_params`, `mhr_model_params`, `scale_params`, `pred_cam_t`,
`focal_length`, etc.) from `sam_3d_body/sam_3d_body_estimator.py`; the
notebook's section 8 prints the *actual* schema from a real call rather
than assuming Task 02's source-derived expectation is exactly right in
practice. Section 9 explicitly re-confirms, against real output, that
`scale_params` (28-dim) is distinct from the decoded scale inside
`mhr_model_params` (204-dim) — the trap Task 02 identified from source.

## Measurements produced

None — no execution has occurred. The notebook, once run, writes both raw
and (optionally) height-calibrated measurements to
`results/measurement_output.json`, mapped to MTM terminology via Task 02's
unchanged `mtm_mapping.py`.

## Raw vs. height-calibrated measurements

Implemented as two independent calls to the same measurement worker: one
with `known_height_cm=None` (raw), one with it set (calibrated) — never
computed by mutating the raw result in place. The calibrated call uses
Task 02's `uniform_rescale_to_height()` unchanged: a single uniform scalar
(`known_height_cm / raw_body_height_cm`) applied to every mesh vertex,
which corrects overall scale only and cannot correct proportions,
posture, or perspective effects — documented both in `rescale.py`'s
docstring and restated in the notebook's section 12.

## Measured compute

None yet — requires actual execution. The notebook's section 14 collects,
from real measurements (not estimates): peak GPU VRAM
(`torch.cuda.max_memory_allocated()`), system RAM, checkpoint size on disk,
checkpoint download time, model load time, SAM 3D Body inference time,
`clad_env` setup time, and measurement-extraction time.

## Blockers/failures (as of this task)

This agent's own attempt: **no GPU present** in this sandbox — matches the
`NO_GPU` failure category and the `GPU_INSUFFICIENT` decision gate exactly
(see Decision gate below). No HF authentication was attempted (would
require the token in this agent's environment, which the task explicitly
prohibits). No dependency install, checkpoint download, or inference was
attempted in this sandbox, consistent with "do not attempt full inference
on CPU" and "never fabricate credentials."

Two design risks carried forward for the human's real Colab run to reveal:
1. The `clad_env` CPU-torch fix (above) is untested — Colab's actual
   network/package-resolution behavior could still surprise us.
2. `sam-3d-body`'s own install (`detectron2` built from a pinned commit,
   `--no-build-isolation`) has real potential to fail against whatever
   CUDA/torch versions a given Colab GPU runtime ships with that week —
   Colab's base image changes over time and is not fully within this
   project's control.

## Licensing notes

No change from Task 02. SAM 3D Body's checkpoint license (Meta's "SAM
License") and MHR's asset-file license (Apache-2.0) are unchanged; both
apply exactly as documented in `LICENSE_AND_COMMERCIAL_USE.md`.

## Exact reproducibility steps for the human

1. Open `notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` in Google Colab
   (upload it, or open directly from GitHub once this branch is pushed:
   `File > Open notebook > GitHub`, paste the repo URL and branch).
2. `Runtime > Change runtime type` → select a GPU (the free T4 tier is
   fine to start).
3. Click the key icon in the left sidebar → add a secret named exactly
   `HF_TOKEN` → paste your Hugging Face **read** token (the one already
   approved for `facebook/sam-3d-body-dinov3` and
   `facebook/sam-3d-body-vith`) → toggle notebook access on.
4. `Runtime > Run all`. Section 1 stops immediately with a clear error if
   no GPU is attached; section 2 stops immediately if `HF_TOKEN` is
   missing/inaccessible; section 5 stops if checkpoint access isn't
   confirmed.
5. Section 6 defaults to the bundled public sample image — no upload
   needed for a first run. Set `USE_SAMPLE_IMAGE = False` in that cell to
   upload your own photo instead.
6. To test known-height calibration, set `KNOWN_HEIGHT_CM` (e.g. `178.0`)
   in the section 7 cell before running it.
7. Section 16 prints the actual decision gate (A–F) for that run, with the
   full recorded state.
8. `results/measurement_output.json` is written to the Colab runtime
   (`/content/results/`); it is copied into the cloned repo only if the
   public sample image was used, and even then is **not** committed or
   pushed automatically — that's a deliberate manual step, left to the
   human, so no result is ever pushed without a human looking at it first.

## Decision gate

**C. GPU_INSUFFICIENT** — for this agent's own execution attempt in this
sandbox, which has no GPU at all. This is the only honest classification
available without fabricating results: the notebook was authored,
structurally validated (valid nbformat, every cell's Python syntax
checked), and reuses/extends Task 02's tested code, but **zero cells were
actually executed** by this agent, and no numeric measurement of any kind
was produced. Do not read this as a claim that the pipeline will fail in
real Colab — it is a direct statement that this agent's own environment
cannot run it, exactly per the notebook's own designed behavior in that
situation. The real classification (plausibly A, B, D, or E depending on
what the human's Colab run actually encounters) can only be determined by
that run; re-running this smoke test with the human's actual Colab output
is the natural next step, not a new task.
