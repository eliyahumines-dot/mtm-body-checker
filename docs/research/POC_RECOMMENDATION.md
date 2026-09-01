# POC Recommendation — What To Test First

This document answers the task's required end-of-task questions and
proposes the concrete first experiment. It does not authorize building the
full system — per the task brief, this phase is research + repo bootstrap
only.

**Task 02 update:** the pipeline recommended below was implemented as a
smoke test (`experiments/sam3d_mhr_clad_smoke/`) and given a decision gate
of **`BLOCKED_BY_ACCESS`** (SAM 3D Body checkpoint requires manual
Hugging Face approval, not available in that task's environment),
compounded by `BLOCKED_BY_COMPUTE` (no GPU present there) and
`BLOCKED_BY_INTEGRATION` (clad-body's native MHR loader segfaulted, likely
a fixable PyTorch-build ABI issue). None of these are licensing or design
blockers — the one real design ambiguity (SAM3D↔clad-body field mapping)
was resolved, implemented, and unit-tested in Task 02. Full detail:
`docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md`.

## 1. Is a useful no-training POC technically plausible?

Yes, with caveats. A no-training pipeline can plausibly be assembled from
existing pretrained components (see Path B in the audit). Whether it hits
*useful* accuracy for MTM-grade discrepancy detection is unproven — no
candidate found publishes credible cm-level anthropometric accuracy
evidence. Plausible ≠ proven; that's exactly what the first benchmark must
establish.

## 2. Strongest plausible open-source pipeline for this use case?

```
Guided front + side smartphone photos + customer-reported height
  → SAM2 (subject silhouette) + MediaPipe/RTMPose (landmarks, capture QA)
  → SAM 3D Body → MHR mesh
  → clad-body → ISO 8559-1-aligned measurements
  → reimplemented clothing-looseness overlap-ratio check (ECON's idea, our code)
  → statistical comparison against customer-reported measurements
```

This is the only surveyed combination that is both technically plausible
end-to-end and avoids the SMPL/SMPL-X commercial-licensing question
entirely. A parallel, more mature but licensing-encumbered alternative
(4D-Humans + SMPL-Anthropometry) exists for internal R&D/benchmarking only,
pending resolution of SMPL's commercial terms.

## 3. Which parts appear usable immediately?

- MediaPipe Pose / RTMPose for capture guidance and landmark QA — mature,
  on-device, Apache-2.0.
- SAM2 for clean subject silhouettes — mature, Apache-2.0.
- Known customer height as the scale anchor — zero-risk, already part of
  MTM intake.
- GeoCalib for camera-intrinsics recovery when EXIF is missing —
  commercially clean, low integration cost.

## 4. Which parts remain scientifically uncertain?

- SAM 3D Body's mesh accuracy in general, and specifically whether it
  reproduces the "regresses to standard body shape" failure mode an
  independent study found — directly relevant since MTM customers are by
  definition often non-average.
- clad-body's measurement accuracy against real tape measurements (its own
  published numbers are self-referential, not ground-truth validated).
- Whether the SAM-3D-Body → clad-body handoff (MHR mesh format
  compatibility) even works without a custom adaptation layer — untested.
- Whether 1–2 casual guided photos give enough signal for girth
  measurements at all, versus needing 3–4 views.
- Whether the known-height scale anchor is accurate enough, or whether a
  depth/intrinsics signal is required to hit useful precision.

## 5. Likely biggest accuracy bottleneck?

Absolute (metric) scale recovery from casual monocular smartphone photos,
compounded by loose-clothing ambiguity in girth measurements (chest, waist,
seat, thigh, bicep). Length measurements (shoulder width, sleeve, inseam)
are geometrically simpler and likely to be more reliable than
circumferences, which require inferring depth/volume the camera never
directly observed.

## 6. Can the proposed POC be tested within $0–$100 total compute?

Yes. All primary candidates in Path B (SAM3D Body, SAM2, MediaPipe/RTMPose,
clad-body, GeoCalib) are inference-only, run on modest/free-tier GPU
compute or on-device, and require no training. A small benchmark (a few
dozen images, a handful of subjects) run on a single rented GPU instance or
free-tier notebook is well within $100. The one item that must NOT be
assumed free is any SMPL/SMPL-X commercial sub-license — that stays outside
this budget line and requires separate, explicit approval with a quoted
price before being pursued.

**Task 02 grounded this estimate in live pricing** (not stale knowledge):
RTX 4090-class instances run **~$0.29–$0.69/hr** on Vast.ai/RunPod as of
~April 2026. A first real benchmark (environment setup, checkpoint
download once HF access is approved, inference over ~10 images, clad-body
measurement) would plausibly take 2–4 hours, i.e. **roughly $0.60–$2.80**
— well under the task's preferred $10 ceiling for the first real
benchmark. Nothing has been purchased or provisioned yet.

## 7. Which 2–3 candidate pipelines should actually be RUN next?

1. **SAM 3D Body → clad-body (Path B core).** Highest priority: commercially
   clean, most likely production-relevant, least validated.
2. **MediaPipe/RTMPose length-only pipeline (Path A).** Cheapest, fastest
   to validate, gives a fallback that ships even if Path B's girth
   measurements underperform.
3. **4D-Humans → SMPL-Anthropometry, for internal benchmarking only (not
   customer-facing).** Run purely to establish an accuracy *reference
   point* — since this is the most mature/well-documented mesh-recovery
   method, it tells us whether Path B's SAM-3D-Body numbers are in the
   right ballpark, independent of the licensing question, which can be
   resolved separately before any production use.

**Task 02 update — what actually needs to happen next to run #1:** two
concrete, non-architectural blockers, both cleared by moving to a
different environment rather than more design work: (a) request and obtain
Hugging Face access approval for `facebook/sam-3d-body-dinov3` or
`facebook/sam-3d-body-vith`, and (b) run in an environment with a GPU and
unrestricted access to the PyTorch wheel index (`download.pytorch.org`),
so `pymomentum-cpu` resolves against a matching torch build instead of the
CUDA-tagged default that caused a native crash in Task 02's sandbox. The
SAM3D→clad-body adapter code itself (`experiments/sam3d_mhr_clad_smoke/`)
is already implemented and unit-tested and needs no further design work
before that run.

## 8. What exact input data will be needed to compare them fairly?

- A small set (start with 5–10) of consenting individuals with real,
  independently taken tape measurements for the full MTM measurement list
  (chest, waist, seat/hip, shoulder width, neck, sleeve, back length, front
  torso length, bicep, wrist, inseam, outseam, thigh) as ground truth.
- Standardized guided photos per subject: front + side at minimum, ideally
  a 3rd (back or 3/4) view, taken at a documented distance under
  documented (fitted, not loose) clothing conditions, with self-reported
  height recorded.
- A repeat capture of a subset of subjects (different clothing, different
  day) to separately measure *within-pipeline consistency* vs.
  *ground-truth accuracy* — the task's own discrepancy-detection goal cares
  about both.
- All raw images and measurement data must NOT be committed to this repo
  (see `.gitignore`); store them outside the project tree and record only
  aggregated results in `docs/experiments/`.

## 9. What measurements should be included in the first benchmark?

All 13 target measurements should be attempted, but expectations should be
set per type going in:
- **Higher-confidence candidates for first benchmark:** shoulder width,
  sleeve length, inseam, neck (length-type or simple-circumference
  measurements with more direct geometric mapping to photo evidence).
- **Higher-uncertainty candidates:** chest, waist, seat/hip, thigh, bicep
  (circumferences requiring volumetric inference — expect larger error and
  larger clothing sensitivity).
- Back length, front torso length, wrist, outseam: include but treat as
  secondary/exploratory given weaker coverage across surveyed measurement
  libraries.

## 10. What should NOT be built yet?

- No production pipeline, API, or customer-facing UI.
- No SMPL/SMPL-X-dependent path for anything customer-facing, until the
  commercial-licensing question is resolved with an actual quote.
- No model training or fine-tuning of any kind, including a "small
  calibration correction" — that's explicitly Path C, later, and needs its
  own approval.
- No multi-view/video/photogrammetry pipeline (COLMAP or otherwise) — Path
  C, only if 1–3 static photos prove insufficient.
- No depth-model integration (MoGe, Depth Anything, Metric3D) until the
  known-height anchor has been benchmarked and found insufficient — don't
  add ML-based scale recovery before checking whether the free, zero-risk
  option is already good enough.

## Recommended immediate next task

Run the SAM 3D Body → clad-body pipeline (Path B core) against a small
(5–10 subject) benchmark set with real tape-measurement ground truth,
alongside the MediaPipe/RTMPose length-only pipeline (Path A) as a
low-cost comparison baseline, and record results in
`docs/experiments/`. Resolve the MHR mesh-format compatibility between
SAM 3D Body and clad-body as the first technical checkpoint — if that
handoff doesn't work cleanly, that alone determines whether Path B needs a
custom adaptation layer before anything else proceeds.
