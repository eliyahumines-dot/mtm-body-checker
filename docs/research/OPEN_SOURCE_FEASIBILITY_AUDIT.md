# Open-Source Feasibility Audit — Smartphone Body Measurement for MTM

Status: initial audit (Task 01). No architecture is selected. All candidates
below remain in the `RESEARCHED` decision state unless noted otherwise (see
`CANDIDATE_MATRIX.md` for the full state table).

Method note: research was performed against primary sources (official
repos, papers, license pages) via parallel research passes per pipeline
stage. Several primary domains (`*.tue.mpg.de`, `arxiv.org` PDFs,
`ai.meta.com`) were unreachable from this environment's network egress at
research time; those claims are explicitly marked `UNVERIFIED-PRIMARY`
(sourced from cached search snippets or reputable mirrors, not a directly
fetched primary page) and should be re-confirmed with a direct fetch before
any decision that depends on them, especially licensing.

## Research question

How close can we get today to useful smartphone-based body measurement and
MTM discrepancy detection by integrating existing open-source pretrained
technology, with no new training? This audit traces the shortest credible
path: **smartphone images → 3D/geometric body representation →
anthropometric measurements → discrepancy detection**, and evaluates
candidates at each stage against REPORTED (author claims) vs REPRODUCED
(what we've actually run — nothing yet) vs RELEVANT (evidence it preserves
measurable body shape, not just pose accuracy) vs UNKNOWN.

## Central finding: licensing, not compute, is the binding constraint

Every strong 3D human mesh recovery method surveyed except one is built on
**SMPL or SMPL-X**, whose model license is "non-commercial scientific
research, education, or artistic projects" only, with an explicit ban on
"incorporation in a commercial product" and "production of other artifacts
for commercial purposes." Whether "artifacts" covers a spreadsheet of
measurements (not a redistributed mesh) is not resolved by any primary
source found — this is a real legal ambiguity, not a technicality, and the
conservative reading is that a commercial MTM pipeline touching
SMPL/SMPL-X/STAR/ICON/ECON needs a paid MPI-IS/Meshcapade commercial
license. Full detail in `LICENSE_AND_COMMERCIAL_USE.md`.

The one architecturally distinct exception is **Meta's SAM 3D Body**
(Nov 2025 / Feb 2026), which uses its own body model (MHR — Momentum Human
Rig, not SMPL) under Meta's "SAM License," which explicitly permits
commercial use. **clad-body**, the most complete open measurement-extraction
library found, targets exactly this family of body models (Naver's "Anny"
and Meta's "MHR"), not SMPL — so a SAM-3D-Body → clad-body pipeline is the
only surveyed combination that plausibly avoids the SMPL licensing question
entirely. It is also the least mature and least validated combination
found (see Path B below and the caveats under SAM 3D Body's evidence card).

## Evidence cards — critical-path candidates

Full evidence cards below are given for the ~10 candidates that most affect
the first POC decision. Everything else surveyed is summarized more briefly
in "Other candidates surveyed" and detailed in `CANDIDATE_MATRIX.md`.

### SAM 3D Body (Meta)
- Repo: `github.com/facebookresearch/sam-3d-body` · Paper: arXiv 2602.15989 (~Feb 2026) · Weights on Hugging Face, Nov 2025 checkpoints, actively new.
- Input: single RGB image (+ optional keypoint/mask prompts). Output: full-body mesh incl. hands/feet, in **MHR** representation (Meta's new parametric rig, decoupling skeleton from surface shape).
- Metric scale: **UNKNOWN** — no explicit real-world-cm claim found in the README/paper excerpts available.
- Camera assumptions/multi-view/video: single view; no native video/multi-view (unofficial follow-ons exist, not evaluated).
- Body model dependency: its own MHR (bundled), not SMPL.
- Weights: HuggingFace, same "SAM License" as code.
- Hardware/VRAM: not documented (UNVERIFIED).
- Code & weight license: Meta's custom **"SAM License"** — explicitly permits commercial and non-commercial use, bans military/surveillance/ITAR use, requires attribution. Not an OSI/SPDX-standard license text; read it directly before relying on the "commercial OK" read.
- REPORTED: robust full-body mesh recovery. REPRODUCED: not attempted by us. RELEVANT: an **independent** paper (arXiv 2601.06035, not Meta) specifically studied anthropometric fidelity and found SAM 3D Body **regresses toward "standardized" body shapes** for atypical bodies (geriatric muscle atrophy, scoliosis, pregnancy, amputation, obesity) — a materially negative signal for MTM, whose value proposition is precisely non-average bodies. UNKNOWN: absolute-cm accuracy on any body type — must be benchmarked.
- Integration complexity: newest of all candidates, least battle-tested; expect rough edges.
- **Update (Task 02):** checkpoint access confirmed gated behind manual Hugging Face approval (`INSTALL.md`), not obtainable without a human requesting and being approved — this blocked actual inference in Task 02's environment. The MHR *body model/rig asset files* (distinct from this inference checkpoint) are separately, publicly downloadable under Apache-2.0 with no gating — see `LICENSE_AND_COMMERCIAL_USE.md`. Metric scale: confirmed from source that SAM 3D Body's demo pipeline uses MoGe2 to estimate camera FOV by default, falling back to a **fixed default FOV constant** (not derived from the photo) if no FOV estimator is supplied — so absolute-scale plausibility depends entirely on whether a real FOV estimator is used. Full detail: `docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md`.
- Exact experiment needed: run on a small set of known-measurement bodies (see POC_RECOMMENDATION.md) and check whether the "regression to standard shape" failure mode described in 2601.06035 reproduces and how badly it corrupts girth measurements. Still not attempted — blocked by checkpoint access in Task 02.

### 4D-Humans / HMR 2.0 (UC Berkeley)
- Repo: `github.com/shubham-goel/4D-Humans` · Paper: ICCV 2023, arXiv 2305.20091. Last commit Feb 2026 — lightly maintained, mature/well-known.
- Input: image or video (incl. YouTube). Output: SMPL pose+shape params; PHALP integration for multi-frame tracking.
- Metric scale: not an explicit feature — weak-perspective, relative-shape output (UNVERIFIED as absolute).
- Body model dependency: **SMPL neutral model** (separate registration required).
- Code license: **MIT**. Commercial use of the *code* is fine, but the SMPL model it loads at runtime is non-commercial-only — the MIT wrapper does not free you from that dependency (confirmed pattern across the whole MPI-IS ecosystem; see licensing doc).
- REPORTED: strong pose/shape recovery, standard MPJPE/PVE benchmarks. REPRODUCED: not attempted. RELEVANT: no cm-level measurement evidence found anywhere in the paper. UNKNOWN — MUST BE BENCHMARKED.
- Known failure modes (from HMR lineage generally, not an exhaustive primary quote): degrades under inter-person occlusion, loose/flowing garments, unusual poses, motion blur.
- Exact experiment needed: same benchmark set as SAM 3D Body, but gated on first resolving whether a commercial SMPL sub-license is obtainable/affordable — do not run this path for anything customer-facing until that's resolved.

### SMPL / SMPL-X / SMPLify (MPI-IS) — dependency, not a pipeline stage on its own
- Sites: `smpl.is.tue.mpg.de`, `smpl-x.is.tue.mpg.de`, `smplify.is.tue.mpg.de` (direct fetch blocked in this session; corroborated via `mmhuman3d`'s license mirror and Meshcapade's wiki — UNVERIFIED-PRIMARY, re-confirm directly before any legal decision).
- License: MPI-IS "non-commercial scientific research, education, or artistic projects" template, reused verbatim across SMPL, SMPL-X, SMPLify, STAR, ICON, ECON. Explicitly bans commercial incorporation, "production of other artifacts for commercial purposes," redistribution, and using the software to train other models for commercial use.
- Carve-out: a narrower **CC-BY 4.0** license exists for the bare "SMPL(-X) Body" neutral mesh/rig geometry (used e.g. in Meshcapade's Blender add-on) — this does NOT clearly extend to the full shape-fitting model or to numeric outputs computed by running it.
- Major uncertainty: Meshcapade (MPI-IS's commercial licensing partner) was reportedly acquired by Epic Games (announced ~Feb 2026); future commercial-licensing terms and pricing for SMPL/SMPL-X are **UNVERIFIED and in flux**. Do not assume a commercial SMPL license is cheap, fast, or even currently being sold, without directly confirming.

### clad-body (datar-psa)
- Repo: `github.com/datar-psa/clad-body`, `pip install clad-body`, Apache-2.0 (confirmed). Actively maintained, backs a live commercial try-on service (Clad).
- Input: **Naver "Anny" or Meta "MHR" mesh** — explicitly NOT SMPL/SMPL-X compatible ("SMPL tooling doesn't port over," per README).
- Output: 25 measurements aligned to ISO 8559-1 — bust/underbust/waist/hip/stomach/thigh/knee/calf/neck/upper-arm/wrist circumferences, height, shoulder width, sleeve length, inseam, crotch length, front/back rise, shirt length, back-neck-to-waist — the best MTM-relevant coverage of any measurement library found (all target measurements covered except an explicitly-named "outseam").
- Algorithm: convex-hull plane-sweep circumference (built to emulate how a physical tape bridges body concavities), partially differentiable (PyTorch autograd) for optimization-based fitting.
- REPORTED: README table claims sub-cm MAE. RELEVANT: **that MAE is the library's own differentiable path benchmarked against its own non-differentiable path — not against real human tape measurements.** No independent ground-truth validation found. UNKNOWN — MUST BE BENCHMARKED against real people.
- Commercial use: Apache-2.0, no restriction. Best-licensed measurement-extraction candidate.
- **Update (Task 02):** the SAM3D→clad-body conversion is now resolved, implemented, and unit-tested — not a drop-in pass-through (SAM 3D Body's own `scale_params` output is 28-dim PCA coefficients that would silently corrupt clad-body's loader if forwarded as-is; the fix is to forward only `shape_params` and `mhr_model_params`). See `experiments/sam3d_mhr_clad_smoke/adapter.py` and `TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md` section A for the full derivation. clad-body itself installs and imports correctly (Python ≥3.12 required — confirmed via its `pyproject.toml`); its native MHR-mesh-loading step (via `pymomentum`) segfaulted in Task 02's test environment, most likely from a CUDA-vs-CPU-tagged PyTorch build mismatch specific to that environment's network policy — not a clad-body API or licensing problem. Still not benchmarked against real tape measurements.
- Exact experiment needed: get the pymomentum/MHR native loader running in an environment with unrestricted PyPI/PyTorch-wheel-index access (or a GPU instance), then compare resulting measurements to a small set of people with known tape measurements.

### SMPL-Anthropometry (David Bojanić)
- Repo: `github.com/DavidBoja/SMPL-Anthropometry`, MIT. Operates on SMPL or SMPL-X (betas+gender, or raw vertices).
- Output: 16 measurements (head/neck/chest/waist/hip/wrist/bicep/forearm/thigh/calf/ankle circumferences; arm length, inside-leg height, shoulder breadth, shoulder-to-crotch height, height) via landmark + plane-slicing.
- REPORTED/RELEVANT: repo's own `evaluate_mae()` compares two runs of the *same* library against each other (self-consistency), not against real tape measurements. UNKNOWN — MUST BE BENCHMARKED.
- Only handles neutral T-pose bodies; posed input is out of scope for this repo.
- Coverage gaps vs. MTM list: no explicit sleeve length (has generic "arm length"), no back length, no front torso length, no outseam.
- Not pip-installable (clone + manual SMPL/SMPL-X `.pkl` placement); research code but easy to adapt. MIT code, but inherits SMPL/SMPL-X's non-commercial model license (see above) — the code alone is fine to use/modify, the dependency is not commercially clean.

### MediaPipe Pose / BlazePose (Google)
- Repo: `github.com/google-ai-edge/mediapipe`, Apache-2.0. Actively maintained, v1.0.0.
- Output: 33 2D/3D landmarks (3D coords in meters, hip-centered) + optional per-pixel person segmentation mask.
- **Only candidate confirmed to run natively on-device** (TFLite; official Android/iOS/Web/desktop deployment) — real-time, zero server cost.
- Landmarks are sparse skeletal points, not contour/silhouette detail — good for pose/scale normalization and capture-guidance (checking the customer is standing correctly, arms away from body, etc.), not for fine garment-boundary/measurement-landmark detection on its own.
- Commercial use: unrestricted under Apache-2.0.
- RTMPose (`github.com/open-mmlab/mmpose`, Apache-2.0) is a credible alternative/complement — proven on-device via ncnn on a Snapdragon 865, similarly license-clean, potentially more accurate; same sparse-landmark limitation.

### SAM2 (Meta)
- Repo: `github.com/facebookresearch/sam2`, Apache-2.0 (code, checkpoints, training code). General promptable segmenter, not human-specific — needs a person detector/prompt.
- Once prompted, produces very clean silhouette/contour masks — useful both as measurement-extraction input and as the "photo silhouette" side of a clothing-looseness overlap check (see ECON below).
- Server/GPU-oriented (ViT encoder); no confirmed on-device path.

### GeoCalib (ETH Zurich)
- Repo: `github.com/cvg/GeoCalib`, ECCV 2024. Code **Apache-2.0**, weights **CC-BY-4.0** — the cleanest commercial license pair found among the depth/geometry candidates.
- Recovers focal length, gravity direction, optional distortion from a single image, without EXIF — directly useful since phone photos shared via messaging apps commonly have EXIF (and hence focal length) stripped.
- Benchmarks are architectural/scene datasets, not human-specific, but calibration accuracy is a general-scene property, not a domain-specific one — more directly transferable than depth-estimation benchmarks are.

### ECON (MPI-IS) — not for direct commercial use, but valuable as a design pattern
- Repo: `github.com/YuliangXiu/ECON`, CVPR'23 Highlight. Non-commercial MPI-IS license (same family as SMPL); not usable in the commercial product as-is.
- **Primary-source design insight worth reusing (re-implemented ourselves, not by using ECON's code/weights):** the ECON paper explicitly treats "loose clothing" as a distinct, measured failure category, and uses the **overlap ratio between the clothing mask and the underlying body mask** (overlap < 0.5 flags "loose clothing") as a heuristic for when reconstruction — and by extension, measurement — should not be trusted. This is exactly the kind of "when NOT to trust a measurement" signal the project needs, and it's simple enough to reimplement with SAM2 (clothed-photo silhouette) + our own mesh-projected silhouette, without touching ECON's non-commercial code.

### Known height as scale anchor (non-ML)
- Not a model — the simplest available metric-scale solution: ask the customer for height (already commonly collected for MTM), use it to rescale a relative-shape mesh or 2D silhouette to real-world units.
- Zero GPU cost, zero model-licensing risk, trivial to implement and to audit. Weakness: depends on honest self-report and can't correct for posture/camera-perspective distortion the way a genuine depth signal could — but as a first-pass anchor it is by far the lowest-risk option and should be the default before reaching for any depth-estimation model.

## NVIDIA `video_to_data` (Isaac) — reference architecture, added Task 02

`nvidia-isaac/video_to_data` is a robot-learning-data pipeline, not a
body-measurement tool; one of its reconstruction modules
(`v2d_sam3d_body`) wraps SAM 3D Body and adds a genuine **joint multi-view
MHR optimization** (shared shape/scale across cameras, per-frame pose,
robust multi-view reprojection loss). Code Apache-2.0, docs CC-BY-4.0
(SAM 3D Body/MHR weights remain separately licensed/gated as above). It
requires **pre-calibrated, synchronized multi-camera rigs** (chessboard
calibration, frame-aligned streams) — there is no support, documented or
otherwise, for casual sequential single-phone front/side/back capture.
Its core joint-optimization *principle* is conceptually reusable for a
guided-smartphone-capture Path C enhancement, but doing so would mean
building our own calibration/pose front end from scratch — a real R&D
project, not a configuration change. `RESEARCHED`, recorded as a reference
architecture, not adopted. Full findings:
`docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md` section C.

## Other candidates surveyed (brief)

- **Multi-HMR** (Naver, SMPL-X): code license is an explicit **non-commercial** custom license — disqualified for commercial use regardless of the SMPL-X question. `RESEARCHED` / effectively `REJECTED` for this phase.
- **NLF** (Neural Localizer Fields, Sárándi & Pons-Moll, NeurIPS 2024): MIT code, but weights restricted to **noncommercial research use**. Technically interesting (explicitly metric-scale-native design, doesn't need known camera intrinsics) but license-blocked. `REJECTED` for this phase, worth revisiting if a commercial-weights release ever appears.
- **CameraHMR** (MPI-IS, 3DV 2025): jointly predicts camera intrinsics + full-perspective SMPL mesh — directly relevant technical idea for phone photos with unknown intrinsics — but standard MPI-IS non-commercial license. `REJECTED` for this phase.
- **Sapiens** (Meta, ECCV 2024): strong human-specific segmentation/normals/pose foundation models, but weights are **CC-BY-NC 4.0**. `REJECTED` for this phase.
- **SCHP** (Self-Correction Human Parsing): MIT, garment/body-part segmentation directly relevant to locating measurement landmarks, but unmaintained since 2021 — treat any claims about robustness on modern photos as unverified.
- **OpenPose**: CMU non-commercial research license, explicitly prohibits commercial use without a separate license, and is superseded in accuracy/mobile-viability by MediaPipe/RTMPose. `REJECTED`.
- **BiRefNet**: best pure high-resolution matting/silhouette candidate found; exact code license not independently confirmed — verify before use.
- **pose-independent-anthropometry / Landmarks2Anthropometry** (David Bojanić): trained on the paid/restricted CAESAR dataset as an operational dependency (not just for validation), and `Landmarks2Anthropometry` has no LICENSE file (default all-rights-reserved). `REJECTED` pending direct author contact.
- **CALVIS**: only 3 circumferences (chest/waist/pelvis), synthetic-only validation (explicitly to avoid CAESAR's cost), unclear license. Too narrow for MTM's measurement list.
- **STAR** (MPI-IS): same non-commercial license family as SMPL, no measurement tooling of its own, no licensing advantage over SMPL — doesn't solve anything SMPL doesn't already offer.
- **ICON**: MPI-IS non-commercial; predecessor to ECON, superseded on loose-clothing handling per ECON's own paper.
- **PIFu**: relicensed **MIT** in 2020 (commercially usable!) but produces clothed-surface geometry only, no decoupled underlying-body estimate — would need an additional body-fitting step to be useful for measurement.
- **PIFuHD** (Meta): **CC-BY-NC 4.0** — blocked commercially, unlike the original PIFu.
- **ETCH / ETCH-X** (2025/2026): code MIT, but pretrained weights non-commercial, and the pipeline outputs a SMPL/SMPL-X fit anyway — inherits that license regardless of ETCH's own code license.
- **MoGe/MoGe-2** (Microsoft): MIT code, weights license unconfirmed, GPU-required (Triton — no macOS path), metric point-map estimation with no human-specific benchmark.
- **UniDepth**: **CC BY-NC 4.0** — blocked commercially.
- **Depth Anything V2**: split licensing — Small model **Apache-2.0** (commercially usable, likely lower accuracy), Base/Large/Giant **CC-BY-NC** (blocked). Separate relative-depth vs. fine-tuned metric-depth checkpoints; do not conflate them.
- **Metric3D v2**: **BSD-2-Clause** code (permissive) but README asks that commercial use route through direct author contact — verify before relying on the permissive read.
- **ARKit / ARCore** (Apple/Google, native APIs, not repos): free, on-device, well-documented. ARKit's LiDAR-backed dense metric depth is real but hardware-gated (iPad Pro 2020+/iPhone 12 Pro+ only — most customers' phones won't have it). ARCore's Depth API works via depth-from-motion even without a ToF sensor and ships on most modern Android devices — the more broadly-available option of the two.
- **COLMAP**: BSD, but bundles GPL-licensed SiftGPU by default (check build config for commercial redistribution); scale-ambiguous without a known reference, and it's unverified whether 1–4 casual customer photos (rather than many overlapping views) would even reconstruct reliably — real feasibility risk independent of licensing.

## Combination analysis (per task section 10)

The strongest technically-plausible **and** commercially-clean combination
found is:

```
Guided front+side photos + known height
  → SAM2 (silhouette) + MediaPipe/RTMPose (landmarks, capture QA)
  → SAM 3D Body (MHR mesh)
  → clad-body (ISO 8559-1 measurements from MHR)
  → our own clothing-looseness heuristic (ECON's overlap-ratio idea, reimplemented)
  → statistical comparison against customer-reported measurements
```

All components above are Apache-2.0/MIT/Meta-SAM-License with no SMPL
dependency. Its weakness is that SAM 3D Body and clad-body are both the
*least experimentally validated* pieces in this whole audit — clean
licensing does not imply proven accuracy, and the one independent
anthropometric-fidelity study available found SAM 3D Body specifically
struggles with atypical bodies, which is a direct concern for an MTM
audience.

**Update (Task 02):** the SAM3D→clad-body handoff, the one genuine
integration-design uncertainty in this combination, is now resolved,
implemented, and unit-tested (`experiments/sam3d_mhr_clad_smoke/`). What
remains is not a design question but an execution-environment one: SAM 3D
Body's checkpoint access is Hugging-Face-gated (blocking the image→params
half) and clad-body's native MHR-mesh-loading step failed with a native
crash in Task 02's sandbox, most likely from a PyTorch build mismatch that
environment's network policy prevented fixing directly (blocking the
params→measurements half). Neither blocker is architectural. Decision gate
for this exact pipeline: `BLOCKED_BY_ACCESS` / `BLOCKED_BY_COMPUTE` /
`BLOCKED_BY_INTEGRATION` — see `TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md` for
the full breakdown and the next environment needed to clear it.

The **most mature but licensing-encumbered** alternative combination is
4D-Humans (SMPL) + SMPL-Anthropometry, both individually well-documented
and MIT-licensed at the code level, but both dependent on the non-commercial
SMPL model — usable for internal, non-customer-facing R&D/benchmarking now,
not for the shipped commercial product without resolving the SMPL licensing
question first (cost currently unknown; do not assume it fits the $100
ceiling).

## Path A / B / C

**Path A — Simplest viable system.** No 3D mesh recovery at all. Guided
front+side photos, known height as scale anchor, MediaPipe Pose landmarks
(on-device, Apache-2.0) to measure length-type proportions (shoulder width,
limb lengths, torso segments) via deterministic pixel-to-cm geometry, then
compare against customer values with plausibility bounds. Zero training,
zero licensing risk, lowest technical risk. Cannot produce girth
measurements (chest/waist/seat/thigh/bicep circumferences) — those need a
volumetric/3D representation, not 2D landmarks alone. Best suited to
catching gross errors in lengths only.

**Path B — Best zero-training system.** The SAM 3D Body → clad-body
combination above, augmented with SAM2 silhouettes and the
clothing-looseness heuristic, targeting the full MTM measurement list.
Zero training, zero GPU spend beyond inference compute (fits the $0–$100
ceiling), but experimentally unproven — must be benchmarked before trusting
any output.

**Path C — Optional enhancements (not needed initially).** Multi-view/video
capture and photogrammetry (COLMAP) for cross-validation; metric depth
models (MoGe-2, Depth Anything V2 Small, GeoCalib intrinsics) to
supplement or replace the known-height anchor; a small, narrowly-scoped
calibration-correction step trained on our own later-collected paired data
(explicitly out of scope now — would need separate approval, as it borders
on "training," even if small).

## Answers to the required end-of-task questions

See `docs/research/POC_RECOMMENDATION.md` — all ten questions from the task
brief are answered there together with the concrete first experiment.
