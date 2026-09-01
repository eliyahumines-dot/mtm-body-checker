# Candidate Matrix

Decision states: `UNASSESSED` → `RESEARCHED` → `RUNNABLE` → `BENCHMARKED` →
`VALIDATED` → `SELECTED` / `REJECTED`. Nothing has been run yet, so nothing
is `RUNNABLE`/`BENCHMARKED`/`VALIDATED`/`SELECTED`. `REJECTED` here means
"disqualified for *this* zero-paid-license phase," not permanently —
mainly candidates with a hard non-commercial license and no budget line for
a commercial license. Full detail and sources in
`OPEN_SOURCE_FEASIBILITY_AUDIT.md`.

## Mesh / body recovery

| Candidate | Body model | Metric scale | Code license | Weight license | Commercial? | State | Key uncertainty |
|---|---|---|---|---|---|---|---|
| SAM 3D Body (Meta) | own MHR | Conditional on FOV estimator (MoGe2 default; falls back to a fixed default FOV, not photo-derived, if none supplied — confirmed from source, Task 02) | SAM License (permissive) | SAM License, checkpoint gated behind manual HF approval (confirmed, Task 02) | Yes (license); access blocked in Task 02's environment | RESEARCHED | Anthropometric fidelity for atypical bodies (independent study found degradation); still not run — checkpoint access blocked |
| 4D-Humans / HMR2.0 | SMPL | UNKNOWN, likely relative | MIT | n/a (loads SMPL) | Blocked via SMPL dep. | RESEARCHED | Needs paid SMPL commercial license, cost unknown |
| Multi-HMR (Naver) | SMPL-X | UNKNOWN | Custom non-commercial | Custom non-commercial | No | REJECTED (license) | — |
| NLF (Neural Localizer Fields) | SMPL-compatible | Designed for metric | MIT | Non-commercial research only | No | REJECTED (license) | Otherwise most metric-scale-native design found |
| CameraHMR | SMPL | UNKNOWN | MPI-IS non-commercial | MPI-IS non-commercial | No | REJECTED (license) | Intrinsics-aware design idea worth reusing conceptually |
| SMPL / SMPL-X / SMPLify (dependency) | — | n/a | Non-commercial research (narrow CC-BY carve-out for bare mesh geometry only) | same | Ambiguous/likely No for full pipeline | RESEARCHED | Whether numeric outputs are covered by license; Meshcapade/Epic acquisition impact on future terms |
| STAR | — | n/a | MPI-IS non-commercial | same | No | REJECTED (license) | No licensing advantage over SMPL |

## Human perception

| Candidate | Output | On-device? | License | Commercial? | State | Key uncertainty |
|---|---|---|---|---|---|---|
| MediaPipe Pose / BlazePose | 33 landmarks + optional seg mask | Yes (TFLite) | Apache-2.0 | Yes | RESEARCHED | Only sparse landmarks, no fine contour |
| RTMPose (OpenMMLab) | 2D keypoints, whole-body variants | Yes (proven via ncnn) | Apache-2.0 | Yes | RESEARCHED | Same sparse-landmark limitation |
| SAM2 (Meta) | Promptable segmentation mask | No (server/GPU) | Apache-2.0 | Yes | RESEARCHED | Needs person-prompting step |
| SCHP (human parsing) | Semantic body/garment parts | No | MIT | Yes | RESEARCHED | Unmaintained since 2021, robustness on modern photos unverified |
| BiRefNet | High-res matte/silhouette | No | Unverified | Unverified | RESEARCHED | Confirm license before use |
| Sapiens (Meta) | Pose/parsing/normals/depth | No | Mixed code; weights CC-BY-NC | No (weights) | REJECTED (license) | — |
| OpenPose (CMU) | 2D keypoints | No | CMU non-commercial | No | REJECTED (license) | Superseded technically too |

## Measurement extraction (mesh → cm)

| Candidate | Target mesh | MTM coverage | Real-tape validation | License | Commercial? | State | Key uncertainty |
|---|---|---|---|---|---|---|---|
| clad-body | Anny / MHR | 10/13 MTM measures covered (back length, front torso length, outseam confirmed as gaps, Task 02); best of all candidates surveyed | None found (self-referential only) | Apache-2.0 | Yes | RESEARCHED — installs and imports correctly (Task 02); SAM3D output adapter implemented and unit-tested; native MHR mesh-loading crashes in Task 02's sandbox (likely torch-build ABI issue, not a clad-body defect) | Accuracy vs. real humans entirely unknown; native MHR loader needs an environment with unrestricted PyPI/PyTorch-wheel access to actually execute |
| SMPL-Anthropometry (Bojanić) | SMPL / SMPL-X | Good but missing sleeve/back/front-torso/outseam explicitly | None found (self-consistency only) | MIT (code); SMPL dep. non-commercial | Blocked via SMPL dep. | RESEARCHED | Same SMPL licensing question |
| Landmarks2Anthropometry / pose-independent-anthropometry | SMPL-derived landmarks | 11 measures, partial | Trained/eval'd on CAESAR (restricted dataset; UNVERIFIED-PRIMARY detail) | No LICENSE file (all rights reserved) | No | REJECTED (license) | Contact author to clarify |
| CALVIS | SMPL | Only chest/waist/pelvis | Synthetic-only (explicitly avoids CAESAR) | Unclear | Unverified | RESEARCHED | Too narrow for MTM list regardless |
| CAESAR (dataset, not a tool) | — | — | Ground-truth source others depend on | Paid/restricted | N/A | RESEARCHED | Confirm current pricing/access if ever needed |

## Metric geometry / depth

| Candidate | Estimates | Human-specific evidence | Code license | Weight license | Commercial? | State | Key uncertainty |
|---|---|---|---|---|---|---|---|
| Known customer height (non-ML) | scale anchor only | n/a | n/a | n/a | Yes | RESEARCHED | Depends on honest self-report, no posture correction |
| GeoCalib | Camera intrinsics (focal length, gravity) | None (scene benchmarks) | Apache-2.0 | CC-BY-4.0 | Yes | RESEARCHED | Cleanest license pair found; scene-only benchmarks |
| MoGe / MoGe-2 (Microsoft) | Metric point map / depth | None (scene benchmarks) | MIT | Unconfirmed | Verify | RESEARCHED | GPU/Triton required, no macOS path |
| Depth Anything V2 (Small) | Metric/relative depth | None | Apache-2.0 | Apache-2.0 (Small only) | Yes (Small only) | RESEARCHED | Base/Large/Giant are CC-BY-NC — pick variant carefully |
| Metric3D v2 | Metric depth + normals | None (driving/indoor benchmarks) | BSD-2-Clause | "Contact authors" caveat | Verify | RESEARCHED | Commercial-use caveat needs direct confirmation |
| UniDepth | Metric depth + intrinsics | None | CC BY-NC 4.0 | CC BY-NC 4.0 | No | REJECTED (license) | — |
| ARKit depth (Apple, native) | LiDAR-backed metric depth | n/a (platform API) | n/a | n/a | Yes | RESEARCHED | Hardware-gated to LiDAR-equipped Pro devices only |
| ARCore Depth API (Google, native) | Depth-from-motion, ToF-fused | n/a (platform API) | n/a | n/a | Yes | RESEARCHED | Broadest device coverage of any depth option |
| COLMAP | Multi-view SfM reconstruction | None | BSD (bundles GPL SiftGPU by default) | n/a | Verify build | RESEARCHED | Scale-ambiguous without reference; unclear if 1-4 casual photos suffice |

## Clothing-aware reconstruction

| Candidate | Approach | License | Commercial? | State | Key uncertainty |
|---|---|---|---|---|---|
| ECON (MPI-IS) | SMPL-X-anchored + explicit loose-clothing detection heuristic | MPI-IS non-commercial | No | REJECTED (license) — but reimplement the overlap-ratio heuristic ourselves | Its clothing-overlap-ratio *idea* is reusable without its code |
| ICON (MPI-IS) | Predecessor to ECON | MPI-IS non-commercial | No | REJECTED (license) | Superseded by ECON per authors' own comparison |
| PIFu (relicensed) | Clothed-surface implicit function | MIT | Yes | RESEARCHED | No decoupled underlying-body output |
| PIFuHD (Meta) | High-res clothed surface | CC-BY-NC 4.0 | No | REJECTED (license) | — |
| ETCH / ETCH-X | Cloth-to-body displacement fitting | Code MIT, weights non-commercial, outputs SMPL/SMPL-X anyway | No | REJECTED (license) | Best technical treatment of loose clothing found; revisit if licensing changes |

## Multi-view / reference architectures (added Task 02)

| Candidate | Approach | License | Commercial? | State | Key uncertainty |
|---|---|---|---|---|---|
| NVIDIA `video_to_data` (`v2d_sam3d_body`) | Joint multi-view MHR optimization (shared shape/scale, per-frame pose, multi-view reprojection loss) inside a broader robot-learning-data pipeline | Apache-2.0 (code), CC-BY-4.0 (docs); SAM 3D Body/MHR weights separately licensed/gated | Yes (V2D's own code); depends on gated SAM 3D Body weights | RESEARCHED — reference architecture, not adopted | Requires pre-calibrated, synchronized multi-camera rigs; no support for casual sequential smartphone capture; reusing the joint-optimization principle for guided phone photos would need a from-scratch calibration/pose front end (real R&D, not configuration) |
