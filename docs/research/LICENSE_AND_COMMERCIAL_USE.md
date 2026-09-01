# Licensing and Commercial-Use Findings

This is the single most consequential section of the audit: the business is
commercial, and licensing — not compute — is the dominant constraint on
which technologies are actually usable. Read this before adopting any
candidate from the audit, even one already marked "commercially usable"
below; verify directly against the primary license text before shipping.

## Access caveat

Several primary license pages (`smpl.is.tue.mpg.de`, `smpl-x.is.tue.mpg.de`,
`smplify.is.tue.mpg.de`, direct arXiv PDFs, `ai.meta.com`) were unreachable
from this research environment's network egress. Findings about them below
are corroborated through cached search snippets and reputable third-party
mirrors (e.g. `mmhuman3d`'s license-aggregation doc, Meshcapade's public
wiki) and are marked `UNVERIFIED-PRIMARY`. **Before any legal decision, a
human must directly re-fetch and read the primary license page.**

## The central issue: SMPL/SMPL-X and derived outputs

SMPL, SMPL-X, SMPLify, and STAR (all MPI-IS) share a near-identical license
template (`UNVERIFIED-PRIMARY`, cross-checked across multiple mirrors):

- Use limited to "the sole purpose of performing non-commercial scientific
  research, non-commercial education, or non-commercial artistic projects."
- Explicit ban: "any other use, in particular any use for commercial,
  pornographic, military, or surveillance, purposes is prohibited,"
  including "incorporation in a commercial product, use in a commercial
  service, or production of other artifacts for commercial purposes."
- Explicit ban on using the software to train other models/algorithms for
  commercial use.
- Explicit ban on redistribution/sub-licensing/resale, in whole or in part.
- A narrower, separately-named **CC-BY 4.0** license exists for the bare
  "SMPL(-X) Body" — a specific neutral mesh/rig geometry artifact (used in
  things like Meshcapade's Blender/Maya add-ons) — but this does not
  clearly extend to the full shape-fitting model or to arbitrary numeric
  outputs (like a measurement) computed by running the restricted model.

**No primary source directly and unambiguously states whether numeric
measurements derived from an SMPL/SMPL-X mesh — as opposed to a
redistributed mesh file — count as a restricted "artifact... for commercial
purposes."** We do not draw a legal conclusion from this ambiguity — that is
not ours to determine from source-reading alone, and Task 02 revised this
section's framing accordingly (2026-09, corrects Task 01's "verdict"
language below).

**Standing engineering/business decision (not a legal conclusion):** SMPL,
SMPL-X, SMPLify, and STAR carry enough licensing uncertainty and potential
cost that they are **excluded from the initial commercial-oriented POC**,
along with every tool that loads them at runtime (SMPL-Anthropometry,
4D-Humans/HMR2.0, ICON, ECON, CameraHMR, Multi-HMR, CALVIS, and the
Landmarks2Anthropometry family) — **unless and until explicit, suitable
commercial licensing is obtained** for the specific model in use. This
decision holds regardless of how the underlying legal question eventually
resolves; it is a project-scope choice, not a claim about what the license
text legally permits. Task 02's implemented pipeline
(`experiments/sam3d_mhr_clad_smoke/`) uses no SMPL-family component for
exactly this reason — it is built entirely on SAM 3D Body's own MHR body
model instead (see the row below).

If this project later wants to use an SMPL/SMPL-X-dependent tool
commercially, the concrete next step is a lawyer's opinion or a confirmed,
priced commercial license — not further source-reading here.

### Additional uncertainty: Meshcapade / Epic Games

MPI-IS's commercial licensing partner for SMPL-family models, Meshcapade,
was reportedly acquired by Epic Games (announced ~Feb 2026; closing
~April 2026 per search snippets — `UNVERIFIED-PRIMARY`). Whatever the
current commercial-licensing process and pricing is, it should be treated
as **actively in flux** and reconfirmed at the time of any real purchase
decision, not assumed stable from older documentation.

## Commercially usable without a paid license (as researched)

These carry permissive licenses on both code and the specific weights
identified, with no SMPL/SMPL-X dependency:

| Component | License(s) | Notes |
|---|---|---|
| SAM 3D Body (Meta) — **inference checkpoint** | Meta "SAM License" (code + weights) | Custom text, not OSI-standard; explicitly permits commercial use, bans military/surveillance/ITAR use. Read the exact license text before shipping — "SAM License" is not a recognized SPDX identifier and could carry Meta-specific conditions not captured by a one-line summary. Checkpoint download is separately **gated behind manual Hugging Face access approval** (confirmed directly from `INSTALL.md`, Task 02) — a licensing/access question distinct from the license terms themselves. |
| MHR (Momentum Human Rig) — **body model/rig assets** | **Apache-2.0**, confirmed by reading the `LICENSE.txt` bundled directly inside the public `assets.zip` release (Task 02) | Distinct artifact from the SAM 3D Body checkpoint above, despite the shared "MHR" name — this is the actual 3D body geometry/rig, publicly downloadable with **no authentication required** from `github.com/facebookresearch/MHR/releases`. Task 01 only characterized "the MHR ecosystem" generally as Apache-2.0; Task 02 confirms this at the asset-file level specifically, which matters because SMPL's restrictive license famously applies at exactly this level (the body model file itself, not just wrapper code). |
| clad-body | Apache-2.0 | Targets Anny/MHR, not SMPL — this is what keeps it out of the SMPL question. Confirmed installable and importable (Task 02); its MHR loading path is currently blocked in our test environment by an unrelated native/dependency issue, not a licensing one — see `TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md`. |
| SAM2 (Meta) | Apache-2.0 | Code, checkpoints, and training code all Apache-2.0 |
| MediaPipe Pose / BlazePose (Google) | Apache-2.0 | On-device, mobile-native |
| RTMPose (OpenMMLab) | Apache-2.0 | Mobile-proven alternative to MediaPipe |
| SCHP (human parsing) | MIT | Unmaintained since 2021 — license is fine, currency of the model is not |
| PIFu (original, relicensed 2020) | MIT | Clothed-surface only, no decoupled body |
| GeoCalib | Apache-2.0 (code), CC-BY-4.0 (weights) | Cleanest license pair among depth/geometry tools |
| Depth Anything V2 — Small variant only | Apache-2.0 | Base/Large/Giant variants are CC-BY-NC — do not substitute without re-checking |
| Metric3D v2 | BSD-2-Clause (code) | README asks commercial users to contact the authors directly — confirm before relying on the permissive code license alone |
| MoGe / MoGe-2 (Microsoft) | MIT (code) | Weights license not independently confirmed — verify on Hugging Face before commercial use |
| COLMAP | BSD (core) | Default build bundles GPL-licensed SiftGPU — check/replace before commercial redistribution |
| ARKit (Apple) / ARCore (Google) | Platform ToS, not open-source | Free to use per platform developer terms; not a licensing blocker, but a hardware-coverage one (see audit) |

## Explicitly blocked for commercial use (non-commercial license, no budget for a paid license)

| Component | License | Note |
|---|---|---|
| SMPL / SMPL-X / SMPLify / STAR | MPI-IS non-commercial (narrow CC-BY carve-out only for bare mesh geometry) | See central issue above |
| 4D-Humans / HMR2.0 | MIT code, but loads SMPL at runtime | Code license alone is not sufficient |
| Multi-HMR (Naver) | Custom "Non-Commercial License" | Explicit, independent of SMPL-X dependency |
| NLF | MIT code, weights "noncommercial research use" | Weights are the blocker |
| CameraHMR | MPI-IS non-commercial | Same family as SMPL |
| ICON / ECON (MPI-IS) | MPI-IS non-commercial | ECON's clothing-looseness *heuristic* is reusable if reimplemented independently; its code/weights are not |
| PIFuHD (Meta) | CC-BY-NC 4.0 | Unlike original PIFu, which is MIT |
| ETCH / ETCH-X | Code MIT, weights non-commercial; outputs SMPL/SMPL-X anyway | Double-blocked |
| Sapiens (Meta) | Weights CC-BY-NC 4.0 | Code partly Apache-2.0, weights are the blocker |
| OpenPose (CMU) | CMU non-commercial research license | Also technically superseded |
| UniDepth | CC BY-NC 4.0 | — |
| Landmarks2Anthropometry | No LICENSE file (default all-rights-reserved) | Contact author before any use |

## Restricted datasets (not licenses on code/weights, but worth tracking)

- **CAESAR** anthropometric dataset: paid/restricted access (SAE
  International / HumanShape / Shape Analysis Ltd as commercial
  distributors), no free-for-research tier found. Relevant because
  `Landmarks2Anthropometry`'s pretrained weights depend on it operationally,
  and it's the standard ground-truth source the field uses for validation —
  we will not have free access to it for our own benchmarking either.

## Standing rule

Per `CLAUDE.md`: **never silently introduce a paid dependency.** If a future
task proposes using any SMPL/SMPL-X-dependent tool, or purchasing a
Meshcapade/MPI-IS commercial license, that proposal must be raised
explicitly, with actual quoted pricing, before being adopted — it is not
assumed to fit inside the $0–$100 initial budget ceiling.
