# CLAUDE.md — Persistent Project Instructions

This file is the standing brief for anyone (human or Claude) working in this
repository. Read it before making architectural or dependency decisions.

## What this project is

An independent, smartphone-photo-based body-measurement **consistency
checker** that supports an online Made-to-Measure (MTM) suit business. It is
NOT a standalone body-scanning product. Its job is to flag customer
measurement errors, inconsistencies, and implausible values, and to
recommend re-measurement — not to replace professional tailoring
measurements.

See `docs/product/PROJECT_SCOPE.md` for the full scope and
`docs/research/` for the technology feasibility audit.

## Standing constraints (do not violate silently)

1. **No new model training** in the initial phase. This is an integration
   project over existing pretrained, open-source components.
2. **No proprietary dataset creation.** Use existing public data only.
3. **No paid body-scanning APIs** in the core solution.
4. **Budget ceiling: ~$100 total**, target ~$0 for cloud/GPU spend. Any
   design that assumes dedicated GPU training infrastructure must justify
   why it is unavoidable, in writing, before being adopted.
5. **No mandatory special hardware.** Must work from an ordinary modern
   smartphone camera. LiDAR/depth sensors may be used opportunistically when
   present, never as a hard requirement.
6. **Open-source-first.** Prefer existing code, pretrained weights, body
   models, and measurement libraries over writing new ML from scratch.

## Working principles

- **Evidence before architecture.** Do not select a final production
  architecture based on research promises. Use the decision-state ladder
  (UNASSESSED → RESEARCHED → RUNNABLE → BENCHMARKED → VALIDATED → SELECTED /
  REJECTED) tracked in `docs/research/CANDIDATE_MATRIX.md`. Nothing skips
  states.
- **Measurement accuracy is never inferred from visual/pose quality.** A
  model's MPJPE, PVE, or rendering quality says nothing about whether it
  produces accurate body circumferences in centimeters. Anthropometric
  accuracy claims must cite measurement-specific evidence or be marked
  `UNKNOWN — MUST BE BENCHMARKED`.
- **Hybrid over end-to-end.** Prefer combining specialized components
  (segmentation, pose, metric depth, parametric body model, measurement
  algorithms, statistical comparison) over a single photo→measurement
  neural network. Use AI only where deterministic geometry can't do the job.
- **Licensing must always be checked, per component, separately.** Code
  license, pretrained-weight license, and body-model license can all
  differ. "Open source" is not sufficient — verify commercial-use and
  redistribution terms explicitly before adoption. See
  `docs/research/LICENSE_AND_COMMERCIAL_USE.md`.
- **Never silently introduce a paid dependency** (API, dataset, compute
  tier). Any such proposal must be flagged explicitly and approved before
  it lands in the architecture.
- **Keep implementation modular.** Each pipeline stage (capture guidance,
  segmentation, mesh recovery, scale anchoring, measurement extraction,
  discrepancy comparison) should be swappable without rewriting the others,
  since candidate technologies are expected to change as the field moves.
- **Document major decisions.** Non-trivial technology choices, rejections,
  and architecture changes go in `docs/decisions/` as short ADRs (Architecture
  Decision Records) — one file per decision, dated, with the alternatives
  considered and why.

## Repository layout

- `docs/product/` — product scope and requirements
- `docs/research/` — technology feasibility audits, candidate comparisons,
  licensing notes, POC recommendations
- `docs/decisions/` — architecture decision records (ADRs)
- `docs/experiments/` — records of runnable experiments/benchmarks and their
  results (not the raw data itself — see `.gitignore`)
- `src/` — implementation code (minimal/empty until a POC is approved)
- `tests/` — automated tests

## Repository hygiene

Do not commit model checkpoints, pretrained weights, large datasets,
caches, vendored third-party repos, generated binaries, or other large
temporary files. Inspect third-party repos outside this project tree. See
`.gitignore`.
