# Project Scope — Independent Smartphone Body Measurement & Consistency Checker

## Business context

The core business is online Made-to-Measure (MTM) suits. Customers submit
their own body measurements. This project is a **second-opinion check**, not
a replacement for those measurements and not a standalone body-scanning
product.

Goal: use smartphone photos (+ optional known height/weight) to produce an
independent body estimate, compare it against customer-submitted
measurements, and flag cases warranting re-measurement.

Example: customer reports chest = 106 cm; vision estimate suggests ~100–102
cm → system flags `SIGNIFICANT DISCREPANCY — RE-MEASURE CHEST`.

## What the system must detect

- Large customer measurement errors
- Internally inconsistent measurement sets (e.g. proportions that don't fit
  together on any plausible body)
- Measurements implausible relative to the observed body
- Body-shape characteristics useful for MTM fitting beyond flat numbers
- Cases that should trigger customer re-measurement

## Target measurements (priority order, not equal difficulty)

Chest, waist, seat/hip, shoulder width, neck, sleeve length, back length,
front torso length, biceps, wrist, inseam, outseam, thigh.

Not every measurement is expected to be equally estimable from photos;
per-measurement feasibility is tracked in the research audit, not assumed.

## Hard constraints

1. No training of new foundation models in the initial phase.
2. No proprietary/large dataset creation.
3. No paid body-scanning APIs in the core solution.
4. Prefer open-source code, public pretrained weights, existing body
   models, existing measurement libraries, classical geometry/optimization,
   and publicly documented smartphone sensor capabilities.
5. Cloud/GPU spend target ~$0; absolute ceiling ~$100 for the initial POC.
6. No architecture that assumes expensive GPU training infrastructure
   unless proven unavoidable.
7. Must work on ordinary modern smartphones — no dedicated scanners,
   calibration rigs, or mandatory depth hardware. LiDAR/ToF may be used
   opportunistically when present.
8. Commercial-use licensing (code, weights, body model, datasets) must be
   verified per component before adoption — this supports a commercial
   business.
9. Popularity is not a proxy for suitability; every candidate is assessed
   on its own evidence.

## Non-goals for this phase

- Building a production pipeline.
- Selecting a final architecture.
- Any model training or dataset collection.
- Replacing professional/tailor measurements.

## Technical principle

Prefer a hybrid pipeline — AI/CV for evidence extraction, geometry/camera
models for consistent spatial reconstruction, a parametric human body prior
to constrain the solution, deterministic measurement algorithms to derive
numbers, and statistical logic to compare against customer-reported values
— over a single end-to-end photo→measurement network. AI is used only where
it provides information deterministic geometry cannot reliably supply.

See `docs/research/OPEN_SOURCE_FEASIBILITY_AUDIT.md` for the technology
audit and `docs/research/POC_RECOMMENDATION.md` for what to test first.
