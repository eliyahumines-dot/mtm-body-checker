# Task 03E — Resolve SAM3D Source Import Completely

Continues from a second real Colab run of Task 03D's notebook, which
reported the *exact same* failure Task 03D was meant to fix:

| Phase | Result |
|---|---|
| SAM3D_CORE_ENVIRONMENT | **PASS** |
| SAM3D_SOURCE_IMPORT | **FAIL** |
| SAM3D_MODEL_LOAD | NOT_ATTEMPTED |
| SAM3D_CORE_INFERENCE | NOT_ATTEMPTED |
| MHR_PARAMS_SERIALIZED | NOT_ATTEMPTED |
| First failing boundary | SAM3D_SOURCE_IMPORT |

Exact error: `ModuleNotFoundError: No module named 'sam_3d_body'` — same
root symptom as Task 03D, meaning Task 03D's fix did not actually resolve
the problem on real Colab hardware.

## Root cause

Task 03D shipped two mechanisms: the notebook set `PYTHONPATH` on the
worker/pre-flight subprocess's `env=`, and the worker additionally did its
own `sys.path.insert()` from an explicit argument. The pre-flight cell
(PHASE A1.5) used only the first mechanism — an inline `subprocess.run([...,
'-c', preflight_script], env={**os.environ, 'PYTHONPATH': ...})`. That is
exactly the mechanism that failed again on real Colab.

This agent's own sandbox has no GPU and could not reproduce the failure
under either mechanism: a fresh `git clone --depth 1
https://github.com/facebookresearch/sam-3d-body.git` plus a real venv with
the full `SAM3D_CORE_PIP_DEPENDENCIES` set installed imported cleanly both
via `env=PYTHONPATH` and via `sys.path.insert()`. Rather than continue
guessing at an unreproducible, Colab-specific environment-variable
propagation quirk, the fix eliminates the `env=PYTHONPATH` dependency
**everywhere** in favor of the one mechanism already proven robust here: a
script inserting its own explicit, validated source root into its own
`sys.path` before importing anything from `sam_3d_body` — never relying on
cwd, ambient `PYTHONPATH`, shell session state, or "accidental" editable
installation.

## What changed

### New standalone pre-flight script

`experiments/sam3d_mhr_clad_smoke/_sam3d_source_import_check.py` — takes
`sam3d_source_root` as an explicit CLI argument (no `env=` reliance
whatsoever), validates it with `sam3d_source_path.validate_sam3d_source_root()`,
inserts it into its own `sys.path`, and performs two independently-reportable
stages:

- `SAM3D_SOURCE_IMPORT` — bare `import sam_3d_body`, then confirms
  `sam_3d_body.__file__` resolves under the given root (not an unrelated
  installed package).
- `SAM3D_MODEL_CODE_IMPORT` (new boundary, one step further than Task 03D)
  — `from sam_3d_body.build_models import load_sam_3d_body` and `from
  sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator`. SAM 3D
  Body's own model-construction Python code importing cleanly, distinct
  both from the bare package import above and from `load_sam_3d_body()`
  actually reading real checkpoint assets from disk
  (`SAM3D_MODEL_LOAD`, which this script never attempts).

Prints `SAM3D_SOURCE_ROOT`, `SAM3D_MODULE_FILE`, then each stage's
PASS/FAIL/NOT_ATTEMPTED, before any checkpoint download or model loading is
even considered. Exit code 0 only if both stages pass.

### `_sam3d_inference_worker.py`

Split what was one conflated `source_import` stage into two
(`source_import`, `model_code_import`), matching the standalone script
exactly. Removed the residual assumption that a caller-supplied
`PYTHONPATH` would help — the worker already did its own
`sys.path.insert()`, so nothing here changed except the stage split and the
telemetry-default fix below.

**Cascading-false-failures fix (the other real bug this task fixes):**
every "not yet attempted" telemetry field (`source_import_ok`,
`model_code_import_ok`, `model_load_ok`, `inference_ok`, `person_detected`,
`interchange_written`) now defaults to `None`, not `False`. Previously, a
`False` default combined with `bool(telemetry.get(...))` coercion in the
notebook's worker-invocation cell silently turned "this stage never ran
because an earlier one failed" into a reported FAIL for every downstream
stage — exactly the failure mode this task's brief calls "cascading false
failures." Each stage now only sets its own field to `False` at the moment
it is actually attempted and fails (including via a generic catch-all
`except Exception` at the bottom, which maps the stage name that was active
at crash time to its telemetry field so a genuine mid-stage crash still
reports `False`, never leaves a real failure looking like NOT_ATTEMPTED
either).

### `decision_gate.py`

- New `FailureCategory.SAM3D_MODEL_CODE_IMPORT_FAILURE` (19 categories
  total, up from 18).
- New `PipelineState.sam3d_model_code_import_ok: bool | None` field.
- `classify()` gained a new precondition, positioned between
  `sam3d_source_import_ok` and `sam3d_model_load_ok`, so a model-code-import
  failure is never misreported as either a source-import or a
  checkpoint/model-load failure.
- `phase_summary()` inserts `SAM3D_MODEL_CODE_IMPORT` as a sixth
  independent Phase A boundary (between `SAM3D_SOURCE_IMPORT` and
  `SAM3D_MODEL_LOAD`); `phase_summary()` now returns 11 keys, up from 10.

### Notebook (`notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb`)

- **PHASE A1.5** now calls `_sam3d_source_import_check.py` as a plain
  subprocess with **no `env=` override at all** (previously
  `env={**os.environ, 'PYTHONPATH': ...}`), reads its JSON telemetry file,
  and prints `SAM3D_SOURCE_ROOT`/`SAM3D_MODULE_FILE`/both stage results
  exactly as this task requires.
- **PHASE A4-A5** (worker invocation) also drops `env=PYTHONPATH` (the
  worker validates and inserts its own source root from its 4th positional
  argument), is now gated on `sam3d_core_deps_ok and sam3d_source_import_ok
  and sam3d_model_code_import_ok`, no longer coerces telemetry through
  `bool()` (uses `is True`/`is False` so a missing/`None` field stays
  NOT_ATTEMPTED rather than reading as FAIL), and reports
  `SAM3D_MODEL_CODE_IMPORT` from the worker's own telemetry as well.
- The now-unused direct import of `sam3d_source_path`'s functions was
  removed from the shared-imports cell (Task 03E: validation now happens
  entirely inside the two subprocess scripts, not the notebook driver).
- The decision-gate printout cell lists `SAM3D_MODEL_CODE_IMPORT` alongside
  the other Phase A boundaries.

## Actual upstream import test performed (real, not mocked)

Built for this task and reused across both diagnostic work and the formal
test suite:

- `/tmp/sam3d_e2e_clone` — a real, fresh `git clone --depth 1
  https://github.com/facebookresearch/sam-3d-body.git` (same commit
  Task 02/03C/03D already established, `b5c765a`).
- `/tmp/env_sam3d_test` — a real `python3.11 -m venv` with the exact
  `SAM3D_CORE_PIP_DEPENDENCIES` list installed (torch via plain PyPI, since
  `download.pytorch.org` is network-blocked in this sandbox — sufficient for
  import-testing; the CUDA wheel pin itself is untouched production code,
  not part of this task's scope, and was not re-verified here).

Running the new standalone script for real against these:

```
$ /tmp/env_sam3d_test/bin/python3.11 _sam3d_source_import_check.py /tmp/sam3d_e2e_clone
SAM3D_SOURCE_ROOT: /tmp/sam3d_e2e_clone
SAM3D_MODULE_FILE: /tmp/sam3d_e2e_clone/sam_3d_body/__init__.py
SAM3D_SOURCE_IMPORT: PASS
SAM3D_MODEL_CODE_IMPORT: PASS
$ echo $?
0
```

Also directly verified (not just asserted): an unrelated fake `sam_3d_body`
package placed earlier on `PYTHONPATH` does **not** override the explicitly
supplied source root (the real clone's `__file__` is still what's printed
and recorded); a source root containing spaces resolves and imports
correctly; the check is independent of the calling process's current
working directory; and running the exact worker subprocess invocation
against this real clone (with a nonexistent checkpoint) correctly leaves
`source_import_ok`/`model_code_import_ok` as `null` (NOT_ATTEMPTED, not
FAIL) when it stops even earlier, at the pre-existing `gpu_check` stage —
this sandbox has no GPU, which is an existing, untouched, orthogonal
limitation, not a regression from this task's changes.

## Tests

23 new tests, all passing, none requiring GPU:

- `tests/test_sam3d_source_import_check.py` (16) — valid/invalid source
  root, `SAM3D_MODEL_CODE_IMPORT_FAILURE` distinct from
  `SAM3D_SOURCE_IMPORT_FAILURE`, downstream boundary stays `None`
  (NOT_ATTEMPTED) after an earlier failure, correct `sys.path` insertion
  (verified via subprocess with `PYTHONPATH` deliberately cleared),
  deterministic behavior, a source root containing spaces, source-root
  independence from cwd, an unrelated module on `PYTHONPATH` failing to
  override the explicitly supplied root, exit codes, and — the most
  important one — `test_real_upstream_sam3d_source_import_check`, which
  runs the script for real against the actual cloned upstream repo and a
  real dependency-complete venv (skipping with an explicit, honest reason
  if that heavy, non-committed fixture isn't present in a given
  environment — it never fabricates a pass).
- `tests/test_decision_gate.py` (+7) — `SAM3D_MODEL_CODE_IMPORT_FAILURE`
  classification and its precedence, that it is never conflated with either
  neighboring failure category, and `phase_summary()`'s new field/count/
  ordering (including a scenario reproducing this task's own source-import
  boundary and one for the model-code-import boundary specifically).

**All 132 tests pass** (109 from Task 02/03B/03C/03D + 23 new), all under
plain Python with `pytest`+`numpy` only for the synthetic-fixture tests; the
one real-upstream test additionally requires (and, in this run, found and
used) a pre-built dependency-complete venv and a real clone.

## What was kept unchanged

Per this task's explicit "do not touch" list: `torch==2.8.0`,
`torchvision==0.23.0`, the `cu129` wheel index, `HF_TOKEN` retrieval/
handling, checkpoint selection logic, the explicit full-image bbox logic,
Detectron2/MoGe exclusion. None of `run.py`, `adapter.py`, `rescale.py`,
`mtm_mapping.py`, `interchange.py`, `_mhr_measure_worker.py`,
`sam3d_env_spec.py`, `install_log.py`, or `sam3d_source_path.py` were
modified in this task.

## Colab/GPU execution

`COLAB_EXECUTION_UNAVAILABLE`. This agent's sandbox has no browser and no
GPU/Colab runtime access in any task in this project so far (Tasks 02
through 03E); reported only after every locally-reproducible boundary above
was made to genuinely pass, per this task's explicit instruction not to
report this until local/CPU work is complete. Authored and structurally
validated (valid nbformat, every cell's Python syntax checked, full
notebook pyflakes-clean, 132 tests passing, and — going beyond static
validation — an actual, non-mocked import preflight run against a real
clone of the upstream repo) but not executed in a real GPU runtime.

## Files changed

- `experiments/sam3d_mhr_clad_smoke/_sam3d_source_import_check.py` (new)
- `experiments/sam3d_mhr_clad_smoke/tests/test_sam3d_source_import_check.py` (new)
- `experiments/sam3d_mhr_clad_smoke/_sam3d_inference_worker.py` (revised:
  source_import/model_code_import stage split, None-safe telemetry
  defaults)
- `experiments/sam3d_mhr_clad_smoke/decision_gate.py` (extended: one new
  `FailureCategory`, one new `PipelineState` field, one new `classify()`
  precondition, `phase_summary()` gains a sixth Phase A field)
- `experiments/sam3d_mhr_clad_smoke/tests/test_decision_gate.py` (extended)
- `notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb` (PHASE A1.5 rewritten to
  call the new standalone script with no `env=PYTHONPATH`; worker
  invocation cell drops `env=PYTHONPATH`, fixes `bool()` coercion, adds
  `SAM3D_MODEL_CODE_IMPORT` reporting; unused `sam3d_source_path` import
  removed from the shared-imports cell; decision-gate printout cell lists
  the new boundary)
- `docs/experiments/TASK03E_SAM3D_SOURCE_IMPORT_HARDENING.md` (this
  document)
- `experiments/sam3d_mhr_clad_smoke/README.md` (updated to describe the new
  script and Task 03E's fix)

`sam3d_source_path.py`, `sam3d_env_spec.py`, `install_log.py`, `run.py`,
`adapter.py`, `rescale.py`, `mtm_mapping.py`, `interchange.py`,
`_mhr_measure_worker.py`: unchanged.

## What human should run next

The same notebook, same setup (GPU runtime, `HF_TOKEN` Colab Secret),
`Runtime > Run all`. Watch PHASE A1.5's `SAM3D_MODULE_FILE` printout — it
should point into `/content/sam-3d-body/sam_3d_body/...`, and both
`SAM3D_SOURCE_IMPORT` and `SAM3D_MODEL_CODE_IMPORT` should print PASS
before any checkpoint download happens. If a failure still occurs at this
boundary, the printed stdout (captured directly from the standalone
script's own PASS/FAIL/error output, not just a generic subprocess
returncode) will show exactly which of the two stages it was and the
specific `ModuleNotFoundError`/other exception text.

## Decision gate (this agent's own attempt)

Same as Task 03/03B/03C/03D: **C. GPU_INSUFFICIENT** — no GPU in this
sandbox. Authored, structurally validated, and — beyond prior tasks — this
time also verified with a real, non-mocked import preflight against the
actual upstream repository. The real classification is whatever the
human's next Colab run reports.
