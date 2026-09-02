"""Standalone SAM3D_SOURCE_ROOT_VALIDATION / SAM3D_SOURCE_IMPORT /
SAM3D_MODEL_CODE_IMPORT pre-flight check -- Task 03E, revised Task 03F.

Replaces Task 03D's notebook-inline `-c` preflight, which launched a
subprocess relying on `env={**os.environ, 'PYTHONPATH': ...}` to expose the
cloned `facebookresearch/sam-3d-body` repo to a separate interpreter. A
second real Colab run still failed with the exact same
`ModuleNotFoundError: No module named 'sam_3d_body'` Task 03D was meant to
fix -- meaning that mechanism was not actually sufficient in real Colab, for
a reason this agent's sandbox could not reproduce or observe directly
(both the `env=PYTHONPATH` mechanism and a `sys.path.insert()`-based one
succeeded identically here, against a real fresh clone of the upstream repo
with the full production dependency set installed). Rather than continue
speculating about an unreproducible Colab-specific environment-propagation
quirk, this script eliminates the dependency on subprocess `env=` entirely:
it takes the source root as an explicit CLI argument and inserts it into
its *own* `sys.path` directly -- deterministic, testable in isolation, and
never dependent on cwd, ambient `PYTHONPATH`, shell session state, or
"accidental" editable-install magic. This mirrors exactly the mechanism
`_sam3d_inference_worker.py` already uses internally (and that this task
verified, with a real clone, actually works).

Task 03F fix: with the source-import path fixed, a *subsequent* real Colab
run got further -- the standalone subprocess actually found and inserted
the right `sys.path` entry -- and then failed *inside* `import sam_3d_body`
itself with `ValueError: Key backend:
'module://matplotlib_inline.backend_inline' is not a valid value for
backend`. Reproduced directly in this task, with a real clone and a real
dependency-complete venv: Colab's interactive kernel sets `MPLBACKEND` to
its own inline-plotting backend, a standalone (non-notebook) subprocess
inherits that same environment variable by default, and something SAM 3D
Body imports transitively (in this task's reproduction: `torchmetrics`, via
`pytorch_lightning`) imports `matplotlib` as a side effect, which then
fails to activate the inherited, notebook-only backend name. See
`sam3d_matplotlib_guard.py` for the fix (force `MPLBACKEND=Agg`,
unconditionally, before anything that might import matplotlib) and the full
root-cause writeup.

Task 03F also splits what was previously folded into `SAM3D_SOURCE_IMPORT`
(source-root validation failing) into its own, earlier boundary,
`SAM3D_SOURCE_ROOT_VALIDATION` -- section 6 of that task: "source-root
validation and module execution are distinct." A root that exists and
contains an importable `sam_3d_body/__init__.py` (root validation PASS) can
still fail to actually *execute* on `import sam_3d_body` for reasons that
have nothing to do with the path being wrong (the Matplotlib backend case
above being the concrete example) -- that must never be described as "repo
root not on module search path."

Reports three independently-observable boundaries, in order, so a failure
is never misclassified as coming from an earlier or later stage:

    SAM3D_SOURCE_ROOT_VALIDATION -- the given root exists and contains an
                                     importable `sam_3d_body/__init__.py`
                                     (Task 03F: distinct from the import
                                     that follows actually succeeding).
    SAM3D_SOURCE_IMPORT          -- bare `import sam_3d_body` succeeds, and
                                     `sam_3d_body.__file__` resolves under
                                     the given root (not an unrelated
                                     installed package shadowing the
                                     intended repo).
    SAM3D_MODEL_CODE_IMPORT      -- `from sam_3d_body.build_models import
                                     load_sam_3d_body` and `from
                                     sam_3d_body.sam_3d_body_estimator
                                     import SAM3DBodyEstimator` succeed --
                                     SAM 3D Body's own model-construction
                                     Python code importing cleanly, distinct
                                     from checkpoint/model assets actually
                                     being loaded from disk (SAM3D_MODEL_LOAD,
                                     which this script never attempts -- it
                                     requires real gated checkpoint files
                                     this script has no reason to need).

Prints, in order:

    Inherited MPLBACKEND: <value this process's environment had for
                           MPLBACKEND before this script overrode it, or
                           None>
    Worker MPLBACKEND: Agg
    SAM3D_SOURCE_ROOT: <resolved absolute path, or the raw input if
                        validation itself failed>
    SAM3D_MODULE_FILE: <sam_3d_body.__file__, or None>
    SAM3D_SOURCE_ROOT_VALIDATION: PASS|FAIL
    SAM3D_SOURCE_IMPORT: PASS|FAIL|NOT_ATTEMPTED
    SAM3D_MODEL_CODE_IMPORT: PASS|FAIL|NOT_ATTEMPTED
    Effective matplotlib backend: <matplotlib.get_backend(), if matplotlib
                                   was imported at all; omitted otherwise>

so the exact boundary is visible in captured stdout even if no telemetry
file is requested. A stage is only ever printed as PASS/FAIL once the stage
before it actually passed -- otherwise it is NOT_ATTEMPTED, never a false
FAIL (Task 03E section 7, carried forward in Task 03F section 7).

Usage:
    python _sam3d_source_import_check.py <sam3d_source_root> [telemetry_output_path]

Exit code 0 only if all three stages pass; 1 otherwise; 2 on a usage error.
Never raises past its own try/except -- a broken import shows up as a
clear PASS/FAIL printout and telemetry record, not a bare traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sam3d_matplotlib_guard import (  # noqa: E402
    HEADLESS_BACKEND,
    classify_import_exception,
    effective_matplotlib_backend,
    force_headless_matplotlib_backend,
)

# Task 03F: force a headless backend BEFORE anything that might import
# matplotlib as a side effect (sam_3d_body, pyrender, torchmetrics via
# pytorch_lightning, etc.) -- must happen before `import sam_3d_body` below,
# which is the earliest point in this script such an import could occur.
_INHERITED_MPLBACKEND = force_headless_matplotlib_backend()

import json  # noqa: E402

from sam3d_source_path import (  # noqa: E402
    Sam3dSourceRootError,
    resolved_module_is_under_root,
    validate_sam3d_source_root,
)


def run_preflight(sam3d_source_root: str) -> dict:
    """Run all three boundaries against `sam3d_source_root`, returning a
    telemetry dict. Never raises -- any failure is captured in the dict.

    `source_root_validation_ok`/`source_import_ok`/`model_code_import_ok`
    are `None` only while a stage has not yet been attempted (e.g.
    `source_import_ok` stays `None` if `source_root_validation_ok` is
    `False`) -- never a `False` standing in for "not attempted" (Task 03E
    section 7, Task 03F section 7).
    """
    telemetry: dict = {
        "sam3d_source_root": sam3d_source_root,
        "sam3d_source_root_resolved": None,
        "sam3d_module_file": None,
        "inherited_mplbackend": _INHERITED_MPLBACKEND,
        "worker_mplbackend": HEADLESS_BACKEND,
        "matplotlib_backend_effective": None,
        "source_root_validation_ok": None,
        "source_import_ok": None,
        "model_code_import_ok": None,
        "failure_category": None,
        "error": None,
    }

    # --- SAM3D_SOURCE_ROOT_VALIDATION (Task 03F: split out of SAM3D_SOURCE_IMPORT) ---
    try:
        validated_root = validate_sam3d_source_root(sam3d_source_root)
    except Sam3dSourceRootError as exc:
        telemetry["source_root_validation_ok"] = False
        telemetry["failure_category"] = "SAM3D_SOURCE_ROOT_VALIDATION_FAILURE"
        telemetry["error"] = f"SAM3D_SOURCE_ROOT_VALIDATION_FAILURE: {exc}"
        return telemetry

    telemetry["source_root_validation_ok"] = True
    telemetry["sam3d_source_root_resolved"] = validated_root

    if validated_root not in sys.path:
        sys.path.insert(0, validated_root)

    # --- SAM3D_SOURCE_IMPORT ---
    try:
        import sam_3d_body
    except Exception as exc:  # noqa: BLE001 -- classified below, never left generic
        telemetry["source_import_ok"] = False
        category = classify_import_exception(exc)
        telemetry["failure_category"] = category
        telemetry["error"] = f"{category}: {type(exc).__name__}: {exc}"
        telemetry["matplotlib_backend_effective"] = effective_matplotlib_backend()
        return telemetry

    telemetry["sam3d_module_file"] = sam_3d_body.__file__
    telemetry["matplotlib_backend_effective"] = effective_matplotlib_backend()

    if not resolved_module_is_under_root(sam_3d_body.__file__, validated_root):
        telemetry["source_import_ok"] = False
        telemetry["failure_category"] = "SAM3D_SOURCE_IMPORT_FAILURE"
        telemetry["error"] = (
            f"SAM3D_SOURCE_IMPORT_FAILURE: sam_3d_body imported from "
            f"{sam_3d_body.__file__}, which is NOT under the intended source root "
            f"{validated_root} -- resolves to an unrelated installed package, not "
            f"the official Meta source tree."
        )
        return telemetry

    telemetry["source_import_ok"] = True

    # --- SAM3D_MODEL_CODE_IMPORT ---
    try:
        from sam_3d_body.build_models import load_sam_3d_body  # noqa: F401
        from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        telemetry["model_code_import_ok"] = False
        telemetry["failure_category"] = "SAM3D_MODEL_CODE_IMPORT_FAILURE"
        telemetry["error"] = f"SAM3D_MODEL_CODE_IMPORT_FAILURE: {type(exc).__name__}: {exc}"
        telemetry["matplotlib_backend_effective"] = effective_matplotlib_backend()
        return telemetry

    telemetry["model_code_import_ok"] = True
    return telemetry


def _status(ok: bool | None, gated_on: bool | None) -> str:
    if gated_on is not True:
        return "NOT_ATTEMPTED"
    return "PASS" if ok else "FAIL"


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(
            "usage: _sam3d_source_import_check.py <sam3d_source_root> [telemetry_output_path]",
            file=sys.stderr,
        )
        return 2

    sam3d_source_root = sys.argv[1]
    telemetry_path = sys.argv[2] if len(sys.argv) == 3 else None

    telemetry = run_preflight(sam3d_source_root)

    print("Inherited MPLBACKEND:", telemetry["inherited_mplbackend"])
    print("Worker MPLBACKEND:", telemetry["worker_mplbackend"])
    print("SAM3D_SOURCE_ROOT:", telemetry["sam3d_source_root_resolved"] or sam3d_source_root)
    print("SAM3D_MODULE_FILE:", telemetry["sam3d_module_file"])
    print("SAM3D_SOURCE_ROOT_VALIDATION:", "PASS" if telemetry["source_root_validation_ok"] else "FAIL")
    print("SAM3D_SOURCE_IMPORT:", _status(telemetry["source_import_ok"], telemetry["source_root_validation_ok"]))
    print("SAM3D_MODEL_CODE_IMPORT:", _status(telemetry["model_code_import_ok"], telemetry["source_import_ok"]))
    if telemetry["matplotlib_backend_effective"] is not None:
        print("Effective matplotlib backend:", telemetry["matplotlib_backend_effective"])

    if telemetry["error"]:
        print(telemetry["error"], file=sys.stderr)

    if telemetry_path:
        with open(telemetry_path, "w") as f:
            json.dump(telemetry, f, indent=2)

    ok = (
        bool(telemetry["source_root_validation_ok"])
        and bool(telemetry["source_import_ok"])
        and bool(telemetry["model_code_import_ok"])
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
