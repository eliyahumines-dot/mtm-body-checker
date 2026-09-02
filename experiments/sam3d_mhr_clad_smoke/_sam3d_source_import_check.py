"""Standalone SAM3D_SOURCE_IMPORT / SAM3D_MODEL_CODE_IMPORT pre-flight check
-- Task 03E.

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

Reports two independently-observable stages, matching
`_sam3d_inference_worker.py`'s own split exactly, so a Python import
failure is never misclassified as a checkpoint/model-loading failure:

    SAM3D_SOURCE_IMPORT      -- bare `import sam_3d_body` succeeds, and
                                 `sam_3d_body.__file__` resolves under the
                                 given root (not an unrelated installed
                                 package shadowing the intended repo).
    SAM3D_MODEL_CODE_IMPORT  -- `from sam_3d_body.build_models import
                                 load_sam_3d_body` and `from
                                 sam_3d_body.sam_3d_body_estimator import
                                 SAM3DBodyEstimator` succeed -- SAM 3D
                                 Body's own model-construction Python code
                                 importing cleanly, distinct from
                                 checkpoint/model assets actually being
                                 loaded from disk (SAM3D_MODEL_LOAD, which
                                 this script never attempts -- it requires
                                 real gated checkpoint files this script has
                                 no reason to need).

Prints, in order:

    SAM3D_SOURCE_ROOT: <resolved absolute path, or the raw input if
                        validation itself failed>
    SAM3D_MODULE_FILE: <sam_3d_body.__file__, or None>
    SAM3D_SOURCE_IMPORT: PASS|FAIL
    SAM3D_MODEL_CODE_IMPORT: PASS|FAIL|NOT_ATTEMPTED

so the exact boundary is visible in captured stdout even if no telemetry
file is requested. `SAM3D_MODEL_CODE_IMPORT` is only ever printed as
PASS/FAIL when SAM3D_SOURCE_IMPORT itself passed -- otherwise it is
NOT_ATTEMPTED, never a false FAIL (Task 03E section 7).

Usage:
    python _sam3d_source_import_check.py <sam3d_source_root> [telemetry_output_path]

Exit code 0 only if both stages pass; 1 otherwise; 2 on a usage error.
Never raises past its own try/except -- a broken import shows up as a
clear PASS/FAIL printout and telemetry record, not a bare traceback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sam3d_source_path import (  # noqa: E402
    Sam3dSourceRootError,
    resolved_module_is_under_root,
    validate_sam3d_source_root,
)


def run_preflight(sam3d_source_root: str) -> dict:
    """Run both import stages against `sam3d_source_root`, returning a
    telemetry dict. Never raises -- any failure is captured in the dict.

    `source_import_ok`/`model_code_import_ok` are `None` only while a
    stage has not yet been attempted (e.g. `model_code_import_ok` stays
    `None` if `source_import_ok` is `False`) -- never a `False` standing
    in for "not attempted" (Task 03E section 7).
    """
    telemetry: dict = {
        "sam3d_source_root": sam3d_source_root,
        "sam3d_source_root_resolved": None,
        "sam3d_module_file": None,
        "source_import_ok": None,
        "model_code_import_ok": None,
        "error": None,
    }

    try:
        validated_root = validate_sam3d_source_root(sam3d_source_root)
    except Sam3dSourceRootError as exc:
        telemetry["source_import_ok"] = False
        telemetry["error"] = f"SAM3D_SOURCE_IMPORT_FAILURE: {exc}"
        return telemetry

    telemetry["sam3d_source_root_resolved"] = validated_root

    if validated_root not in sys.path:
        sys.path.insert(0, validated_root)

    try:
        import sam_3d_body
    except Exception as exc:  # noqa: BLE001 -- any import-time failure here is SAM3D_SOURCE_IMPORT_FAILURE
        telemetry["source_import_ok"] = False
        telemetry["error"] = f"SAM3D_SOURCE_IMPORT_FAILURE: {type(exc).__name__}: {exc}"
        return telemetry

    telemetry["sam3d_module_file"] = sam_3d_body.__file__

    if not resolved_module_is_under_root(sam_3d_body.__file__, validated_root):
        telemetry["source_import_ok"] = False
        telemetry["error"] = (
            f"SAM3D_SOURCE_IMPORT_FAILURE: sam_3d_body imported from "
            f"{sam_3d_body.__file__}, which is NOT under the intended source root "
            f"{validated_root} -- resolves to an unrelated installed package, not "
            f"the official Meta source tree."
        )
        return telemetry

    telemetry["source_import_ok"] = True

    try:
        from sam_3d_body.build_models import load_sam_3d_body  # noqa: F401
        from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        telemetry["model_code_import_ok"] = False
        telemetry["error"] = f"SAM3D_MODEL_CODE_IMPORT_FAILURE: {type(exc).__name__}: {exc}"
        return telemetry

    telemetry["model_code_import_ok"] = True
    return telemetry


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

    print("SAM3D_SOURCE_ROOT:", telemetry["sam3d_source_root_resolved"] or sam3d_source_root)
    print("SAM3D_MODULE_FILE:", telemetry["sam3d_module_file"])
    print("SAM3D_SOURCE_IMPORT:", "PASS" if telemetry["source_import_ok"] else "FAIL")
    if telemetry["source_import_ok"]:
        print("SAM3D_MODEL_CODE_IMPORT:", "PASS" if telemetry["model_code_import_ok"] else "FAIL")
    else:
        print("SAM3D_MODEL_CODE_IMPORT: NOT_ATTEMPTED")

    if telemetry["error"]:
        print(telemetry["error"], file=sys.stderr)

    if telemetry_path:
        with open(telemetry_path, "w") as f:
            json.dump(telemetry, f, indent=2)

    ok = bool(telemetry["source_import_ok"]) and bool(telemetry["model_code_import_ok"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
