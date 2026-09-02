"""Environment A worker: run real, minimal-core SAM 3D Body inference and
write the interchange file (Task 03B, revised Task 03C/03D/03E/03F).

Invoked as a subprocess using Environment A's own dedicated venv Python --
see notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb (PHASE A0-A5) and
docs/experiments/TASK03D_SAM3D_IMPORT_PATH_FIX.md,
docs/experiments/TASK03E_SAM3D_SOURCE_IMPORT_HARDENING.md,
docs/experiments/TASK03F_MATPLOTLIB_BACKEND_FIX.md.

Task 03D fix: a real Colab run with a fully working Environment A (torch/
CUDA/GPU/pin all verified) still failed with `ModuleNotFoundError: No
module named 'sam_3d_body'`, because the upstream `facebookresearch/
sam-3d-body` repo root was never on this interpreter's module search path
-- that repo has no `pyproject.toml`/`setup.py`/`setup.cfg` (confirmed
directly against a fresh clone), so it cannot be `pip install -e .`'d;
its root must be used directly on `sys.path`. This worker takes that
root as an explicit, validated argument (`sam3d_source_root`) and performs
a pre-flight import check -- `import sam_3d_body`, resolve
`sam_3d_body.__file__`, confirm it actually falls under the given root --
BEFORE attempting checkpoint/model loading.

Task 03E fix: the *same* Colab failure recurred on a second run, meaning
Task 03D's fix (which relied partly on the notebook also setting
`PYTHONPATH` on the subprocess `env=`) was not actually sufficient in real
Colab -- a mechanism this agent's sandbox cannot reproduce a failure in
either way, so rather than continue guessing at Colab-specific env-var
propagation, every remaining reliance on `env=PYTHONPATH` has been removed
in favor of *only* this script's own deterministic `sys.path.insert()`
from the explicit CLI argument (never cwd, never ambient `PYTHONPATH`).
Task 03E also splits what was one conflated `source_import` stage into
two independently-reportable ones, matching the standalone
`_sam3d_source_import_check.py` pre-flight script exactly:

    SAM3D_SOURCE_IMPORT      -- bare `import sam_3d_body` resolves, and
                                 under the given root (not an unrelated
                                 installed package).
    SAM3D_MODEL_CODE_IMPORT  -- `from sam_3d_body.build_models import
                                 load_sam_3d_body`, `from sam_3d_body.
                                 sam_3d_body_estimator import
                                 SAM3DBodyEstimator`, and this project's
                                 own `sam_3d_body.data.utils.io.load_image`
                                 dependency -- SAM 3D Body's own Python code
                                 constructing without error, distinct from
                                 checkpoint/model *loading* itself
                                 (SAM3D_MODEL_LOAD, which needs real
                                 checkpoint assets).

Every "not yet attempted" telemetry field now defaults to `None`, not
`False` -- Task 03E section 7's "cascading false failures" bug: a `False`
default, combined with `bool(telemetry.get(...))` coercion in the notebook,
silently turned "this stage never ran because an earlier one failed" into
a reported FAIL rather than NOT_ATTEMPTED for every downstream stage.

Task 03F fix: a *subsequent* real Colab run got past source-root resolution
entirely (Task 03E's fix was correct) and then failed *inside*
`import sam_3d_body` itself: `ValueError: Key backend:
'module://matplotlib_inline.backend_inline' is not a valid value for
backend`. Reproduced directly in this task against a real clone and a real
dependency-complete venv: Colab's interactive kernel sets `MPLBACKEND` to
its own inline-plotting backend; a standalone (non-notebook) subprocess
inherits that same environment variable by default; something SAM 3D Body
imports transitively (in this task's reproduction: `torchmetrics`, via
`pytorch_lightning`) imports `matplotlib` as a side effect, which then
fails to activate the inherited, notebook-only backend name. Fixed via
`sam3d_matplotlib_guard.force_headless_matplotlib_backend()`, called at the
very top of this module -- before anything that could import matplotlib --
which forces `MPLBACKEND=Agg` unconditionally (never `setdefault()`: the
invalid inherited value already exists and must be overridden, not
preserved).

Task 03F also splits what was previously folded into `SAM3D_SOURCE_IMPORT`
(source-root validation failing) into its own, earlier stage,
`source_root_validation` -- source-root validation and module *execution*
are distinct (a root can be perfectly valid and still fail to import for
unrelated runtime reasons, the Matplotlib case above being the concrete
example), and import failures are now classified via
`sam3d_matplotlib_guard.classify_import_exception()` into
`SAM3D_MATPLOTLIB_BACKEND_FAILURE` or the more general
`SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE` rather than always being
described as a source-root/path problem.

Task 03C deliberately removes every optional component from the actual
inference call: no human_detector (no Detectron2/ViTDet), no
human_segmentor (no SAM2/SAM3), no fov_estimator (no MoGe).
`SAM3DBodyEstimator.__init__` supports all three as `None` by design (it
prints "No human detector is used...", "Mask-condition inference is not
supported...", "No FOV estimator... Using the default FOV!" -- these are
expected, first-class states, not degraded fallbacks this script works
around).

Bounding box: passed EXPLICITLY as the full image
(``[[0, 0, width, height]]``), matching this project's controlled
single-centered-subject assumption (Task 03C section 2/8). Source reading
of `sam_3d_body/sam_3d_body_estimator.py::process_one_image` confirms this
is functionally identical to leaving `bboxes=None` with no detector, which
falls back to the exact same array internally (`self.is_crop` differs
between the two paths but is assigned and never read anywhere in the
repo -- confirmed by a full-repo grep, so this has no observable effect).
Explicit is chosen anyway so the box actually used is visible in telemetry
and validated (section 8) rather than left implicit.

Reports each Phase A boundary independently via
`telemetry["stage"]`/`telemetry["status"]`, so a failure at any one stage
can be told apart from the others:

    SAM3D_SOURCE_ROOT_VALIDATION -- given root exists, has sam_3d_body/__init__.py (Task 03F)
    SAM3D_SOURCE_IMPORT          -- `import sam_3d_body` actually executes and resolves under
                                     the given root (Task 03D; split from root validation Task 03F)
    SAM3D_MODEL_CODE_IMPORT      -- build_models/sam_3d_body_estimator/data.utils.io import (Task 03E)
    SAM3D_MODEL_LOAD             -- load_sam_3d_body() succeeds
    SAM3D_CORE_INFERENCE         -- process_one_image() succeeds, person found
    MHR_PARAMS_SERIALIZED        -- write_interchange() succeeds

(SAM3D_CORE_ENVIRONMENT -- whether this venv itself is usable at all -- is
necessarily determined by the caller before this script can even run, so
it is not one of this script's own stages.)

Usage:
    python _sam3d_inference_worker.py <image_path> <checkpoint_dir> \
        <sam3d_source_root> <interchange_output_path> <telemetry_output_path>

Always writes a JSON telemetry report to <telemetry_output_path>, whether
or not inference succeeded. Writes the interchange .npz only on success.
Exits 0 on success, 1 on any recorded failure -- never raises past its own
try/except, so a crash here shows up as a clear telemetry status, not an
opaque non-zero/negative return code.
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
# the earliest point in this module such an import could occur.
_INHERITED_MPLBACKEND = force_headless_matplotlib_backend()

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

from sam3d_source_path import (  # noqa: E402
    Sam3dSourceRootError,
    resolved_module_is_under_root,
    validate_sam3d_source_root,
)


def _write(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: _sam3d_inference_worker.py <image_path> <checkpoint_dir> "
            "<sam3d_source_root> <interchange_output_path> <telemetry_output_path>",
            file=sys.stderr,
        )
        return 2

    image_path, checkpoint_dir, sam3d_source_root, interchange_path, telemetry_path = sys.argv[1:6]

    # Task 03E: every "not yet attempted" field defaults to None, never False --
    # a False default here previously got coerced by the notebook's bool(telemetry.get(...))
    # into a reported FAIL for a stage that never even ran (cascading false failures).
    telemetry = {
        "status": "error",
        "stage": "startup",
        "python_version": sys.version.split()[0],
        "sam3d_source_root": sam3d_source_root,
        "inherited_mplbackend": _INHERITED_MPLBACKEND,
        "worker_mplbackend": HEADLESS_BACKEND,
        "matplotlib_backend_effective": None,
        "source_root_validation_ok": None,
        "source_import_ok": None,
        "model_code_import_ok": None,
        "failure_category": None,
        "sam3d_module_file": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "gpu_name": None,
        "model_load_ok": None,
        "sam3d_load_time_s": None,
        "inference_ok": None,
        "sam3d_inference_time_s": None,
        "peak_vram_mb": None,
        "person_detected": None,
        "bbox_used": None,
        "image_width": None,
        "image_height": None,
        "output_schema": {},
        "interchange_written": None,
        "error": None,
    }

    try:
        import torch

        telemetry["torch_version"] = torch.__version__
        telemetry["torch_cuda_version"] = torch.version.cuda

        if not torch.cuda.is_available():
            telemetry["status"] = "no_gpu"
            telemetry["stage"] = "gpu_check"
            telemetry["error"] = "NO_GPU: Environment A's venv could not see a CUDA device via torch"
            _write(telemetry_path, telemetry)
            return 1

        telemetry["gpu_name"] = torch.cuda.get_device_name(0)

        # --- SAM3D_SOURCE_ROOT_VALIDATION (Task 03F: split out of SAM3D_SOURCE_IMPORT) ---
        # Must succeed BEFORE any checkpoint/model-loading is attempted, so a
        # missing-root failure here is never misreported as SAM3D_MODEL_LOAD_FAILURE.
        # Task 03E: relies ONLY on this script's own sys.path.insert() from the explicit
        # sam3d_source_root argument -- never on an ambient/caller-supplied PYTHONPATH.
        telemetry["stage"] = "source_root_validation"
        try:
            validated_root = validate_sam3d_source_root(sam3d_source_root)
        except Sam3dSourceRootError as exc:
            telemetry["source_root_validation_ok"] = False
            telemetry["status"] = "source_root_validation_failure"
            telemetry["failure_category"] = "SAM3D_SOURCE_ROOT_VALIDATION_FAILURE"
            telemetry["error"] = f"SAM3D_SOURCE_ROOT_VALIDATION_FAILURE: {exc}"
            _write(telemetry_path, telemetry)
            return 1

        telemetry["source_root_validation_ok"] = True

        if validated_root not in sys.path:
            sys.path.insert(0, validated_root)

        # --- SAM3D_SOURCE_IMPORT ---
        # Distinct from root validation above: the root can be perfectly valid (exists,
        # has sam_3d_body/__init__.py) and `import sam_3d_body` can still fail for
        # unrelated runtime reasons (Task 03F: an inherited, invalid Colab MPLBACKEND
        # value crashing a transitive matplotlib import) -- never described as a
        # source-root/path problem unless it actually is one.
        telemetry["stage"] = "source_import"
        try:
            import sam_3d_body
        except Exception as exc:  # noqa: BLE001 -- classified below, never left generic
            telemetry["source_import_ok"] = False
            telemetry["status"] = "source_import_failure"
            category = classify_import_exception(exc)
            telemetry["failure_category"] = category
            telemetry["error"] = f"{category}: {type(exc).__name__}: {exc}"
            telemetry["matplotlib_backend_effective"] = effective_matplotlib_backend()
            _write(telemetry_path, telemetry)
            return 1

        telemetry["sam3d_module_file"] = sam_3d_body.__file__
        telemetry["matplotlib_backend_effective"] = effective_matplotlib_backend()
        print("sam_3d_body.__file__ =", sam_3d_body.__file__)

        if not resolved_module_is_under_root(sam_3d_body.__file__, validated_root):
            telemetry["source_import_ok"] = False
            telemetry["status"] = "source_import_failure"
            telemetry["failure_category"] = "SAM3D_SOURCE_IMPORT_FAILURE"
            telemetry["error"] = (
                f"SAM3D_SOURCE_IMPORT_FAILURE: sam_3d_body imported from "
                f"{sam_3d_body.__file__}, which is NOT under the intended source root "
                f"{validated_root} -- resolves to an unrelated installed package, not "
                f"the official Meta source tree."
            )
            _write(telemetry_path, telemetry)
            return 1

        telemetry["source_import_ok"] = True

        # --- SAM3D_MODEL_CODE_IMPORT (Task 03E) ---
        # Distinct from SAM3D_SOURCE_IMPORT above: the bare `sam_3d_body` package
        # importing cleanly does not guarantee its model-construction submodules do too
        # (e.g. a missing sub-dependency only reachable from build_models/estimator code).
        # Also distinct from SAM3D_MODEL_LOAD below: this is Python code constructing,
        # not checkpoint/model assets being loaded from disk.
        telemetry["stage"] = "model_code_import"
        try:
            from sam_3d_body.build_models import load_sam_3d_body
            from sam_3d_body.data.utils.io import load_image
            from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
        except Exception as exc:  # noqa: BLE001
            telemetry["model_code_import_ok"] = False
            telemetry["status"] = "model_code_import_failure"
            telemetry["failure_category"] = "SAM3D_MODEL_CODE_IMPORT_FAILURE"
            telemetry["error"] = f"SAM3D_MODEL_CODE_IMPORT_FAILURE: {type(exc).__name__}: {exc}"
            telemetry["matplotlib_backend_effective"] = effective_matplotlib_backend()
            _write(telemetry_path, telemetry)
            return 1

        telemetry["model_code_import_ok"] = True

        import numpy as np

        from adapter import AdapterError
        from interchange import write_interchange

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()

        # --- SAM3D_MODEL_LOAD ---
        telemetry["stage"] = "model_load"
        t0 = time.time()
        ckpt_path = f"{checkpoint_dir}/model.ckpt"
        mhr_asset_path = f"{checkpoint_dir}/assets/mhr_model.pt"
        model, model_cfg = load_sam_3d_body(ckpt_path, device=device, mhr_path=mhr_asset_path)
        telemetry["sam3d_load_time_s"] = round(time.time() - t0, 1)
        telemetry["model_load_ok"] = True

        # Minimal core estimator: no detector, no segmentor, no FOV estimator.
        # These print upstream's own "No human detector is used...", "Mask-condition
        # inference is not supported...", "No FOV estimator... Using the default FOV!"
        # -- expected, first-class states for this smoke test, not silently degraded ones.
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )

        # --- bounding box validation (Task 03C section 8) ---
        telemetry["stage"] = "bbox_validation"
        img = load_image(image_path, backend="cv2", image_format="bgr")
        height, width = img.shape[:2]
        telemetry["image_width"] = int(width)
        telemetry["image_height"] = int(height)
        if width <= 0 or height <= 0:
            raise ValueError(f"Loaded image has invalid dimensions: {width}x{height}")
        bbox = np.array([[0, 0, width, height]], dtype=np.float32)
        telemetry["bbox_used"] = bbox.tolist()

        # --- SAM3D_CORE_INFERENCE ---
        telemetry["stage"] = "inference"
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        outputs = estimator.process_one_image(image_path, bboxes=bbox)
        telemetry["sam3d_inference_time_s"] = round(time.time() - t1, 2)
        telemetry["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

        if not outputs:
            telemetry["status"] = "ok_no_person_detected"
            telemetry["inference_ok"] = True  # ran without error; just produced no output
            telemetry["person_detected"] = False  # attempted and genuinely found nobody, not "not attempted"
            _write(telemetry_path, telemetry)
            return 0

        telemetry["inference_ok"] = True
        person_output = outputs[0]
        telemetry["person_detected"] = True

        for k, v in person_output.items():
            try:
                if hasattr(v, "shape"):
                    shape = list(v.shape)
                elif hasattr(v, "__len__"):
                    shape = [len(v)]
                else:
                    shape = None
            except Exception:
                shape = None
            dtype = str(v.dtype) if hasattr(v, "dtype") else type(v).__name__
            telemetry["output_schema"][k] = {"shape": shape, "dtype": dtype}

        # --- MHR_PARAMS_SERIALIZED ---
        telemetry["stage"] = "interchange_write"
        try:
            write_summary = write_interchange(interchange_path, person_output, source_checkpoint=checkpoint_dir)
            telemetry["interchange_written"] = True
            telemetry["interchange_fields"] = write_summary
            telemetry["status"] = "ok"
        except AdapterError as exc:
            telemetry["interchange_written"] = False
            telemetry["status"] = "mhr_schema_failure"
            telemetry["error"] = f"MHR_SCHEMA_FAILURE: {exc}"

    except Exception as exc:  # noqa: BLE001 -- this is a boundary worker, must never propagate a bare crash
        telemetry["status"] = "error"
        telemetry["error"] = f"[{telemetry['stage']}] {type(exc).__name__}: {exc}"
        # Task 03E: a stage that was actually reached and crashed must report False, not
        # leave its field at the None default -- a None here would misreport a real,
        # attempted failure as NOT_ATTEMPTED (the same cascading-false-failure class of
        # bug this task fixes, just in the opposite direction: never mask FAIL as
        # NOT_ATTEMPTED either). Stages with no dedicated field (e.g. bbox_validation)
        # correctly leave the next stage's field at None -- it genuinely never started.
        _stage_field = {
            "model_load": "model_load_ok",
            "inference": "inference_ok",
            "interchange_write": "interchange_written",
        }.get(telemetry["stage"])
        if _stage_field is not None and telemetry.get(_stage_field) is None:
            telemetry[_stage_field] = False
        traceback.print_exc()

    _write(telemetry_path, telemetry)
    return 0 if telemetry["status"] in ("ok", "ok_no_person_detected") else 1


if __name__ == "__main__":
    sys.exit(main())
