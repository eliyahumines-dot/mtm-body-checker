"""Environment A worker: run real, minimal-core SAM 3D Body inference and
write the interchange file (Task 03B, revised Task 03C, revised Task 03D).

Invoked as a subprocess using Environment A's own dedicated venv Python --
see notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb (PHASE A0-A5) and
docs/experiments/TASK03D_SAM3D_IMPORT_PATH_FIX.md.

Task 03D fix: a real Colab run with a fully working Environment A (torch/
CUDA/GPU/pin all verified) still failed with `ModuleNotFoundError: No
module named 'sam_3d_body'`, because the upstream `facebookresearch/
sam-3d-body` repo root was never on this interpreter's module search path
-- that repo has no `pyproject.toml`/`setup.py`/`setup.cfg` (confirmed
directly against a fresh clone), so it cannot be `pip install -e .`'d;
its root must be used directly on `sys.path`. This worker now takes that
root as an explicit, validated argument (`sam3d_source_root`) and performs
a pre-flight import check -- `import sam_3d_body`, resolve
`sam_3d_body.__file__`, confirm it actually falls under the given root --
BEFORE attempting checkpoint/model loading, so "Python cannot import
sam_3d_body" (`SAM3D_SOURCE_IMPORT_FAILURE`) is never conflated with
"load_sam_3d_body() failed after a successful import"
(`SAM3D_MODEL_LOAD_FAILURE`, Task 03D section 6). The calling notebook
also sets `PYTHONPATH` when launching this subprocess (the "preferred
solution" per Task 03D section 3) -- this script's own `sys.path.insert()`
is a second, independently-verifiable mechanism, not a replacement for it.

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

    SAM3D_SOURCE_IMPORT     -- `import sam_3d_body` resolves under the given root (Task 03D)
    SAM3D_MODEL_LOAD        -- load_sam_3d_body() succeeds
    SAM3D_CORE_INFERENCE    -- process_one_image() succeeds, person found
    MHR_PARAMS_SERIALIZED   -- write_interchange() succeeds

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

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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

    telemetry = {
        "status": "error",
        "stage": "startup",
        "python_version": sys.version.split()[0],
        "sam3d_source_root": sam3d_source_root,
        "source_import_ok": False,
        "sam3d_module_file": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "gpu_name": None,
        "model_load_ok": False,
        "sam3d_load_time_s": None,
        "inference_ok": False,
        "sam3d_inference_time_s": None,
        "peak_vram_mb": None,
        "person_detected": False,
        "bbox_used": None,
        "image_width": None,
        "image_height": None,
        "output_schema": {},
        "interchange_written": False,
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

        # --- SAM3D_SOURCE_IMPORT (Task 03D) ---
        # Must succeed BEFORE any checkpoint/model-loading is attempted, so a
        # missing-import failure here is never misreported as SAM3D_MODEL_LOAD_FAILURE.
        telemetry["stage"] = "source_import"
        try:
            validated_root = validate_sam3d_source_root(sam3d_source_root)
        except Sam3dSourceRootError as exc:
            telemetry["status"] = "source_import_failure"
            telemetry["error"] = f"SAM3D_SOURCE_IMPORT_FAILURE: {exc}"
            _write(telemetry_path, telemetry)
            return 1

        if validated_root not in sys.path:
            sys.path.insert(0, validated_root)  # belt-and-suspenders alongside the notebook's PYTHONPATH

        try:
            import sam_3d_body
            from sam_3d_body.build_models import load_sam_3d_body
            from sam_3d_body.data.utils.io import load_image
            from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
        except Exception as exc:  # noqa: BLE001 -- any import-time failure here is SAM3D_SOURCE_IMPORT_FAILURE
            telemetry["status"] = "source_import_failure"
            telemetry["error"] = f"SAM3D_SOURCE_IMPORT_FAILURE: {type(exc).__name__}: {exc}"
            _write(telemetry_path, telemetry)
            return 1

        telemetry["sam3d_module_file"] = sam_3d_body.__file__
        print("sam_3d_body.__file__ =", sam_3d_body.__file__)

        if not resolved_module_is_under_root(sam_3d_body.__file__, validated_root):
            telemetry["status"] = "source_import_failure"
            telemetry["error"] = (
                f"SAM3D_SOURCE_IMPORT_FAILURE: sam_3d_body imported from "
                f"{sam_3d_body.__file__}, which is NOT under the intended source root "
                f"{validated_root} -- resolves to an unrelated installed package, not "
                f"the official Meta source tree."
            )
            _write(telemetry_path, telemetry)
            return 1

        telemetry["source_import_ok"] = True

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
            telemetry["status"] = "mhr_schema_failure"
            telemetry["error"] = f"MHR_SCHEMA_FAILURE: {exc}"

    except Exception as exc:  # noqa: BLE001 -- this is a boundary worker, must never propagate a bare crash
        telemetry["status"] = "error"
        telemetry["error"] = f"[{telemetry['stage']}] {type(exc).__name__}: {exc}"
        traceback.print_exc()

    _write(telemetry_path, telemetry)
    return 0 if telemetry["status"] in ("ok", "ok_no_person_detected") else 1


if __name__ == "__main__":
    sys.exit(main())
