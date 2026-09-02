"""Environment A worker: run real, minimal-core SAM 3D Body inference and
write the interchange file (Task 03B, revised Task 03C).

Invoked as a subprocess using Environment A's own dedicated venv Python --
see notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb (PHASE A0-A5) and
docs/experiments/TASK03C_MINIMAL_CORE_INFERENCE.md.

Task 03C deliberately removes every optional component from this call:
no human_detector (no Detectron2/ViTDet), no human_segmentor (no SAM2/
SAM3), no fov_estimator (no MoGe). `SAM3DBodyEstimator.__init__` supports
all three as `None` by design (it prints "No human detector is used...",
"Mask-condition inference is not supported...", "No FOV estimator...
Using the default FOV!" -- these are expected, first-class states, not
degraded fallbacks this script works around).

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

Reports each of Task 03C's four required Phase A boundaries independently
via `telemetry["stage"]`/`telemetry["status"]`, rather than one
undifferentiated status, so a failure at model load can be told apart from
a failure during inference or during interchange serialization:

    SAM3D_MODEL_LOAD        -- load_sam_3d_body() succeeds
    SAM3D_CORE_INFERENCE    -- process_one_image() succeeds, person found
    MHR_PARAMS_SERIALIZED   -- write_interchange() succeeds

(SAM3D_CORE_ENVIRONMENT -- whether this venv itself is usable at all -- is
necessarily determined by the caller before this script can even run, so
it is not one of this script's own stages.)

Usage:
    python _sam3d_inference_worker.py <image_path> <checkpoint_dir> \
        <interchange_output_path> <telemetry_output_path>

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


def _write(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: _sam3d_inference_worker.py <image_path> <checkpoint_dir> "
            "<interchange_output_path> <telemetry_output_path>",
            file=sys.stderr,
        )
        return 2

    image_path, checkpoint_dir, interchange_path, telemetry_path = sys.argv[1:5]

    telemetry = {
        "status": "error",
        "stage": "startup",
        "python_version": sys.version.split()[0],
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
        import numpy as np
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

        from adapter import AdapterError
        from interchange import write_interchange
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from sam_3d_body.data.utils.io import load_image

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
