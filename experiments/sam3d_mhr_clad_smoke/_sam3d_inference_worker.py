"""Environment A worker: run real SAM 3D Body inference and write the
interchange file (Task 03B).

Invoked as a subprocess using Environment A's own dedicated venv Python --
see notebooks/TASK03_SAM3D_MHR_CLAD_COLAB.ipynb, PHASE A, and
docs/experiments/TASK03B_DEPENDENCY_RESOLUTION.md for why Environment A is
isolated from Environment B (they must never share torch/CUDA ABI state).

This script is the only thing in the whole pipeline that needs `torch` and
`sam_3d_body` importable; `interchange.py` and `adapter.py` (both plain
Python + numpy) are imported from the shared experiments/ directory so the
same validated field-mapping logic Task 02 wrote is reused unchanged, not
reimplemented for Environment A.

Usage:
    python _sam3d_inference_worker.py <image_path> <checkpoint_dir> \
        <interchange_output_path> <telemetry_output_path>

Always writes a JSON telemetry/status report to <telemetry_output_path>,
whether or not inference succeeded, so the calling (ambient-kernel)
process can inspect exactly what happened without parsing stderr. Writes
the interchange .npz to <interchange_output_path> only on success. Exits 0
on success, 1 on any recorded failure (never raises past its own
try/except -- a crash here should show up as a clear telemetry status, not
an opaque non-zero/negative return code the way Task 02's pymomentum
crash did).
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
        "python_version": sys.version.split()[0],
        "torch_version": None,
        "torch_cuda_version": None,
        "gpu_name": None,
        "sam3d_load_time_s": None,
        "sam3d_inference_time_s": None,
        "peak_vram_mb": None,
        "person_detected": False,
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
            telemetry["error"] = "NO_GPU: Environment A's venv could not see a CUDA device via torch"
            _write(telemetry_path, telemetry)
            return 1

        telemetry["gpu_name"] = torch.cuda.get_device_name(0)

        from adapter import AdapterError
        from interchange import write_interchange
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        ckpt_path = f"{checkpoint_dir}/model.ckpt"
        mhr_asset_path = f"{checkpoint_dir}/assets/mhr_model.pt"
        model, model_cfg = load_sam_3d_body(ckpt_path, device=device, mhr_path=mhr_asset_path)
        telemetry["sam3d_load_time_s"] = round(time.time() - t0, 1)

        estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)

        t1 = time.time()
        outputs = estimator.process_one_image(image_path)
        telemetry["sam3d_inference_time_s"] = round(time.time() - t1, 2)
        telemetry["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

        if not outputs:
            telemetry["status"] = "ok_no_person_detected"
            _write(telemetry_path, telemetry)
            return 0

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
        telemetry["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    _write(telemetry_path, telemetry)
    return 0 if telemetry["status"] in ("ok", "ok_no_person_detected") else 1


if __name__ == "__main__":
    sys.exit(main())
