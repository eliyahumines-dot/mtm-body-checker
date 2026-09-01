#!/usr/bin/env python3
"""SAM 3D Body -> MHR -> clad-body smoke-test CLI.

Proves (or disproves) that the pipeline
    image -> SAM 3D Body -> MHR params -> clad-body -> measurements
runs end-to-end, and reports exactly which stage worked, which stage was
blocked, and why -- per docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md.

This is a SANITY/INTEROPERABILITY TEST, not an accuracy benchmark. See that
document's "Measurement extraction is not body accuracy" section: a
successful run here proves clad-body can compute measurements from a given
MHR mesh, not that SAM 3D Body reconstructed anyone's true body shape.

Two ways to get MHR params into the pipeline:
  1. --image + --sam3d-checkpoint: run real SAM 3D Body inference (requires
     the `sam_3d_body` package and a downloaded, license-gated checkpoint --
     see INSTALL.md in the upstream repo). If either is unavailable, this
     stage reports a "blocked_*" status and the run continues using
     --mhr-params-json if one was also given.
  2. --mhr-params-json: skip inference and measure an existing SAM3D-style
     params JSON directly (e.g. one of clad-body's own bundled test
     fixtures, or a real SAM 3D Body run captured elsewhere). This is how
     this task validated the MHR -> clad-body half of the pipeline in an
     environment where the SAM 3D Body checkpoint could not be downloaded.

Usage:
    python run.py --mhr-params-json <path> [--known-height-cm 180] --output <path>
    python run.py --image <path> --sam3d-checkpoint <path> [--known-height-cm 180] --output <path>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from adapter import (  # noqa: E402
    AdapterError,
    sam3d_output_to_clad_params,
    warn_if_scale_params_would_be_misused,
)

HERE = Path(__file__).parent


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _device_string() -> str:
    if _package_available("torch"):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "unknown (torch not installed)"


def run_sam3d_inference(image_path: str, checkpoint_path: str | None, warnings: list, failures: list):
    """Attempt real SAM 3D Body inference. Returns (status, person_output_or_None)."""
    if not _package_available("sam_3d_body"):
        return "blocked_no_package: `sam_3d_body` is not installed in this environment", None
    if not checkpoint_path:
        return (
            "blocked_no_checkpoint: sam_3d_body is installed but no --sam3d-checkpoint "
            "was given. SAM 3D Body checkpoints require manual Hugging Face access "
            "approval and cannot be auto-downloaded -- see TASK02_ENVIRONMENT.md",
            None,
        )
    if not os.path.exists(checkpoint_path):
        return f"blocked_no_checkpoint: checkpoint path does not exist: {checkpoint_path}", None

    try:
        import torch
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_cfg = load_sam_3d_body(checkpoint_path, device=device)
        estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)
        outputs = estimator.process_one_image(image_path)
        if not outputs:
            warnings.append("SAM 3D Body ran but detected no person in the image")
            return "ok_no_person_detected", None
        return "ok", outputs[0]
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, this is a smoke test
        failures.append(f"sam3d_inference_error: {exc!r}")
        return f"error: {exc!r}", None


def measure_via_subprocess(
    params_path: str,
    known_height_cm: float | None,
    warnings: list,
    failures: list,
    python_executable: str | None = None,
):
    """Run clad-body's MHR loader + measure() in an isolated subprocess.

    See _mhr_measure_worker.py's docstring for why this is a subprocess and
    not an in-process call: the loader has been observed to segfault the
    whole interpreter (native crash inside pymomentum's FBX loader), and we
    do not want that to take down run.py itself.

    Args:
        python_executable: which Python to run the worker under. Defaults
            to ``sys.executable`` (Task 02's behavior, unchanged). Task 03
            (Colab/GPU) evidence showed clad-body's `pymomentum-cpu`
            dependency needs a torch build ABI-matched to a CPU-only
            install, which conflicts with the CUDA-enabled torch build the
            main SAM 3D Body inference process needs -- so on a real GPU
            environment, pass the python executable of a *separate*,
            CPU-torch venv here instead of leaving this as the default. See
            docs/experiments/TASK03_COLAB_END_TO_END_SMOKE_TEST.md.
    """
    if not _package_available("clad_body") and python_executable is None:
        return "blocked_no_clad_body: `clad_body` is not installed in this environment", None

    python_bin = python_executable or sys.executable

    with tempfile.TemporaryDirectory() as td:
        output_path = os.path.join(td, "result.json")
        cmd = [python_bin, str(HERE / "_mhr_measure_worker.py"), params_path, output_path]
        if known_height_cm is not None:
            cmd.append(str(known_height_cm))

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            failures.append("measurement_extraction_timeout: worker did not finish within 180s")
            return "blocked_timeout", None

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-1500:]
            if proc.returncode < 0:
                status = (
                    f"blocked_native_crash: worker subprocess was killed by signal "
                    f"{-proc.returncode} (likely a native/C++ extension crash, e.g. "
                    f"SIGSEGV=11 -- see TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md for the "
                    f"reproduced pymomentum/MHR loader crash in this environment)"
                )
            else:
                status = f"error: worker exited with code {proc.returncode}"
            failures.append(f"measurement_extraction_stderr_tail: {stderr_tail}")
            return status, None

        with open(output_path) as f:
            result = json.load(f)
        return "ok", result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="Path to a smartphone-style RGB input image")
    parser.add_argument("--sam3d-checkpoint", help="Path to a downloaded SAM 3D Body checkpoint (optional)")
    parser.add_argument(
        "--mhr-params-json",
        help="Path to an existing SAM3D-style MHR params JSON (skips SAM 3D Body inference)",
    )
    parser.add_argument("--known-height-cm", type=float, default=None, help="Optional known customer height in cm")
    parser.add_argument(
        "--clad-python-executable",
        default=None,
        help=(
            "Python executable to run the clad-body/pymomentum measurement worker under "
            "(e.g. a separate CPU-torch venv). Defaults to this process's own interpreter."
        ),
    )
    parser.add_argument("--subject-id", default=None, help="Optional identifier for this subject/run")
    parser.add_argument("--output", required=True, help="Path to write the machine-readable JSON result")
    args = parser.parse_args()

    if not args.image and not args.mhr_params_json:
        parser.error("must provide --image or --mhr-params-json")

    t_start = time.time()
    warnings: list[str] = []
    failures: list[str] = []
    runtime: dict[str, float] = {}

    result = {
        "subject_id": args.subject_id or (Path(args.image).stem if args.image else Path(args.mhr_params_json).stem),
        "source_image": args.image,
        "sam3d_inference_status": None,
        "mhr_params_available": False,
        "mhr_params_source": None,
        "measurement_extraction_status": None,
        "raw_body_height_cm": None,
        "known_height_cm": args.known_height_cm,
        "rescale_applied": False,
        "rescale_factor": None,
        "measurements_cm": None,
        "runtime_seconds": runtime,
        "device": _device_string(),
        "warnings": warnings,
        "failures": failures,
        "python_version": platform.python_version(),
        "note": (
            "A successful measurement_extraction_status only proves clad-body "
            "computed a circumference/length from the given MHR mesh. It does NOT "
            "prove SAM 3D Body reconstructed the photographed person's true body "
            "shape -- see TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md."
        ),
    }

    params_path = None
    tmp_params_file = None

    if args.mhr_params_json:
        result["mhr_params_source"] = "provided_json"
        result["sam3d_inference_status"] = "skipped_params_provided"
        params_path = args.mhr_params_json
        result["mhr_params_available"] = os.path.exists(params_path)
        if not result["mhr_params_available"]:
            failures.append(f"mhr_params_json path does not exist: {params_path}")
    else:
        t0 = time.time()
        status, person_output = run_sam3d_inference(args.image, args.sam3d_checkpoint, warnings, failures)
        runtime["sam3d_inference"] = round(time.time() - t0, 3)
        result["sam3d_inference_status"] = status
        if person_output is not None:
            warn = warn_if_scale_params_would_be_misused(person_output)
            if warn:
                warnings.append(warn)
            try:
                clad_params = sam3d_output_to_clad_params(person_output)
            except AdapterError as exc:
                failures.append(f"adapter_error: {exc}")
                clad_params = None

            if clad_params is not None:
                tmp_params_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix="_sam3d_mhr_params.json", delete=False
                )
                json.dump(clad_params, tmp_params_file)
                tmp_params_file.close()
                params_path = tmp_params_file.name
                result["mhr_params_source"] = "sam3d_inference"
                result["mhr_params_available"] = True

    if params_path and result["mhr_params_available"]:
        t1 = time.time()
        m_status, m_result = measure_via_subprocess(
            params_path, args.known_height_cm, warnings, failures,
            python_executable=args.clad_python_executable,
        )
        runtime["measurement_extraction"] = round(time.time() - t1, 3)
        result["measurement_extraction_status"] = m_status
        if m_result is not None:
            result["raw_body_height_cm"] = m_result["raw_body_height_cm"]
            result["rescale_applied"] = m_result["rescale_applied"]
            result["rescale_factor"] = m_result["rescale_factor"]
            result["measurements_cm"] = m_result["measurements_cm"]
    else:
        result["measurement_extraction_status"] = "skipped_no_mhr_params"

    if tmp_params_file:
        try:
            os.unlink(tmp_params_file.name)
        except OSError:
            pass

    runtime["total"] = round(time.time() - t_start, 3)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
