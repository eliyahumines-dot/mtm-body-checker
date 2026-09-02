"""Deterministic decision-gate classification for a pipeline run.

Pure logic, no GPU/torch/clad-body import required -- so it can be unit
tested without any heavy dependency, and reused identically by both a CLI
run and the Colab notebook's final cell (Task 03 section 15/18), rather
than reimplementing the same classification inline in notebook prose.

The eleven ``FailureCategory`` values and six ``DecisionGate`` values are
taken verbatim from Task 03's specification. ``classify()`` maps a
:class:`PipelineState` (a plain record of what happened at each stage) to
exactly one gate, plus a short human-readable reason -- so the notebook
never has to eyeball a pile of booleans to decide which of A-F applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailureCategory(str, Enum):
    NO_GPU = "NO_GPU"
    HF_AUTH_FAILURE = "HF_AUTH_FAILURE"
    CHECKPOINT_ACCESS_FAILURE = "CHECKPOINT_ACCESS_FAILURE"
    INSTALL_FAILURE = "INSTALL_FAILURE"
    CUDA_PYTORCH_MISMATCH = "CUDA_PYTORCH_MISMATCH"
    PYMOMENTUM_FAILURE = "PYMOMENTUM_FAILURE"
    SAM3D_INFERENCE_FAILURE = "SAM3D_INFERENCE_FAILURE"
    MHR_SCHEMA_FAILURE = "MHR_SCHEMA_FAILURE"
    MHR_RECONSTRUCTION_FAILURE = "MHR_RECONSTRUCTION_FAILURE"
    CLAD_BODY_FAILURE = "CLAD_BODY_FAILURE"
    METRIC_SCALE_UNKNOWN = "METRIC_SCALE_UNKNOWN"


class DecisionGate(str, Enum):
    END_TO_END_MEASUREMENTS_PRODUCED = "A"
    SAM3D_RUNS_BUT_MEASUREMENT_EXTRACTION_BLOCKED = "B"
    GPU_INSUFFICIENT = "C"
    DEPENDENCY_ENVIRONMENT_BLOCKED = "D"
    CHECKPOINT_ACCESS_BLOCKED = "E"
    PIPELINE_NOT_TECHNICALLY_VIABLE = "F"


# Failure categories that, if present, indicate an environment/dependency
# problem rather than a fundamental pipeline-design problem.
_ENVIRONMENT_FAILURES = {
    FailureCategory.INSTALL_FAILURE,
    FailureCategory.CUDA_PYTORCH_MISMATCH,
    FailureCategory.PYMOMENTUM_FAILURE,
}


@dataclass
class PipelineState:
    """Record of what actually happened at each pipeline stage.

    Each ``Optional[bool]`` is ``None`` if that stage was never reached,
    ``True``/``False`` if it was attempted and its outcome is known. This
    mirrors how a notebook run naturally proceeds: later fields stay
    ``None`` if an earlier stage blocked the run.
    """

    gpu_available: bool = False
    hf_auth_ok: bool | None = None
    checkpoint_downloaded: bool | None = None
    dependencies_installed: bool | None = None  # Task 03B: Environment A (SAM 3D Body) build, specifically
    sam3d_inference_ok: bool | None = None
    mhr_schema_valid: bool | None = None
    mhr_clad_environment_ok: bool | None = None  # Task 03B: Environment B build + bundled-fixture self-test
    mhr_reconstruction_ok: bool | None = None  # Task 03B: real interchange-derived data, not the self-test
    clad_body_measure_ok: bool | None = None  # Task 03B: real interchange-derived data, not the self-test
    measurements: dict | None = None
    failure_categories: list[str] = field(default_factory=list)

    def add_failure(self, category: FailureCategory) -> None:
        if category.value not in self.failure_categories:
            self.failure_categories.append(category.value)


def classify(state: PipelineState) -> tuple[DecisionGate, str]:
    """Return (gate, reason) for the given pipeline state.

    Evaluated in the same order the pipeline itself executes -- the first
    unmet precondition determines the gate. A later stage's success can
    never "undo" an earlier stage's recorded failure.
    """
    if not state.gpu_available:
        return (
            DecisionGate.GPU_INSUFFICIENT,
            "No GPU detected. Per Task 03 section 1, full inference is not "
            "attempted on CPU -- the notebook is designed to stop here "
            "rather than produce a slow or misleading CPU run.",
        )

    if state.hf_auth_ok is False:
        return (
            DecisionGate.CHECKPOINT_ACCESS_BLOCKED,
            "Hugging Face authentication failed (HF_TOKEN missing, invalid, "
            "or lacking approved access to the SAM 3D Body checkpoints).",
        )

    if state.checkpoint_downloaded is False:
        return (
            DecisionGate.CHECKPOINT_ACCESS_BLOCKED,
            "HF auth succeeded but the checkpoint itself could not be "
            "downloaded (access not yet approved for this specific repo, "
            "network failure, or disk space exhausted mid-download).",
        )

    if state.dependencies_installed is False:
        env_failures = [f for f in state.failure_categories if f in {c.value for c in _ENVIRONMENT_FAILURES}]
        detail = f" ({', '.join(env_failures)})" if env_failures else ""
        return (
            DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED,
            f"Dependency installation did not converge to a working GPU-compatible "
            f"environment{detail}.",
        )

    if state.sam3d_inference_ok is False:
        if FailureCategory.SAM3D_INFERENCE_FAILURE.value in state.failure_categories:
            return (
                DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED,
                "SAM 3D Body inference raised an error after loading -- treated as an "
                "environment/dependency issue unless independently confirmed as a "
                "fundamental incompatibility (in which case reclassify as "
                "PIPELINE_NOT_TECHNICALLY_VIABLE manually, with the evidence recorded).",
            )
        return (
            DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED,
            "SAM 3D Body inference did not complete successfully.",
        )

    if state.sam3d_inference_ok is True:
        if state.mhr_schema_valid is False:
            return (
                DecisionGate.SAM3D_RUNS_BUT_MEASUREMENT_EXTRACTION_BLOCKED,
                "SAM 3D Body ran and produced output, but its schema did not match "
                "what the adapter/clad-body expects (missing or wrong-shaped "
                "shape_params/mhr_model_params).",
            )
        if state.mhr_reconstruction_ok is False:
            return (
                DecisionGate.SAM3D_RUNS_BUT_MEASUREMENT_EXTRACTION_BLOCKED,
                "SAM 3D Body ran successfully, but MHR mesh reconstruction "
                "(clad_body.load.load_mhr_from_params) failed.",
            )
        if state.clad_body_measure_ok is False:
            return (
                DecisionGate.SAM3D_RUNS_BUT_MEASUREMENT_EXTRACTION_BLOCKED,
                "SAM 3D Body ran and the MHR mesh reconstructed, but "
                "clad_body.measure.measure() failed.",
            )
        if state.clad_body_measure_ok is True and state.measurements:
            return (
                DecisionGate.END_TO_END_MEASUREMENTS_PRODUCED,
                "SAM 3D Body inference, MHR reconstruction, and clad-body "
                "measurement extraction all completed and produced numeric "
                "measurements from an actual SAM 3D Body inference result.",
            )

    return (
        DecisionGate.PIPELINE_NOT_TECHNICALLY_VIABLE,
        "Pipeline state is inconclusive or incomplete in a way not covered by an "
        "earlier, more specific gate -- treated conservatively rather than "
        "assumed successful.",
    )


def _phase_status(value: bool | None) -> str:
    if value is None:
        return "NOT_ATTEMPTED"
    return "PASS" if value else "FAIL"


def phase_summary(state: PipelineState) -> dict:
    """Task 03B section 11: report the five phases independently
    (SAM3D_ENVIRONMENT, SAM3D_INFERENCE, MHR_CLAD_ENVIRONMENT,
    MHR_CLAD_EXTRACTION, END_TO_END), each as PASS / FAIL / NOT_ATTEMPTED,
    plus the first exact failing boundary -- deliberately not collapsed
    into a single letter grade the way :func:`classify` is, since Task 03B
    asks for these to be visible separately.

    Does not replace :func:`classify`; both read the same
    :class:`PipelineState` and are safe to call together.
    """
    fields = {
        "SAM3D_ENVIRONMENT": _phase_status(state.dependencies_installed),
        "SAM3D_INFERENCE": _phase_status(state.sam3d_inference_ok),
        "MHR_CLAD_ENVIRONMENT": _phase_status(state.mhr_clad_environment_ok),
        "MHR_CLAD_EXTRACTION": _phase_status(state.clad_body_measure_ok),
    }

    end_to_end = (
        fields["SAM3D_ENVIRONMENT"] == "PASS"
        and fields["SAM3D_INFERENCE"] == "PASS"
        and fields["MHR_CLAD_ENVIRONMENT"] == "PASS"
        and fields["MHR_CLAD_EXTRACTION"] == "PASS"
        and bool(state.measurements)
    )
    fields["END_TO_END"] = "PASS" if end_to_end else "FAIL"

    ordered_phases = ["SAM3D_ENVIRONMENT", "SAM3D_INFERENCE", "MHR_CLAD_ENVIRONMENT", "MHR_CLAD_EXTRACTION"]
    fields["first_failing_boundary"] = next((p for p in ordered_phases if fields[p] == "FAIL"), None)

    return fields
