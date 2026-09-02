"""Deterministic decision-gate classification for a pipeline run.

Pure logic, no GPU/torch/clad-body import required -- so it can be unit
tested without any heavy dependency, and reused identically by both a CLI
run and the Colab notebook's final cell (Task 03 section 15/18), rather
than reimplementing the same classification inline in notebook prose.

The original eleven ``FailureCategory`` values and six ``DecisionGate``
values are taken verbatim from Task 03's specification; Task 03C added six
more failure categories (``WRONG_DEPENDENCY_CHUMPY``,
``TORCH_VERSION_DRIFT``, ``TORCHVISION_VERSION_DRIFT``,
``SAM3D_CORE_DEPENDENCY_FAILURE``, ``SAM3D_MODEL_LOAD_FAILURE``,
``SAM3D_CORE_INFERENCE_FAILURE``) for more precise Phase A diagnostics,
after a real Colab run reported "Recorded failure categories: none"
despite two observed build failures -- every install command's failure
now gets a specific category (see ``install_log.py``), never silently
just a `False` boolean. Task 03D adds one more,
``SAM3D_SOURCE_IMPORT_FAILURE``, after a real Colab run with a working
Environment A still failed with ``ModuleNotFoundError: No module named
'sam_3d_body'`` -- the upstream repo root was never on the worker
interpreter's module search path. This is now a distinct, earlier stage
from ``SAM3D_MODEL_LOAD_FAILURE`` (Task 03D section 6: "A. Python cannot
import sam_3d_body" vs. "B. load_sam_3d_body() imports successfully but
checkpoint/model construction fails" -- only B is
``SAM3D_MODEL_LOAD_FAILURE``). ``classify()`` maps a :class:`PipelineState`
(a plain record of what happened at each stage) to exactly one gate, plus
a short human-readable reason -- so the notebook never has to eyeball a
pile of booleans to decide which of A-F applies.
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

    # Task 03C additions -- more precise Phase A diagnostics.
    WRONG_DEPENDENCY_CHUMPY = "WRONG_DEPENDENCY_CHUMPY"
    TORCH_VERSION_DRIFT = "TORCH_VERSION_DRIFT"
    TORCHVISION_VERSION_DRIFT = "TORCHVISION_VERSION_DRIFT"
    SAM3D_CORE_DEPENDENCY_FAILURE = "SAM3D_CORE_DEPENDENCY_FAILURE"
    SAM3D_MODEL_LOAD_FAILURE = "SAM3D_MODEL_LOAD_FAILURE"
    SAM3D_CORE_INFERENCE_FAILURE = "SAM3D_CORE_INFERENCE_FAILURE"

    # Task 03D addition -- distinct from SAM3D_MODEL_LOAD_FAILURE (section 6):
    # this is Python failing to even import sam_3d_body (wrong/missing repo
    # root on sys.path), not load_sam_3d_body() failing after a successful import.
    SAM3D_SOURCE_IMPORT_FAILURE = "SAM3D_SOURCE_IMPORT_FAILURE"


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
    FailureCategory.WRONG_DEPENDENCY_CHUMPY,
    FailureCategory.TORCH_VERSION_DRIFT,
    FailureCategory.TORCHVISION_VERSION_DRIFT,
    FailureCategory.SAM3D_CORE_DEPENDENCY_FAILURE,
    FailureCategory.SAM3D_SOURCE_IMPORT_FAILURE,
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
    sam3d_source_import_ok: bool | None = None  # Task 03D: `import sam_3d_body` itself, distinct from model load
    sam3d_model_load_ok: bool | None = None  # Task 03C: load_sam_3d_body(), distinct from inference itself
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

    if state.sam3d_source_import_ok is False:
        return (
            DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED,
            "SAM 3D Body's environment built, but `import sam_3d_body` itself failed "
            "(the upstream repo root was not on the worker interpreter's module search "
            "path, or resolved to an unrelated package) -- this precedes and is distinct "
            "from load_sam_3d_body() failing after a successful import.",
        )

    if state.sam3d_model_load_ok is False:
        return (
            DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED,
            "SAM 3D Body's environment built, but load_sam_3d_body() itself failed "
            "(checkpoint/asset path mismatch, or a version incompatibility not caught "
            "by the torch/torchvision pin check) -- this is distinct from an inference-"
            "time failure, which never got the chance to run.",
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
    """Report every phase independently, each as PASS / FAIL /
    NOT_ATTEMPTED, plus the first exact failing boundary and a derived
    ``PHASE_A_SUCCESSFUL`` flag -- deliberately not collapsed into a
    single letter grade the way :func:`classify` is.

    Task 03C section 7 split what Task 03B called ``SAM3D_ENVIRONMENT``/
    ``SAM3D_INFERENCE`` into four Phase A boundaries; Task 03D inserts a
    fifth, ``SAM3D_SOURCE_IMPORT``, between environment and model load --
    a real Colab run showed environment build (torch/CUDA/GPU all working)
    can succeed while `import sam_3d_body` itself still fails (wrong repo
    root on the module search path), which is a distinct, earlier failure
    than `load_sam_3d_body()` raising after a successful import. Each of
    the five Phase A boundaries (``SAM3D_CORE_ENVIRONMENT``,
    ``SAM3D_SOURCE_IMPORT``, ``SAM3D_MODEL_LOAD``, ``SAM3D_CORE_INFERENCE``,
    ``MHR_PARAMS_SERIALIZED``) is independently reportable. Phase B's two
    fields (``MHR_CLAD_ENVIRONMENT``, ``MHR_CLAD_EXTRACTION``) are
    unchanged from Task 03B.

    Does not replace :func:`classify`; both read the same
    :class:`PipelineState` and are safe to call together.
    """
    fields = {
        "SAM3D_CORE_ENVIRONMENT": _phase_status(state.dependencies_installed),
        "SAM3D_SOURCE_IMPORT": _phase_status(state.sam3d_source_import_ok),
        "SAM3D_MODEL_LOAD": _phase_status(state.sam3d_model_load_ok),
        "SAM3D_CORE_INFERENCE": _phase_status(state.sam3d_inference_ok),
        "MHR_PARAMS_SERIALIZED": _phase_status(state.mhr_schema_valid),
        "MHR_CLAD_ENVIRONMENT": _phase_status(state.mhr_clad_environment_ok),
        "MHR_CLAD_EXTRACTION": _phase_status(state.clad_body_measure_ok),
    }

    phase_a_fields = [
        "SAM3D_CORE_ENVIRONMENT", "SAM3D_SOURCE_IMPORT", "SAM3D_MODEL_LOAD",
        "SAM3D_CORE_INFERENCE", "MHR_PARAMS_SERIALIZED",
    ]
    fields["PHASE_A_SUCCESSFUL"] = "PASS" if all(fields[p] == "PASS" for p in phase_a_fields) else "FAIL"

    end_to_end = (
        fields["PHASE_A_SUCCESSFUL"] == "PASS"
        and fields["MHR_CLAD_ENVIRONMENT"] == "PASS"
        and fields["MHR_CLAD_EXTRACTION"] == "PASS"
        and bool(state.measurements)
    )
    fields["END_TO_END"] = "PASS" if end_to_end else "FAIL"

    ordered_phases = phase_a_fields + ["MHR_CLAD_ENVIRONMENT", "MHR_CLAD_EXTRACTION"]
    fields["first_failing_boundary"] = next((p for p in ordered_phases if fields[p] == "FAIL"), None)

    return fields
