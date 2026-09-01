"""Tests for decision_gate.py -- pure logic, no GPU/torch/clad-body needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from decision_gate import DecisionGate, FailureCategory, PipelineState, classify


def test_no_gpu_is_gpu_insufficient():
    state = PipelineState(gpu_available=False)
    gate, _ = classify(state)
    assert gate == DecisionGate.GPU_INSUFFICIENT


def test_no_gpu_wins_even_if_later_fields_look_successful():
    """gpu_available=False must short-circuit regardless of anything else recorded --
    a stale/inconsistent state must never be read as success."""
    state = PipelineState(
        gpu_available=False,
        hf_auth_ok=True,
        checkpoint_downloaded=True,
        dependencies_installed=True,
        sam3d_inference_ok=True,
        mhr_schema_valid=True,
        mhr_reconstruction_ok=True,
        clad_body_measure_ok=True,
        measurements={"height_cm": 170.0},
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.GPU_INSUFFICIENT


def test_hf_auth_failure_is_checkpoint_access_blocked():
    state = PipelineState(gpu_available=True, hf_auth_ok=False)
    gate, reason = classify(state)
    assert gate == DecisionGate.CHECKPOINT_ACCESS_BLOCKED
    assert "auth" in reason.lower()


def test_checkpoint_download_failure_is_checkpoint_access_blocked():
    state = PipelineState(gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=False)
    gate, _ = classify(state)
    assert gate == DecisionGate.CHECKPOINT_ACCESS_BLOCKED


def test_dependency_install_failure_is_dependency_environment_blocked():
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=False,
    )
    state.add_failure(FailureCategory.CUDA_PYTORCH_MISMATCH)
    gate, reason = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED
    assert "CUDA_PYTORCH_MISMATCH" in reason


def test_sam3d_inference_failure_is_dependency_environment_blocked_by_default():
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_inference_ok=False,
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED


def test_mhr_schema_failure_is_measurement_extraction_blocked():
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_inference_ok=True, mhr_schema_valid=False,
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.SAM3D_RUNS_BUT_MEASUREMENT_EXTRACTION_BLOCKED


def test_mhr_reconstruction_failure_is_measurement_extraction_blocked():
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_inference_ok=True, mhr_schema_valid=True,
        mhr_reconstruction_ok=False,
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.SAM3D_RUNS_BUT_MEASUREMENT_EXTRACTION_BLOCKED


def test_clad_body_measure_failure_is_measurement_extraction_blocked():
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_inference_ok=True, mhr_schema_valid=True,
        mhr_reconstruction_ok=True, clad_body_measure_ok=False,
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.SAM3D_RUNS_BUT_MEASUREMENT_EXTRACTION_BLOCKED


def test_full_success_is_end_to_end_measurements_produced():
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_inference_ok=True, mhr_schema_valid=True,
        mhr_reconstruction_ok=True, clad_body_measure_ok=True,
        measurements={"height_cm": 170.0, "bust_cm": 92.0},
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.END_TO_END_MEASUREMENTS_PRODUCED


def test_success_flags_but_empty_measurements_is_not_gate_a():
    """clad_body_measure_ok=True with no actual measurements dict must never
    be read as END_TO_END_MEASUREMENTS_PRODUCED -- guards against a bug where
    the flag is set true without real numeric output."""
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_inference_ok=True, mhr_schema_valid=True,
        mhr_reconstruction_ok=True, clad_body_measure_ok=True,
        measurements=None,
    )
    gate, _ = classify(state)
    assert gate != DecisionGate.END_TO_END_MEASUREMENTS_PRODUCED


def test_incomplete_state_is_not_technically_viable_not_success():
    """A state with everything left as None (nothing attempted/recorded) must
    not default to success."""
    state = PipelineState(gpu_available=True)
    gate, _ = classify(state)
    assert gate != DecisionGate.END_TO_END_MEASUREMENTS_PRODUCED


def test_add_failure_is_idempotent():
    state = PipelineState()
    state.add_failure(FailureCategory.NO_GPU)
    state.add_failure(FailureCategory.NO_GPU)
    assert state.failure_categories == [FailureCategory.NO_GPU.value]


def test_all_failure_categories_are_distinct_strings():
    values = [c.value for c in FailureCategory]
    assert len(values) == len(set(values)) == 11


def test_all_decision_gates_are_distinct_single_letters():
    values = [g.value for g in DecisionGate]
    assert values == ["A", "B", "C", "D", "E", "F"]
