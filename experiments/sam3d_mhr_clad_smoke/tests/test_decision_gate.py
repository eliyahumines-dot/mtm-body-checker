"""Tests for decision_gate.py -- pure logic, no GPU/torch/clad-body needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from decision_gate import DecisionGate, FailureCategory, PipelineState, classify, phase_summary


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
    assert len(values) == len(set(values)) == 17  # 11 original + 6 Task 03C additions


def test_task_03c_failure_categories_are_present():
    names = {c.name for c in FailureCategory}
    assert {
        "WRONG_DEPENDENCY_CHUMPY", "TORCH_VERSION_DRIFT", "TORCHVISION_VERSION_DRIFT",
        "SAM3D_CORE_DEPENDENCY_FAILURE", "SAM3D_MODEL_LOAD_FAILURE", "SAM3D_CORE_INFERENCE_FAILURE",
    }.issubset(names)


def test_model_load_failure_is_dependency_environment_blocked():
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_model_load_ok=False,
    )
    state.add_failure(FailureCategory.SAM3D_MODEL_LOAD_FAILURE)
    gate, reason = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED
    assert "load_sam_3d_body" in reason


def test_model_load_failure_takes_precedence_over_unset_inference():
    """sam3d_model_load_ok=False must be caught before falling through to the
    (unset) sam3d_inference_ok check -- model load never got the chance to
    let inference run at all."""
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_model_load_ok=False,
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED


def test_model_load_success_does_not_block_later_classification():
    """sam3d_model_load_ok=True must not, by itself, short-circuit anything --
    a subsequent inference failure should still be reachable."""
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_model_load_ok=True,
        sam3d_inference_ok=False,
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED


def test_all_decision_gates_are_distinct_single_letters():
    values = [g.value for g in DecisionGate]
    assert values == ["A", "B", "C", "D", "E", "F"]


# --- Task 03B/03C: phase_summary() ---

def test_phase_summary_all_not_attempted_on_fresh_state():
    fields = phase_summary(PipelineState())
    assert fields["SAM3D_CORE_ENVIRONMENT"] == "NOT_ATTEMPTED"
    assert fields["SAM3D_MODEL_LOAD"] == "NOT_ATTEMPTED"
    assert fields["SAM3D_CORE_INFERENCE"] == "NOT_ATTEMPTED"
    assert fields["MHR_PARAMS_SERIALIZED"] == "NOT_ATTEMPTED"
    assert fields["MHR_CLAD_ENVIRONMENT"] == "NOT_ATTEMPTED"
    assert fields["MHR_CLAD_EXTRACTION"] == "NOT_ATTEMPTED"
    assert fields["PHASE_A_SUCCESSFUL"] == "FAIL"
    assert fields["END_TO_END"] == "FAIL"
    assert fields["first_failing_boundary"] is None  # nothing FAILED, just not attempted


def test_phase_summary_matches_observed_task03b_state():
    """Reproduces the exact state reported from the human's real Colab run
    (Task 03B): gpu/hf/checkpoint all fine, dependencies_installed False
    (Environment A build itself failed), everything downstream never attempted."""
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=False,
    )
    fields = phase_summary(state)
    assert fields["SAM3D_CORE_ENVIRONMENT"] == "FAIL"
    assert fields["SAM3D_MODEL_LOAD"] == "NOT_ATTEMPTED"
    assert fields["SAM3D_CORE_INFERENCE"] == "NOT_ATTEMPTED"
    assert fields["MHR_CLAD_ENVIRONMENT"] == "NOT_ATTEMPTED"
    assert fields["PHASE_A_SUCCESSFUL"] == "FAIL"
    assert fields["END_TO_END"] == "FAIL"
    assert fields["first_failing_boundary"] == "SAM3D_CORE_ENVIRONMENT"


def test_phase_summary_matches_task03c_model_load_boundary():
    """Task 03C's own scenario: environment builds fine this time, but
    model load is where it fails -- must be distinguishable from an
    environment-build failure."""
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_model_load_ok=False,
    )
    fields = phase_summary(state)
    assert fields["SAM3D_CORE_ENVIRONMENT"] == "PASS"
    assert fields["SAM3D_MODEL_LOAD"] == "FAIL"
    assert fields["SAM3D_CORE_INFERENCE"] == "NOT_ATTEMPTED"
    assert fields["first_failing_boundary"] == "SAM3D_MODEL_LOAD"


def test_phase_summary_first_failing_boundary_is_earliest_not_first_recorded():
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_model_load_ok=True,
        sam3d_inference_ok=True, mhr_schema_valid=True,
        mhr_clad_environment_ok=False, clad_body_measure_ok=None,
    )
    fields = phase_summary(state)
    assert fields["first_failing_boundary"] == "MHR_CLAD_ENVIRONMENT"


def test_phase_summary_full_success_is_end_to_end_pass():
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_model_load_ok=True, sam3d_inference_ok=True,
        mhr_schema_valid=True, mhr_clad_environment_ok=True, mhr_reconstruction_ok=True,
        clad_body_measure_ok=True, measurements={"height_cm": 170.0},
    )
    fields = phase_summary(state)
    assert fields["PHASE_A_SUCCESSFUL"] == "PASS"
    assert fields["END_TO_END"] == "PASS"
    assert fields["first_failing_boundary"] is None


def test_phase_summary_end_to_end_fails_if_measurements_empty_despite_flags_true():
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_model_load_ok=True,
        sam3d_inference_ok=True, mhr_schema_valid=True,
        mhr_clad_environment_ok=True, clad_body_measure_ok=True, measurements=None,
    )
    fields = phase_summary(state)
    assert fields["PHASE_A_SUCCESSFUL"] == "PASS"  # Phase A itself was fine
    assert fields["END_TO_END"] == "FAIL"  # but no real measurements exist


def test_phase_summary_returns_exactly_nine_keys():
    fields = phase_summary(PipelineState())
    assert set(fields.keys()) == {
        "SAM3D_CORE_ENVIRONMENT", "SAM3D_MODEL_LOAD", "SAM3D_CORE_INFERENCE",
        "MHR_PARAMS_SERIALIZED", "MHR_CLAD_ENVIRONMENT", "MHR_CLAD_EXTRACTION",
        "PHASE_A_SUCCESSFUL", "END_TO_END", "first_failing_boundary",
    }
