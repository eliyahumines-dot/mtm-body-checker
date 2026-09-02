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
    assert len(values) == len(set(values)) == 22  # 11 original + 6 Task 03C + 1 Task 03D + 1 Task 03E + 3 Task 03F


def test_task_03d_failure_category_is_present():
    names = {c.name for c in FailureCategory}
    assert "SAM3D_SOURCE_IMPORT_FAILURE" in names


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


# --- Task 03D: SAM3D_SOURCE_IMPORT_FAILURE, distinct from SAM3D_MODEL_LOAD_FAILURE ---

def test_source_import_failure_is_dependency_environment_blocked():
    """Reproduces the exact real Colab failure: environment built fine,
    but `import sam_3d_body` itself failed (ModuleNotFoundError) --
    Environment A never got the chance to attempt load_sam_3d_body()."""
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_source_import_ok=False,
    )
    state.add_failure(FailureCategory.SAM3D_SOURCE_IMPORT_FAILURE)
    gate, reason = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED
    assert "import sam_3d_body" in reason


def test_source_import_failure_takes_precedence_over_unset_model_load():
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_source_import_ok=False,
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED


def test_source_import_success_does_not_block_later_model_load_failure():
    """sam3d_source_import_ok=True must not, by itself, short-circuit anything --
    a subsequent model-load failure should still be independently reachable,
    and classified as SAM3D_MODEL_LOAD_FAILURE territory, not source-import."""
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_source_import_ok=True,
        sam3d_model_load_ok=False,
    )
    gate, reason = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED
    assert "load_sam_3d_body" in reason


def test_source_import_and_model_load_failures_are_distinguishable_categories():
    """The two failure categories must never be conflated -- this is the
    entire point of Task 03D's A vs. B distinction (section 6)."""
    state_a = PipelineState(sam3d_source_import_ok=False)
    state_a.add_failure(FailureCategory.SAM3D_SOURCE_IMPORT_FAILURE)
    state_b = PipelineState(sam3d_source_import_ok=True, sam3d_model_load_ok=False)
    state_b.add_failure(FailureCategory.SAM3D_MODEL_LOAD_FAILURE)
    assert state_a.failure_categories != state_b.failure_categories
    assert FailureCategory.SAM3D_SOURCE_IMPORT_FAILURE.value not in state_b.failure_categories
    assert FailureCategory.SAM3D_MODEL_LOAD_FAILURE.value not in state_a.failure_categories


# --- Task 03E: SAM3D_MODEL_CODE_IMPORT_FAILURE, distinct from both
# SAM3D_SOURCE_IMPORT_FAILURE and SAM3D_MODEL_LOAD_FAILURE ---

def test_task_03e_failure_category_is_present():
    names = {c.name for c in FailureCategory}
    assert "SAM3D_MODEL_CODE_IMPORT_FAILURE" in names


def test_model_code_import_failure_is_dependency_environment_blocked():
    """The bare `sam_3d_body` package imports fine, but importing its
    model-construction submodules fails -- must never fall through to
    SAM3D_MODEL_LOAD_FAILURE territory, since load_sam_3d_body() itself
    never got the chance to even be imported."""
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_source_import_ok=True,
        sam3d_model_code_import_ok=False,
    )
    state.add_failure(FailureCategory.SAM3D_MODEL_CODE_IMPORT_FAILURE)
    gate, reason = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED
    assert "model-construction code" in reason


def test_model_code_import_failure_takes_precedence_over_unset_model_load():
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_source_import_ok=True,
        sam3d_model_code_import_ok=False,
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED


def test_model_code_import_success_does_not_block_later_model_load_failure():
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_source_import_ok=True,
        sam3d_model_code_import_ok=True, sam3d_model_load_ok=False,
    )
    gate, reason = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED
    assert "load_sam_3d_body" in reason


def test_source_import_model_code_import_and_model_load_failures_are_distinguishable():
    """The three failure categories must never be conflated."""
    state_a = PipelineState(sam3d_source_import_ok=False)
    state_a.add_failure(FailureCategory.SAM3D_SOURCE_IMPORT_FAILURE)
    state_b = PipelineState(sam3d_source_import_ok=True, sam3d_model_code_import_ok=False)
    state_b.add_failure(FailureCategory.SAM3D_MODEL_CODE_IMPORT_FAILURE)
    state_c = PipelineState(
        sam3d_source_import_ok=True, sam3d_model_code_import_ok=True, sam3d_model_load_ok=False,
    )
    state_c.add_failure(FailureCategory.SAM3D_MODEL_LOAD_FAILURE)
    all_categories = [state_a.failure_categories, state_b.failure_categories, state_c.failure_categories]
    for i, cats_i in enumerate(all_categories):
        for j, cats_j in enumerate(all_categories):
            if i != j:
                assert not set(cats_i) & set(cats_j)


def test_phase_summary_matches_task03e_model_code_import_boundary():
    """Reproduces a hypothetical (but structurally identical to the real
    Task 03D Colab report) scenario: source import passes, but the
    model-construction submodules fail to import -- must be distinguishable
    from both SAM3D_SOURCE_IMPORT and SAM3D_MODEL_LOAD."""
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_source_import_ok=True,
        sam3d_model_code_import_ok=False,
    )
    fields = phase_summary(state)
    assert fields["SAM3D_CORE_ENVIRONMENT"] == "PASS"
    assert fields["SAM3D_SOURCE_IMPORT"] == "PASS"
    assert fields["SAM3D_MODEL_CODE_IMPORT"] == "FAIL"
    assert fields["SAM3D_MODEL_LOAD"] == "NOT_ATTEMPTED"
    assert fields["first_failing_boundary"] == "SAM3D_MODEL_CODE_IMPORT"


def test_phase_summary_source_import_failure_leaves_model_code_import_not_attempted():
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_source_import_ok=False,
    )
    fields = phase_summary(state)
    assert fields["SAM3D_SOURCE_IMPORT"] == "FAIL"
    assert fields["SAM3D_MODEL_CODE_IMPORT"] == "NOT_ATTEMPTED"
    assert fields["SAM3D_MODEL_LOAD"] == "NOT_ATTEMPTED"
    assert fields["first_failing_boundary"] == "SAM3D_SOURCE_IMPORT"


# --- Task 03F: SAM3D_SOURCE_ROOT_VALIDATION_FAILURE, distinct from SAM3D_SOURCE_IMPORT_FAILURE,
# and the finer SAM3D_MATPLOTLIB_BACKEND_FAILURE / SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE ---

def test_task_03f_failure_categories_are_present():
    names = {c.name for c in FailureCategory}
    assert {
        "SAM3D_SOURCE_ROOT_VALIDATION_FAILURE",
        "SAM3D_MATPLOTLIB_BACKEND_FAILURE",
        "SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE",
    }.issubset(names)


def test_source_root_validation_failure_is_dependency_environment_blocked():
    """Reproduces a root that simply doesn't exist/isn't a valid checkout --
    must be caught before `import sam_3d_body` is even attempted."""
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_source_root_validation_ok=False,
    )
    state.add_failure(FailureCategory.SAM3D_SOURCE_ROOT_VALIDATION_FAILURE)
    gate, reason = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED
    assert "source root" in reason.lower()


def test_source_root_validation_failure_takes_precedence_over_unset_source_import():
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_source_root_validation_ok=False,
    )
    gate, _ = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED


def test_source_root_validation_success_does_not_block_later_source_import_failure():
    """Reproduces the exact real Colab scenario this task fixes: the root
    validates fine, but `import sam_3d_body` itself still fails (the
    inherited-Matplotlib-backend case) -- must still be independently
    reachable and not misreported via the root-validation reason text."""
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_source_root_validation_ok=True,
        sam3d_source_import_ok=False,
    )
    gate, reason = classify(state)
    assert gate == DecisionGate.DEPENDENCY_ENVIRONMENT_BLOCKED
    assert "source root itself is invalid" not in reason  # must not misreport once root DID validate
    assert "import sam_3d_body" in reason


def test_source_root_validation_and_source_import_failures_are_distinguishable():
    state_a = PipelineState(sam3d_source_root_validation_ok=False)
    state_a.add_failure(FailureCategory.SAM3D_SOURCE_ROOT_VALIDATION_FAILURE)
    state_b = PipelineState(sam3d_source_root_validation_ok=True, sam3d_source_import_ok=False)
    state_b.add_failure(FailureCategory.SAM3D_MATPLOTLIB_BACKEND_FAILURE)
    assert not set(state_a.failure_categories) & set(state_b.failure_categories)


def test_phase_summary_matches_task03f_source_root_validation_boundary():
    """Reproduces the real Colab failure this task fixes: SAM3D_CORE_ENVIRONMENT
    PASS, SAM3D_SOURCE_ROOT_VALIDATION never even attempted before an earlier
    stage blocks -- and the mirror case, where root validation PASSES but
    SAM3D_SOURCE_IMPORT itself is where it actually fails (the inherited
    Matplotlib backend case)."""
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_source_root_validation_ok=True,
        sam3d_source_import_ok=False,
    )
    fields = phase_summary(state)
    assert fields["SAM3D_CORE_ENVIRONMENT"] == "PASS"
    assert fields["SAM3D_SOURCE_ROOT_VALIDATION"] == "PASS"
    assert fields["SAM3D_SOURCE_IMPORT"] == "FAIL"
    assert fields["SAM3D_MODEL_CODE_IMPORT"] == "NOT_ATTEMPTED"
    assert fields["first_failing_boundary"] == "SAM3D_SOURCE_IMPORT"


def test_phase_summary_source_root_validation_failure_leaves_source_import_not_attempted():
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_source_root_validation_ok=False,
    )
    fields = phase_summary(state)
    assert fields["SAM3D_SOURCE_ROOT_VALIDATION"] == "FAIL"
    assert fields["SAM3D_SOURCE_IMPORT"] == "NOT_ATTEMPTED"
    assert fields["SAM3D_MODEL_CODE_IMPORT"] == "NOT_ATTEMPTED"
    assert fields["first_failing_boundary"] == "SAM3D_SOURCE_ROOT_VALIDATION"


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
        gpu_available=True, dependencies_installed=True, sam3d_source_import_ok=True,
        sam3d_model_load_ok=True, sam3d_inference_ok=True, mhr_schema_valid=True,
        mhr_clad_environment_ok=False, clad_body_measure_ok=None,
    )
    fields = phase_summary(state)
    assert fields["first_failing_boundary"] == "MHR_CLAD_ENVIRONMENT"


def test_phase_summary_full_success_is_end_to_end_pass():
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_source_root_validation_ok=True,
        sam3d_source_import_ok=True,
        sam3d_model_code_import_ok=True, sam3d_model_load_ok=True,
        sam3d_inference_ok=True, mhr_schema_valid=True, mhr_clad_environment_ok=True,
        mhr_reconstruction_ok=True, clad_body_measure_ok=True, measurements={"height_cm": 170.0},
    )
    fields = phase_summary(state)
    assert fields["PHASE_A_SUCCESSFUL"] == "PASS"
    assert fields["END_TO_END"] == "PASS"
    assert fields["first_failing_boundary"] is None


def test_phase_summary_end_to_end_fails_if_measurements_empty_despite_flags_true():
    state = PipelineState(
        gpu_available=True, dependencies_installed=True, sam3d_source_root_validation_ok=True,
        sam3d_source_import_ok=True,
        sam3d_model_code_import_ok=True, sam3d_model_load_ok=True,
        sam3d_inference_ok=True, mhr_schema_valid=True,
        mhr_clad_environment_ok=True, clad_body_measure_ok=True, measurements=None,
    )
    fields = phase_summary(state)
    assert fields["PHASE_A_SUCCESSFUL"] == "PASS"  # Phase A itself was fine
    assert fields["END_TO_END"] == "FAIL"  # but no real measurements exist


def test_phase_summary_returns_exactly_twelve_keys():
    fields = phase_summary(PipelineState())
    assert set(fields.keys()) == {
        "SAM3D_CORE_ENVIRONMENT", "SAM3D_SOURCE_ROOT_VALIDATION", "SAM3D_SOURCE_IMPORT",
        "SAM3D_MODEL_CODE_IMPORT", "SAM3D_MODEL_LOAD", "SAM3D_CORE_INFERENCE",
        "MHR_PARAMS_SERIALIZED", "MHR_CLAD_ENVIRONMENT",
        "MHR_CLAD_EXTRACTION", "PHASE_A_SUCCESSFUL", "END_TO_END", "first_failing_boundary",
    }


def test_phase_summary_matches_task03d_source_import_boundary():
    """Reproduces the exact real Colab result: environment PASS, but
    SAM3D_SOURCE_IMPORT (not model load) is where it actually failed."""
    state = PipelineState(
        gpu_available=True, hf_auth_ok=True, checkpoint_downloaded=True,
        dependencies_installed=True, sam3d_source_import_ok=False,
    )
    fields = phase_summary(state)
    assert fields["SAM3D_CORE_ENVIRONMENT"] == "PASS"
    assert fields["SAM3D_SOURCE_IMPORT"] == "FAIL"
    assert fields["SAM3D_MODEL_LOAD"] == "NOT_ATTEMPTED"
    assert fields["first_failing_boundary"] == "SAM3D_SOURCE_IMPORT"
