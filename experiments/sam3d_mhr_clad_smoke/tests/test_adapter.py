"""Tests for adapter.py -- our own SAM3D->clad-body field-mapping code.

Deliberately uses plain lists/synthetic dicts, not real SAM 3D Body or
clad-body objects, so this runs without any heavy ML dependency or
checkpoint download (Task 02 requirement).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from adapter import (
    AdapterError,
    EXPECTED_MHR_MODEL_PARAMS_LEN,
    EXPECTED_SHAPE_PARAMS_LEN,
    sam3d_output_to_clad_params,
    warn_if_scale_params_would_be_misused,
)


def _fake_person_output(**overrides):
    base = {
        "shape_params": [0.1] * EXPECTED_SHAPE_PARAMS_LEN,
        "mhr_model_params": [0.0] * EXPECTED_MHR_MODEL_PARAMS_LEN,
        "scale_params": [0.2] * 28,  # SAM 3D Body's raw 28-dim PCA scale coeffs
        "pred_cam_t": [0.0, 1.0, 1.5],
        "focal_length": 3000.0,
    }
    base.update(overrides)
    return base


def test_happy_path_keeps_only_the_two_required_fields():
    out = sam3d_output_to_clad_params(_fake_person_output())
    assert set(out.keys()) == {"shape_params", "mhr_model_params"}
    assert len(out["shape_params"]) == EXPECTED_SHAPE_PARAMS_LEN
    assert len(out["mhr_model_params"]) == EXPECTED_MHR_MODEL_PARAMS_LEN


def test_drops_scale_params_deliberately():
    """The 28-dim SAM3D scale_params must never end up in clad-body's input --
    see the module docstring for why (dimension/space mismatch trap)."""
    out = sam3d_output_to_clad_params(_fake_person_output())
    assert "scale_params" not in out


def test_drops_pose_camera_hand_face_fields():
    out = sam3d_output_to_clad_params(_fake_person_output())
    for dropped in ("pred_cam_t", "focal_length", "pred_pose_raw", "global_rot"):
        assert dropped not in out


def test_missing_shape_params_raises():
    bad = _fake_person_output()
    del bad["shape_params"]
    with pytest.raises(AdapterError, match="shape_params"):
        sam3d_output_to_clad_params(bad)


def test_missing_mhr_model_params_raises():
    bad = _fake_person_output()
    del bad["mhr_model_params"]
    with pytest.raises(AdapterError, match="mhr_model_params"):
        sam3d_output_to_clad_params(bad)


def test_wrong_length_shape_params_raises():
    bad = _fake_person_output(shape_params=[0.1] * 10)
    with pytest.raises(AdapterError, match="length 10"):
        sam3d_output_to_clad_params(bad)


def test_wrong_length_mhr_model_params_raises():
    bad = _fake_person_output(mhr_model_params=[0.0] * 50)
    with pytest.raises(AdapterError, match="length 50"):
        sam3d_output_to_clad_params(bad)


def test_accepts_numpy_arrays_and_produces_json_safe_lists():
    np = pytest.importorskip("numpy")
    out = sam3d_output_to_clad_params(
        _fake_person_output(
            shape_params=np.zeros(EXPECTED_SHAPE_PARAMS_LEN, dtype="float32"),
            mhr_model_params=np.zeros(EXPECTED_MHR_MODEL_PARAMS_LEN, dtype="float32"),
        )
    )
    assert isinstance(out["shape_params"], list)
    assert isinstance(out["mhr_model_params"], list)
    assert isinstance(out["shape_params"][0], float)


def test_warns_on_raw_28dim_scale_params():
    warning = warn_if_scale_params_would_be_misused(_fake_person_output())
    assert warning is not None
    assert "28" in warning


def test_no_warning_when_scale_params_already_68dim():
    out = _fake_person_output(scale_params=[0.0] * 68)
    assert warn_if_scale_params_would_be_misused(out) is None


def test_no_warning_when_scale_params_absent():
    out = _fake_person_output()
    del out["scale_params"]
    assert warn_if_scale_params_would_be_misused(out) is None
