"""Tests for interchange.py -- the Environment A <-> Environment B file
contract. Uses only numpy + synthetic dicts; no torch, GPU, clad-body, or
checkpoint required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from adapter import AdapterError, EXPECTED_MHR_MODEL_PARAMS_LEN, EXPECTED_SHAPE_PARAMS_LEN
from interchange import (
    SCHEMA_VERSION,
    InterchangeError,
    interchange_to_clad_params,
    read_interchange,
    write_interchange,
)


def _fake_person_output(**overrides):
    base = {
        "shape_params": [0.1] * EXPECTED_SHAPE_PARAMS_LEN,
        "mhr_model_params": [0.0] * EXPECTED_MHR_MODEL_PARAMS_LEN,
        "scale_params": [0.2] * 28,
        "pred_cam_t": [0.0, 1.0, 1.5],
        "focal_length": 3000.0,
        "global_rot": [0.1, 0.2, 0.3],
        "body_pose_params": [0.0] * 133,
    }
    base.update(overrides)
    return base


def test_round_trip_preserves_required_fields(tmp_path):
    path = str(tmp_path / "interchange.npz")
    write_interchange(path, _fake_person_output(), source_checkpoint="facebook/sam-3d-body-dinov3")
    record = read_interchange(path)
    assert len(record["shape_params"]) == EXPECTED_SHAPE_PARAMS_LEN
    assert len(record["mhr_model_params"]) == EXPECTED_MHR_MODEL_PARAMS_LEN
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["source_checkpoint"] == "facebook/sam-3d-body-dinov3"


def test_scale_params_never_written_to_interchange_file(tmp_path):
    """The 28-dim raw scale_params must never appear in the interchange file
    -- adapter.py already excludes it, and this must hold through write+read."""
    path = str(tmp_path / "interchange.npz")
    write_interchange(path, _fake_person_output(), source_checkpoint="x")
    with np.load(path) as raw:
        assert "scale_params" not in raw.files
    record = read_interchange(path)
    assert "scale_params" not in record


def test_optional_provenance_fields_round_trip(tmp_path):
    path = str(tmp_path / "interchange.npz")
    write_interchange(path, _fake_person_output(), source_checkpoint="x")
    record = read_interchange(path)
    assert len(record["pred_cam_t"]) == 3
    assert len(record["global_rot"]) == 3
    assert len(record["body_pose_params"]) == 133
    assert record["focal_length"] == pytest.approx(3000.0)


def test_optional_provenance_fields_absent_when_not_in_source(tmp_path):
    out = _fake_person_output()
    for f in ("pred_cam_t", "focal_length", "global_rot", "body_pose_params"):
        del out[f]
    path = str(tmp_path / "interchange.npz")
    write_interchange(path, out, source_checkpoint="x")
    record = read_interchange(path)
    for f in ("pred_cam_t", "focal_length", "global_rot", "body_pose_params"):
        assert f not in record


def test_write_raises_adapter_error_on_missing_required_field(tmp_path):
    bad = _fake_person_output()
    del bad["shape_params"]
    path = str(tmp_path / "interchange.npz")
    with pytest.raises(AdapterError):
        write_interchange(path, bad, source_checkpoint="x")


def test_write_raises_adapter_error_on_wrong_length(tmp_path):
    bad = _fake_person_output(mhr_model_params=[0.0] * 10)
    path = str(tmp_path / "interchange.npz")
    with pytest.raises(AdapterError):
        write_interchange(path, bad, source_checkpoint="x")


def test_read_raises_on_missing_schema_version(tmp_path):
    path = str(tmp_path / "bad.npz")
    np.savez(path, shape_params=np.zeros(45), mhr_model_params=np.zeros(204))
    with pytest.raises(InterchangeError, match="schema_version"):
        read_interchange(path)


def test_read_raises_on_mismatched_schema_version(tmp_path):
    path = str(tmp_path / "bad.npz")
    np.savez(
        path,
        schema_version=np.array("some_other_version"),
        shape_params=np.zeros(45),
        mhr_model_params=np.zeros(204),
    )
    with pytest.raises(InterchangeError, match="schema_version"):
        read_interchange(path)


def test_read_raises_on_missing_required_field(tmp_path):
    path = str(tmp_path / "bad.npz")
    np.savez(path, schema_version=np.array(SCHEMA_VERSION), shape_params=np.zeros(45))
    with pytest.raises(InterchangeError, match="mhr_model_params"):
        read_interchange(path)


def test_written_file_loads_without_allow_pickle(tmp_path):
    """The whole point of using npz for the contract is that it never needs
    allow_pickle=True -- assert this stays true."""
    path = str(tmp_path / "interchange.npz")
    write_interchange(path, _fake_person_output(), source_checkpoint="x")
    with np.load(path, allow_pickle=False) as data:
        assert "shape_params" in data.files  # would have raised already if pickle were required


def test_interchange_to_clad_params_extracts_only_the_two_required_keys(tmp_path):
    path = str(tmp_path / "interchange.npz")
    write_interchange(path, _fake_person_output(), source_checkpoint="x")
    record = read_interchange(path)
    clad_params = interchange_to_clad_params(record)
    assert set(clad_params.keys()) == {"shape_params", "mhr_model_params"}


def test_write_summary_reports_shapes(tmp_path):
    path = str(tmp_path / "interchange.npz")
    summary = write_interchange(path, _fake_person_output(), source_checkpoint="x")
    assert summary["shape_params"] == [EXPECTED_SHAPE_PARAMS_LEN]
    assert summary["mhr_model_params"] == [EXPECTED_MHR_MODEL_PARAMS_LEN]
