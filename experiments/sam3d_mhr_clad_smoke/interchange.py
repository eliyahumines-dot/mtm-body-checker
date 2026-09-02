"""Deterministic file-based interchange contract between Environment A
(SAM 3D Body GPU inference) and Environment B (MHR + clad-body measurement
extraction) -- Task 03B.

The two environments are separate Python processes/venvs by design (see
docs/experiments/TASK03B_DEPENDENCY_RESOLUTION.md for why) and must never
share Python-specific state -- no pickle, no torch tensors, no arbitrary
objects crossing the boundary. Only a small, versioned ``.npz`` file of
plain numpy float arrays and unicode-string arrays, loadable with
``numpy.load(path, allow_pickle=False)`` from either side regardless of
which torch build (or whether torch at all) is installed there.

Required fields (the only two clad-body's current loader actually reads,
per Task 02's source-level finding -- see adapter.py's docstring for the
``scale_params`` vs. ``mhr_model_params`` trap this avoids by construction,
since this module builds on top of ``adapter.sam3d_output_to_clad_params``):

- ``shape_params``: float32[45] -- MHR identity coefficients
- ``mhr_model_params``: float32[204] -- decoded pose+scale vector,
  ``[136:204]`` is the scale clad-body's loader actually uses

Metadata fields (always written, small, string-typed):

- ``schema_version``: identifies this exact contract version
- ``source_checkpoint``: which SAM 3D Body checkpoint repo produced this

Optional provenance fields (written only if present in the source SAM 3D
Body output; NOT read by clad-body's current rest-pose loader -- carried
only so a human or a future measurement stage can inspect what SAM 3D Body
actually predicted for pose/camera):

- ``body_pose_params``, ``global_rot``, ``pred_cam_t``, ``focal_length``
"""

from __future__ import annotations

import numpy as np

from adapter import AdapterError, sam3d_output_to_clad_params

SCHEMA_VERSION = "sam3d_mhr_interchange_v1"

REQUIRED_FIELDS = ("shape_params", "mhr_model_params")
OPTIONAL_PROVENANCE_FIELDS = ("body_pose_params", "global_rot", "pred_cam_t", "focal_length")


class InterchangeError(ValueError):
    """Raised for a malformed or version-mismatched interchange file."""


def write_interchange(path: str, person_output: dict, source_checkpoint: str) -> dict:
    """Validate a SAM 3D Body ``process_one_image()`` person dict and write
    it as a deterministic ``.npz`` interchange file.

    Reuses ``adapter.sam3d_output_to_clad_params`` for validation, so
    Environment A fails loudly here (raising ``AdapterError``) rather than
    writing a file Environment B cannot use -- the same length/presence
    checks Task 02 already established, not reimplemented.

    Returns a small dict of {field: shape_or_"scalar"} for logging, so a
    caller doesn't need torch to summarize what was written.
    """
    clad_params = sam3d_output_to_clad_params(person_output)  # raises AdapterError on bad input

    payload = {
        "schema_version": np.array(SCHEMA_VERSION),
        "source_checkpoint": np.array(str(source_checkpoint)),
        "shape_params": np.asarray(clad_params["shape_params"], dtype=np.float32),
        "mhr_model_params": np.asarray(clad_params["mhr_model_params"], dtype=np.float32),
    }
    for field in OPTIONAL_PROVENANCE_FIELDS:
        if field in person_output:
            value = person_output[field]
            if hasattr(value, "tolist"):
                value = value.tolist()
            payload[field] = np.asarray(value, dtype=np.float32)

    np.savez(path, **payload)

    return {k: (list(v.shape) if v.shape else "scalar") for k, v in payload.items()}


def read_interchange(path: str) -> dict:
    """Load a ``.npz`` interchange file into a plain, JSON-serializable dict.

    Raises ``InterchangeError`` (not a generic exception) if the file has
    no/mismatched ``schema_version`` or is missing a required field --
    Environment B should fail loudly and early, before attempting any MHR
    reconstruction, rather than pass a malformed dict deeper into
    clad-body where the failure would be harder to diagnose.
    """
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)

        if "schema_version" not in keys:
            raise InterchangeError(f"{path} has no 'schema_version' field -- not a valid interchange file")
        version = str(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise InterchangeError(
                f"{path} has schema_version={version!r}, expected {SCHEMA_VERSION!r} -- "
                f"Environment A and Environment B are running mismatched interchange code"
            )

        missing = [f for f in REQUIRED_FIELDS if f not in keys]
        if missing:
            raise InterchangeError(f"{path} is missing required field(s): {missing}")

        result = {
            "schema_version": version,
            "source_checkpoint": str(data["source_checkpoint"]) if "source_checkpoint" in keys else None,
            "shape_params": data["shape_params"].tolist(),
            "mhr_model_params": data["mhr_model_params"].tolist(),
        }
        for field in OPTIONAL_PROVENANCE_FIELDS:
            if field in keys:
                result[field] = data[field].tolist()
        return result


def interchange_to_clad_params(record: dict) -> dict:
    """Extract just the two fields clad-body's loader needs from a dict
    already produced by :func:`read_interchange`."""
    return {"shape_params": record["shape_params"], "mhr_model_params": record["mhr_model_params"]}
