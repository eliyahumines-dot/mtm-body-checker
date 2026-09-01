"""SAM 3D Body -> clad-body MHR params adapter.

This is our own glue code, not a reimplementation of either upstream
project. It exists because the two upstream field names/shapes do not line
up cleanly (see docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md for
the full investigation):

- SAM 3D Body's ``SAM3DBodyEstimator.process_one_image()`` returns a dict
  per detected person containing, among other things, a ``scale_params``
  field that is 28-dim PCA scale *coefficients* (``num_scale_comps = 28``
  in ``mhr_head.py``), and a ``mhr_model_params`` field that is the full
  204-dim decoded pose+scale vector (``[pose(136) | scale(68)]``), where
  the scale segment is already ``scale_mean + scale_params @ scale_comps``.
- clad-body's ``load_mhr_from_params`` reads either ``mhr_model_params``
  (and uses its ``[136:204]`` slice directly as the 68-dim decoded scale)
  or a ``scale_params`` key that it *assumes* is already the 68-dim
  decoded vector, not 28-dim PCA coefficients.

Passing SAM 3D Body's own ``scale_params`` straight through would silently
under-fill clad-body's expected 68-dim slot with 28 wrong-space values and
zero-pad the rest -- no shape/dimension error is raised by clad-body's
loader, so this would corrupt every measurement without any visible
failure. The fix is to always use ``mhr_model_params`` and to never forward
SAM 3D Body's ``scale_params`` field to clad-body.

clad-body's rest-pose loader (``load_mhr_from_params``) also does not
consume pose, camera, hand, face, or mask fields at all -- it always
reconstructs a zeroed rest-pose mesh from ``shape_params`` and the scale
slice of ``mhr_model_params``. So this adapter only forwards those two
fields; everything else in SAM 3D Body's output is dropped deliberately,
not by oversight.
"""

from __future__ import annotations

from typing import Any, Mapping


REQUIRED_SAM3D_FIELDS = ("shape_params", "mhr_model_params")

EXPECTED_SHAPE_PARAMS_LEN = 45
EXPECTED_MHR_MODEL_PARAMS_LEN = 204
EXPECTED_SCALE_SLICE_LEN = 68
SCALE_SLICE_START = 136


class AdapterError(ValueError):
    """Raised when a SAM 3D Body output dict cannot be converted."""


def _to_list(value: Any) -> list:
    """Convert a numpy array / torch tensor / list to a plain JSON-safe list."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def sam3d_output_to_clad_params(person_output: Mapping[str, Any]) -> dict:
    """Convert one person's SAM 3D Body ``process_one_image`` output dict into
    the params dict clad-body's ``load_mhr_from_params`` / ``load_mhr_from_params_dict``
    expects.

    Args:
        person_output: one element of the list returned by
            ``SAM3DBodyEstimator.process_one_image(image_path)`` -- a dict
            with at least ``shape_params`` (45-dim) and ``mhr_model_params``
            (204-dim) entries (numpy arrays, torch tensors, or lists).

    Returns:
        A plain-Python, JSON-serializable dict with exactly the two keys
        clad-body's MHR loader reads: ``shape_params`` and
        ``mhr_model_params``. Deliberately excludes ``scale_params`` (see
        module docstring -- it is the wrong representation) and all
        pose/camera/hand/face/mask fields, which clad-body's rest-pose
        loader ignores.

    Raises:
        AdapterError: if a required field is missing or has an unexpected
            length. We check shape explicitly because clad-body's loader
            does not -- it will silently zero-pad/truncate a
            wrong-length vector instead of raising.
    """
    missing = [f for f in REQUIRED_SAM3D_FIELDS if f not in person_output]
    if missing:
        raise AdapterError(
            f"SAM 3D Body output is missing required field(s): {missing}"
        )

    shape_params = _to_list(person_output["shape_params"])
    mhr_model_params = _to_list(person_output["mhr_model_params"])

    if len(shape_params) != EXPECTED_SHAPE_PARAMS_LEN:
        raise AdapterError(
            f"shape_params has length {len(shape_params)}, "
            f"expected {EXPECTED_SHAPE_PARAMS_LEN}"
        )
    if len(mhr_model_params) != EXPECTED_MHR_MODEL_PARAMS_LEN:
        raise AdapterError(
            f"mhr_model_params has length {len(mhr_model_params)}, "
            f"expected {EXPECTED_MHR_MODEL_PARAMS_LEN}"
        )

    return {
        "shape_params": shape_params,
        "mhr_model_params": mhr_model_params,
    }


def warn_if_scale_params_would_be_misused(person_output: Mapping[str, Any]) -> str | None:
    """Return a human-readable warning if the caller's dict has a
    ``scale_params`` field shaped like raw SAM 3D Body PCA output (28-dim)
    rather than an already-decoded 68-dim vector.

    This does not raise -- it exists so ``run.py`` can surface the trap
    described in the module docstring as an explicit warning in the
    machine-readable output, rather than a silent corruption or a crash.
    """
    if "scale_params" not in person_output:
        return None
    n = len(_to_list(person_output["scale_params"]))
    if n == EXPECTED_SCALE_SLICE_LEN:
        return None
    return (
        f"person_output['scale_params'] has length {n}, not "
        f"{EXPECTED_SCALE_SLICE_LEN}. This looks like SAM 3D Body's raw "
        f"28-dim PCA scale coefficients, which clad-body's loader would "
        f"silently misinterpret if passed as 'scale_params'. This adapter "
        f"does not forward 'scale_params' at all -- it uses "
        f"'mhr_model_params' instead, which already contains the "
        f"correctly-decoded scale in its [136:204] slice."
    )
