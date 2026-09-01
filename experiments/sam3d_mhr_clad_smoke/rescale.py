"""Optional deterministic known-height rescale.

Investigation (see docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md,
"Metric scale investigation") found that clad-body's MHR loader
(``load_mhr_from_params``) does not use any camera/FOV information at all
-- it reconstructs a rest-pose mesh purely from ``shape_params`` and the
scale slice of ``mhr_model_params``, decoded through the MHR body model's
own learned parameterization. Whatever real-world scale the resulting mesh
has comes entirely from how well SAM 3D Body's regression network predicts
those parameters -- not from a per-photo camera-based correction.

There is no per-image metric-scale correction step already wired into this
pipeline. This module adds ONE explicit, optional, deterministic step: if
the customer's real height is known (as MTM intake commonly already
collects), uniformly rescale the *entire* mesh (and derived joints) so its
own computed height matches the known height, before measuring.

This is a uniform isotropic scale only. It does NOT correct for:
- Camera perspective distortion (a body closer to the camera reads
  differently than one farther away, at a given pixel height).
- Posture (a slouched vs. upright subject changes standing height without
  changing the underlying skeletal proportions the same way).
- Non-uniform errors (e.g. SAM 3D Body might get torso length
  proportionally wrong even if its overall vertical extent happens to
  match reality) -- a single scalar cannot fix that.

It is best understood as "given the model got proportions roughly right
but scale wrong, use the one reliable number we already have (self-reported
height) to fix scale" -- not as a general accuracy correction.
"""

from __future__ import annotations

import numpy as np


class RescaleError(ValueError):
    pass


def compute_mesh_height_cm(vertices_m: np.ndarray, up_axis: int = 2) -> float:
    """Height of a mesh in cm, given vertices in metres with the body
    already oriented so ``up_axis`` is the vertical axis and feet are at
    the minimum of that axis (clad-body's canonical convention: Z-up,
    metres, feet at Z=0 -- see ``clad_body.load.mhr._mhr_yup_cm_to_canonical``).
    """
    if vertices_m.ndim != 2 or vertices_m.shape[1] != 3:
        raise RescaleError(f"expected an (N, 3) vertex array, got shape {vertices_m.shape}")
    span_m = float(vertices_m[:, up_axis].max() - vertices_m[:, up_axis].min())
    return span_m * 100.0


def uniform_rescale_to_height(
    vertices_m: np.ndarray,
    known_height_cm: float,
    up_axis: int = 2,
) -> tuple[np.ndarray, float]:
    """Uniformly scale ``vertices_m`` about the origin so the mesh's own
    height (per :func:`compute_mesh_height_cm`) matches ``known_height_cm``.

    Args:
        vertices_m: (N, 3) vertex array in metres, canonical orientation.
        known_height_cm: customer-reported height in cm. Must be positive
            and within a broad plausible adult-human range -- this is a
            sanity check, not a validated bound.
        up_axis: which column is the vertical axis (2 = Z, clad-body's
            canonical convention).

    Returns:
        (rescaled_vertices, scale_factor) -- the scale factor actually
        applied, so callers can log/report it.

    Raises:
        RescaleError: if the input mesh has zero/degenerate height, or if
            ``known_height_cm`` is outside a plausible adult range
            (roughly 100-230 cm) -- this catches obvious unit-confusion
            bugs (e.g. passing metres or inches by mistake) rather than
            asserting any real anthropometric bound.
    """
    if not (100.0 <= known_height_cm <= 230.0):
        raise RescaleError(
            f"known_height_cm={known_height_cm} is outside the plausible "
            f"adult range [100, 230] cm -- refusing to rescale, this is "
            f"almost certainly a unit or input error"
        )

    current_height_cm = compute_mesh_height_cm(vertices_m, up_axis=up_axis)
    if current_height_cm <= 0:
        raise RescaleError(
            f"mesh has non-positive computed height ({current_height_cm} cm); "
            f"cannot rescale a degenerate mesh"
        )

    scale_factor = known_height_cm / current_height_cm
    rescaled = vertices_m * scale_factor
    return rescaled, scale_factor
