"""Tests for rescale.py -- our own optional known-height correction.

Uses synthetic vertex arrays (a unit cube-ish "body"), not a real MHR mesh,
so this runs without pymomentum/clad-body being importable at all.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from rescale import RescaleError, compute_mesh_height_cm, uniform_rescale_to_height


def _fake_body_vertices(height_m: float) -> np.ndarray:
    """A trivial 'body': 8 corners of a box, Z from 0 to height_m, canonical
    Z-up / feet-at-zero convention (matches clad-body's MhrBody)."""
    xs = [-0.2, 0.2]
    ys = [-0.1, 0.1]
    zs = [0.0, height_m]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype="float64")


def test_compute_mesh_height_cm_matches_known_input():
    verts = _fake_body_vertices(1.70)  # 170 cm
    assert compute_mesh_height_cm(verts) == pytest.approx(170.0, abs=1e-6)


def test_compute_mesh_height_cm_rejects_wrong_shape():
    with pytest.raises(Exception):
        compute_mesh_height_cm(np.zeros((10, 4)))


def test_uniform_rescale_matches_target_height():
    verts = _fake_body_vertices(1.60)  # model predicted 160 cm
    rescaled, factor = uniform_rescale_to_height(verts, known_height_cm=180.0)
    assert compute_mesh_height_cm(rescaled) == pytest.approx(180.0, abs=1e-6)
    assert factor == pytest.approx(180.0 / 160.0)


def test_uniform_rescale_is_isotropic_not_just_vertical():
    """A uniform scale must also stretch X/Y by the same factor, not just Z --
    this catches an accidental 'scale Z only' bug."""
    verts = _fake_body_vertices(1.50)
    original_x_span = verts[:, 0].max() - verts[:, 0].min()
    rescaled, factor = uniform_rescale_to_height(verts, known_height_cm=165.0)
    new_x_span = rescaled[:, 0].max() - rescaled[:, 0].min()
    assert new_x_span == pytest.approx(original_x_span * factor)


def test_rejects_implausible_known_height_too_small():
    verts = _fake_body_vertices(1.70)
    with pytest.raises(RescaleError, match="plausible adult range"):
        uniform_rescale_to_height(verts, known_height_cm=50.0)


def test_rejects_implausible_known_height_too_large():
    verts = _fake_body_vertices(1.70)
    with pytest.raises(RescaleError, match="plausible adult range"):
        uniform_rescale_to_height(verts, known_height_cm=400.0)


def test_rejects_degenerate_zero_height_mesh():
    verts = _fake_body_vertices(0.0)
    with pytest.raises(RescaleError, match="non-positive"):
        uniform_rescale_to_height(verts, known_height_cm=170.0)


def test_scale_factor_of_one_is_a_noop():
    verts = _fake_body_vertices(1.75)
    rescaled, factor = uniform_rescale_to_height(verts, known_height_cm=175.0)
    assert factor == pytest.approx(1.0)
    np.testing.assert_allclose(rescaled, verts)
