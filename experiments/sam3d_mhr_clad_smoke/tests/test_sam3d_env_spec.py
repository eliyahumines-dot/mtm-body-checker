"""Tests for sam3d_env_spec.py -- Task 03C's Environment A dependency
specification. Pure data/string checks, no GPU, no network, no torch."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sam3d_env_spec import (
    EXCLUDED_FROM_MINIMAL_PHASE_A,
    SAM3D_CORE_PIP_DEPENDENCIES,
    TORCH_PIN,
    ExcludedDependencyError,
    pip_install_command,
    torch_install_command,
    validate_no_excluded_dependencies,
)


def test_chumpy_is_not_in_the_dependency_list():
    """The exact Task 03B bug: 'chumpy' must never appear -- the official
    name, verified against a fresh clone of sam-3d-body's INSTALL.md, is
    'chump'."""
    assert "chumpy" not in [d.lower() for d in SAM3D_CORE_PIP_DEPENDENCIES]


def test_chump_is_in_the_dependency_list():
    assert "chump" in SAM3D_CORE_PIP_DEPENDENCIES


def test_detectron2_is_not_in_the_dependency_list():
    lowered = [d.lower() for d in SAM3D_CORE_PIP_DEPENDENCIES]
    assert not any("detectron2" in d for d in lowered)


def test_moge_is_not_in_the_dependency_list():
    lowered = [d.lower() for d in SAM3D_CORE_PIP_DEPENDENCIES]
    assert not any("moge" in d for d in lowered)


def test_no_sam2_or_sam3_in_the_dependency_list():
    lowered = [d.lower() for d in SAM3D_CORE_PIP_DEPENDENCIES]
    assert not any("sam2" in d or "sam-2" in d or "sam3" in d or "sam-3" in d for d in lowered)


def test_validate_no_excluded_dependencies_passes_on_current_spec():
    validate_no_excluded_dependencies()  # must not raise


def test_validate_raises_if_an_excluded_package_is_present(monkeypatch):
    import sam3d_env_spec

    monkeypatch.setattr(sam3d_env_spec, "SAM3D_CORE_PIP_DEPENDENCIES", ("chumpy", "yacs"))
    with pytest.raises(ExcludedDependencyError, match="chumpy"):
        sam3d_env_spec.validate_no_excluded_dependencies()


def test_torch_pin_matches_task_03c_specification():
    assert TORCH_PIN["torch"] == "2.8.0"
    assert TORCH_PIN["torchvision"] == "0.23.0"
    assert TORCH_PIN["index_url"] == "https://download.pytorch.org/whl/cu129"


def test_torch_install_command_uses_the_pin():
    cmd = torch_install_command("/content/env_sam3d/bin/python3")
    assert "torch==2.8.0" in cmd
    assert "torchvision==0.23.0" in cmd
    assert "https://download.pytorch.org/whl/cu129" in cmd
    assert "/content/env_sam3d/bin/python3" in cmd


def test_pip_install_command_contains_chump_not_chumpy():
    cmd = pip_install_command("/content/env_sam3d/bin/python3")
    assert " chump " in f" {cmd} "
    assert "chumpy" not in cmd


def test_pip_install_command_has_no_unbounded_upgrade_flag():
    cmd = pip_install_command("/content/env_sam3d/bin/python3")
    assert "-U" not in cmd.split()
    assert "--upgrade" not in cmd


def test_excluded_list_and_dependency_list_do_not_trivially_overlap_by_typo():
    """Sanity check on the test data itself: every excluded name really is
    absent, not just coincidentally similar to something present."""
    lowered = [d.lower() for d in SAM3D_CORE_PIP_DEPENDENCIES]
    for excluded in EXCLUDED_FROM_MINIMAL_PHASE_A:
        assert not any(excluded in d for d in lowered), excluded
