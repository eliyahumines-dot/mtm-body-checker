"""Tests for sam3d_source_path.py -- Task 03D's source-root validation and
PYTHONPATH construction. Pure filesystem/string checks, no GPU, no torch,
no real sam-3d-body checkout required (built with tmp_path fixtures)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sam3d_source_path import (
    Sam3dSourceRootError,
    build_pythonpath,
    resolved_module_is_under_root,
    validate_sam3d_source_root,
)


def _make_valid_source_root(tmp_path) -> Path:
    root = tmp_path / "sam-3d-body"
    pkg = root / "sam_3d_body"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    return root


def test_validate_accepts_a_real_sam3d_checkout_layout(tmp_path):
    root = _make_valid_source_root(tmp_path)
    result = validate_sam3d_source_root(str(root))
    assert result == os.path.abspath(str(root))


def test_validate_rejects_empty_path():
    with pytest.raises(Sam3dSourceRootError, match="empty"):
        validate_sam3d_source_root("")


def test_validate_rejects_nonexistent_path(tmp_path):
    with pytest.raises(Sam3dSourceRootError, match="does not exist"):
        validate_sam3d_source_root(str(tmp_path / "does_not_exist"))


def test_validate_rejects_directory_without_sam_3d_body_subdir(tmp_path):
    """This is the exact Task 03D bug scenario: a directory that exists but
    doesn't actually contain the sam_3d_body package."""
    empty_dir = tmp_path / "not_a_sam3d_checkout"
    empty_dir.mkdir()
    with pytest.raises(Sam3dSourceRootError, match="sam_3d_body"):
        validate_sam3d_source_root(str(empty_dir))


def test_validate_rejects_sam_3d_body_dir_missing_init(tmp_path):
    root = tmp_path / "sam-3d-body"
    (root / "sam_3d_body").mkdir(parents=True)
    # no __init__.py written
    with pytest.raises(Sam3dSourceRootError, match="__init__"):
        validate_sam3d_source_root(str(root))


def test_validate_returns_absolute_path_even_for_relative_input(tmp_path, monkeypatch):
    root = _make_valid_source_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = validate_sam3d_source_root("sam-3d-body")
    assert os.path.isabs(result)


def test_build_pythonpath_with_no_existing_pythonpath():
    assert build_pythonpath("/content/sam-3d-body") == "/content/sam-3d-body"


def test_build_pythonpath_preserves_existing_pythonpath():
    result = build_pythonpath("/content/sam-3d-body", existing_pythonpath="/some/other/path")
    assert result == f"/content/sam-3d-body{os.pathsep}/some/other/path"
    assert "/some/other/path" in result  # existing PYTHONPATH must never be dropped


def test_build_pythonpath_with_empty_string_existing_pythonpath_is_treated_as_absent():
    assert build_pythonpath("/content/sam-3d-body", existing_pythonpath="") == "/content/sam-3d-body"


def test_build_pythonpath_is_deterministic():
    """Same inputs must always produce the same output -- no environment-
    dependent ordering or hidden state."""
    r1 = build_pythonpath("/a/b", existing_pythonpath="/c/d")
    r2 = build_pythonpath("/a/b", existing_pythonpath="/c/d")
    assert r1 == r2


def test_build_pythonpath_puts_source_root_first():
    """The new source root must take precedence over whatever else is on
    PYTHONPATH, in case of a name collision."""
    result = build_pythonpath("/content/sam-3d-body", existing_pythonpath="/other")
    assert result.startswith("/content/sam-3d-body")


def test_resolved_module_is_under_root_true_case(tmp_path):
    root = _make_valid_source_root(tmp_path)
    module_file = str(root / "sam_3d_body" / "__init__.py")
    assert resolved_module_is_under_root(module_file, str(root)) is True


def test_resolved_module_is_under_root_false_case_unrelated_package(tmp_path):
    """Guards against Task 03D section 4's specific concern: sam_3d_body.__file__
    resolving to some unrelated installed package rather than the intended
    official source tree."""
    root = _make_valid_source_root(tmp_path)
    unrelated = tmp_path / "site-packages" / "sam_3d_body" / "__init__.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("")
    assert resolved_module_is_under_root(str(unrelated), str(root)) is False
