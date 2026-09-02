"""SAM 3D Body upstream source-tree location -- Task 03D.

Root cause of the real Colab failure this fixes: `ModuleNotFoundError: No
module named 'sam_3d_body'`. Environment A (`/content/env_sam3d`) built
successfully, torch/CUDA/GPU all verified working, but the worker
subprocess was launched with no `PYTHONPATH` and no `sys.path` entry
pointing at the cloned `facebookresearch/sam-3d-body` repo root -- the
interpreter simply had nowhere to find the `sam_3d_body` package.

A fresh clone of the pinned commit (`b5c765a`) has no `pyproject.toml`,
`setup.py`, or `setup.cfg` anywhere in the repo root -- confirmed directly,
not assumed. `pip install -e .` is therefore not an option; inventing
packaging metadata for a third-party repository is explicitly out of
scope for this task. The repo root is used directly on the Python module
search path instead, which is exactly what running `demo.py` from that
directory has always relied on upstream.

Two mechanisms, deliberately redundant: the notebook sets `PYTHONPATH`
(preserving whatever was already set) when launching the worker
subprocess -- the "preferred solution" per Task 03D section 3, since it
requires no code change to how Python resolves the import. The worker
*additionally* takes an explicit `sam3d_source_root` argument, validates
it, and prepends it to `sys.path` itself before importing anything from
`sam_3d_body` -- so a pre-flight import check can run and fail loudly and
specifically (`SAM3D_SOURCE_IMPORT_FAILURE`) even if `PYTHONPATH` alone
didn't propagate for some reason, rather than falling through to a
generic, harder-to-diagnose `ModuleNotFoundError` deep inside model
loading.
"""

from __future__ import annotations

import os


class Sam3dSourceRootError(ValueError):
    """Raised when a candidate SAM 3D Body source root doesn't actually
    contain the sam_3d_body package."""


def validate_sam3d_source_root(path: str) -> str:
    """Validate that `path` is a directory containing an importable
    `sam_3d_body` package (i.e. `<path>/sam_3d_body/__init__.py` exists).

    Returns the validated absolute path (never invents or guesses one --
    raises instead) so a caller can pass the return value straight to
    `sys.path.insert()` or `build_pythonpath()` with no further checks.

    Raises:
        Sam3dSourceRootError: if `path` does not exist, or does not
            contain a `sam_3d_body` package directory with `__init__.py`.
    """
    if not path:
        raise Sam3dSourceRootError("sam3d_source_root is empty")

    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise Sam3dSourceRootError(f"sam3d_source_root does not exist or is not a directory: {abs_path}")

    package_dir = os.path.join(abs_path, "sam_3d_body")
    if not os.path.isdir(package_dir):
        raise Sam3dSourceRootError(
            f"{abs_path} does not contain a 'sam_3d_body/' directory -- "
            f"this is not a valid facebookresearch/sam-3d-body checkout"
        )

    init_file = os.path.join(package_dir, "__init__.py")
    if not os.path.isfile(init_file):
        raise Sam3dSourceRootError(
            f"{package_dir} exists but has no __init__.py -- not an importable package"
        )

    return abs_path


def build_pythonpath(source_root: str, existing_pythonpath: str | None = None) -> str:
    """Deterministically build a PYTHONPATH value with `source_root`
    prepended, preserving whatever was already set.

    Does not validate `source_root` itself -- call
    `validate_sam3d_source_root()` first and pass its return value in, so
    this function stays a pure, trivially-testable string operation.
    """
    if existing_pythonpath:
        return f"{source_root}{os.pathsep}{existing_pythonpath}"
    return source_root


def resolved_module_is_under_root(module_file: str, source_root: str) -> bool:
    """True if an imported module's `__file__` actually resolves to
    somewhere under `source_root`, not some unrelated installed package
    that happens to share the `sam_3d_body` name on `sys.path` (Task 03D
    section 4: "Do not proceed if it resolves to some unrelated installed
    package.")."""
    real_module_file = os.path.realpath(module_file)
    real_root = os.path.realpath(source_root)
    return os.path.commonpath([real_module_file, real_root]) == real_root
