"""Tests that `_sam3d_inference_worker.py` and `_sam3d_source_import_check.py`
each force a headless Matplotlib backend at module-import time, and that
Task 03F's fix was NOT implemented by adding `matplotlib-inline` as a new
dependency (that would only mask this one inherited value and leave a
headless subprocess depending on an interactive-kernel-only package for no
reason -- see sam3d_matplotlib_guard.py's docstring).

Importing either module directly (not running it as `__main__`) is safe
under a plain pytest/numpy environment: neither module imports torch,
cv2, or sam_3d_body at module scope -- those only happen inside `main()`/
`run_preflight()`, deep past the point where MPLBACKEND has already been
forced. This lets the override itself be verified without any heavy
dependency.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

COLAB_INHERITED_VALUE = "module://matplotlib_inline.backend_inline"


@pytest.fixture(autouse=True)
def _restore_mplbackend_env():
    original = os.environ.get("MPLBACKEND")
    yield
    if original is None:
        os.environ.pop("MPLBACKEND", None)
    else:
        os.environ["MPLBACKEND"] = original


@pytest.mark.parametrize("module_name", ["_sam3d_inference_worker", "_sam3d_source_import_check"])
def test_module_forces_headless_backend_at_import_time_in_process(module_name):
    """Importing the module (as a fresh subprocess would, at its own
    top-level) must force MPLBACKEND to Agg before any heavier import can
    occur -- verified here via a genuinely separate subprocess (not just
    re-importing into this already-running test process, which would hit
    Python's module cache and not re-execute the module body)."""
    proc = subprocess.run(
        [
            sys.executable, "-c",
            f"import os; os.environ['MPLBACKEND'] = {COLAB_INHERITED_VALUE!r}\n"
            f"import {module_name}\n"
            f"print('MPLBACKEND_AFTER_IMPORT=' + os.environ.get('MPLBACKEND', ''))\n"
            f"print('INHERITED_RECORDED=' + str({module_name}._INHERITED_MPLBACKEND))\n",
        ],
        cwd=str(EXPERIMENT_DIR),
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MPLBACKEND_AFTER_IMPORT=Agg" in proc.stdout
    assert f"INHERITED_RECORDED={COLAB_INHERITED_VALUE}" in proc.stdout


def test_worker_module_imports_cleanly_without_torch_or_cv2():
    """Sanity check that the worker module can be loaded at all under a
    plain pytest/numpy environment (no torch, no cv2, no sam_3d_body) --
    proves the MPLBACKEND override and its imports happen before anything
    heavy, since a heavy import here would raise ModuleNotFoundError."""
    if "_sam3d_inference_worker" in sys.modules:
        del sys.modules["_sam3d_inference_worker"]
    module = importlib.import_module("_sam3d_inference_worker")
    assert hasattr(module, "_INHERITED_MPLBACKEND")
    assert module.HEADLESS_BACKEND == "Agg"


# --- no matplotlib-inline workaround dependency was added ---

def test_matplotlib_inline_was_not_added_as_a_dependency():
    """Task 03F section 2: 'Do not solve this by installing matplotlib-inline
    merely to satisfy an inherited notebook setting.' Environment A's pinned
    dependency list is the single source of truth for what gets installed
    (sam3d_env_spec.py) -- assert it directly rather than trusting a
    docstring claim."""
    from sam3d_env_spec import SAM3D_CORE_PIP_DEPENDENCIES

    lowered = [dep.lower() for dep in SAM3D_CORE_PIP_DEPENDENCIES]
    assert not any("matplotlib-inline" in dep or "matplotlib_inline" in dep for dep in lowered)
