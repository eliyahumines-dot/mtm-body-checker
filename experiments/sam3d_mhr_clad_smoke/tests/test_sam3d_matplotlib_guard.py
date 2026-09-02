"""Tests for sam3d_matplotlib_guard.py -- Task 03F's headless-Matplotlib-
backend fix for standalone Environment A worker/check subprocesses.

Root cause under test: Colab's interactive kernel sets `MPLBACKEND` to its
own inline-plotting backend (`module://matplotlib_inline.backend_inline`),
a standalone subprocess inherits that value by default, and something SAM
3D Body imports transitively (this task's real reproduction: `torchmetrics`
via `pytorch_lightning`) crashes trying to activate it. Pure logic here --
no torch, no GPU, no real checkout required.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sam3d_matplotlib_guard import (  # noqa: E402
    HEADLESS_BACKEND,
    SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE,
    SAM3D_MATPLOTLIB_BACKEND_FAILURE,
    classify_import_exception,
    effective_matplotlib_backend,
    force_headless_matplotlib_backend,
    sanitized_subprocess_env,
)

# The exact value observed on real Colab hardware (Task 03F's reported failure).
COLAB_INHERITED_VALUE = "module://matplotlib_inline.backend_inline"


@pytest.fixture(autouse=True)
def _restore_mplbackend_env():
    """force_headless_matplotlib_backend() mutates process os.environ --
    isolate that mutation to each test."""
    original = os.environ.get("MPLBACKEND")
    yield
    if original is None:
        os.environ.pop("MPLBACKEND", None)
    else:
        os.environ["MPLBACKEND"] = original


# --- force_headless_matplotlib_backend() ---

def test_headless_backend_constant_is_agg():
    assert HEADLESS_BACKEND == "Agg"


def test_invalid_inherited_mplbackend_is_overridden():
    """The exact scenario this task fixes: an inherited, notebook-only
    backend value already exists in the environment and must be replaced,
    not left in place."""
    os.environ["MPLBACKEND"] = COLAB_INHERITED_VALUE
    inherited = force_headless_matplotlib_backend()
    assert inherited == COLAB_INHERITED_VALUE
    assert os.environ["MPLBACKEND"] == "Agg"


def test_unset_mplbackend_returns_none_and_still_forces_agg():
    os.environ.pop("MPLBACKEND", None)
    inherited = force_headless_matplotlib_backend()
    assert inherited is None
    assert os.environ["MPLBACKEND"] == "Agg"


def test_does_not_use_setdefault_semantics():
    """Explicit requirement (Task 03F section 3): must assign, not
    setdefault -- an already-existing (even already-correct-looking, but
    still notebook-specific) value must always be replaced by the call."""
    os.environ["MPLBACKEND"] = "some_other_previously_set_value"
    force_headless_matplotlib_backend()
    assert os.environ["MPLBACKEND"] == "Agg"


# --- sanitized_subprocess_env() -- for the NOTEBOOK's own subprocess launches ---

def test_sanitized_subprocess_env_forces_mplbackend():
    env = sanitized_subprocess_env({"MPLBACKEND": COLAB_INHERITED_VALUE})
    assert env["MPLBACKEND"] == "Agg"


def test_sanitized_subprocess_env_preserves_unrelated_variables():
    base = {"MPLBACKEND": COLAB_INHERITED_VALUE, "HF_TOKEN_PRESENT": "1", "PATH": "/usr/bin", "SOME_VAR": "keep-me"}
    env = sanitized_subprocess_env(base)
    assert env["HF_TOKEN_PRESENT"] == "1"
    assert env["PATH"] == "/usr/bin"
    assert env["SOME_VAR"] == "keep-me"
    assert env["MPLBACKEND"] == "Agg"


def test_sanitized_subprocess_env_does_not_mutate_input():
    base = {"MPLBACKEND": COLAB_INHERITED_VALUE}
    sanitized_subprocess_env(base)
    assert base["MPLBACKEND"] == COLAB_INHERITED_VALUE  # caller's dict untouched


def test_sanitized_subprocess_env_defaults_to_current_os_environ(monkeypatch):
    monkeypatch.setenv("SOME_AMBIENT_VAR", "ambient-value")
    monkeypatch.setenv("MPLBACKEND", COLAB_INHERITED_VALUE)
    env = sanitized_subprocess_env()
    assert env["SOME_AMBIENT_VAR"] == "ambient-value"
    assert env["MPLBACKEND"] == "Agg"


# --- effective_matplotlib_backend() -- never force-imports matplotlib ---

def test_effective_backend_is_none_if_matplotlib_never_imported():
    assert "matplotlib" not in sys.modules  # sanity: this test process never imported it
    assert effective_matplotlib_backend() is None


def test_effective_backend_reports_actual_value_once_matplotlib_is_imported():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use(HEADLESS_BACKEND, force=True)
    assert effective_matplotlib_backend() == matplotlib.get_backend()


# --- classify_import_exception() -- correct failure classification ---

def test_classifies_the_exact_observed_matplotlib_backend_error():
    exc = ValueError(
        "Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend"
    )
    assert classify_import_exception(exc) == SAM3D_MATPLOTLIB_BACKEND_FAILURE


def test_classifies_unrelated_import_error_as_general_runtime_dependency_failure():
    exc = ModuleNotFoundError("No module named 'some_other_thing'")
    assert classify_import_exception(exc) == SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE


def test_classification_is_case_insensitive():
    exc = ValueError("BACKEND error mentions Matplotlib in some other phrasing")
    assert classify_import_exception(exc) == SAM3D_MATPLOTLIB_BACKEND_FAILURE


def test_classify_never_raises_on_an_exception_with_no_message():
    exc = RuntimeError()
    assert classify_import_exception(exc) == SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE
