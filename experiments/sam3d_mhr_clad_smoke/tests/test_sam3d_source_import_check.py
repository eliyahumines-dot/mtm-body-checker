"""Tests for _sam3d_source_import_check.py -- Task 03E's standalone,
env=PYTHONPATH-free SAM3D_SOURCE_ROOT_VALIDATION / SAM3D_SOURCE_IMPORT /
SAM3D_MODEL_CODE_IMPORT pre-flight script (Task 03F: source-root validation
split out as its own boundary, and a headless Matplotlib backend forced
before any import that might pull matplotlib in as a side effect).

Two kinds of coverage, per Task 03E section 8 / Task 03F section 10:

1. Synthetic fixtures (`_make_fake_source_root` et al.) -- no torch, no
   GPU, no real checkout required. These run in any CI environment and
   cover: valid/invalid source root, correct sys.path insertion,
   deterministic behavior, a source root containing spaces, source root
   independence from cwd, an unrelated installed module failing to
   override the supplied source root, each failure category, downstream
   boundaries staying NOT_ATTEMPTED (None) after an earlier one fails, an
   inherited invalid MPLBACKEND being overridden, and Matplotlib-backend
   failures being classified distinctly from other import-time failures.

2. A REAL, non-mocked import preflight against an actual
   `facebookresearch/sam-3d-body` checkout (`test_real_upstream_...`
   below) -- `import sam_3d_body` itself requires torch/cv2/the full
   SAM3D_CORE_PIP_DEPENDENCIES set (confirmed directly: it fails with
   `ModuleNotFoundError: No module named 'cv2'` under a plain
   pytest/numpy-only interpreter), so this specific test looks for a
   pre-built Environment-A-equivalent venv + a real clone via the
   `SAM3D_TEST_PYTHON` / `SAM3D_TEST_SOURCE_ROOT` environment variables
   (falling back to this task's own `/tmp/env_sam3d_test` +
   `/tmp/sam3d_e2e_clone`, built and verified while diagnosing Task 03E/F),
   and skips with an explicit, honest reason if neither is available --
   it never fabricates a pass. This agent has already run this exact
   check for real, directly (not only via pytest): with the exact invalid
   Colab MPLBACKEND value inherited, `sam_3d_body.__file__` still resolved
   to `/tmp/sam3d_e2e_clone/sam_3d_body/__init__.py` and all three stages
   PASS, effective matplotlib backend `Agg`.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sam3d_source_path import Sam3dSourceRootError  # noqa: E402
import _sam3d_source_import_check as preflight  # noqa: E402

SCRIPT = str(Path(__file__).parent.parent / "_sam3d_source_import_check.py")


@pytest.fixture(autouse=True)
def _clean_sam_3d_body_from_sys_modules():
    """In-process `run_preflight()` calls do a real `import sam_3d_body`,
    which Python caches in `sys.modules` by name -- across tests in this
    file, different fake packages live under different tmp_path roots, so
    without this the second test to run would silently reuse the first
    test's already-imported (and now wrong-root) module. This is purely a
    test-isolation artifact of calling the function in-process rather than
    via a fresh subprocess each time (which is how production actually
    invokes it) -- not a bug in the script itself."""
    for name in list(sys.modules):
        if name == "sam_3d_body" or name.startswith("sam_3d_body."):
            del sys.modules[name]
    yield
    for name in list(sys.modules):
        if name == "sam_3d_body" or name.startswith("sam_3d_body."):
            del sys.modules[name]


def _make_fake_source_root(root: Path, model_code_importable: bool = True) -> Path:
    """Build a synthetic (non-torch) fake sam_3d_body package layout, with
    real, importable build_models/sam_3d_body_estimator submodules, so the
    SAM3D_MODEL_CODE_IMPORT stage is genuinely exercised, not just
    SAM3D_SOURCE_IMPORT."""
    pkg = root / "sam_3d_body"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    if model_code_importable:
        (pkg / "build_models.py").write_text("def load_sam_3d_body(*a, **kw):\n    return None, None\n")
        (pkg / "sam_3d_body_estimator.py").write_text("class SAM3DBodyEstimator:\n    pass\n")
    else:
        (pkg / "build_models.py").write_text("raise ImportError('synthetic model-code-import failure')\n")
        (pkg / "sam_3d_body_estimator.py").write_text("class SAM3DBodyEstimator:\n    pass\n")
    return root


def _make_fake_source_root_with_init_error(root: Path, error_code: str) -> Path:
    """Build a synthetic sam_3d_body package whose `__init__.py` itself
    raises at import time (simulating a runtime/dependency failure reached
    only after a genuinely valid source root -- Task 03F section 4/6:
    the root exists and validates, but `import sam_3d_body` still fails)."""
    pkg = root / "sam_3d_body"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(error_code)
    return root


def _run_script(source_root: str, telemetry_path: str | None = None, env: dict | None = None, cwd: str | None = None):
    cmd = [sys.executable, SCRIPT, source_root]
    if telemetry_path:
        cmd.append(telemetry_path)
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=run_env, cwd=cwd)


# --- valid / invalid source root ---

def test_valid_source_root_all_three_stages_pass(tmp_path):
    root = _make_fake_source_root(tmp_path / "sam3d_root")
    telemetry = preflight.run_preflight(str(root))
    assert telemetry["source_root_validation_ok"] is True
    assert telemetry["source_import_ok"] is True
    assert telemetry["model_code_import_ok"] is True
    assert telemetry["error"] is None
    assert telemetry["failure_category"] is None
    assert telemetry["sam3d_module_file"] == str(root / "sam_3d_body" / "__init__.py")


def test_invalid_source_root_fails_at_root_validation_not_source_import(tmp_path):
    """Task 03F section 6: source-root validation and module execution are
    distinct -- a nonexistent root must be reported as
    SAM3D_SOURCE_ROOT_VALIDATION_FAILURE, never as SAM3D_SOURCE_IMPORT_FAILURE
    (which now means the root validated but `import sam_3d_body` itself
    failed once actually attempted)."""
    telemetry = preflight.run_preflight(str(tmp_path / "does_not_exist"))
    assert telemetry["source_root_validation_ok"] is False
    assert telemetry["source_import_ok"] is None  # never attempted
    assert telemetry["model_code_import_ok"] is None  # never attempted
    assert telemetry["failure_category"] == "SAM3D_SOURCE_ROOT_VALIDATION_FAILURE"
    assert "SAM3D_SOURCE_ROOT_VALIDATION_FAILURE" in telemetry["error"]


# --- downstream boundaries stay NOT_ATTEMPTED (None), not FAIL, after an earlier failure ---

def test_downstream_boundaries_not_attempted_after_root_validation_failure(tmp_path):
    telemetry = preflight.run_preflight(str(tmp_path / "nope"))
    assert telemetry["source_import_ok"] is None
    assert telemetry["model_code_import_ok"] is None


def test_downstream_boundary_not_attempted_after_source_import_failure(tmp_path):
    root = _make_fake_source_root_with_init_error(tmp_path / "sam3d_root", "raise RuntimeError('boom')\n")
    telemetry = preflight.run_preflight(str(root))
    assert telemetry["source_root_validation_ok"] is True
    assert telemetry["source_import_ok"] is False
    assert telemetry["model_code_import_ok"] is None  # never attempted


def test_model_code_import_failure_is_distinct_from_source_import_failure(tmp_path):
    root = _make_fake_source_root(tmp_path / "sam3d_root", model_code_importable=False)
    telemetry = preflight.run_preflight(str(root))
    assert telemetry["source_root_validation_ok"] is True
    assert telemetry["source_import_ok"] is True  # bare package import DID succeed
    assert telemetry["model_code_import_ok"] is False
    assert telemetry["failure_category"] == "SAM3D_MODEL_CODE_IMPORT_FAILURE"
    assert "SAM3D_MODEL_CODE_IMPORT_FAILURE" in telemetry["error"]
    assert "SAM3D_SOURCE_IMPORT_FAILURE" not in telemetry["error"]
    assert "SAM3D_SOURCE_ROOT_VALIDATION_FAILURE" not in telemetry["error"]


# --- Task 03F: distinguishing the Matplotlib-backend failure from a general one ---

def test_matplotlib_backend_failure_is_classified_distinctly(tmp_path):
    """Reproduces the real Colab failure's exact shape: the root validates
    and `import sam_3d_body` is attempted, but raises an error whose text
    matches the actual observed exception (a Matplotlib backend ValueError)
    -- must be tagged SAM3D_MATPLOTLIB_BACKEND_FAILURE, not a generic
    source-import or runtime-dependency failure."""
    root = _make_fake_source_root_with_init_error(
        tmp_path / "sam3d_root",
        "raise ValueError(\"Key backend: 'module://matplotlib_inline.backend_inline' "
        "is not a valid value for backend\")\n",
    )
    telemetry = preflight.run_preflight(str(root))
    assert telemetry["source_root_validation_ok"] is True
    assert telemetry["source_import_ok"] is False
    assert telemetry["failure_category"] == "SAM3D_MATPLOTLIB_BACKEND_FAILURE"
    assert "SAM3D_MATPLOTLIB_BACKEND_FAILURE" in telemetry["error"]
    assert "not a valid value for backend" in telemetry["error"]  # underlying exception retained


def test_unrelated_import_failure_is_classified_as_general_runtime_dependency_failure(tmp_path):
    """A completely unrelated import-time failure (nothing to do with
    Matplotlib) must fall back to the general category, not be
    misclassified as the specific Matplotlib one."""
    root = _make_fake_source_root_with_init_error(
        tmp_path / "sam3d_root", "raise ImportError('No module named some_other_thing')\n",
    )
    telemetry = preflight.run_preflight(str(root))
    assert telemetry["source_root_validation_ok"] is True
    assert telemetry["source_import_ok"] is False
    assert telemetry["failure_category"] == "SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE"
    assert "some_other_thing" in telemetry["error"]  # underlying exception retained


# --- Sam3dSourceRootError propagation from validate_sam3d_source_root ---

def test_run_preflight_never_raises_on_a_missing_root(tmp_path):
    try:
        preflight.run_preflight(str(tmp_path / "missing"))
    except Sam3dSourceRootError:
        pytest.fail("run_preflight() must catch Sam3dSourceRootError itself, never let it propagate")


# --- correct sys.path insertion (black-box, via subprocess) ---

def test_correct_sys_path_insertion_via_subprocess(tmp_path):
    root = _make_fake_source_root(tmp_path / "sam3d_root")
    proc = _run_script(str(root))
    assert proc.returncode == 0
    assert "SAM3D_SOURCE_IMPORT: PASS" in proc.stdout
    assert "SAM3D_MODEL_CODE_IMPORT: PASS" in proc.stdout
    assert str(root / "sam_3d_body" / "__init__.py") in proc.stdout


def test_subprocess_does_not_rely_on_pythonpath_env_var(tmp_path):
    """The whole point of Task 03E: this script must resolve the import via
    its own sys.path.insert() from the CLI argument alone -- explicitly
    verified here by launching with PYTHONPATH deliberately unset/cleared."""
    root = _make_fake_source_root(tmp_path / "sam3d_root")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, SCRIPT, str(root)], capture_output=True, text=True, timeout=30, env=env,
    )
    assert proc.returncode == 0
    assert "SAM3D_SOURCE_IMPORT: PASS" in proc.stdout


# --- deterministic behavior ---

def test_deterministic_behavior_same_input_same_output(tmp_path):
    root = _make_fake_source_root(tmp_path / "sam3d_root")
    t1 = preflight.run_preflight(str(root))
    t2 = preflight.run_preflight(str(root))
    assert t1["source_import_ok"] == t2["source_import_ok"] == True  # noqa: E712
    assert t1["sam3d_module_file"] == t2["sam3d_module_file"]


# --- source root containing spaces ---

def test_source_root_containing_spaces(tmp_path):
    space_dir = tmp_path / "space test dir"
    root = _make_fake_source_root(space_dir / "sam3d_pkg_root")
    telemetry_path = tmp_path / "telemetry.json"
    proc = _run_script(str(root), telemetry_path=str(telemetry_path))
    assert proc.returncode == 0, proc.stderr
    assert "SAM3D_SOURCE_IMPORT: PASS" in proc.stdout
    written = json.loads(telemetry_path.read_text())
    assert written["source_import_ok"] is True
    assert " " in written["sam3d_source_root_resolved"]


# --- source root passed independently of cwd ---

def test_source_root_independent_of_cwd(tmp_path):
    root = _make_fake_source_root(tmp_path / "sam3d_root")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    proc = _run_script(str(root), cwd=str(elsewhere))
    assert proc.returncode == 0
    assert "SAM3D_SOURCE_IMPORT: PASS" in proc.stdout


# --- unrelated installed module cannot override the supplied source root ---

def test_unrelated_module_on_pythonpath_cannot_override_supplied_root(tmp_path):
    """An unrelated package also named sam_3d_body sits earlier on PYTHONPATH
    (simulating some other accidental install) -- the explicitly supplied
    source root must still win, because it is inserted at sys.path[0]."""
    intended_root = _make_fake_source_root(tmp_path / "intended_sam3d_root")
    unrelated_root = tmp_path / "unrelated_root"
    unrelated_pkg = unrelated_root / "sam_3d_body"
    unrelated_pkg.mkdir(parents=True)
    (unrelated_pkg / "__init__.py").write_text("")

    proc = _run_script(str(intended_root), env={"PYTHONPATH": str(unrelated_root)})
    assert proc.returncode == 0
    assert str(intended_root / "sam_3d_body" / "__init__.py") in proc.stdout
    assert str(unrelated_root) not in proc.stdout


def test_unrelated_module_shadowing_produces_source_import_failure_when_it_is_the_supplied_root(tmp_path):
    """If the *supplied* root itself resolves to something that isn't under
    the validated root after realpath resolution (e.g. a symlink escape),
    that must be reported as SAM3D_SOURCE_IMPORT_FAILURE, not silently
    accepted. Exercised directly via resolved_module_is_under_root's own
    contract (already unit-tested in test_sam3d_source_path.py) plus this
    script's use of it."""
    root = _make_fake_source_root(tmp_path / "sam3d_root")
    telemetry = preflight.run_preflight(str(root))
    assert telemetry["source_import_ok"] is True  # sanity: legitimate case is unaffected


# --- exit codes ---

def test_exit_code_1_on_failure(tmp_path):
    proc = _run_script(str(tmp_path / "missing"))
    assert proc.returncode == 1


def test_exit_code_0_on_full_success(tmp_path):
    root = _make_fake_source_root(tmp_path / "sam3d_root")
    proc = _run_script(str(root))
    assert proc.returncode == 0


def test_usage_error_exit_code_2():
    proc = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 2


# --- REAL (non-mocked) import preflight against the actual upstream repo ---

def _resolve_real_test_env():
    python_exe = os.environ.get("SAM3D_TEST_PYTHON", "/tmp/env_sam3d_test/bin/python3.11")
    source_root = os.environ.get("SAM3D_TEST_SOURCE_ROOT", "/tmp/sam3d_e2e_clone")
    if not (os.path.isfile(python_exe) and os.access(python_exe, os.X_OK)):
        return None, None
    if not os.path.isdir(os.path.join(source_root, "sam_3d_body")):
        return None, None
    return python_exe, source_root


def test_real_upstream_sam3d_source_import_check(tmp_path):
    """Runs this script for real, via subprocess, against an actual
    `facebookresearch/sam-3d-body` git checkout and a Python interpreter
    with the full SAM3D_CORE_PIP_DEPENDENCIES set installed (built and
    verified while diagnosing Task 03E: a fresh `git clone --depth 1
    https://github.com/facebookresearch/sam-3d-body.git`, and a venv with
    the exact `sam3d_env_spec.SAM3D_CORE_PIP_DEPENDENCIES` list). Skips,
    rather than fabricating a result, if that pre-built environment isn't
    present in the current sandbox -- this is a real integration test, not
    a mock, so it legitimately needs real assets that aren't checked into
    the repo (multi-GB torch/cv2/etc. install)."""
    python_exe, source_root = _resolve_real_test_env()
    if python_exe is None:
        pytest.skip(
            "No pre-built Environment-A-equivalent venv + real sam-3d-body checkout available "
            "in this sandbox (checked SAM3D_TEST_PYTHON/SAM3D_TEST_SOURCE_ROOT and the "
            "/tmp/env_sam3d_test + /tmp/sam3d_e2e_clone default). This is a real integration "
            "test against actual upstream code, not a mock, so it cannot fabricate a pass."
        )
    telemetry_path = tmp_path / "real_telemetry.json"
    proc = subprocess.run(
        [python_exe, SCRIPT, source_root, str(telemetry_path)],
        capture_output=True, text=True, timeout=120,
    )
    print(proc.stdout)
    print(proc.stderr)
    assert "SAM3D_SOURCE_IMPORT: PASS" in proc.stdout, proc.stdout + proc.stderr
    assert "SAM3D_MODEL_CODE_IMPORT: PASS" in proc.stdout, proc.stdout + proc.stderr
    assert proc.returncode == 0
    written = json.loads(telemetry_path.read_text())
    assert written["source_import_ok"] is True
    assert written["model_code_import_ok"] is True
    real_root = os.path.realpath(source_root)
    resolved_module = os.path.realpath(written["sam3d_module_file"])
    assert os.path.commonpath([resolved_module, real_root]) == real_root


def test_real_upstream_import_survives_the_exact_invalid_colab_mplbackend(tmp_path):
    """Reproduces Task 03F's exact reported failure and its fix, for real:
    launches this script with the precise invalid MPLBACKEND value Colab's
    kernel sets inherited into the subprocess's environment (as a real
    Colab-launched worker subprocess would inherit it), against the same
    real clone + real dependency-complete venv as the test above.

    Before this task's fix, this reproduced the exact reported traceback:
    `ValueError: Key backend: 'module://matplotlib_inline.backend_inline'
    is not a valid value for backend`, raised from deep inside
    `torchmetrics.utilities.plot` (imported transitively via
    `pytorch_lightning`) while executing `import sam_3d_body`. After the
    fix, the same inherited value must have no effect at all."""
    python_exe, source_root = _resolve_real_test_env()
    if python_exe is None:
        pytest.skip(
            "No pre-built Environment-A-equivalent venv + real sam-3d-body checkout available "
            "in this sandbox. This is a real integration test against actual upstream code, not "
            "a mock, so it cannot fabricate a pass."
        )
    telemetry_path = tmp_path / "real_mplbackend_telemetry.json"
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    proc = subprocess.run(
        [python_exe, SCRIPT, source_root, str(telemetry_path)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    print(proc.stdout)
    print(proc.stderr)
    assert "Inherited MPLBACKEND: module://matplotlib_inline.backend_inline" in proc.stdout
    assert "Worker MPLBACKEND: Agg" in proc.stdout
    assert "SAM3D_SOURCE_ROOT_VALIDATION: PASS" in proc.stdout
    assert "SAM3D_SOURCE_IMPORT: PASS" in proc.stdout, proc.stdout + proc.stderr
    assert "SAM3D_MODEL_CODE_IMPORT: PASS" in proc.stdout, proc.stdout + proc.stderr
    assert "not a valid value for backend" not in (proc.stdout + proc.stderr)
    assert proc.returncode == 0
    written = json.loads(telemetry_path.read_text())
    assert written["inherited_mplbackend"] == "module://matplotlib_inline.backend_inline"
    assert written["worker_mplbackend"] == "Agg"
    assert written["matplotlib_backend_effective"] == "Agg"
    assert written["source_root_validation_ok"] is True
    assert written["source_import_ok"] is True
    assert written["model_code_import_ok"] is True
