"""Tests for install_log.py -- structured command-failure logging (Task 03C).
Pure dataclasses, no subprocess/GPU/network needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from install_log import InstallLog


def test_successful_command_recorded_but_not_a_failure():
    log = InstallLog()
    log.record("pip install numpy", ok=True, returncode=0)
    assert log.failures == []
    assert not log.has_failures()


def test_failed_command_appears_in_failures():
    log = InstallLog()
    log.record("pip install chumpy", ok=False, returncode=1, stderr_tail="error: failed building wheel",
                category="WRONG_DEPENDENCY_CHUMPY")
    assert log.has_failures()
    assert len(log.failures) == 1
    assert log.failures[0].command == "pip install chumpy"
    assert log.failures[0].category == "WRONG_DEPENDENCY_CHUMPY"


def test_category_is_none_for_a_successful_command_even_if_passed():
    """Guards against accidentally tagging a category on a command that
    actually succeeded -- a category only ever means 'this failed because of X'."""
    log = InstallLog()
    log.record("pip install numpy", ok=True, returncode=0, category="SHOULD_BE_IGNORED")
    assert log.records[0].category is None


def test_stderr_tail_truncated_to_1500_chars():
    log = InstallLog()
    long_stderr = "x" * 5000
    log.record("pip install foo", ok=False, returncode=1, stderr_tail=long_stderr, category="INSTALL_FAILURE")
    assert len(log.failures[0].stderr_tail) == 1500
    assert log.failures[0].stderr_tail == long_stderr[-1500:]


def test_categories_returns_distinct_values_in_first_seen_order():
    log = InstallLog()
    log.record("cmd1", ok=False, category="A")
    log.record("cmd2", ok=False, category="B")
    log.record("cmd3", ok=False, category="A")  # duplicate, must not repeat
    log.record("cmd4", ok=True)
    assert log.categories() == ["A", "B"]


def test_categories_skips_failures_with_no_category():
    log = InstallLog()
    log.record("cmd1", ok=False, category=None)
    log.record("cmd2", ok=False, category="B")
    assert log.categories() == ["B"]


def test_summary_reports_only_failures_with_full_detail():
    log = InstallLog()
    log.record("pip install numpy", ok=True, returncode=0)
    log.record(
        "pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9'",
        ok=False, returncode=1, stderr_tail="error: subprocess-exited-with-error",
        category="SAM3D_CORE_DEPENDENCY_FAILURE",
    )
    summary = log.summary()
    assert len(summary) == 1
    assert summary[0]["command"].startswith("pip install 'git+")
    assert summary[0]["returncode"] == 1
    assert summary[0]["category"] == "SAM3D_CORE_DEPENDENCY_FAILURE"
    assert "subprocess-exited" in summary[0]["stderr_tail"]


def test_reproduces_the_task_03b_bug_scenario_now_fixed():
    """Task 03B's real Colab run had two observed build failures
    (chumpy, detectron2) but printed 'Recorded failure categories: none'.
    This test proves InstallLog would have surfaced both."""
    log = InstallLog()
    log.record("pip install ... chumpy ...", ok=False, returncode=1,
                stderr_tail="ERROR: Failed building wheel for chumpy",
                category="WRONG_DEPENDENCY_CHUMPY")
    log.record(
        "pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps",
        ok=False, returncode=1, stderr_tail="ERROR: Failed building wheel for detectron2",
        category="SAM3D_CORE_DEPENDENCY_FAILURE",
    )
    assert log.has_failures()
    assert log.categories() == ["WRONG_DEPENDENCY_CHUMPY", "SAM3D_CORE_DEPENDENCY_FAILURE"]
    assert len(log.summary()) == 2
