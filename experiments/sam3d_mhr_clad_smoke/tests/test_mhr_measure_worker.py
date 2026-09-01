"""Tests for _mhr_measure_worker.parse_last_stage -- pure string parsing,
no clad-body/pymomentum/torch import required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from _mhr_measure_worker import parse_last_stage


def test_no_marker_returns_none():
    assert parse_last_stage("some traceback\nwith no markers\n") is None


def test_empty_string_returns_none():
    assert parse_last_stage("") is None


def test_none_input_returns_none():
    assert parse_last_stage(None) is None


def test_single_marker_is_returned():
    text = "STAGE=mhr_reconstruction\nTraceback...\nSegmentation fault\n"
    assert parse_last_stage(text) == "mhr_reconstruction"


def test_last_marker_wins_when_multiple_present():
    text = (
        "STAGE=mhr_reconstruction\n"
        "some load output\n"
        "STAGE=rescale\n"
        "STAGE=clad_body_measure\n"
        "Traceback (most recent call last):\n"
        "  ValueError: bad key\n"
    )
    assert parse_last_stage(text) == "clad_body_measure"


def test_marker_with_surrounding_whitespace_is_stripped():
    text = "  STAGE=mhr_reconstruction  \n"
    assert parse_last_stage(text) == "mhr_reconstruction"
