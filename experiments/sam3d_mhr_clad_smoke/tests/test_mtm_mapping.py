"""Tests for mtm_mapping.py -- structural checks on our own terminology map.

These check the map is internally consistent (no duplicate names, every
entry has a note, confidence values are from the allowed set) -- not a
claim about measurement accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mtm_mapping import MTM_MEASUREMENT_MAP, as_lookup

EXPECTED_MTM_MEASUREMENTS = {
    "height",
    "chest / bust circumference",
    "waist circumference",
    "seat / hip circumference",
    "shoulder width",
    "sleeve / arm length",
    "upper-arm / bicep circumference",
    "neck circumference",
    "back length",
    "front torso length",
    "wrist circumference",
    "inseam",
    "outseam",
    "thigh circumference",
}


def test_covers_all_task_required_measurements():
    names = {m.mtm_name for m in MTM_MEASUREMENT_MAP}
    assert names == EXPECTED_MTM_MEASUREMENTS


def test_no_duplicate_mtm_names():
    names = [m.mtm_name for m in MTM_MEASUREMENT_MAP]
    assert len(names) == len(set(names))


def test_every_entry_has_a_nonempty_note():
    for m in MTM_MEASUREMENT_MAP:
        assert m.note and len(m.note.strip()) > 0, f"{m.mtm_name} has an empty note"


def test_confidence_is_from_allowed_set():
    for m in MTM_MEASUREMENT_MAP:
        assert m.confidence in {"direct", "review", "gap"}, m.mtm_name


def test_gap_entries_have_no_clad_body_key():
    for m in MTM_MEASUREMENT_MAP:
        if m.confidence == "gap":
            assert m.clad_body_key is None, f"{m.mtm_name} is marked gap but has a key"


def test_non_gap_entries_have_a_clad_body_key():
    for m in MTM_MEASUREMENT_MAP:
        if m.confidence != "gap":
            assert m.clad_body_key is not None, f"{m.mtm_name} is not gap but has no key"


def test_as_lookup_round_trips():
    lookup = as_lookup()
    assert len(lookup) == len(MTM_MEASUREMENT_MAP)
    for m in MTM_MEASUREMENT_MAP:
        assert lookup[m.mtm_name] is m
