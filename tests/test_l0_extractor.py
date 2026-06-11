"""Phase 7 tests: L0 regex extraction."""

from __future__ import annotations

from mnemo.extraction import extract_l0


def test_extract_email() -> None:
    facts = extract_l0("Reach me at alex@example.com thanks")
    assert len(facts) == 1
    assert facts[0].predicate == "email"
    assert facts[0].value == "alex@example.com"
    assert facts[0].write_level == 0


def test_extract_name_my_name_is() -> None:
    facts = extract_l0("Hi, my name is Alex Smith")
    names = [f for f in facts if f.predicate == "name"]
    assert len(names) == 1
    assert names[0].value == "Alex Smith"


def test_extract_location() -> None:
    facts = extract_l0("I moved to Pune last month")
    locs = [f for f in facts if f.predicate == "location"]
    assert len(locs) == 1
    assert locs[0].value == "Pune"


def test_extract_iso_date() -> None:
    facts = extract_l0("Meeting on 2026-06-04 confirmed")
    dates = [f for f in facts if f.predicate == "date"]
    assert len(dates) == 1
    assert dates[0].value == "2026-06-04"


def test_empty_text_returns_nothing() -> None:
    assert extract_l0("Hello, how are you?") == []


def test_i_am_happy_not_extracted_as_name() -> None:
    facts = extract_l0("I am happy today")
    assert all(f.predicate != "name" for f in facts)


def test_same_turn_name_correction_last_wins_in_extraction() -> None:
    facts = extract_l0("My name is Alex and my name is Bob")
    names = [f for f in facts if f.predicate == "name"]
    assert len(names) == 1
    assert names[0].value == "Bob"
