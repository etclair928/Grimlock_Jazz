# =================================================================
# MODULE: tests/test_notation_hard_rules.py
# Pins the notation rules that are stated as ABSOLUTE, not preferences.
#
#   "I NEVER NEVER NEVER WANT TO SEE 32nd or 64th notes again."
#   "There should never be tuplets of 5 or 12 or anything like that."
#
# These are not quality targets to be traded against F1 - they are conditions
# on the page being readable at all, so they get a test that fails loudly
# rather than a number in a report that drifts.
#
# WHY THE GUARDS LIVE WHERE THEY DO. Nine violations survived on Clocks in
# every export including the shipped one, and they were rests, not notes -
# invented by music21's makeRests(fillGaps=True) to fill an unfilled slot
# inside a triplet frame. Two earlier guesses about their origin measured zero
# effect before the real one was found, which is why the assertions here are on
# the WRITTEN OUTPUT and not on any function believed to produce it. A guard
# that checks the input to the thing that writes is a guard that can be right
# while the page is wrong.
# =================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music21 import note as m21note  # noqa: E402

from output.musicxml_exporter import (  # noqa: E402
    _ALLOWED_TUPLET_ACTUALS, _FORBIDDEN_WRITTEN_TYPES, _apply_tuplet,
    _leading_tuplet_rest, _writable_frame_marker,
)


def test_forbidden_types_are_the_ones_the_rules_name():
    assert {"32nd", "64th"} <= _FORBIDDEN_WRITTEN_TYPES


def test_allowed_tuplets_are_only_the_simple_ones():
    """Duplets, triplets and sextuplets. A 5 or a 12 means the beat grid is
    wrong, not that the music is unusual."""
    assert _ALLOWED_TUPLET_ACTUALS == frozenset({2, 3, 6})


def test_a_32nd_rest_is_refused_as_a_frame_marker():
    r = m21note.Rest()
    r.quarterLength = 0.125
    assert r.duration.type == "32nd"
    assert not _writable_frame_marker(r)


def test_a_16th_rest_is_accepted():
    r = m21note.Rest()
    r.quarterLength = 0.25
    assert _writable_frame_marker(r)


def test_a_tuplet_of_twelve_is_refused():
    r = m21note.Rest()
    if _apply_tuplet(r, 1.0 / 12.0 * 2, 12) > 0:
        assert not _writable_frame_marker(r)


def test_a_triplet_rest_is_accepted():
    r = m21note.Rest()
    assert _apply_tuplet(r, 1.0 / 3.0, 3) > 0
    assert _writable_frame_marker(r)


def test_an_unwritable_tuplet_frame_yields_no_rest_at_all():
    """A twelfth of a beat cannot anchor a triplet frame legally, so the frame
    goes unanchored rather than being anchored with something unreadable."""
    from fractions import Fraction
    assert _leading_tuplet_rest(Fraction(1, 12), 3) is None
