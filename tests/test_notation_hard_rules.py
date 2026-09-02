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


# ---------------------------------------------------------------- grid cap
# MAX_SUBDIVISION refuses a grid finer than the material supports. Measured on
# Hubristic against Basic Pitch's own read of the same audio - same detector,
# so every difference is our notation layer - capping at an eighth took
# on-beat placement from 46.9% to 73.0% (Basic Pitch: 59.7%), sixteenth
# positions from 26.2% to 0.0%, rests from 29.3% to 20.4% (Basic Pitch: 20.3%)
# and illegal tuplets from 5 to 0, while ATTACKS were preserved (1891 -> 1894)
# and voice quality was untouched (mean leap 8.08 -> 8.02).

def test_polish_is_on_by_default():
    """Measured across six songs it cut rests by a third to a half and raised
    on-beat placement 9-23 points while attacks stayed put, so it ships on."""
    import output.musicxml_exporter as mx
    assert mx.MAX_SUBDIVISION == 2
    assert mx.POLISH_SUBDIVISION_CAP == 2


def test_cap_limits_the_binary_grid():
    import output.musicxml_exporter as mx
    before = mx.MAX_SUBDIVISION
    try:
        mx.MAX_SUBDIVISION = 2
        # Under an eighth cap every value must land on a multiple of 0.5.
        # 0.75 is a sixteenth-grid value and sits exactly between two eighth
        # points, so it rounds to 1.0 - the assertion is that it leaves the
        # sixteenth grid, not that it rounds any particular way.
        for want in (0.3, 0.75, 1.25, 1.75):
            got = mx._snap_quarter_length(want, False, None, 4)
            assert abs(got * 2 - round(got * 2)) < 1e-9, (want, got)
        mx.MAX_SUBDIVISION = None      # unpolished
        assert mx._snap_quarter_length(0.75, False, None, 4) == 0.75
    finally:
        mx.MAX_SUBDIVISION = before


def test_cap_below_three_also_suppresses_tuplets():
    """A triplet is a finer division than an eighth cap allows. Capping the
    binary grid alone let the divisor path route around the ceiling and LEFT
    MORE onsets on triplet positions than before the cap."""
    import output.musicxml_exporter as mx
    before = mx.MAX_SUBDIVISION
    try:
        mx.MAX_SUBDIVISION = 2
        capped = mx._snap_quarter_length(1.0 / 3.0, True, 3, 2)
        assert abs(capped * 2 - round(capped * 2)) < 1e-9, "must land on the eighth grid"
    finally:
        mx.MAX_SUBDIVISION = before


# ------------------------------------------------------- polish is gated
# Polish is right for material with a kit and WRONG without one. On the Chopin
# Nocturne an eighth ceiling took the page from 34.5% sixteenths to 0.5% - real
# passagework rewritten as eighths - while on Clocks it took 6.1% to 0.3% and
# landed on a human transcription that writes none at all.

class _FakePart:
    def __init__(self, is_drum): self.is_drum = is_drum


class _FakeScore:
    def __init__(self, *drums): self.parts = [_FakePart(d) for d in drums]


def test_polish_runs_when_the_score_has_a_kit():
    import output.musicxml_exporter as mx
    assert mx.polish_cap_for(_FakeScore(False, False, True)) == mx.MAX_SUBDIVISION


def test_polish_stands_down_without_percussion():
    """Solo piano keeps its own subdivision - its sixteenths are the music."""
    import output.musicxml_exporter as mx
    assert mx.polish_cap_for(_FakeScore(False, False)) is None


def test_the_global_switch_still_wins():
    """Turning Polish off entirely must not be overridden by the gate."""
    import output.musicxml_exporter as mx
    before = mx.MAX_SUBDIVISION
    try:
        mx.MAX_SUBDIVISION = None
        assert mx.polish_cap_for(_FakeScore(True)) is None
    finally:
        mx.MAX_SUBDIVISION = before


def test_an_empty_score_does_not_crash_the_gate():
    import output.musicxml_exporter as mx
    assert mx.polish_cap_for(_FakeScore()) == mx.MAX_SUBDIVISION
    assert mx.polish_cap_for(object()) == mx.MAX_SUBDIVISION


# ------------------------------------------------- the writer cannot be broken
# A page that will not OPEN is worse than any notation choice inside it. The
# Chopin intermediate carried a note music21 read as 18-in-the-time-of-13 with
# quarterLength 13/36, ending 5/36 short of the next onset - a gap nothing can
# write - so the MusicXML writer emitted a "2048th" and raised
# MusicXMLExportException, refusing the ENTIRE FILE. Polish had been masking it
# by snapping everything onto the eighth grid; standing Polish down for solo
# piano re-exposed it on exactly the repertoire that needs it.

def test_an_illegal_tuplet_is_not_writable():
    from music21 import note as m21note, duration as m21duration
    from output.musicxml_exporter import _writable_frame_marker
    n = m21note.Note()
    n.duration = m21duration.Duration(0.5)
    n.duration.appendTuplet(m21duration.Tuplet(18, 13))
    assert not _writable_frame_marker(n)


def test_the_safety_net_repairs_what_makenotation_invents():
    """It runs AFTER makeNotation because that is the pass that invents these -
    the same one that turned an unfilled triplet slot into a 32nd rest."""
    from music21 import stream, note as m21note, duration as m21duration
    from output.musicxml_exporter import _make_part_writable, _writable_frame_marker
    p = stream.Part()
    good = m21note.Note(); good.quarterLength = 0.5
    bad = m21note.Note()
    bad.duration = m21duration.Duration(13.0 / 36.0)
    bad.duration.appendTuplet(m21duration.Tuplet(18, 13))
    p.append(good); p.append(bad)
    fixed = _make_part_writable(p)
    assert fixed >= 1
    for el in p.recurse().notesAndRests:
        assert _writable_frame_marker(el), (el.duration.type, el.duration.tuplets)


def test_the_safety_net_leaves_a_clean_part_alone():
    from music21 import stream, note as m21note
    from output.musicxml_exporter import _make_part_writable
    p = stream.Part()
    for ql in (1.0, 0.5, 0.25, 2.0):
        n = m21note.Note(); n.quarterLength = ql; p.append(n)
    assert _make_part_writable(p) == 0
