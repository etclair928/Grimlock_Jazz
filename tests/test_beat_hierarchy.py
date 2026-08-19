# =================================================================
# MODULE: tests/test_beat_hierarchy.py
# The user's specification, turned into assertions (2026-08-18).
#
# Each test names the rule it pins. The comparison table from the spec -
# "Flawed / Floating Rhythm" versus "Hierarchical / Standard Metric
# Alignment" - is reproduced case for case below, because that table is the
# acceptance criterion and it should be impossible to regress it silently.
# =================================================================

from __future__ import annotations

import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from output.beat_hierarchy import (  # noqa: E402
    LEVEL_BARLINE, LEVEL_BEAT, LEVEL_PRIMARY, LEVEL_SUBBEAT,
    position_level, primary_pulse_offsets, respects_hierarchy, split_for_hierarchy,
)


# --- Level 0: the barline is absolute -------------------------------------

def test_nothing_crosses_a_barline():
    """A half note starting on beat 4 of 4/4 runs into the next bar."""
    segs = split_for_hierarchy(F(3), F(2), 4, 4)
    assert segs == [(F(3), F(1)), (F(4), F(1))], segs


def test_a_tuplet_bracket_never_reaches_a_barline():
    """Tuplets are anchored to one beat upstream, so they cannot reach a
    boundary - and are never split, because a split bracket is exactly the
    un-notatable fragment the anchoring rule exists to prevent."""
    segs = split_for_hierarchy(F(3), F(1, 3), 4, 4, is_tuplet=True)
    assert len(segs) == 1


def test_long_notes_split_at_every_barline():
    segs = split_for_hierarchy(F(0), F(9), 4, 4)
    assert [s for s, _d in segs] == [F(0), F(4), F(8)]
    assert sum(d for _s, d in segs) == F(9)


# --- Level 1: the half measure is an invisible barline --------------------

def test_half_note_on_beat_two_is_split_to_keep_beat_three_visible():
    """The spec's headline case: a half note starting on beat 2 hides beat 3.
    It must read as quarter tied to quarter."""
    segs = split_for_hierarchy(F(1), F(2), 4, 4)
    assert segs == [(F(1), F(1)), (F(2), F(1))], segs


def test_dotted_half_on_beat_one_is_left_alone():
    """It crosses the half measure, but it STARTS on the barline, which is
    stronger - so it is allowed, and a dotted half on beat 1 is exactly what a
    copyist writes."""
    assert respects_hierarchy(F(0), F(3), 4, 4)


def test_half_note_on_beat_one_and_on_beat_three_are_both_fine():
    assert respects_hierarchy(F(0), F(2), 4, 4)
    assert respects_hierarchy(F(2), F(2), 4, 4)


# --- Level 3: durations complete their beat -------------------------------

def test_quarter_on_the_and_of_two_becomes_two_tied_eighths():
    """The spec's second case: a quarter starting at beat 2.5 sustaining to
    3.5 must read as an 8th tied to an 8th, keeping beat 3 visible."""
    segs = split_for_hierarchy(F(3, 2), F(1), 4, 4)
    assert segs == [(F(3, 2), F(1, 2)), (F(2), F(1, 2))], segs


def test_offbeat_eighth_inside_one_beat_is_left_alone():
    """Not everything off-beat is split - only what crosses a boundary."""
    assert respects_hierarchy(F(1, 2), F(1, 2), 4, 4)
    assert respects_hierarchy(F(5, 2), F(1, 2), 4, 4)


def test_every_split_is_contiguous_and_conserves_duration():
    for start, dur in ((F(1, 2), F(7, 2)), (F(3, 4), F(5)), (F(7, 2), F(1, 4)),
                       (F(0), F(16)), (F(5, 3), F(2))):
        segs = split_for_hierarchy(start, dur, 4, 4)
        assert segs[0][0] == start
        assert sum(d for _s, d in segs) == dur
        for (s0, d0), (s1, _d1) in zip(segs, segs[1:]):
            assert s0 + d0 == s1, (start, dur, segs)


# --- the rule of thumb, stated directly -----------------------------------

def test_no_untied_duration_cuts_through_a_beat_line():
    """A performer must be able to draw vertical lines through beats 1-4
    without cutting an un-tied duration. Exhaustive over a bar of 16ths."""
    for numer in (4, 3):
        for i in range(numer * 4):
            start = F(i, 4)
            for j in range(1, numer * 4 + 1):
                dur = F(j, 4)
                for seg_start, seg_dur in split_for_hierarchy(start, dur, numer, 4):
                    seg_end = seg_start + seg_dur
                    start_level = position_level(seg_start, numer, 4)
                    for k in range(int(seg_start) + 1, int(seg_end) + 1):
                        if F(k) < seg_end:
                            crossed = position_level(F(k), numer, 4)
                            assert crossed >= start_level, (
                                f"{numer}/4: segment {seg_start}+{seg_dur} crosses a "
                                f"level-{crossed} boundary at {k} from a level-"
                                f"{start_level} start")


# --- metric levels themselves ---------------------------------------------

def test_levels_in_common_time():
    assert position_level(F(0), 4, 4) == LEVEL_BARLINE
    assert position_level(F(2), 4, 4) == LEVEL_PRIMARY
    assert position_level(F(1), 4, 4) == LEVEL_BEAT
    assert position_level(F(3), 4, 4) == LEVEL_BEAT
    assert position_level(F(1, 2), 4, 4) == LEVEL_SUBBEAT


def test_compound_meters_group_in_threes():
    """6/8 is two dotted quarters, 9/8 three, 12/8 four - the group is the
    primary pulse, not the written eighth."""
    assert primary_pulse_offsets(6, 8) == (F(3),)
    assert primary_pulse_offsets(9, 8) == (F(3), F(6))
    assert primary_pulse_offsets(12, 8) == (F(3), F(6), F(9))
    assert position_level(F(3), 6, 8) == LEVEL_PRIMARY


def test_odd_meters_get_no_invented_half_measure():
    """3/4, 5/4 and 7/8 have no unambiguous half. Inventing one would put a
    boundary where no reader expects it; levels 0/2/3 still apply."""
    assert primary_pulse_offsets(3, 4) == ()
    assert primary_pulse_offsets(5, 4) == ()
    assert primary_pulse_offsets(7, 8) == ()
    # ...and a note still may not cross the barline in 7/8
    segs = split_for_hierarchy(F(6), F(3), 7, 8)
    assert [s for s, _d in segs] == [F(6), F(7)]


def test_six_eight_splits_at_the_dotted_quarter_group():
    """A note starting on the 2nd eighth and running past the group boundary
    hides the second dotted-quarter pulse."""
    segs = split_for_hierarchy(F(1), F(3), 6, 8)
    assert segs == [(F(1), F(2)), (F(3), F(1))], segs
