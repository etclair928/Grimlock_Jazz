# =================================================================
# MODULE: tests/test_playability.py
# Pins output/playability.py.
#
# WHAT THESE ARE REALLY GUARDING. This model's whole value is that it can be
# WRONG in a checkable way: a published edition is by definition playable, so
# any false positive here is a false positive anything built on top inherits.
# The rewrite loosened the model in three places (strike instants instead of
# sounding sets, a 17-semitone default span, one permitted hand crossing),
# and every loosening is a chance to make the model unfalsifiable instead of
# accurate. So the refusals are tested as hard as the catches: a five-note
# cluster spanning a tenth must still fail, and free interleaving must still
# be rejected even though single crossing is allowed.
# =================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from output.playability import (  # noqa: E402
    IMPOSSIBLE, PLAYABLE, PROFILES, ROLLED, PlayabilityConfig, assess,
    assign_hands, build_strike_timeline, build_timeline, find_strain,
    hand_feasible, two_hand_feasible,
)

CFG = PlayabilityConfig()


# --- one hand --------------------------------------------------------------

def test_a_real_tenth_is_reachable():
    """C-E-G-C: one wide thumb-to-index gap, everything else close. This is
    the shape the inner-span rule exists to admit."""
    assert hand_feasible([60, 64, 67, 72], CFG)[0]


def test_a_shape_within_the_span_can_still_be_unreachable():
    """C-G#-E: a minor sixth between EVERY adjacent finger, total span 16st,
    comfortably inside the 17st bracket. Total span alone cannot tell this
    from a real tenth, which is exactly why total span alone was not enough."""
    ok, why = hand_feasible([60, 68, 76], CFG)
    assert not ok
    assert "inner gap" in why


def test_an_augmented_shape_is_reachable_and_must_stay_so():
    """C-E-G#-C, major thirds throughout. A large hand plays this; an
    inner-gap rule set too tight would reject real Liszt and Debussy."""
    assert hand_feasible([60, 64, 68, 72], CFG)[0]


def test_the_one_wide_gap_may_sit_anywhere_in_the_hand():
    """Dropping the LARGEST gap rather than assuming it is the first is what
    handles a hand whose stretch is at the top."""
    assert hand_feasible([60, 62, 64, 74], CFG)[0]     # stretch at the top
    assert hand_feasible([60, 70, 72, 74], CFG)[0]     # stretch at the bottom


def test_six_notes_is_never_one_hand():
    assert not hand_feasible([60, 61, 62, 63, 64, 65], CFG)[0]


def test_span_beyond_the_bracket_fails():
    ok, why = hand_feasible([60, 79], CFG)             # a nineteenth
    assert not ok and "span" in why


# --- two hands -------------------------------------------------------------

def test_an_ordinary_two_hand_chord():
    ok, split, _why = two_hand_feasible([48, 55, 64, 67, 72], CFG)
    assert ok and split is not None


def test_crossing_never_rescues_a_hand_from_its_own_span():
    """A crossed hand still has to span ITS OWN extremes. Left hand on a low
    bass and reaching over the right for a high melody note is a real gesture,
    but it is a SEQUENTIAL one - held simultaneously, that hand spans four
    octaves and no permission to cross can help it."""
    pitches = [36, 60, 62, 64, 88]
    assert not two_hand_feasible(pitches, PlayabilityConfig(max_crossings=0))[0]
    assert not two_hand_feasible(pitches, PlayabilityConfig(max_crossings=1))[0]


def test_crossing_can_only_ever_help_inside_one_hand_span():
    """The whole reach of the crossing relaxation: the outer hand takes both
    extremes, so it helps only when the FULL chord already fits one hand's
    span. That is a narrow window, and saying so is the point - it is why
    free interleaving was not worth the loss of falsifiability."""
    wide = [40, 60, 62, 64, 84]
    ok_free = two_hand_feasible(wide, PlayabilityConfig(max_crossings=1))[0]
    assert not ok_free


def test_no_hand_is_ever_more_than_two_runs():
    """The bound itself. Free subset assignment would wave through almost any
    ten notes inside three octaves; this asserts the search never offers a
    hand three separate registers to hold at once."""
    from output.playability import _splits
    n = 7
    for a_idx, b_idx in _splits(n, 1):
        for group in (a_idx, b_idx):
            runs = 1 if group else 0
            for x, y in zip(group, group[1:]):
                if y != x + 1:
                    runs += 1
            assert runs <= 2, group


def test_more_than_ten_notes_is_never_playable():
    assert not two_hand_feasible(list(range(60, 72)), CFG)[0]


def test_an_uncrossed_reading_is_preferred_when_one_exists():
    ok, hands, why = assign_hands([48, 52, 55, 67, 71, 74], CFG)
    assert ok and why == "ok"
    assert max(hands[0]) < min(hands[1])


def test_silence_is_playable():
    assert two_hand_feasible([], CFG)[0]


# --- strikes vs sounding: the pedal argument -------------------------------

def test_a_pedalled_arpeggio_is_playable_struck_and_impossible_sounding():
    """THE reason the module was rewritten. Struck one at a time across four
    octaves and left ringing: ordinary under pedal, and the sounding sweep
    calls it impossible."""
    notes = [(0.0, 8.0, 36), (1.0, 8.0, 48), (2.0, 8.0, 60),
             (3.0, 8.0, 72), (4.0, 8.0, 84)]
    struck = build_strike_timeline(notes, CFG)
    assert all(i.playable for i in struck)

    sounding = build_timeline(notes, CFG)
    assert any(not i.playable for i in sounding)


def test_the_same_notes_struck_together_are_impossible():
    """The converse, so the test above is not just proving the check is off."""
    notes = [(0.0, 8.0, p) for p in (36, 48, 60, 72, 84)]
    struck = build_strike_timeline(notes, CFG)
    assert any(i.verdict == IMPOSSIBLE for i in struck)


def test_a_rolled_chord_reads_as_rolled_not_impossible():
    """Inside one gesture but not coincident: an arpeggiando, held by the
    pedal, which is how a Rachmaninoff spread is actually played."""
    notes = [(0.00, 4.0, 36), (0.03, 4.0, 55), (0.06, 4.0, 64), (0.09, 4.0, 91)]
    timeline = build_strike_timeline(notes, CFG)
    assert len(timeline) == 1
    assert timeline[0].verdict == ROLLED
    assert timeline[0].playable


def test_grouping_is_anchored_so_a_run_does_not_become_one_chord():
    """Anchoring on the previous onset instead of the group's first would let
    a scale collapse into a single impossible 'instant'."""
    notes = [(i * 0.1, 4.0, 60 + i) for i in range(12)]
    timeline = build_strike_timeline(notes, CFG)
    assert len(timeline) > 1
    assert all(i.playable for i in timeline)


# --- difficulty is not impossibility ---------------------------------------

def test_a_stride_leap_is_strain_and_never_fails_an_instant():
    """Low root on 1, mid chord on 2 - two octaves of travel. Art Tatum is
    hard, not impossible, and a model that failed him is measuring the wrong
    thing."""
    notes = [(0.0, 0.4, 36), (0.0, 0.4, 48),
             (0.5, 0.9, 64), (0.5, 0.9, 68), (0.5, 0.9, 71)]
    timeline = build_strike_timeline(notes, CFG)
    assert all(i.playable for i in timeline)
    strain = find_strain(timeline, CFG)
    assert any(s.kind == "leap" for s in strain)


def test_strain_stays_out_of_the_pass_rate():
    notes = [(0.0, 0.4, 36), (0.0, 0.4, 48),
             (0.5, 0.9, 64), (0.5, 0.9, 68), (0.5, 0.9, 71)]
    rep = assess(notes, CFG)
    assert rep.time_pass_rate == 1.0
    assert rep.leaps


def test_repetition_is_reported_and_can_actually_fire():
    """It is gated on tempo, so it is asserted at a tempo where it must -
    the previous version's gate was unsatisfiable and the rule never fired
    on any source at all."""
    fast = PlayabilityConfig(beats_per_second=4.0, max_repetition_hz=12.0)
    notes = [(i * 0.25, i * 0.25 + 0.2, 60) for i in range(8)]
    rep = assess(notes, fast)
    assert rep.repetitions


# --- the acceptance property -----------------------------------------------

def test_the_default_config_is_the_calibrated_bracket():
    """The default has to be the setting where a positive means something.
    Calibration puts the published edition at 1.4% false positives under a
    14st span and 0.1% under 17st."""
    assert PlayabilityConfig().max_span_semitones == 17
    assert PROFILES["comfortable"].max_span_semitones < 17


def test_tighter_profiles_are_never_more_permissive():
    """Monotonicity - a comfort bracket that admitted something the virtuoso
    bracket rejected would mean the profiles are not a bracket at all."""
    chord = [48, 52, 55, 64, 67, 72]
    if two_hand_feasible(chord, PROFILES["comfortable"])[0]:
        assert two_hand_feasible(chord, PROFILES["virtuoso"])[0]


def test_report_shape_survives_for_existing_callers():
    """tools/playability_report.py and tools/engraving_experiments.py read
    these by name."""
    notes = [(0.0, 1.0, 60), (0.0, 1.0, 64), (1.0, 2.0, 67)]
    rep = assess(notes)
    for attr in ("time_pass_rate", "instant_pass_rate", "mean_simultaneity",
                 "max_simultaneity", "max_span", "sounding_beats",
                 "playable_beats", "unplayable_beats_finger_count",
                 "unplayable_beats_span_or_split", "excess_notehead_beats",
                 "simultaneity_beats", "worst"):
        assert hasattr(rep, attr), attr


def test_empty_input_is_not_an_error():
    rep = assess([])
    assert rep.time_pass_rate == 1.0
    assert rep.sounding_instants == 0
