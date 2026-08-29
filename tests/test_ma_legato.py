# =================================================================
# MODULE: tests/test_ma_legato.py
# Pins output/ma_legato.py - the gate that decides whether a rest is real.
#
# WHY THIS NEEDS PINNING. The module can only ever LENGTHEN a note into a gap
# that already exists, so its failure mode is not a crash, it is a page that
# sustains through silence nobody performed. Every test here is therefore about
# a bound: silence still blocks, the hold tier is capped, and the middle case
# is left exactly as it was.
#
# TWO OF THESE EXIST BECAUSE OF SPECIFIC MISTAKES.
#
# `test_cap_is_read_at_call_time` guards the bug that made the hold tier look
# dead on arrival: `max_hold` was a default argument, so it bound MAX_HOLD_QL
# when the function was DEFINED and no swept value ever reached it. Every row
# of the first sweep silently used the same cap and produced identical output,
# which reads exactly like a tier that never fires.
#
# `test_middle_case_is_untouched` guards the three-way policy the module was
# built on: a witness that is only sometimes informative must not decide when
# it is not. The hold tier is a FOURTH outcome added above the others, not a
# replacement for them.
# =================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from output.ma_legato import (  # noqa: E402
    MaGate, RINGING_REACH, calibrate_from_notes,
)


def gate():
    """resonance 0.0-1.0 uniform, so ring_at=0.25 and hold_at=0.75."""
    res = [i / 1000.0 for i in range(1001)]
    void = [i / 1000.0 for i in range(1001)]
    return MaGate.calibrate(res, void)


BASE = 0.5


def test_calibration_places_the_three_thresholds():
    g = gate()
    assert abs(g.ring_at - 0.25) < 0.01
    assert abs(g.hold_at - 0.75) < 0.01
    assert g.void_at is not None


# ------------------------------------------------------------------- bounds

def test_genuine_silence_still_blocks_everything():
    """The one outcome that must never be weakened - a real rest stays."""
    g = gate()
    assert g.limit(0.99, 0.99, BASE, gap=8.0) == 0.0


def test_hold_tier_is_capped_not_unbounded():
    """A voice can be silent for 74 quarters on this material. 'Hold to the
    next attack' without a cap writes absurdities."""
    g = gate()
    assert g.limit(0.9, 0.0, BASE, gap=74.0, max_hold=2.0) == 2.0


def test_hold_tier_reaches_the_attack_when_it_is_close():
    g = gate()
    assert g.limit(0.9, 0.0, BASE, gap=1.5, max_hold=2.0) == 1.5


def test_hold_tier_never_returns_less_than_the_ringing_window():
    """A tiny gap must not make a strongly-ringing note reach LESS far than a
    merely-ringing one would."""
    g = gate()
    assert g.limit(0.9, 0.0, BASE, gap=0.1, max_hold=2.0) >= BASE * RINGING_REACH


def test_cap_is_read_at_call_time():
    """Regression: max_hold was a default argument bound at definition, so a
    swept constant never reached it and the tier looked inert."""
    import output.ma_legato as ma
    g = gate()
    before = ma.MAX_HOLD_QL
    try:
        ma.MAX_HOLD_QL = 3.0
        assert g.limit(0.9, 0.0, BASE, gap=9.0) == 3.0
    finally:
        ma.MAX_HOLD_QL = before


# ----------------------------------------------------------- unchanged path

def test_middle_case_is_untouched():
    g = gate()
    assert g.limit(0.10, 0.0, BASE, gap=8.0) == BASE


def test_ringing_but_not_strongly_keeps_the_old_window():
    g = gate()
    assert g.limit(0.50, 0.0, BASE, gap=8.0) == BASE * RINGING_REACH


def test_no_gap_means_the_old_behaviour_exactly():
    """Callers that do not supply a gap must behave as they did before the
    hold tier existed."""
    g = gate()
    assert g.limit(0.9, 0.0, BASE) == BASE * RINGING_REACH


def test_blind_gate_returns_the_base_window():
    g = MaGate()
    assert g.blind
    assert g.limit(0.9, 0.0, BASE, gap=8.0) == BASE


def test_too_few_samples_stays_blind():
    assert MaGate.calibrate([0.5] * 3, [0.5] * 3).blind


# -------------------------------------------------------------- fill values

def test_fill_value_never_lengthens():
    from output.musicxml_exporter import _fill_value_at_most
    for want in (0.3, 0.5, 1.25, 1.75, 2.5, 3.9):
        assert _fill_value_at_most(want) <= want + 1e-9


def test_fill_value_keeps_what_a_tie_can_render():
    """1.25 and 2.5 are not single note values, but split_for_hierarchy writes
    them as ordinary ties - rounding them to 1.0 and 2.0 threw reach away."""
    from output.musicxml_exporter import _fill_value_at_most
    assert _fill_value_at_most(1.25) == 1.0
    assert _fill_value_at_most(2.5) == 2.5
    assert _fill_value_at_most(1.75) == 1.5


def test_fill_value_of_nothing_is_nothing():
    from output.musicxml_exporter import _fill_value_at_most
    assert _fill_value_at_most(0.0) == 0.0
    assert _fill_value_at_most(-1.0) == 0.0


# ---------------------------------------------------------------- voicing
# _monophonic_share decides whether a part is a LINE. It is here rather than in
# its own file because it is the other half of the same problem: both voicers
# allocate by when a note ENDS, so sustaining a note harder (which ma_legato
# now does) makes it overlap the next one and buys a second voice that then
# rests through everything the first voice plays.

def _ev(start, end, pitch=60):
    from output.notation_score import NotationNote
    return [NotationNote(pitch=pitch, start_ms=start, end_ms=end, velocity=80,
                         source_note_id=f"n{start}")]


def test_a_clean_line_measures_fully_monophonic():
    from output.musicxml_exporter import _monophonic_share
    evs = [_ev(0, 100), _ev(100, 200), _ev(200, 300)]
    assert _monophonic_share(evs) == 1.0


def test_full_overlap_measures_polyphonic():
    from output.musicxml_exporter import _monophonic_share
    evs = [_ev(0, 200), _ev(0, 200, 64)]
    assert _monophonic_share(evs) < 0.55


def test_slight_overlap_still_reads_as_a_line():
    """A line whose notes ring a little into the next one is still a line -
    that is exactly the sustain this pass now writes, and it must not cost a
    second voice."""
    from output.musicxml_exporter import (_monophonic_share,
                                          MONOPHONIC_VOICE_SHARE)
    evs = [_ev(0, 110), _ev(100, 210), _ev(200, 310)]
    assert _monophonic_share(evs) >= MONOPHONIC_VOICE_SHARE


def test_sparseness_is_not_polyphony():
    """Long silences between notes must not make a part look polyphonic;
    silence is a different question and is measured over sounding time."""
    from output.musicxml_exporter import _monophonic_share
    evs = [_ev(0, 100), _ev(5000, 5100), _ev(9000, 9100)]
    assert _monophonic_share(evs) == 1.0


def test_collapse_keeps_every_event():
    """The collapse may re-voice, never discard - measured on both songs it
    changed the rest count by a third and the attack count by zero."""
    from output.musicxml_exporter import _collapse_to_one_voice
    evs = [_ev(200, 300), _ev(0, 100), _ev(100, 200)]
    out = _collapse_to_one_voice(evs)
    assert len(out) == 1
    assert len(out[0]) == 3
    starts = [min(n.start_ms for n in ev) for ev in out[0]]
    assert starts == sorted(starts)


def test_single_event_is_monophonic():
    from output.musicxml_exporter import _monophonic_share
    assert _monophonic_share([_ev(0, 100)]) == 1.0
    assert _monophonic_share([]) == 1.0
