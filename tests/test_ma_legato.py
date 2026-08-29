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
