# =================================================================
# MODULE: tests/test_meter_grouping.py
# Pins the grouping rule in epistemic/referee.py's resolve_meter.
#
# THE DISTINCTION THIS DEFENDS. Duple-or-triple is a different KIND of question
# from how many beats go in a bar. 6/4 where 3/4 belongs is a barring choice
# and a reader can follow either; 6/4 where 4/4 belongs makes every bar after
# it wrong. Summed confidence cannot tell those apart, so the category is now
# settled by counting witnesses and only the bar length is settled by
# confidence.
#
# THE CASE THAT FOUND IT. On HRV two independent witnesses said 4/4 (0.333,
# 0.158) and one said 6/4 (0.877). 0.877 beat 0.491 and the page had been in
# 6/4 since 2026-08-06. Scored against the kit and the bass - the two parts
# that carry meter - HRV's onsets accent duple (drums 1.23 at g=2, bass 1.27 at
# g=4) and not triple (1.02 at g=3).
#
# The Hopeful case is here for the opposite reason: 3/4, 3/4 and 6/4 are ALL
# triple, so there is no category conflict and the rule must keep its hands off
# and let confidence pick 6/4.
# =================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epistemic.referee import resolve_meter  # noqa: E402


def test_a_duple_majority_is_not_outvoted_by_one_confident_triple():
    """HRV, exactly as measured."""
    r = resolve_meter([(4, 4, 0.333), (4, 4, 0.158), (6, 4, 0.877)])
    assert (r.numerator, r.denominator) == (4, 4)
    assert r.contention["grouping"]["chose"] == "duple"


def test_a_triple_majority_is_not_outvoted_by_one_confident_duple():
    r = resolve_meter([(3, 4, 0.2), (6, 4, 0.15), (4, 4, 0.9)])
    assert r.numerator % 3 == 0
    assert r.contention["grouping"]["chose"] == "triple"


def test_all_triple_witnesses_leave_the_bar_length_to_confidence():
    """Hopeful: 3/4, 3/4 and 6/4 are all triple, so there is nothing for the
    grouping rule to settle and the most confident reading must win."""
    r = resolve_meter([(3, 4, 0.136), (3, 4, 0.059), (6, 4, 0.533)])
    assert (r.numerator, r.denominator) == (6, 4)
    assert (r.contention or {}).get("grouping") is None


def test_all_duple_witnesses_are_left_alone():
    r = resolve_meter([(4, 4, 0.3), (2, 4, 0.8)])
    assert (r.numerator, r.denominator) == (2, 4)
    assert (r.contention or {}).get("grouping") is None


def test_an_even_split_falls_back_to_confidence():
    """One against one is not a majority. The old behaviour is the right
    default when the witnesses genuinely split."""
    assert resolve_meter([(4, 4, 0.9), (3, 4, 0.2)]).numerator == 4
    assert resolve_meter([(4, 4, 0.2), (3, 4, 0.9)]).numerator == 3


def test_grey_is_unchanged():
    """2/4 against 12/4 is one witness each, so confidence decides and 2/4
    stays - the reading its own run produced."""
    r = resolve_meter([(2, 4, 0.087), (12, 4, 0.025)])
    assert (r.numerator, r.denominator) == (2, 4)


def test_a_single_candidate_still_passes_through():
    r = resolve_meter([(4, 4, 0.9)])
    assert (r.numerator, r.denominator) == (4, 4)
    assert r.contention is None


def test_the_reason_is_recorded_for_the_trail():
    r = resolve_meter([(4, 4, 0.333), (4, 4, 0.158), (6, 4, 0.877)])
    g = r.contention["grouping"]
    assert g["duple_witnesses"] == 2 and g["triple_witnesses"] == 1
    assert "witness count" in g["why"]
