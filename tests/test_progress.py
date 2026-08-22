# =================================================================
# MODULE: tests/test_progress.py
# Pins app/progress.py.
#
# A progress indicator is the easiest thing in an application to make dishonest,
# and this pipeline makes it easy in a specific way: it runs for thirty to
# seventy minutes and emits about thirty events, almost none of them during the
# expensive stages. So the properties tested here are the ones that stop the
# panel lying - the clock must advance without any events at all, the bar must
# never claim completion it has not reached, and a long silence must be
# reported as work rather than looking like a hang.
# =================================================================

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.progress import (  # noqa: E402
    MILESTONES, OPTIONAL, QUIET_AFTER_SECONDS, STAGE_ORDER, RunProgress,
    format_elapsed,
)


def decision(stage, kind):
    return {"event_type": "decision", "stage": stage,
            "payload": {"decision_type": kind, "reasoning": "because"}}


# --- the clock, which is the load-bearing part ----------------------------

def test_the_clock_advances_with_no_events_at_all():
    """The whole point. During Demucs and Basic Pitch nothing is emitted for
    many minutes, and that is exactly when a user needs to see movement."""
    p = RunProgress()
    p.started_at = time.time() - 90
    assert p.elapsed_seconds >= 90
    assert "1m 30s" in p.headline() or "1m 3" in p.headline()


def test_silence_is_reported_as_work_not_as_a_hang():
    p = RunProgress()
    p.observe(decision("separation_engine", "separated"))
    p.last_event_at = time.time() - (QUIET_AFTER_SECONDS + 60)
    assert p.is_quiet
    assert "working" in p.headline()


def test_a_run_that_is_talking_is_not_reported_as_quiet():
    p = RunProgress()
    p.observe(decision("conductor", "session_start"))
    assert not p.is_quiet
    assert "working, nothing reported" not in p.headline()


def test_elapsed_formats_hours_when_a_run_is_long():
    assert format_elapsed(0) == "0m 00s"
    assert format_elapsed(94) == "1m 34s"
    assert format_elapsed(3 * 3600 + 5 * 60 + 7) == "3h 05m 07s"


# --- the bar, which must not overclaim ------------------------------------

def test_the_bar_starts_empty_and_never_exceeds_one():
    p = RunProgress()
    assert p.fraction == 0.0
    for stage, kind in MILESTONES:
        p.observe(decision(stage, kind))
    assert p.fraction <= 1.0


def test_optional_milestones_cannot_hold_the_bar_below_full():
    """A solo run skips Demucs, so its separation milestones never arrive.
    Counting them would cap a perfectly complete run below 100%."""
    p = RunProgress()
    for stage, kind in MILESTONES:
        if (stage, kind) in OPTIONAL:
            continue
        p.observe(decision(stage, kind))
    assert p.fraction == 1.0


def test_finishing_fills_the_bar_even_if_milestones_were_missed():
    p = RunProgress()
    p.observe(decision("conductor", "session_start"))
    p.observe({"event_type": "run_end", "stage": "conductor", "payload": {"ok": True}})
    assert p.finished
    assert p.fraction == 1.0
    assert "finished in" in p.headline()


def test_an_unknown_decision_does_not_inflate_the_count():
    """Milestones are a known list. A decision nobody catalogued is real work
    but must not be counted as progress toward a total it is not in."""
    p = RunProgress()
    before = p.milestones_reached
    p.observe(decision("quantization", "something_new_nobody_listed"))
    assert p.milestones_reached == before


# --- stages ---------------------------------------------------------------

def test_the_current_stage_follows_the_feed():
    p = RunProgress()
    p.observe(decision("separation_engine", "separated"))
    assert p.current_stage == "separation_engine"
    p.observe(decision("rhythm_engine", "meter_grid"))
    assert p.current_stage == "rhythm_engine"


def test_a_stage_that_speaks_again_is_not_left_marked_finished():
    """Stages interleave - key_intelligence appears three times in a real run -
    so 'finished' has to mean 'moved past for now', not 'done forever'."""
    p = RunProgress()
    p.observe(decision("key_intelligence", "key_detected"))
    p.observe(decision("quantization", "onsets_refined"))
    assert p.stages["key_intelligence"].finished
    p.observe(decision("key_intelligence", "key_stability"))
    assert not p.stages["key_intelligence"].finished


def test_every_stage_in_the_order_is_known_up_front():
    """The chips are laid out before a run starts, so the order cannot depend
    on what happens to arrive."""
    assert STAGE_ORDER[0] == "conductor"
    assert "separation_engine" in STAGE_ORDER
    assert "scribe_engraver" in STAGE_ORDER
    p = RunProgress()
    assert set(p.stages) == set(STAGE_ORDER)


def test_stage_elapsed_is_zero_before_it_starts():
    p = RunProgress()
    assert p.stages["epistemic"].elapsed == 0.0
