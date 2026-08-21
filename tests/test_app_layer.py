# =================================================================
# MODULE: tests/test_app_layer.py
# Pins app/diagnostics.py and app/probe.py - the front end's library layer.
#
# THE PROPERTY THAT MATTERS MOST HERE IS AGREEMENT. These modules exist to be
# the single place a health number comes from, replacing scratch scripts that
# were rewritten several times in one session. That is worth nothing if they
# quietly disagree with the shipped rule they claim to report: the first draft
# of check_tuplets reimplemented the anchoring rule and found 44 defects on a
# page with 9, because it demanded a whole-beat anchor the published edition
# disproves. So the tests below assert DELEGATION, not just plausible output.
# =================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))

from app.diagnostics import (  # noqa: E402
    FAIL, INFO, PASS, HealthCheck, HealthReport, check_tuplets, diagnose,
)


# --- the report container -------------------------------------------------

def test_a_failed_rule_makes_the_report_fail():
    r = HealthReport(source="x")
    r.add("barline_crossings_notes", PASS, 0, reference=0)
    assert r.rules_hold
    r.add("offgrid_onsets", FAIL, 7, reference=0)
    assert not r.rules_hold
    assert [c.name for c in r.failures] == ["offgrid_onsets"]


def test_a_measurement_never_fails_the_report():
    """INFO checks are numbers needing a reference, not rules. A junk-ratio
    count of 28 is bad news, but it is not a broken invariant, and conflating
    the two would make `rules_hold` meaningless."""
    r = HealthReport(source="x")
    r.add("junk_tuplet_ratios", INFO, 28, reference=0)
    assert r.rules_hold


def test_measurements_carry_their_reference():
    """A bare number invites false confidence: 0.3% impossible means nothing
    without the edition's 0.1% beside it."""
    from app.diagnostics import EDITION_IMPOSSIBLE_FRACTION
    assert EDITION_IMPOSSIBLE_FRACTION > 0
    c = HealthCheck("playability_impossible", INFO, 0.003,
                    reference=EDITION_IMPOSSIBLE_FRACTION)
    assert c.reference is not None
    assert not c.is_rule


def test_rule_checks_are_distinguishable_from_measurements():
    assert HealthCheck("r", PASS, 0).is_rule
    assert HealthCheck("r", FAIL, 1).is_rule
    assert not HealthCheck("m", INFO, 1).is_rule


# --- delegation, the load-bearing property --------------------------------

def _page():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("Chopin_Solo_FULLRUN", "Hopeful_FULLRUN", "HRV_FULLRUN"):
        path = os.path.join(here, "transcriptions", f"{name}.musicxml")
        if os.path.exists(path):
            return path
    return None


def test_tuplet_check_agrees_with_the_audit_it_delegates_to():
    """Not 'returns something plausible' - returns THE SAME THING. Two tools
    disagreeing about one file is the failure this layer exists to prevent."""
    page = _page()
    if page is None:
        return                                   # no engraved page available
    from tuplet_audit import measure_tuplets
    assert check_tuplets(page) == measure_tuplets(page)


def test_the_audit_cli_and_its_measuring_core_cannot_drift():
    """`audit` prints; `measure_tuplets` returns. The printer must have no
    rules of its own - it was split precisely so there is one implementation."""
    import inspect
    import tuplet_audit
    body = inspect.getsource(tuplet_audit.audit)
    assert "measure_tuplets(" in body
    for rule_token in ("_LEGAL_NORMALS", "denominator & ", "barDuration"):
        assert rule_token not in body, f"{rule_token} is a RULE and belongs in measure_tuplets"


# --- diagnose end to end --------------------------------------------------

def test_diagnose_survives_missing_inputs():
    r = diagnose(None, None)
    assert r.checks == [] and r.rules_hold


def test_diagnose_collects_read_errors_instead_of_raising():
    r = diagnose("nope_does_not_exist.pkl", "nope_does_not_exist.musicxml")
    assert r.rules_hold
    assert r.checks == []


def test_diagnose_reports_the_hard_rules_when_given_a_page():
    page = _page()
    if page is None:
        return
    r = diagnose(None, page)
    for rule in ("barline_crossings_notes", "offgrid_onsets"):
        check = r.get(rule)
        assert check is not None, rule
        assert check.is_rule, f"{rule} must be PASS/FAIL, not a bare number"


# --- probe ----------------------------------------------------------------

def test_the_plan_says_what_a_run_will_do_under_an_override():
    """Guided mode is a hard lock, and the plan text has to reflect that -
    a user who overrode the probe should not be told what the probe thought."""
    from app.probe import ProbeResult
    from separation_engine import ENSEMBLE, SOLO_PIANO
    r = ProbeResult(audio_path="x.wav", duration_seconds=10.0, solo=None)
    assert "SKIPPED" in r.plan(SOLO_PIANO)
    assert "not consulted" in r.plan(SOLO_PIANO)
    assert "WILL run" in r.plan(ENSEMBLE)


def test_the_plan_without_an_override_reports_the_probe():
    from app.probe import ProbeResult
    from separation_engine import SoloVerdict, SOLO_PIANO, ENSEMBLE
    solo = SoloVerdict(SOLO_PIANO, 0.8, 0.01, 0.1, 0.02, "reason", 0.65)
    r = ProbeResult(audio_path="x.wav", duration_seconds=10.0, solo=solo)
    assert r.will_skip_separation
    assert "SKIPPED" in r.plan()

    band = SoloVerdict(ENSEMBLE, 0.8, 0.2, 0.4, 0.1, "reason", 0.0)
    r2 = ProbeResult(audio_path="x.wav", duration_seconds=10.0, solo=band)
    assert not r2.will_skip_separation
    assert r2.planned_stem is None
    assert "WILL run" in r2.plan()
