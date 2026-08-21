# =================================================================
# MODULE: app/diagnostics.py
# IS THIS TRANSCRIPTION TRUSTWORTHY? AS DATA, NOT AS PRINTED TEXT.
#
# WHY THIS EXISTS. The checks below were all written during one working
# session, as throwaway scripts that printed to stdout - and several were
# rewritten three or four times because nothing kept them. Worse, two of them
# printed numbers that later turned out to be measuring the wrong thing, and
# the printing is part of why: a number on a terminal cannot be compared with
# the same number from yesterday, so a regression is invisible until someone
# happens to look. This module returns them instead.
#
# TWO KINDS OF CHECK, AND THEY MUST NOT BE CONFLATED.
#
#   A HARD RULE either holds or does not. "Tuplets must never cross barlines.
#   NEVER" is a rule; so is "an onset sits on a legal metric position". These
#   render as PASS or FAIL and a FAIL is a defect, full stop.
#
#   A MEASUREMENT is a number that means nothing without a reference.
#   "Playability: 0.3% impossible" is not good or bad until you know the
#   published Chopin edition scores 0.1% under the same model. Every
#   measurement here therefore carries its own reference, in the same object.
#   A dashboard of bare numbers invites false confidence, which is the exact
#   failure this session kept running into.
#
# NOTHING HERE RE-DERIVES A RULE IT IS CHECKING. Each check calls the shipped
# function - find_octave_stacks, analyze_key_stability, playability.assess -
# rather than reimplementing it. An earlier version of the octave audit
# reimplemented its own rule and reported a number the shipped code does not
# produce; a tool that re-derives what it audits measures the tool.
# =================================================================

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

PASS = "pass"
FAIL = "fail"
INFO = "info"

# Positions inside a beat a notated onset may legally occupy. Anything else is
# an artifact - see the k/24 investigation, where onsets drifted onto k/24
# because a note had been written 1/24 too long and displaced everything after
# it.
LEGAL_ONSET_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16)

# The published edition's own score under the calibrated playability model.
# It is by definition playable, so this is the model's false-positive floor
# and the only sane reference for our own number.
EDITION_IMPOSSIBLE_FRACTION = 0.001


@dataclass(frozen=True)
class HealthCheck:
    """One check. `status` is PASS/FAIL for a rule, INFO for a measurement."""
    name: str
    status: str
    value: Any
    reference: Optional[Any] = None
    detail: str = ""

    @property
    def is_rule(self) -> bool:
        return self.status in (PASS, FAIL)


@dataclass
class HealthReport:
    source: str
    checks: List[HealthCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add(self, *args, **kwargs) -> None:
        self.checks.append(HealthCheck(*args, **kwargs))

    def get(self, name: str) -> Optional[HealthCheck]:
        for check in self.checks:
            if check.name == name:
                return check
        return None

    @property
    def failures(self) -> List[HealthCheck]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def rules_hold(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------
# score-side checks (need the engraved MusicXML)
# ---------------------------------------------------------------------

def _measures(score):
    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            bar = float(measure.barDuration.quarterLength)
            for voice in (list(measure.voices) or [measure]):
                yield part, measure, voice, bar


def check_barlines(score) -> Tuple[int, int, List[str]]:
    """Nothing may end past its barline. Notes and rests counted separately.

    THE SPLIT MATTERS. After the duration_witness ceiling fix, no NOTE crosses
    a barline on any song in the corpus - every survivor is a rest of exactly
    one bar's length starting mid-bar, which is music21's writer padding a
    short voice at serialization time. Reporting one number would hide a
    genuine result behind someone else's artifact.
    """
    notes = rests = 0
    examples: List[str] = []
    for part, measure, voice, bar in _measures(score):
        for el in voice.notesAndRests:
            over = float(el.offset) + float(el.quarterLength) - bar
            if over <= 1e-6:
                continue
            if el.isRest:
                rests += 1
            else:
                notes += 1
            if len(examples) < 5:
                examples.append(
                    f"{part.partName} m{measure.number} "
                    f"{'rest' if el.isRest else 'note'} "
                    f"off={float(el.offset):.4f} ql={float(el.quarterLength):.4f} "
                    f"over={over:.4f}")
    return notes, rests, examples


def check_onset_grid(score) -> Tuple[int, int, Dict[str, int]]:
    """Every notated onset must sit on a legal position within its beat."""
    total = 0
    offenders: Dict[str, int] = {}
    for _part, _measure, voice, _bar in _measures(score):
        for el in voice.notes:
            total += 1
            frac = Fraction(float(el.offset)).limit_denominator(96) % 1
            if frac.denominator not in LEGAL_ONSET_DENOMINATORS:
                offenders[str(frac)] = offenders.get(str(frac), 0) + 1
    return total, sum(offenders.values()), offenders


def check_tuplets(musicxml_path: str) -> Dict[str, Any]:
    """Ratio vocabulary and beat anchoring - DELEGATED, not re-derived.

    The first version of this function reimplemented the rules and reported 44
    off-beat groups on a page that has 9: it demanded a whole-beat anchor,
    which the published edition disproves (4:3 and 8:2 groups legitimately
    anchor to the second eighth of a beat), and it read offset-within-part
    where the rule is about beat-within-measure. tools/tuplet_audit.py already
    encodes the calibrated version, having had three bugs beaten out of it by
    being made to pass the edition first. Calling it is the whole point.
    """
    import os
    import sys
    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from tuplet_audit import measure_tuplets
    return measure_tuplets(musicxml_path)


# ---------------------------------------------------------------------
# note-side checks (need the run's .pkl)
# ---------------------------------------------------------------------

def check_key(notes) -> Dict[str, Any]:
    from key_intelligence.key_stability import analyze_key_stability
    st = analyze_key_stability(notes)
    return {"key": st.key, "raw_confidence": st.raw_confidence,
            "confidence": st.confidence, "agreement": st.agreement,
            "modulates": st.modulates,
            "windows": [k for _s, _e, k in st.windows],
            "cadence_key": st.cadence_key, "cadence_agrees": st.cadence_agrees,
            "detail": st.notes}


def check_playability(notes, tempo_bpm: float) -> Dict[str, Any]:
    import dataclasses
    from output.playability import PROFILES, assess
    pitched = [n for n in notes if n.stem.value != "drums"]
    if not pitched or tempo_bpm <= 0:
        return {"impossible_fraction": 0.0, "instants": 0, "leaps": 0}
    ms_per_beat = 60000.0 / tempo_bpm
    cfg = dataclasses.replace(PROFILES["virtuoso"], beats_per_second=tempo_bpm / 60.0)
    rep = assess([(n.start_ms / ms_per_beat, n.end_ms / ms_per_beat, int(n.pitch))
                  for n in pitched], cfg)
    bad = rep.sounding_instants - rep.playable_instants
    return {"impossible_fraction": bad / max(rep.sounding_instants, 1),
            "instants": rep.sounding_instants,
            "rolled": rep.rolled_instants, "leaps": len(rep.leaps)}


def check_octave_stacks(notes) -> Dict[str, Any]:
    from acoustic_witness.octave_stack import find_octave_stacks
    pitched = [n for n in notes if n.stem.value != "drums"]
    flagged = find_octave_stacks(pitched)
    return {"flagged": len(flagged), "pitched": len(pitched)}


def check_stems(notes) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for n in notes:
        counts[n.stem.value] = counts.get(n.stem.value, 0) + 1
    return counts


# ---------------------------------------------------------------------
# the whole panel
# ---------------------------------------------------------------------

def diagnose(pkl_path: Optional[str] = None,
             musicxml_path: Optional[str] = None) -> HealthReport:
    """Everything the health panel shows, for one run.

    Either path may be omitted; the checks that need it are skipped rather
    than guessed at. Failures to read are collected on the report instead of
    raised, so a panel can render what it has.
    """
    report = HealthReport(source=str(musicxml_path or pkl_path or "?"))

    notes: Sequence = ()
    tempo_bpm = 0.0
    if pkl_path and os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)
            notes = data.get("all_notes", ())
            tempo_bpm = float(data.get("tempo_bpm") or 0.0)
            report.add("tempo", INFO, tempo_bpm, detail="bpm, as resolved")
            report.add("time_signature", INFO, tuple(data.get("time_signature") or ()))
        except Exception as exc:
            report.errors.append(f"could not read {pkl_path}: {exc}")

    if notes:
        stems = check_stems(notes)
        report.add("notes_by_stem", INFO, stems,
                   detail="a solo recording showing drums/bass/vocals is stem bleed")
        report.add("total_notes", INFO, sum(stems.values()))

        key = check_key(notes)
        report.add("key", INFO, key["key"],
                   reference=f"agreement {key['agreement']:.0%}",
                   detail=key["detail"])
        report.add("key_confidence", INFO, round(key["confidence"], 3),
                   reference=round(key["raw_confidence"], 3),
                   detail="damped by how much of the span agrees; raw is undamped")
        report.add("key_windows", INFO, key["windows"],
                   detail="one reading per window, in order")

        play = check_playability(notes, tempo_bpm)
        report.add("playability_impossible", INFO,
                   round(play["impossible_fraction"], 4),
                   reference=EDITION_IMPOSSIBLE_FRACTION,
                   detail="fraction of struck instants no two hands can play; "
                          "the reference is the published edition's own score, "
                          "which is the model's false-positive floor")
        report.add("stride_leaps", INFO, play["leaps"],
                   detail="hard, NOT impossible - a property of the music")

        oct_stack = check_octave_stacks(notes)
        report.add("octave_stacks", INFO, oct_stack["flagged"],
                   detail="interior of a 3+-octave stack; only worth dropping "
                          "where the input carries stem bleed")

    if musicxml_path and os.path.exists(musicxml_path):
        try:
            from music21 import converter
            score = converter.parse(musicxml_path)
        except Exception as exc:
            report.errors.append(f"could not parse {musicxml_path}: {exc}")
            return report

        note_x, rest_x, examples = check_barlines(score)
        report.add("barline_crossings_notes", PASS if note_x == 0 else FAIL,
                   note_x, reference=0,
                   detail="HARD RULE: no rhythm may cross a barline; a held "
                          "note is written with a tie instead"
                          + (f" | e.g. {examples[0]}" if examples and note_x else ""))
        report.add("barline_crossings_rests", PASS if rest_x == 0 else INFO,
                   rest_x, reference=0,
                   detail="music21's writer pads short voices with a full-bar "
                          "rest at serialization; not our rhythm, and not "
                          "fixable before the file is written")

        total, off, offenders = check_onset_grid(score)
        report.add("offgrid_onsets", PASS if off == 0 else FAIL, off, reference=0,
                   detail=f"HARD RULE: onsets sit on legal beat positions "
                          f"({total} onsets checked). Measured on Chopin, what "
                          f"the engraver BUILDS is clean - 2326 onsets, none "
                          f"off-grid - and music21's serialization pass then "
                          f"adds a note and places it off-grid (2327, one at "
                          f"17/24). A count of 1 here is that artifact; a "
                          f"larger count is ours"
                          + (f" | {offenders}" if offenders else ""))

        tup = check_tuplets(musicxml_path)
        report.add("tuplet_notes", INFO, tup["tuplet_notes"],
                   detail=f"ratios: {tup['ratios']}")
        report.add("junk_tuplet_ratios", INFO, tup["junk_count"], reference=0,
                   detail=(f"un-notatable ratios: {tup['junk_ratios']}"
                           if tup["junk_ratios"] else "every ratio is notatable"))
        report.add("tuplets_off_beat", INFO, len(tup["unanchored"]), reference=0,
                   detail="a group must start on a binary subdivision - beat, "
                          "half-beat or quarter-beat. The edition anchors 4:3 "
                          "and 8:2 groups to the second eighth, so a whole-beat "
                          "rule would be stricter than the music")
        report.add("tuplet_barline_crossings",
                   PASS if not tup["crossings"] else FAIL, len(tup["crossings"]),
                   reference=0, detail="HARD RULE: a tuplet never crosses a barline")

    return report


__all__ = ["HealthReport", "HealthCheck", "diagnose", "PASS", "FAIL", "INFO",
           "check_barlines", "check_onset_grid", "check_tuplets", "check_key",
           "check_playability", "check_octave_stacks", "check_stems",
           "EDITION_IMPOSSIBLE_FRACTION"]
