# =================================================================
# MODULE: university/apply.py
# What APPLY mode actually does (GRIMLOCK_UNIVERSITY.md §4, amended by
# the user 2026-08-01: "turned on it should help transcribe better or
# overwrite things/patterns it sees").
#
# THE LINE THIS DOES NOT CROSS: a frozen Note is still never mutated,
# never replaced, never re-created. "Overwrite" happens on the NOTATION
# CLOCK only - the page. The performance clock (MIDI, playback) is
# untouched, exactly as the three-timeline law requires. That is what
# keeps 6.0's playback faithful while the page gets smarter, and it is
# why APPLY is reversible: delete the annotations and the page returns.
#
# Two actions, both written as Annotations that the notation view may
# honor:
#
#   1. PAGE SUPPRESS (the real transcription win)
#      A `rearticulation` run - one pitch re-struck several times inside
#      a fraction of a beat - is the measured signature of Basic Pitch
#      fragmenting ONE sustained note (52% of adjacent notes in a voice
#      were same-pitch on the produced page). When the fragments are
#      also LOW CONFIDENCE, the page keeps the first strike, extends it
#      over the run, and suppresses the echoes. Mirrors
#      note_consolidation's primary/absorbed shape deliberately - and
#      only ever sees notes consolidation did NOT already absorb, so the
#      two are complementary, never fighting.
#      The AND-gate (pattern + low confidence) is the precision guard,
#      the same discipline note_support uses.
#
#   2. VOICE COHESION (the page-layout win)
#      Notes inside one scale run / arpeggio / sequence / Alberti figure
#      are ONE musical gesture and should not be scattered across voices.
#      Recorded as a cohesion group id the voicer can honor.
#
# Real tremolo exists, and a real repeated note exists. That is exactly
# why suppression requires low confidence as well as the pattern, and
# why every suppressed note keeps a reason string and stays reversible.
# =================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from university.observation_types import (
    PAGE_SUPPRESS_ANNOTATION_KIND, PATTERN_STUDY_ANNOTATION_KIND,
    VOICE_COHESION_ANNOTATION_KIND, PatternObservation, StudyReport,
)

# A fragment must be at least this weak before the page will drop it.
# Conservative on purpose: a confident repeated note is a real repeated
# note, and the cost of deleting real music is far higher than the cost
# of leaving a stutter on the page.
SUPPRESS_MAX_CONFIDENCE = 0.60

# Patterns whose notes are one gesture and should share a voice.
# ONLY unanimously-graduated detectors may alter the page. `sequence`,
# `gap_fill` and `chord` are studied and logged, but graded NOISE on at
# least one song, so they do not get to move a notehead - the §7 rule
# applied where it actually costs something.
from university.detectors import GRADUATED as _GRADUATED
_COHESIVE_PATTERNS = tuple(p for p in _GRADUATED if p != "ostinato") + ("alberti_bass",)


def _add(annotations, note_id: str, kind: str, value: Dict[str, Any],
         confidence: float = 1.0) -> None:
    from core import Annotation, Provenance
    source = getattr(Provenance, "GRIMLOCK_UNIVERSITY", None) or Provenance.CHECK
    annotations.add(Annotation(note_id=note_id, kind=kind, value=value,
                               source=source, confidence=confidence))


def write_study_annotations(report: StudyReport, annotations) -> int:
    """STUDY mode's whole footprint: one `pattern_study` annotation per
    note per observation it belongs to. Evidence, nothing more - no
    consumer reads this kind, so output is unchanged."""
    written = 0
    for obs in report.observations:
        payload = {
            "pattern": obs.pattern, "department": obs.department,
            "confidence": obs.confidence, "size": obs.size,
            "evidence": obs.evidence, "falsifier": obs.falsifier,
        }
        for note_id in obs.note_ids:
            _add(annotations, note_id, PATTERN_STUDY_ANNOTATION_KIND,
                 payload, confidence=obs.confidence)
            written += 1
    return written


def apply_observations(
        report: StudyReport,
        annotations,
        notes_by_id: Dict[str, Any],
        suppress_max_confidence: float = SUPPRESS_MAX_CONFIDENCE,
) -> Dict[str, int]:
    """APPLY mode. Writes the two actionable annotation kinds and returns
    counts for the MusicBox trail. Still read-only w.r.t. Notes."""
    suppressed = 0
    primaries = 0
    cohesive = 0

    # ---- 1. page suppression of low-confidence rearticulation echoes ----
    for obs in report.observations:
        if obs.pattern != "rearticulation" or obs.size < 2:
            continue
        members = [notes_by_id.get(nid) for nid in obs.note_ids]
        members = [m for m in members if m is not None]
        if len(members) < 2:
            continue
        members.sort(key=lambda n: n.start_ms)
        primary, echoes = members[0], members[1:]

        weak = [e for e in echoes
                if float(getattr(e, "confidence", 1.0)) <= suppress_max_confidence]
        if not weak:
            continue                       # confident repeats are real music - leave them

        # The page keeps ONE note spanning what the ear hears as one sound.
        end_ms = max(float(m.end_ms) for m in members)
        _add(annotations, primary.id, PAGE_SUPPRESS_ANNOTATION_KIND, {
            "role": "primary", "pattern": obs.pattern, "end_ms": end_ms,
            "absorbed": len(weak),
            "reason": (f"University: {len(weak)} low-confidence re-strikes of pitch "
                       f"{int(primary.pitch)} inside "
                       f"{obs.evidence.get('max_ioi_beats')} beat - held as one note"),
        }, confidence=obs.confidence)
        primaries += 1

        for echo in weak:
            _add(annotations, echo.id, PAGE_SUPPRESS_ANNOTATION_KIND, {
                "role": "absorbed", "pattern": obs.pattern,
                "primary_note_id": primary.id,
                "reason": (f"University: re-strike of pitch {int(echo.pitch)} at "
                           f"confidence {float(getattr(echo, 'confidence', 1.0)):.2f} "
                           f"<= {suppress_max_confidence} - page keeps the first strike"),
            }, confidence=obs.confidence)
            suppressed += 1

    # ---- 2. voice cohesion for real gestures ----
    for idx, obs in enumerate(report.observations):
        if obs.pattern not in _COHESIVE_PATTERNS or obs.size < 3:
            continue
        group_id = f"{obs.pattern}:{idx}"
        for note_id in obs.note_ids:
            _add(annotations, note_id, VOICE_COHESION_ANNOTATION_KIND, {
                "group": group_id, "pattern": obs.pattern,
                "reason": f"University: one {obs.pattern} gesture - keep in one voice",
            }, confidence=obs.confidence)
            cohesive += 1

    return {"suppressed": suppressed, "primaries_extended": primaries,
            "cohesion_tagged": cohesive}


def page_suppression_map(annotations, note_ids) -> Tuple[set, Dict[str, float]]:
    """Read back what APPLY decided, for the notation builder.
    Returns (ids_to_drop, {primary_id: extended_end_ms})."""
    drop: set = set()
    extend: Dict[str, float] = {}
    for note_id in note_ids:
        value = annotations.latest_value(note_id, PAGE_SUPPRESS_ANNOTATION_KIND)
        if not value:
            continue
        if value.get("role") == "absorbed":
            drop.add(note_id)
        elif value.get("role") == "primary" and value.get("end_ms") is not None:
            extend[note_id] = float(value["end_ms"])
    return drop, extend


__all__ = ["write_study_annotations", "apply_observations", "page_suppression_map",
           "SUPPRESS_MAX_CONFIDENCE"]
