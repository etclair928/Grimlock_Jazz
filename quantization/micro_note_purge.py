# =================================================================
# MODULE: quantization/micro_note_purge.py
# Ports Symphony's MicroNotePurge (agents/quantization/ritornello.py,
# Pass 3) - judges whether a very short, low-confidence note is a
# detection artifact or a real, deliberately-played grace note/ghost
# note. Symphony's version DELETED the note outright when it judged
# "artifact"; 6.0's law is annotation, not mutation (§2.2) - this port
# never removes a Note. It writes a `legitimacy_verdict` Annotation
# ("purge_candidate" or "legitimate") that Scribe Engraver may
# optionally read at export time; the Note itself always survives
# upstream of that, so nothing downstream loses access to what Basic
# Pitch actually detected.
#
# The original's harmonic-series-match evidence path (a note whose
# partials line up with a real harmonic series is spared even if short
# and quiet - overtone structure a hallucinated blip wouldn't have) is
# kept as an optional parameter rather than invented: Jazz has no
# harmonic-series-analysis pass yet (grep confirms nothing in this
# codebase currently produces that ratio). Absent that evidence, this
# uses ONLY the stricter "no-evidence floor" - the detection model
# already vouched for the note's existence, so overriding it requires
# hallucination-tier confidence, not merely "below the normal ceiling".
# Wire in a real harmonic_series_match_ratio once such a pass exists,
# per this project's "don't build ahead of need" discipline (see
# quantization/duration_witness.py's module docstring for the same
# call on the same underlying gap).
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from core import Note

MICRO_NOTE_PURGE_ANNOTATION_KIND = "legitimacy_verdict"

MICRO_NOTE_MIN_DURATION_MS = 60.0             # Below this + low confidence => candidate for purge
MICRO_NOTE_PURGE_CONFIDENCE_CEILING = 0.35    # Only purge if confidence below this (when harmonic evidence exists)
MICRO_NOTE_HARMONIC_JUSTIFICATION_RATIO = 0.5 # harmonic_series_match_ratio above this spares the note
MICRO_NOTE_PURGE_NO_EVIDENCE_CONFIDENCE_FLOOR = 0.15  # Stricter floor used absent harmonic evidence

PURGE_CANDIDATE = "purge_candidate"
LEGITIMATE = "legitimate"


@dataclass(frozen=True)
class LegitimacyVerdict:
    note_id: str
    verdict: str  # PURGE_CANDIDATE or LEGITIMATE
    reason: str


def evaluate_legitimacy(note: Note, harmonic_series_match_ratio: Optional[float] = None) -> LegitimacyVerdict:
    """Judges one note. `harmonic_series_match_ratio` is None until a
    harmonic-series-analysis pass exists in Jazz (see module docstring) -
    that's the normal, expected case today, not a missing-data bug."""
    is_short = note.duration_ms < MICRO_NOTE_MIN_DURATION_MS

    if harmonic_series_match_ratio is None:
        should_purge = is_short and note.confidence < MICRO_NOTE_PURGE_NO_EVIDENCE_CONFIDENCE_FLOOR
        reason = (f"duration={note.duration_ms:.0f}ms confidence={note.confidence:.2f} "
                  f"below no-evidence floor {MICRO_NOTE_PURGE_NO_EVIDENCE_CONFIDENCE_FLOOR:.2f}")
    else:
        is_low_conf = note.confidence < MICRO_NOTE_PURGE_CONFIDENCE_CEILING
        harmonic_ok = harmonic_series_match_ratio >= MICRO_NOTE_HARMONIC_JUSTIFICATION_RATIO
        should_purge = is_short and is_low_conf and not harmonic_ok
        reason = (f"duration={note.duration_ms:.0f}ms confidence={note.confidence:.2f}, "
                  f"harmonic ratio {harmonic_series_match_ratio:.2f} below justification threshold")

    return LegitimacyVerdict(
        note_id=note.id,
        verdict=PURGE_CANDIDATE if should_purge else LEGITIMATE,
        reason=reason,
    )


def evaluate_legitimacy_batch(
        notes: List[Note],
        harmonic_ratios_by_note_id: Optional[Dict[str, float]] = None,
) -> List[LegitimacyVerdict]:
    """Convenience wrapper for the Conductor's per-stem note loop."""
    harmonic_ratios_by_note_id = harmonic_ratios_by_note_id or {}
    return [
        evaluate_legitimacy(note, harmonic_ratios_by_note_id.get(note.id))
        for note in notes
    ]


__all__ = [
    "LegitimacyVerdict",
    "evaluate_legitimacy",
    "evaluate_legitimacy_batch",
    "MICRO_NOTE_PURGE_ANNOTATION_KIND",
    "PURGE_CANDIDATE",
    "LEGITIMATE",
]
