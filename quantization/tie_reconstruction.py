# =================================================================
# MODULE: quantization/tie_reconstruction.py
# Ports Symphony's TieReconstruction (agents/quantization/ritornello.py,
# Pass 4) - identifies same-pitch note pairs whose gap straddles a beat
# line, the signature of Basic Pitch splitting ONE held note into two
# fragments right at a barline rather than a genuine re-articulation.
#
# The discriminator, kept exactly: a real re-articulation carries its
# own full-strength onset; a detection artifact splitting a held note
# leaves a weaker second fragment (decaying sustain). Only that decay
# pattern (next.confidence <= note.confidence) justifies treating an
# onset the detection model explicitly reported as spurious - the
# "amplify, don't fight" law applied to a segmentation question rather
# than a pitch or timing one.
#
# 6.0 adaptation: Symphony's version MERGED the two NoteEvents into one.
# This port never does - it writes a `tie_candidate` Annotation on the
# earlier note pointing at the later one's id; both Notes survive
# untouched. Because nothing is consumed/merged here, a chain of three
# or more fragments can each carry a tie_candidate to the next, which a
# reader can walk end-to-end - a strict improvement over the original's
# single-pass pairwise consumption, which existed only to avoid
# double-merging indices, a constraint that doesn't apply once nothing
# is actually being merged.
#
# Beat-line proximity reuses Jazz's existing TempoMeter.beat_times_ms
# (already the resolved, cross-witness beat grid every other stage
# reads) instead of porting Symphony's separate MeasureMap class -
# there's nothing a bespoke measure map would tell this pass that the
# existing beat grid doesn't already answer for "is this near a beat".
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from core import Note

TIE_RECONSTRUCTION_ANNOTATION_KIND = "tie_candidate"

TIE_RECONSTRUCTION_BEAT_PROXIMITY_MS = 30.0   # "Near" a beat line
TIE_RECONSTRUCTION_MAX_GAP_MS = 40.0          # Max gap between same-pitch notes to tie across a beat


@dataclass(frozen=True)
class TieCandidate:
    note_id: str            # the earlier fragment
    tied_to_note_id: str    # the later fragment it should be considered continuous with
    boundary_ms: float
    reason: str


def _is_near_beat(time_ms: float, beat_times_ms: Sequence[float], tolerance_ms: float) -> bool:
    if not beat_times_ms:
        return False
    nearest = min(beat_times_ms, key=lambda b: abs(b - time_ms))
    return abs(nearest - time_ms) <= tolerance_ms


def find_tie_candidates(
        notes: List[Note],
        beat_times_ms: Sequence[float],
        beat_proximity_ms: float = TIE_RECONSTRUCTION_BEAT_PROXIMITY_MS,
        max_gap_ms: float = TIE_RECONSTRUCTION_MAX_GAP_MS,
) -> List[TieCandidate]:
    """Scans `notes` (one stem/voice's worth - mixing pitches/stems here
    would let unrelated notes look adjacent) in time order for same-
    pitch pairs whose gap straddles a beat line. Returns only the pairs
    that qualify; the Conductor writes each as a `tie_candidate`
    Annotation on the earlier note."""
    if len(notes) < 2 or not beat_times_ms:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.start_ms)
    candidates: List[TieCandidate] = []
    for note, nxt in zip(sorted_notes, sorted_notes[1:]):
        gap = nxt.start_ms - note.end_ms
        if not (note.pitch == nxt.pitch
                and 0 <= gap <= max_gap_ms
                and nxt.confidence <= note.confidence):
            continue
        boundary_ms = note.end_ms + gap / 2.0
        if _is_near_beat(boundary_ms, beat_times_ms, beat_proximity_ms):
            candidates.append(TieCandidate(
                note_id=note.id, tied_to_note_id=nxt.id, boundary_ms=boundary_ms,
                reason=f"gap {gap:.1f}ms straddles beat at {boundary_ms:.0f}ms",
            ))

    return candidates


__all__ = [
    "TieCandidate",
    "find_tie_candidates",
    "TIE_RECONSTRUCTION_ANNOTATION_KIND",
    "TIE_RECONSTRUCTION_BEAT_PROXIMITY_MS",
    "TIE_RECONSTRUCTION_MAX_GAP_MS",
]
