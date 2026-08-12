# =================================================================
# MODULE: university/streams.py
# Melodic-line extraction built FOR PATTERN DETECTION.
#
# WHY THIS EXISTS (measured): the curriculum was reading lines from
# instrument_attribution.voice_continuity.stream_into_lines, which is a
# CONSOLIDATOR for instrument identity, not a melodic-line finder. Its
# rules break pattern detection in three specific ways:
#
#   1. ANY overlap starts a new line. Basic Pitch emits overlapping,
#      sustained notes constantly, so a legato melodic line gets cut at
#      almost every note (the same weakness that once produced 776 lines
#      for 2183 notes). A detector needing >=4 CONSECUTIVE notes in one
#      line then almost never fires.
#   2. MAX_GAP_MS = 300 is absolute. At 148bpm one beat is 405ms, so any
#      rest longer than ~3/4 of a beat severs the line - and the same
#      threshold means something completely different at 62bpm.
#   3. It is tuned to stop an instrument's identity fragmenting, which is
#      a different objective from "find the melodic thread".
#
# The fix, and the one thing that must NOT regress: notes struck TOGETHER
# are a chord, not a melodic succession. If simultaneous notes were
# allowed to chain, every block chord would read as an "arpeggio". So a
# continuation requires the onset to be genuinely LATER (> CHORD_WINDOW),
# while sustain overlap is tolerated.
#
# voice_continuity is left completely untouched - the pipeline's
# instrument attribution keeps using it exactly as before.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# Onsets closer than this are struck together -> a chord, never a melodic
# continuation. This is the guard that stops chords reading as arpeggios.
CHORD_WINDOW_MS = 45.0

# Maximum silence between onsets, in BEATS (not ms): the same musical
# meaning at 62 and 150 bpm. Two beats lets a line survive a breath or a
# short rest without stitching together unrelated phrases.
MAX_ONSET_GAP_BEATS = 2.0

# A melodic step/leap beyond an octave is a new idea, not a continuation.
MAX_PITCH_JUMP = 12

# Cost weighting: pitch proximity dominates, time breaks ties.
_PITCH_W = 1.0
_TIME_W = 2.0


@dataclass
class MelodicLine:
    notes: List = field(default_factory=list)

    @property
    def last(self):
        return self.notes[-1]

    def __len__(self) -> int:
        return len(self.notes)


def extract_lines(
        notes: Sequence,
        beat_ms: float,
        max_gap_beats: float = MAX_ONSET_GAP_BEATS,
        chord_window_ms: float = CHORD_WINDOW_MS,
        max_pitch_jump: int = MAX_PITCH_JUMP,
) -> List[MelodicLine]:
    """Group notes into melodic threads by onset succession.

    A note continues the line whose last note is closest in pitch, subject
    to: the onset must be later than that note's onset by more than
    `chord_window_ms` (else it is a chord tone, not a continuation), the
    onset gap must be within `max_gap_beats`, and the leap must be within
    `max_pitch_jump`. SUSTAIN OVERLAP IS ALLOWED - a held note does not
    sever the melodic thread."""
    if not notes:
        return []

    max_gap_ms = max_gap_beats * max(beat_ms, 1.0)
    ordered = sorted(notes, key=lambda n: (n.start_ms, n.pitch))
    lines: List[MelodicLine] = []

    for note in ordered:
        best: Optional[MelodicLine] = None
        best_cost = float("inf")
        for line in lines:
            last = line.last
            onset_gap = note.start_ms - last.start_ms
            if onset_gap <= chord_window_ms:
                continue                       # struck together -> chord, not a continuation
            if onset_gap > max_gap_ms:
                continue                       # the thread has gone cold
            jump = abs(int(note.pitch) - int(last.pitch))
            if jump > max_pitch_jump:
                continue
            cost = _PITCH_W * jump + _TIME_W * (onset_gap / max_gap_ms) * max_pitch_jump
            if cost < best_cost:
                best_cost = cost
                best = line
        if best is not None:
            best.notes.append(note)
        else:
            lines.append(MelodicLine(notes=[note]))

    return lines


def line_stats(lines: Sequence[MelodicLine]) -> dict:
    if not lines:
        return {"lines": 0, "notes": 0, "mean_len": 0.0, "max_len": 0, "usable_ge4": 0}
    lengths = [len(l) for l in lines]
    return {
        "lines": len(lines),
        "notes": sum(lengths),
        "mean_len": round(sum(lengths) / len(lengths), 2),
        "max_len": max(lengths),
        "usable_ge4": sum(1 for x in lengths if x >= 4),
    }


__all__ = ["MelodicLine", "extract_lines", "line_stats",
           "CHORD_WINDOW_MS", "MAX_ONSET_GAP_BEATS", "MAX_PITCH_JUMP"]
