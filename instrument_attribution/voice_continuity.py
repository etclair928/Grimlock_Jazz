# =================================================================
# MODULE: instrument_attribution/voice_continuity.py
# Step 2 of "fingerprint -> stream -> resolve" (GRIMLOCK_6.0_DESIGN_
# DECISIONS.md §5). Groups a stem's notes into monophonic-ish "lines" -
# continuous melodic threads - by real assignment cost (pitch distance
# + time gap), not by clustering on absolute pitch (a legato bassline
# sliding an octave is still one instrument playing one line).
#
# This is a CONSOLIDATOR, not a splitter: its job is to stop one
# instrument's line from fragmenting into multiple identities (the
# "Klangio problem" - one sax phrase mislabeled sax/violin/sax across
# consecutive notes) - not to aggressively carve polyphony apart. Two
# simultaneous notes that can't share one line (monophonic constraint)
# each start their own line; that's correct, not over-splitting.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4

from core import Note

MAX_PITCH_JUMP_SEMITONES = 12
MAX_GAP_MS = 300.0
GAP_COST_PER_MS = 1.0 / 50.0  # 50ms of silence ~= 1 semitone of pitch-jump cost
MAX_ASSIGNMENT_COST = 12.0    # roughly "one octave, or ~600ms of gap, or a blend"


@dataclass
class VoiceLine:
    line_id: str
    notes: List[Note] = field(default_factory=list)

    @property
    def last_note(self) -> Note:
        return self.notes[-1]


def _assignment_cost(line: VoiceLine, note: Note) -> Optional[float]:
    last = line.last_note
    if note.start_ms < last.end_ms:
        return None  # overlaps this line's last note - can't be the same monophonic thread
    gap_ms = note.start_ms - last.end_ms
    pitch_jump = abs(note.pitch - last.pitch)
    if pitch_jump > MAX_PITCH_JUMP_SEMITONES:
        return None
    return pitch_jump + gap_ms * GAP_COST_PER_MS


def stream_into_lines(notes: List[Note]) -> List[VoiceLine]:
    """Greedy nearest-continuation streaming: each note joins whichever
    open line continues it most cheaply (lowest pitch-jump + gap cost),
    or starts a new line if no open line can take it under
    MAX_ASSIGNMENT_COST / MAX_GAP_MS."""
    lines: List[VoiceLine] = []
    # Lines that can still take a note. A line whose last note ended more than
    # MAX_GAP_MS ago can never be chosen again (notes arrive in time order), so
    # RETIRE it instead of re-testing it for every remaining note. The old loop
    # scanned every line ever opened - with the ~177-line floor this greedy
    # streamer hits on real polyphony, that is a large constant on top of an
    # O(n x lines) walk, for candidates that were rejected by the first `if`
    # every single time (2026-08-17 audit). Same output, by construction: the
    # retired lines are exactly the ones the guard already skipped.
    open_lines: List[VoiceLine] = []

    for note in sorted(notes, key=lambda n: n.start_ms):
        still_open: List[VoiceLine] = []
        best_line: Optional[VoiceLine] = None
        best_cost = float("inf")

        for line in open_lines:
            if note.start_ms - line.last_note.end_ms > MAX_GAP_MS:
                continue  # gone cold - drop it from the active set for good
            still_open.append(line)
            cost = _assignment_cost(line, note)
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_line = line
        open_lines = still_open

        if best_line is not None and best_cost <= MAX_ASSIGNMENT_COST:
            best_line.notes.append(note)
        else:
            new_line = VoiceLine(line_id=uuid4().hex, notes=[note])
            lines.append(new_line)
            open_lines.append(new_line)

    return lines


__all__ = ["VoiceLine", "stream_into_lines", "MAX_PITCH_JUMP_SEMITONES", "MAX_GAP_MS"]
