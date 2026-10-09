# =================================================================
# MODULE: instrument_attribution/voice_continuity.py
# Step 2 of "fingerprint -> stream -> resolve". Groups a stem's notes
# into monophonic-ish "lines" - continuous melodic threads - by real
# assignment cost (pitch distance + time gap), not by clustering on
# absolute pitch (a legato bassline sliding an octave is still one
# instrument playing one line).
#
# This is a CONSOLIDATOR, not a splitter: its job is to stop one
# instrument's line from fragmenting into multiple identities (the
# "Klangio problem") - not to aggressively carve polyphony apart. Two
# simultaneous notes that can't share one line each start their own
# line; that's correct, not over-splitting.
#
# WHY THIS MATTERS MORE NOW: resolve.py median-pools the overtone
# fingerprints across each line. Long, pure lines make that pooling
# reliable; fragmented lines starve it of notes, and lines that jump
# between instruments blur it.
#
# CHANGES in this version:
#  1. Notes that start together (a chord) are assigned to lines JOINTLY
#     with an optimal assignment (Hungarian algorithm). The old greedy
#     loop let whichever chord note came first grab the best line, and
#     the other chord notes got the leftovers - which swaps voices.
#  2. Small overlaps are allowed (OVERLAP_TOLERANCE_MS). Basic Pitch
#     often makes legato notes overlap by a few tens of ms; the old
#     hard "no overlap at all" rule split those melodies into pieces.
#  3. Deterministic ordering and line ids. The same notes now always give
#     the same lines (the old uuid4 changed on every run, which made
#     before/after comparisons impossible). Ids come from the first
#     note's id, so they stay unique across different stems.
# =================================================================

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from core import Note

MAX_PITCH_JUMP_SEMITONES = 12
MAX_GAP_MS = 300.0
GAP_COST_PER_MS = 1.0 / 50.0   # 50ms of silence ~= 1 semitone of pitch-jump cost
MAX_ASSIGNMENT_COST = 12.0     # roughly "one octave, or ~600ms of gap, or a blend"

OVERLAP_TOLERANCE_MS = 30.0    # a new note may start this much before the line's last note ends
OVERLAP_COST_PER_MS = 1.0 / 30.0  # small extra cost so a clean continuation beats an overlapping one
ONSET_GROUP_MS = 25.0          # notes starting within this window are one "chord" assigned together

_INFEASIBLE = 1e6              # stand-in for "cannot join this line" inside the cost matrix

_LINE_ID_NAMESPACE = uuid.UUID("6f1d0f2e-6a58-4b1e-9d1c-3c2f7a9b0e11")


@dataclass
class VoiceLine:
    line_id: str
    notes: List[Note] = field(default_factory=list)

    @property
    def last_note(self) -> Note:
        return self.notes[-1]


def _assignment_cost(line: VoiceLine, note: Note) -> Optional[float]:
    """Cost of continuing `line` with `note`, or None if it cannot."""
    last = line.last_note
    overlap_ms = last.end_ms - note.start_ms
    if overlap_ms > OVERLAP_TOLERANCE_MS:
        return None  # sounding together with this line's last note - not one monophonic thread
    pitch_jump = abs(note.pitch - last.pitch)
    if pitch_jump > MAX_PITCH_JUMP_SEMITONES:
        return None
    if overlap_ms > 0:
        return pitch_jump + overlap_ms * OVERLAP_COST_PER_MS
    gap_ms = -overlap_ms
    if gap_ms > MAX_GAP_MS:
        return None
    return pitch_jump + gap_ms * GAP_COST_PER_MS


def _new_line(note: Note) -> VoiceLine:
    return VoiceLine(line_id=uuid.uuid5(_LINE_ID_NAMESPACE, str(note.id)).hex, notes=[note])


def _group_by_onset(notes: List[Note]) -> List[List[Note]]:
    """Split time-sorted notes into chord groups (onsets within ONSET_GROUP_MS)."""
    groups: List[List[Note]] = []
    for note in notes:
        if groups and note.start_ms - groups[-1][0].start_ms <= ONSET_GROUP_MS:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def stream_into_lines(notes: List[Note]) -> List[VoiceLine]:
    """Streams notes into voice lines.

    Notes are processed chord by chord. Within a chord, the best
    matching of notes to open lines is chosen jointly (lowest total
    cost); a note whose match is infeasible or costs more than
    MAX_ASSIGNMENT_COST starts a new line. Output order is the order
    lines were opened."""
    lines: List[VoiceLine] = []
    open_lines: List[VoiceLine] = []  # lines that can still take a note

    ordered = sorted(notes, key=lambda n: (n.start_ms, n.pitch, str(n.id)))

    for group in _group_by_onset(ordered):
        group_start = group[0].start_ms

        # Retire lines that have gone cold. Notes arrive in time order, so a
        # line this old can never be chosen again.
        open_lines = [ln for ln in open_lines if group_start - ln.last_note.end_ms <= MAX_GAP_MS]

        assigned = set()  # indices into `group`
        taken_lines = []  # lines extended by this group

        if open_lines:
            cost = np.full((len(group), len(open_lines)), _INFEASIBLE, dtype=float)
            for i, note in enumerate(group):
                for j, line in enumerate(open_lines):
                    c = _assignment_cost(line, note)
                    if c is not None and c <= MAX_ASSIGNMENT_COST:
                        cost[i, j] = c

            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < _INFEASIBLE:
                    open_lines[j].notes.append(group[i])
                    assigned.add(i)
                    taken_lines.append(open_lines[j])

        # Everything left over starts its own line.
        for i, note in enumerate(group):
            if i not in assigned:
                new_line = _new_line(note)
                lines.append(new_line)
                open_lines.append(new_line)

    return lines


__all__ = [
    "VoiceLine",
    "stream_into_lines",
    "MAX_PITCH_JUMP_SEMITONES",
    "MAX_GAP_MS",
    "OVERLAP_TOLERANCE_MS",
]
