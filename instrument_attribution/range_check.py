# =================================================================
# MODULE: instrument_attribution/range_check.py
# Pitch-range plausibility (GRIMLOCK_6.0_OPEN_PROBLEMS.md §XVII.6).
#
# WHY (measured 2026-08-06): the head-to-head against Klangio on the
# SAME prospering recording showed our bass part running midi 27-77
# while theirs sat in a tight, realistic 34-55. Eight of our bass notes
# were above G4 - implausible for any bass instrument, and the kind of
# octave/overtone artifact a physical constraint catches for free.
# Symphony carried exactly this check (FAMILY_PITCH_RANGE_MIDI in
# agents/analysis/voice_continuity.py); Jazz had no equivalent.
#
# LAW: annotation-only. Nothing here deletes, moves, or re-pitches a
# note - it writes a verdict the engraver (or a later reducer) may act
# on, exactly like note_support and micro_note_purge. A flagged note is
# still exported; it is simply marked implausible for its own stem.
#
# Keyed by STEM, not by the timbre family: on the merged harmonic stem
# the family labels are brightness buckets (bright_lead/mid_body/...),
# not instruments, so they cannot support a range claim (§XVI.4). The
# stem IS the reliable identity for bass/vocals - which is where the
# real constraint lives anyway.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

RANGE_ANNOTATION_KIND = "range_plausibility"
IMPLAUSIBLE = "implausible"
PLAUSIBLE = "plausible"

# (lo, hi) inclusive MIDI bounds per stem, deliberately GENEROUS - the job
# is to catch octave/overtone artifacts, not to police unusual playing.
#   bass   : B0 (5-string low B = 23) .. G4 (67). Even a soloing bassist
#            with harmonics rarely writes above G4 in a transcription.
#   vocals : C2 (36) .. E6 (88). The HIGH bound was loosened 84 -> 88 after
#            measuring: only 1-2 notes per song sit above it, and high
#            harmony/falsetto is real, so a tight ceiling would flag music.
#            The LOW bound is where the value is - measured 51 (prospering)
#            and 61 (HRV) notes below C2, with minima at midi 21 and 27
#            (A0, D#1). No human voice produces those: that is bass bleed
#            into the vocals stem, or an octave error.
# Everything else (the merged harmonic stem, guitar, piano) is left
# unconstrained on purpose: it legitimately contains several instruments
# across the full keyboard, so any bound we set would be a guess.
STEM_PITCH_RANGE_MIDI: Dict[str, Tuple[int, int]] = {
    "bass": (23, 67),
    "vocals": (36, 88),
}


@dataclass(frozen=True)
class RangeVerdict:
    note_id: str
    verdict: str
    reason: str
    stem: str
    pitch: int


def check_range(notes: List, stem_of=None) -> List[RangeVerdict]:
    """Flag notes whose pitch falls outside their own stem's plausible
    range. Returns ONLY the implausible ones (the common case is silence)."""
    out: List[RangeVerdict] = []
    for note in notes:
        stem = stem_of(note) if stem_of else getattr(
            getattr(note, "stem", None), "value", str(getattr(note, "stem", "")))
        bounds = STEM_PITCH_RANGE_MIDI.get(stem)
        if bounds is None:
            continue
        lo, hi = bounds
        pitch = int(note.pitch)
        if lo <= pitch <= hi:
            continue
        where = "below" if pitch < lo else "above"
        out.append(RangeVerdict(
            note_id=note.id, verdict=IMPLAUSIBLE, stem=stem, pitch=pitch,
            reason=(f"pitch {pitch} is {where} the plausible {stem} range "
                    f"{lo}-{hi} - likely an octave or overtone artifact"),
        ))
    return out


def summarize(verdicts: List[RangeVerdict], total: int) -> Dict[str, object]:
    by_stem: Dict[str, int] = {}
    for v in verdicts:
        by_stem[v.stem] = by_stem.get(v.stem, 0) + 1
    return {"implausible": len(verdicts), "of_total": total, "by_stem": by_stem}


__all__ = ["check_range", "summarize", "RangeVerdict", "RANGE_ANNOTATION_KIND",
           "STEM_PITCH_RANGE_MIDI", "IMPLAUSIBLE", "PLAUSIBLE"]
