# =================================================================
# MODULE: quantization/duration_witness.py
# Ports the CORE of Symphony's QuaverIntelligence (epistemic/
# quaver_intelligence.py) - a symbolic-duration witness that proposes
# competing hypotheses for what a note's duration "really is"
# (quarter/eighth/dotted/triplet/etc.), scored by how closely the
# note's raw duration matches each candidate against the resolved
# tempo. "Does NOT quantize. Does NOT decide. Only TESTIFIES" - per
# its own module docstring, already annotation-shaped by design; this
# port keeps that law literally (a function returning a value, never
# touching a Note).
#
# Deliberately reduced from the original's full "adversarial evidence"
# gathering (which cross-examines 8 other subsystems - AnechoicMa
# resonance, per-note drum-anchor metrical evidence, voice-continuity
# phrase grouping, groove-field per-note deviation - none of which
# exist in Jazz yet in the shape QuaverIntelligence expects). This
# keeps the always-available core signal (raw duration vs. the beat
# grid) rather than faking evidence from subsystems that aren't built.
# Extend this once those subsystems exist and have something real to
# contribute, per this project's "don't build ahead of need" discipline.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from core import Note

SYMBOLIC_DURATIONS = {
    "whole": 4.000,
    "dotted_half": 3.000,
    "half": 2.000,
    "dotted_quarter": 1.500,
    "quarter": 1.000,
    "dotted_eighth": 0.750,
    "eighth": 0.500,
    "dotted_sixteenth": 0.375,
    "sixteenth": 0.250,
    "thirty_second": 0.125,
}
TUPLET_RATIOS = {
    "triplet": 2.0 / 3.0,
    "quintuplet": 4.0 / 5.0,
    "septuplet": 4.0 / 7.0,
    "compound_triplet": 3.0 / 4.0,
}
DEFAULT_TOLERANCE_FRACTION = 0.08  # 8% of the beat


@dataclass(frozen=True)
class DurationHypothesis:
    symbolic_value: str
    duration_ms: float
    probability: float


@dataclass(frozen=True)
class DurationTestimony:
    note_id: str
    raw_duration_ms: float
    hypotheses: Tuple[DurationHypothesis, ...]

    @property
    def primary(self) -> DurationHypothesis:
        return max(self.hypotheses, key=lambda h: h.probability)

    @property
    def has_contention(self) -> bool:
        """True if the top two hypotheses are within 0.15 of each
        other in probability - the note's duration is genuinely
        ambiguous between two symbolic readings."""
        if len(self.hypotheses) < 2:
            return False
        ranked = sorted(self.hypotheses, key=lambda h: -h.probability)
        return (ranked[0].probability - ranked[1].probability) < 0.15


def infer_duration(note: Note, beat_ms: float, tolerance_fraction: float = DEFAULT_TOLERANCE_FRACTION) -> DurationTestimony:
    """Generates competing symbolic-duration hypotheses for `note`
    against the resolved beat grid. Falls back to a single low-
    confidence "unknown" hypothesis when nothing scores above zero
    (a very short or noisy note)."""
    raw_dur = note.duration_ms

    candidates: List[Tuple[str, float]] = [(name, mult * beat_ms) for name, mult in SYMBOLIC_DURATIONS.items()]
    candidates += [(name, ratio * beat_ms) for name, ratio in TUPLET_RATIOS.items()]

    scored = []
    for name, candidate_ms in candidates:
        if candidate_ms <= 0:
            continue
        error = abs(raw_dur - candidate_ms) / candidate_ms
        if error > tolerance_fraction * 3:
            continue  # clearly not this candidate, don't dilute the vote
        score = max(0.0, 1.0 - error / (tolerance_fraction * 3))
        scored.append((name, candidate_ms, score))

    if not scored:
        return DurationTestimony(
            note_id=note.id, raw_duration_ms=raw_dur,
            hypotheses=(DurationHypothesis("unknown", raw_dur, 0.3),),
        )

    total_score = sum(s for _, _, s in scored)
    hypotheses = tuple(
        DurationHypothesis(name, candidate_ms, s / total_score if total_score > 0 else 0.0)
        for name, candidate_ms, s in scored
    )
    return DurationTestimony(note_id=note.id, raw_duration_ms=raw_dur, hypotheses=hypotheses)


__all__ = ["DurationHypothesis", "DurationTestimony", "infer_duration", "SYMBOLIC_DURATIONS", "TUPLET_RATIOS"]
