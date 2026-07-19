# =================================================================
# MODULE: quantization/trouble_map.py
# Ports Symphony's TroubleMap (agents/quantization/ritornello.py, Pass
# 7) - pure diagnosis, no mutation to adapt away from: flags measures
# with low average confidence or unusually dense note packing
# ("fragmented"), for anyone (a human listening back, a future repair
# pass) who wants to know where the transcription is shakiest without
# walking every Note's Annotations by hand.
#
# Symphony's version read measure boundaries from a separate MeasureMap
# class (itself built from BeatGrid + TimeSignatureCandidate). Jazz has
# no such class and doesn't need one: TempoMeter already carries
# tempo_bpm, time_signature_numerator, and beat_times_ms, which is
# everything measure-boundary math needs. Jazz also doesn't populate
# downbeat_times_ms yet (nothing produces it), so measure boundaries
# here are approximated as consecutive groups of time_signature_
# numerator beats starting at the first detected beat - exactly right
# once a real downbeat phase is threaded through (meter.py's
# estimate_time_signature already computes one internally; see task
# tracking this ticket for wiring it into TempoMeter), off by at most
# one partial measure (an anacrusis/pickup) until then.
#
# Symphony's paired pass, CorrectionScribe ("re-analyze the flagged
# measure via Basic Pitch and inject corrected notes"), is deliberately
# NOT ported - it re-runs detection and substitutes its own answer,
# which is exactly the "fight the models" pattern 6.0's laws forbid.
# TroubleMap's diagnosis stands on its own without it.
# =================================================================

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from core import Confidence, Note, TempoMeter

TROUBLE_MAP_MIN_CONFIDENCE = Confidence.LOW.value                  # Measure avg confidence below this => flagged
TROUBLE_MAP_MAX_FRAGMENTATION_NOTES_PER_BEAT = 6.0                 # Notes-per-beat above this => flagged as fragmented


@dataclass(frozen=True)
class TroubleMeasure:
    """Diagnostics - a measure flagged as low-confidence and/or
    fragmented. Read-only reporting; nothing consumes this to change a
    Note or write an Annotation."""
    measure_number: int
    start_ms: float
    end_ms: float
    avg_confidence: float
    reasons: Tuple[str, ...] = field(default_factory=tuple)


def _measure_index_at(time_ms: float, first_beat_ms: float, measure_duration_ms: float) -> int:
    if measure_duration_ms <= 0:
        return 0
    return max(0, int((time_ms - first_beat_ms) / measure_duration_ms))


def _measure_bounds(measure_index: int, first_beat_ms: float, measure_duration_ms: float) -> Tuple[float, float]:
    start = first_beat_ms + measure_index * measure_duration_ms
    return start, start + measure_duration_ms


def build_trouble_map(
        notes: List[Note],
        tempo_meter: TempoMeter,
        min_confidence: float = TROUBLE_MAP_MIN_CONFIDENCE,
        max_fragmentation_notes_per_beat: float = TROUBLE_MAP_MAX_FRAGMENTATION_NOTES_PER_BEAT,
) -> List[TroubleMeasure]:
    """One stem's (or the whole pitched set's) notes, bucketed into
    approximate measures against the resolved tempo, flagged for low
    confidence or dense fragmentation. Returns only flagged measures."""
    if not notes or tempo_meter.tempo_bpm <= 0:
        return []

    first_beat_ms = tempo_meter.beat_times_ms[0] if tempo_meter.beat_times_ms else 0.0
    measure_duration_ms = tempo_meter.measure_duration_ms

    by_measure: Dict[int, List[Note]] = defaultdict(list)
    for note in notes:
        idx = _measure_index_at(note.start_ms, first_beat_ms, measure_duration_ms)
        by_measure[idx].append(note)

    trouble: List[TroubleMeasure] = []
    for idx, group in sorted(by_measure.items()):
        avg_conf = sum(n.confidence for n in group) / len(group)
        start_ms, end_ms = _measure_bounds(idx, first_beat_ms, measure_duration_ms)
        duration_beats = max(1e-6, (end_ms - start_ms) / tempo_meter.beat_duration_ms)
        notes_per_beat = len(group) / duration_beats

        reasons = []
        if avg_conf < min_confidence:
            reasons.append(f"low_confidence:{avg_conf:.2f}")
        if notes_per_beat > max_fragmentation_notes_per_beat:
            reasons.append(f"fragmented:{notes_per_beat:.1f}_notes_per_beat")

        if reasons:
            trouble.append(TroubleMeasure(
                measure_number=idx, start_ms=start_ms, end_ms=end_ms,
                avg_confidence=avg_conf, reasons=tuple(reasons),
            ))

    return trouble


__all__ = [
    "TroubleMeasure",
    "build_trouble_map",
    "TROUBLE_MAP_MIN_CONFIDENCE",
    "TROUBLE_MAP_MAX_FRAGMENTATION_NOTES_PER_BEAT",
]
