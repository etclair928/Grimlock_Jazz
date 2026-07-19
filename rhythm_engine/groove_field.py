# =================================================================
# MODULE: rhythm_engine/groove_field.py
# Ports Symphony's GrooveField (agents/analysis/groove_field.py) -
# "Relational Physics": measures the phase delta between bass and kick
# events to detect intentional "wide swing" or "Dilla pocket" feel,
# rather than treating it as timing error. "A note's value is defined
# by its distance to its neighbor, not its distance to the grid" - if
# bass and kick are consistently 12ms apart, that is a physical law of
# the performance the engine must respect, not noise to quantize away.
#
# Genuinely distinct from pulse_field.py (single-track, multi-
# hypothesis grid tracking over time) and rhythm_engine/groove.py
# (measures one stem's onset position within the beat interval) - this
# is CROSS-INSTRUMENT: it needs both a rhythm-bearing pitched stem's
# note onsets (bass, typically) and the drum stem's kick hits.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

from core import Note

SWING_THRESHOLD_MS = 8.0
DILLA_VARIANCE_THRESHOLD_MS = 4.0
MIN_PHASE_DELTA_TO_CONSIDER_MS = 2.0
MAX_PAIRING_DISTANCE_MS = 500.0  # a bass note whose nearest kick is farther than
                                  # this isn't a real rhythmic pair (e.g. a
                                  # kick-less passage) - pairing it in would let
                                  # one outlier drag the whole estimate to nonsense


class GrooveType(str, Enum):
    TIGHT_POCKET = "tight_pocket"
    WIDE_SWING = "wide_swing"
    DILLA_POCKET = "dilla_pocket"
    LAID_BACK = "laid_back"
    RUSHED = "rushed"
    FLUID = "fluid"
    ERRATIC = "erratic"


@dataclass(frozen=True)
class PhaseDeltaResult:
    deltas_ms: Tuple[float, ...]
    average_ms: float
    std_ms: float
    consistency: float
    is_significant: bool
    sample_count: int
    groove_type: GrooveType
    is_wide_swing: bool
    is_dilla_pocket: bool


def compute_phase_deltas(bass_notes: List[Note], kick_notes: List[Note]) -> PhaseDeltaResult:
    """For each bass note, finds the nearest kick and computes the
    offset (positive = bass after kick, negative = bass before kick),
    then classifies the resulting distribution into a groove type."""
    empty = PhaseDeltaResult((), 0.0, 0.0, 0.0, False, 0, GrooveType.ERRATIC, False, False)
    if not bass_notes or not kick_notes:
        return empty

    kick_times = np.array([k.start_ms for k in kick_notes])
    deltas = []
    for bass in bass_notes:
        idx = int(np.argmin(np.abs(kick_times - bass.start_ms)))
        delta = bass.start_ms - kick_times[idx]
        if abs(delta) > MAX_PAIRING_DISTANCE_MS:
            continue
        deltas.append(delta)

    if not deltas:
        return empty

    deltas_arr = np.array(deltas)
    avg_delta = float(np.mean(deltas_arr))
    std_delta = float(np.std(deltas_arr))
    variance = float(np.var(deltas_arr)) if len(deltas) > 1 else 0.0
    consistency = 1.0 / (1.0 + variance / 100) if variance > 0 else 1.0
    is_significant = abs(avg_delta) > MIN_PHASE_DELTA_TO_CONSIDER_MS

    groove_type, is_wide_swing, is_dilla_pocket = _classify_groove(avg_delta, std_delta, consistency, len(deltas))

    return PhaseDeltaResult(
        deltas_ms=tuple(deltas), average_ms=avg_delta, std_ms=std_delta,
        consistency=consistency, is_significant=is_significant, sample_count=len(deltas),
        groove_type=groove_type, is_wide_swing=is_wide_swing, is_dilla_pocket=is_dilla_pocket,
    )


def _classify_groove(avg_delta: float, std_delta: float, consistency: float, sample_count: int) -> Tuple[GrooveType, bool, bool]:
    if sample_count < 2:
        return GrooveType.ERRATIC, False, False

    is_wide_swing = abs(avg_delta) > SWING_THRESHOLD_MS and consistency > 0.6
    is_dilla_pocket = (
            DILLA_VARIANCE_THRESHOLD_MS * 0.5 <= std_delta <= DILLA_VARIANCE_THRESHOLD_MS * 1.5
            and consistency > 0.5
            and abs(avg_delta) < SWING_THRESHOLD_MS
    )

    if is_dilla_pocket:
        groove_type = GrooveType.DILLA_POCKET
    elif is_wide_swing:
        groove_type = GrooveType.WIDE_SWING
    elif abs(avg_delta) < 2 and consistency > 0.8:
        groove_type = GrooveType.TIGHT_POCKET
    elif avg_delta > 0:
        groove_type = GrooveType.LAID_BACK
    elif avg_delta < 0:
        groove_type = GrooveType.RUSHED
    elif consistency < 0.3:
        groove_type = GrooveType.ERRATIC
    else:
        groove_type = GrooveType.FLUID

    return groove_type, is_wide_swing, is_dilla_pocket


__all__ = ["GrooveType", "PhaseDeltaResult", "compute_phase_deltas"]
