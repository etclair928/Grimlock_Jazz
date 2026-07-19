# =================================================================
# MODULE: core/tempo_types.py
# The ONE canonical tempo/meter type. GRIMLOCK_6.0_DESIGN_DECISIONS.md §9:
# Symphony split this fact across THREE types - TempoTestimony, TempoMap,
# and BeatGrid - with tempo_bpm stored independently in all three (they
# could and did drift out of sync), confidence as a raw float in one and
# the Confidence enum in another for the same measurement, and time
# signature as numerator/denominator ints in one but a formatted string
# ("4/4") in another, forcing a pointless parse/format translation
# between them. One TempoMeter replaces all three.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from core.confidence import clamp_confidence
from core.source_types import Provenance


@dataclass(frozen=True)
class TempoMeter:
    """Tempo + meter + beat grid for a track (or a guided override of one).

    confidence is a continuous float (see confidence.py's module docstring
    for why raw-float vs the Confidence enum are different concepts) -
    bucket it with Confidence.from_float() only at the point some stage
    needs a gate decision, never store a second copy alongside this one.

    beat_times_ms/downbeat_times_ms are tuples, not lists: a frozen
    dataclass only blocks reassigning a field, not mutating a mutable
    object it holds - a List[float] field would still be append()-able in
    place. Tuples close that gap for real immutability.
    """
    tempo_bpm: float
    confidence: float
    time_signature_numerator: int
    time_signature_denominator: int
    beat_times_ms: Tuple[float, ...] = field(default_factory=tuple)
    downbeat_times_ms: Tuple[float, ...] = field(default_factory=tuple)
    source: Provenance = Provenance.TEMPO_INTELLIGENCE
    is_guided: bool = False  # hard lock - see §2.7; no witness may re-arbitrate

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", clamp_confidence(self.confidence))
        if self.tempo_bpm <= 0:
            raise ValueError(f"tempo_bpm must be positive, got {self.tempo_bpm}")
        if self.time_signature_numerator <= 0 or self.time_signature_denominator <= 0:
            raise ValueError(
                f"time signature must be positive ints, got "
                f"{self.time_signature_numerator}/{self.time_signature_denominator}"
            )

    @property
    def time_signature(self) -> Tuple[int, int]:
        return (self.time_signature_numerator, self.time_signature_denominator)

    @property
    def beat_duration_ms(self) -> float:
        return 60000.0 / self.tempo_bpm

    @property
    def measure_duration_ms(self) -> float:
        return self.beat_duration_ms * self.time_signature_numerator
