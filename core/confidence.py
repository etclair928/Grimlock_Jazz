# =================================================================
# MODULE: core/confidence.py
# Canonical confidence representations. See GRIMLOCK_6.0_DESIGN_DECISIONS.md
# §9 for the audit finding this fixes: Symphony stored the SAME tempo
# estimate's confidence as a raw float in one object (TempoTestimony) and
# as the Confidence enum in another (TempoMap) - one fact, two shapes,
# because tempo was split across three types instead of one. That bug goes
# away once each fact has exactly one home (see tempo_types.py), not by
# forcing every confidence in the system into one representation.
#
# Two representations are kept because they answer different questions:
#   - a bare float in [0, 1]: "how confident is THIS specific measurement,"
#     continuous, computed by a detector. Note.confidence, TempoMeter.
#     confidence, etc. all use plain float for this reason - a 5-bucket
#     enum would throw away real precision a model actually computed.
#   - Confidence (this enum): "how should this measurement be BUCKETED for
#     a gating/threshold decision" - Scribe validation gates, "is this
#     witness HIGH tier or better." Never stored redundantly alongside a
#     float for the same fact - convert with Confidence.from_float() at
#     the point of the decision, don't carry both around.
# =================================================================

from __future__ import annotations

from enum import Enum
from typing import Union


class Confidence(Enum):
    """
    Bucketed confidence tier for gating/threshold decisions.

    LAW: Does NOT inherit from float - prevents accidental arithmetic on a
    tier (e.g. summing two Confidence values as if they were probabilities).
    Use .value_f for numeric comparisons/arithmetic.
    """
    HALLUCINATION = 0.0
    LOW = 0.25
    MEDIUM = 0.50
    HIGH = 0.80
    PERFECT = 1.0

    @property
    def value_f(self) -> float:
        return self._value_

    @classmethod
    def from_float(cls, value: float) -> "Confidence":
        """Bucket a continuous confidence into the nearest tier at or below
        it - used ONLY at the point a stage needs a gating decision, never
        to store a second copy of a measurement's own float confidence."""
        value = max(0.0, min(1.0, float(value)))
        ordered = sorted(cls, key=lambda c: c.value_f, reverse=True)
        for tier in ordered:
            if value >= tier.value_f:
                return tier
        return cls.HALLUCINATION

    def __ge__(self, other: Union[float, "Confidence"]) -> bool:
        if isinstance(other, Confidence):
            other = other.value_f
        return self.value_f >= other

    def __le__(self, other: Union[float, "Confidence"]) -> bool:
        if isinstance(other, Confidence):
            other = other.value_f
        return self.value_f <= other

    def __gt__(self, other: Union[float, "Confidence"]) -> bool:
        if isinstance(other, Confidence):
            other = other.value_f
        return self.value_f > other

    def __lt__(self, other: Union[float, "Confidence"]) -> bool:
        if isinstance(other, Confidence):
            other = other.value_f
        return self.value_f < other


def clamp_confidence(value: float) -> float:
    """Clamp a continuous confidence into the valid [0, 1] range. Every
    canonical type that stores a raw-float confidence should run new
    values through this rather than trusting the caller."""
    return max(0.0, min(1.0, float(value)))
