# =================================================================
# MODULE: core/separation_types.py
# The ONE canonical separation-stage result. GRIMLOCK_6.0_DESIGN_DECISIONS.
# md §9: Symphony had two objects for this one fact - SeparationResult
# (the real stem audio + `.separator_used`) and SeparationTestimony (report
# metadata + `.model_used`) - same concept, two field names. A real bug
# shipped from exactly this split (main.py read the wrong attribute off
# the wrong class and silently fell back to a hardcoded default).
#
# `stems` holds whatever the Audio Engine's view type is once that layer
# is built (an immutable, read-only slice - §7 Audio Engine); until then
# it's a bridge Dict[StemType, Any]. Either way, Separation itself never
# copies or owns the audio - the Audio Engine does. This type is metadata
# plus references.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from core.confidence import clamp_confidence
from core.stem_types import StemType


@dataclass(frozen=True)
class Separation:
    stems: Dict[StemType, Any]
    model_used: str
    confidence: float
    separation_time_seconds: float
    stems_hallucinated: Tuple[StemType, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", clamp_confidence(self.confidence))

    def has_stem(self, stem: StemType) -> bool:
        return stem in self.stems and stem not in self.stems_hallucinated

    def get_stem(self, stem: StemType) -> Any:
        return self.stems.get(stem)
