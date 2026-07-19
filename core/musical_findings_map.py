# =================================================================
# MODULE: core/musical_findings_map.py
# MusicalFindingsMap (GRIMLOCK_6.0_DESIGN_DECISIONS.md §7 Core): the
# blackboard - shared state where agents publish conclusions. Passive
# container, single source of truth. It holds nothing notes need
# (those live on Note/AnnotationStore) - only track-level scalar
# findings (tempo/meter/groove/key) that Rhythm Engine/Key Intelligence
# produce and Epistemic/Output/Instrument Attribution read.
#
# Deliberately minimal (§9's lesson from Symphony: a findings map that
# accumulates undeclared attributes and triplicated scalar copies is
# exactly the disease 6.0 exists to avoid). Fields are added only when
# a real producer/consumer needs them - not pre-declared for a future
# layer this build doesn't include yet. (key/harmonic detection WAS
# that future layer until key_intelligence/ was built - see key/
# key_confidence below.)
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.stem_types import StemType
from core.tempo_types import TempoMeter

if TYPE_CHECKING:
    # Import deferred to type-checking only: quantization imports from
    # core (Note, TempoMeter, Confidence), so a real runtime import here
    # would be circular. `from __future__ import annotations` (above)
    # already makes every annotation in this file a lazy string, so this
    # guard is all that's needed for the type hint below to resolve
    # correctly under a type checker without ever executing at import time.
    from quantization.trouble_map import TroubleMeasure


@dataclass
class MusicalFindingsMap:
    """Passive blackboard. No logic lives here - it's read and written
    by other layers, never decides anything itself."""
    tempo_meter: Optional[TempoMeter] = None

    # Set only when Epistemic actually had to resolve a tempo dispute
    # between witnesses (see epistemic/referee.py) - None means no
    # contention was ever recorded, not that it was checked and passed.
    tempo_contention: Optional[Dict[str, Any]] = None

    swing_ratio: float = 0.0
    groove_confidence: float = 0.0

    # Key Intelligence's song-level key detection (key_intelligence.
    # key_detector) - e.g. "C", "Am" (trailing 'm' means minor). None
    # means key detection never ran, not that it found nothing.
    key: Optional[str] = None
    key_confidence: float = 0.0

    # Per-stem list of instrument-family names Instrument Attribution
    # resolved onto at least one voice line - a summary for anything
    # that wants a quick answer without walking every Note's Annotations.
    instrument_families: Dict[StemType, List[str]] = field(default_factory=dict)

    # TroubleMap's diagnosis (quantization.trouble_map) - measures
    # flagged as low-confidence and/or fragmented. Pure reporting, same
    # spirit as tempo_contention: read by anything that wants a quick
    # answer, never itself a decision point.
    trouble_measures: List["TroubleMeasure"] = field(default_factory=list)

    def __repr__(self) -> str:
        tempo = f"{self.tempo_meter.tempo_bpm:.1f}bpm" if self.tempo_meter else "unset"
        key = f"{self.key} (conf={self.key_confidence:.2f})" if self.key else "unset"
        return (
            f"MusicalFindingsMap(tempo={tempo}, key={key}, swing={self.swing_ratio:.2f}, "
            f"families={ {k.value: v for k, v in self.instrument_families.items()} })"
        )


__all__ = ["MusicalFindingsMap"]
