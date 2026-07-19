# =================================================================
# MODULE: core/note_types.py
# The ONE canonical Note type. Immutable detection floor (GRIMLOCK_6.0_
# DESIGN_DECISIONS.md §2.1). Nothing downstream of detection may construct
# a new Note with a different pitch/existence than what the detector said -
# repair/presentation passes attach Annotations (annotation_types.py)
# instead. See §9 for the concrete Symphony bugs this fixes:
#   - NoteEvent was a plain mutable dataclass every stage appended to in
#     place ("NoteEvent is a plain, non-frozen dataclass" justified this
#     repeatedly). Note here is frozen=True.
#   - stem identity was reconstructed by string-parsing reasoning_chain
#     for a substring like "stem:bass" (7 call sites). Note.stem is now a
#     real StemType field.
#   - tag-propagation code keyed notes by (pitch, start_ms) alone, with no
#     stem in the key - two different notes at the same pitch/timestamp
#     from different stems could silently share tags. Note.id is a real,
#     unique, stable identity so annotation lookups never collide.
#
# A frozen dataclass only blocks REASSIGNING a field - a List[str] field
# would still be mutable in place even if frozen (`note.tags.append(...)`
# still works). That's why there is no list/dict field on Note at all:
# anything that accumulates opinions about a note lives in AnnotationStore
# (annotation_types.py), keyed by Note.id, never on the Note itself.
# =================================================================

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from core.confidence import clamp_confidence
from core.stem_types import StemType
from core.source_types import Provenance


def _new_note_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Note:
    """A single detected note. This is the transcription - Basic Pitch's
    (or CREPE's, for bass) own reading, never blended or overwritten.

    id: stable identity for AnnotationStore lookups - never reconstruct
        this from (pitch, start_ms) or any other composite key; two real,
        distinct notes can share both.
    """
    pitch: int
    start_ms: float
    end_ms: float
    velocity: int
    confidence: float
    stem: StemType
    source: Provenance
    fundamental_freq_hz: Optional[float] = None
    id: str = field(default_factory=_new_note_id)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", clamp_confidence(self.confidence))
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"Note {self.id}: end_ms ({self.end_ms}) < start_ms ({self.start_ms})"
            )
        if not (0 <= self.pitch <= 127):
            raise ValueError(f"Note {self.id}: pitch {self.pitch} outside MIDI 0-127")

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms
