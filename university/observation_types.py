# =================================================================
# MODULE: university/observation_types.py
# Types for Grimlock University (GRIMLOCK_UNIVERSITY.md).
#
# THE FIX that makes this layer joinable: an observation references
# REAL Note.id values, never bare pitches. The pasted proposal stored
# List[int] pitches, so its patterns could never be mapped back to the
# notes that produced them - its voice assignment was a silent no-op.
# Everything downstream (annotations, apply, the corpus) depends on
# note_ids being real.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Annotation kinds this layer writes. Distinct from every existing kind
# so nothing already in the pipeline can collide with, or accidentally
# read, university output.
PATTERN_STUDY_ANNOTATION_KIND = "pattern_study"
VOICE_COHESION_ANNOTATION_KIND = "pattern_voice_cohesion"
PAGE_SUPPRESS_ANNOTATION_KIND = "pattern_page_suppress"


class UniversityMode(str, Enum):
    """How much the University is allowed to do.

    OFF   - nothing runs. Grimlock Jazz behaves EXACTLY as it does today,
            byte-for-byte. This is the default and the safety net.
    STUDY - detectors run; findings are annotated + logged + written to a
            corpus file. The pipeline's OUTPUT IS UNCHANGED (asserted by
            test). This is the "observe, don't act" mode.
    APPLY - everything STUDY does, plus the notation VIEW honors what was
            found: pattern-cohesive notes are kept in one voice, and
            notes a pattern's context says are spurious are dropped from
            the page. Frozen Notes are still never mutated - the
            performance clock (MIDI) is untouched, only the page changes.
    """
    OFF = "off"
    STUDY = "study"
    APPLY = "apply"

    @classmethod
    def parse(cls, value: "UniversityMode | str | None") -> "UniversityMode":
        if value is None:
            return cls.OFF
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.OFF

    @property
    def runs(self) -> bool:
        return self is not UniversityMode.OFF

    @property
    def applies(self) -> bool:
        return self is UniversityMode.APPLY


@dataclass(frozen=True)
class PatternObservation:
    """One thing a detector claims to have seen.

    `note_ids` are real Note.id values - that is what makes an
    observation joinable back to the frozen notes and their annotations.
    `falsifier` is a shipping requirement: a detector that cannot say
    what would prove it wrong does not belong here.
    """
    pattern: str                       # "scale_run", "arpeggio", "alberti_bass", ...
    department: str                    # "universal" | "keyboard" | "rhythmic"
    note_ids: Tuple[str, ...]
    start_ms: float
    end_ms: float
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    falsifier: str = ""
    stem: Optional[str] = None

    @property
    def span_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)

    @property
    def size(self) -> int:
        return len(self.note_ids)


@dataclass
class StudyReport:
    """Everything one study() pass found. Pure data - no logic, nothing
    acts on this unless the mode says APPLY."""
    observations: List[PatternObservation] = field(default_factory=list)
    notes_studied: int = 0
    mode: UniversityMode = UniversityMode.OFF

    @property
    def covered_note_ids(self) -> set:
        out: set = set()
        for obs in self.observations:
            out.update(obs.note_ids)
        return out

    @property
    def coverage(self) -> float:
        """Fraction of studied notes that any pattern claims. The headline
        number for the null-model comparison (null_model.py)."""
        if not self.notes_studied:
            return 0.0
        return len(self.covered_note_ids) / self.notes_studied

    def by_pattern(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obs in self.observations:
            counts[obs.pattern] = counts.get(obs.pattern, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "notes_studied": self.notes_studied,
            "observations": len(self.observations),
            "coverage": round(self.coverage, 4),
            "by_pattern": self.by_pattern(),
        }


__all__ = [
    "UniversityMode", "PatternObservation", "StudyReport",
    "PATTERN_STUDY_ANNOTATION_KIND", "VOICE_COHESION_ANNOTATION_KIND",
    "PAGE_SUPPRESS_ANNOTATION_KIND",
]
