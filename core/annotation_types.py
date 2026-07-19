# =================================================================
# MODULE: core/annotation_types.py
# "Annotation, not mutation" (GRIMLOCK_6.0_DESIGN_DECISIONS.md §2.2),
# applied literally: even a note's own audit trail doesn't live on the
# note. Every opinion any pass has about a Note - instrument family, voice
# grouping, articulation, a proposed quantized onset, a duration
# hypothesis - is a typed, append-only Annotation keyed by Note.id in an
# AnnotationStore, never a field mutated (or a list appended to) on the
# Note itself.
#
# This is also the one place Music_Box-style traceability lives at the
# per-note level: `store.for_note(note.id)` gives the complete, ordered
# history of every opinion any pass ever formed about that specific note -
# "I can explain every note" (§7 Conductor principle), for free, from the
# data structure itself rather than a separate forensic reconstruction.
# =================================================================

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.source_types import Provenance


@dataclass(frozen=True)
class Annotation:
    """One typed opinion about one note. Immutable - once written, an
    annotation is never edited; a pass that changes its mind writes a NEW
    annotation with the same (note_id, kind), and AnnotationStore.latest()
    returns the most recent one. The old one stays in the ledger."""
    note_id: str
    kind: str  # e.g. "instrument_family", "voice_line", "articulation"
    value: Any
    source: Provenance
    confidence: float = 1.0
    contested: bool = False  # e.g. a voice-line's timbre vote was bimodal


class AnnotationStore:
    """Append-only side-table. The single place every pass's opinions
    about notes accumulate. Never mutates a Note; never deletes an
    Annotation once written."""

    def __init__(self) -> None:
        self._by_note: Dict[str, List[Annotation]] = defaultdict(list)

    def add(self, annotation: Annotation) -> None:
        self._by_note[annotation.note_id].append(annotation)

    def for_note(self, note_id: str) -> List[Annotation]:
        """Full ordered history of every annotation for this note."""
        return list(self._by_note.get(note_id, []))

    def latest(self, note_id: str, kind: str) -> Optional[Annotation]:
        """Most recent annotation of a given kind for this note, or None."""
        matches = [a for a in self._by_note.get(note_id, []) if a.kind == kind]
        return matches[-1] if matches else None

    def latest_value(self, note_id: str, kind: str, default: Any = None) -> Any:
        ann = self.latest(note_id, kind)
        return ann.value if ann is not None else default

    def all_of_kind(self, kind: str) -> List[Annotation]:
        """Every annotation of a given kind across every note - e.g. to
        gather all instrument_family votes when resolving a voice line."""
        return [a for anns in self._by_note.values() for a in anns if a.kind == kind]
