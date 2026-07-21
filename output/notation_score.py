# =================================================================
# MODULE: output/notation_score.py
# The NotationScore - the symbolic object that sits between the
# pipeline's evidence (frozen Notes + AnnotationStore) and the
# exporters (see GRIMLOCK_6.0_NOTATION_GRAPH.md).
#
# It owns exactly the relationships a copyist needs and MIDI cannot
# carry. This first slice owns ONE of them - VOICE membership - because
# that is the one the user named ("guitar is a mess of polyphony... a
# trouble spot") and the one whose input (VoiceLine.line_id) the
# pipeline already computes and then, until now, discarded.
#
# The law is unchanged: nothing here mutates a Note. The score is
# BUILT from notes + annotations and READ by serializers. Every future
# edge (ties, tuplets, beams, spelling, gesture groups) becomes another
# field on NotationPart/NotationScore, added one measured slice at a
# time - never a global solver, never a magic number, never a guess
# that fails silently.
# =================================================================

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core import AnnotationStore, Note, StemType
from instrument_attribution.resolve import ANNOTATION_KIND as FAMILY_ANNOTATION_KIND
from instrument_attribution.resolve import VOICE_ANNOTATION_KIND
from quantization import ONSET_REFINEMENT_ANNOTATION_KIND, SUSTAIN_RECOVERY_ANNOTATION_KIND

# A voice's notes are monophonic by construction (stream_into_lines only
# joins a note to a line when it does NOT overlap that line's last note).
# So "hold each note until the next one in its own voice" is not a policy
# choice with a magic threshold - it is what a monophonic line IS. The
# only real decision is when to break the legato with a rest, and that is
# driven by SustainRecovery's posteriorgram evidence (the extended end),
# not by a constant. This is exactly why voice separation and clean
# rhythm are the SAME move, not two.
_DRUM_FAMILY = "drums"


@dataclass
class NotationNote:
    """One note as the page will see it. The physical Note is untouched;
    these are the resolved values a serializer reads."""
    pitch: int
    start_ms: float           # onset the page uses (refined when available)
    end_ms: float             # end the page uses (sustain-extended when available)
    velocity: int
    source_note_id: str       # provenance back to the frozen Note

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass
class NotationPart:
    """One staff. A single monophonic voice of a single instrument
    family - the unit that stops the polyphony mush."""
    family: str
    voice_id: str
    stem: StemType
    notes: List[NotationNote] = field(default_factory=list)
    is_drum: bool = False

    @property
    def mean_pitch(self) -> float:
        return sum(n.pitch for n in self.notes) / len(self.notes) if self.notes else 60.0


@dataclass
class NotationScore:
    """The resolved symbolic world, as a page. One part per voice."""
    parts: List[NotationPart] = field(default_factory=list)
    tempo_bpm: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    key: Optional[str] = None

    @property
    def total_notes(self) -> int:
        return sum(len(p.notes) for p in self.parts)


def _page_timing(note: Note, annotations: AnnotationStore) -> Tuple[float, float]:
    """The (start, end) the page should use, reading the same evidence
    the engraver's groove timeline reads: the refined onset (a detector-
    latency correction, not an interpretive choice) and the sustain-
    recovery end (posteriorgram-backed, replaces Basic Pitch's known-
    early note-off). Falls back to the frozen values when absent."""
    refined = annotations.latest_value(note.id, ONSET_REFINEMENT_ANNOTATION_KIND)
    start_ms = refined.get("start_ms", note.start_ms) if refined else note.start_ms

    end_ms = note.end_ms
    sustain = annotations.latest_value(note.id, SUSTAIN_RECOVERY_ANNOTATION_KIND)
    if sustain:
        end_ms = max(end_ms, sustain.get("end_ms", end_ms))

    # A refined onset moves the START only; if it landed past the end,
    # keep the note non-degenerate rather than inverting it.
    if start_ms >= end_ms:
        end_ms = start_ms + max(1.0, note.end_ms - note.start_ms)
    return start_ms, end_ms


def build_notation_score(
        notes: List[Note],
        annotations: AnnotationStore,
        tempo_bpm: float,
        time_signature: Tuple[int, int],
        key: Optional[str] = None,
        use_voices: bool = True,
) -> NotationScore:
    """Groups notes into parts and resolves each note's page timing.

    `use_voices=True` (default): one part per (family, voice), using the
    `voice` annotation from voice_continuity. This is only as good as the
    voice separation upstream - while that over-fragments, it yields one
    part per greedy line (many tiny staves).

    `use_voices=False`: one part per FAMILY, leaving all of an
    instrument's polyphony together. The exporter then splits that
    polyphony into the fewest voice-legal staves itself. Until voice
    separation is fixed this is the more readable of the two - a handful
    of dense staves instead of hundreds of fragments.

    Notes with no `voice` annotation (e.g. drums) always fall back to a
    single per-family part - they are never dropped."""
    buckets: Dict[Tuple[str, str, StemType], List[NotationNote]] = defaultdict(list)
    drum_flag: Dict[Tuple[str, str, StemType], bool] = {}

    for note in notes:
        family = annotations.latest_value(note.id, FAMILY_ANNOTATION_KIND)
        is_drum = note.stem == StemType.DRUMS or family == _DRUM_FAMILY
        if family is None:
            family = _DRUM_FAMILY if is_drum else "unknown"

        voice_id = annotations.latest_value(note.id, VOICE_ANNOTATION_KIND) if use_voices else None
        if voice_id is None:
            # Drums, un-voiced notes, and family-only mode collapse to one
            # part per family - a single bucket the exporter then voices
            # legally. Correct for percussion, and honest ("no voice
            # evidence used here") for the rest.
            voice_id = f"{family}::all"

        key_tuple = (family, voice_id, note.stem)
        start_ms, end_ms = _page_timing(note, annotations)
        buckets[key_tuple].append(NotationNote(
            pitch=note.pitch, start_ms=start_ms, end_ms=end_ms,
            velocity=note.velocity, source_note_id=note.id,
        ))
        drum_flag[key_tuple] = is_drum

    parts: List[NotationPart] = []
    for (family, voice_id, stem), nnotes in buckets.items():
        nnotes.sort(key=lambda n: n.start_ms)
        parts.append(NotationPart(
            family=family, voice_id=voice_id, stem=stem,
            notes=nnotes, is_drum=drum_flag[(family, voice_id, stem)],
        ))

    # Stable, readable part order: non-drums by descending register (so a
    # score reads treble-to-bass top-to-bottom), drums last.
    parts.sort(key=lambda p: (p.is_drum, -p.mean_pitch))
    return NotationScore(
        parts=parts, tempo_bpm=tempo_bpm,
        time_signature=time_signature, key=key,
    )


__all__ = [
    "NotationNote",
    "NotationPart",
    "NotationScore",
    "build_notation_score",
]
