# =================================================================
# MODULE: university/study.py
# The read-only study pass (GRIMLOCK_UNIVERSITY.md §4-§6).
#
# CONTRACT: study() takes notes + annotations + resolved params and
# returns a StudyReport. It NEVER returns Notes, never constructs one,
# never calls replace() on one. The frozen detection floor is untouched
# by construction - which is why OFF and STUDY produce byte-identical
# pipeline output.
#
# It consumes what Jazz already knows rather than re-deriving it:
#   quantization.note_consolidation  -> skip absorbed fragments, so we
#                                       study MUSIC, not Basic Pitch
#                                       confetti (measured: 52% of
#                                       adjacent notes were same-pitch
#                                       rearticulations)
#   instrument_attribution.resolve   -> per-note family (dialect gate)
#   instrument_attribution.voice_
#     continuity.stream_into_lines   -> real monophonic streams, instead
#                                       of the pasted proposal's
#                                       re-implemented gap heuristic
#   rhythm_engine (via tempo_bpm)    -> BEAT-RELATIVE detector grammar
# =================================================================

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from university.detectors import (
    DETECTORS, KEYBOARD_DETECTORS, UNIVERSAL_DETECTORS, VERTICAL_DETECTORS,
)
from university.observation_types import (
    PatternObservation, StudyReport, UniversityMode,
)

# Families/stems that unlock the keyboard department. On a BAND mix the
# merged harmonic stem is the honest home for keyboard-shaped figures;
# instrument-specific dialects (flamenco, metal, classical piano) are
# deliberately NOT offered here because stem_merge measured 53%
# cross-stem duplication - the instrument label is not trustworthy on
# band material (GRIMLOCK_UNIVERSITY.md §6.2).
_KEYBOARD_STEMS = {"other", "piano", "guitar"}
_KEYBOARD_FAMILIES = {"piano", "guitar", "keyboard", "keys", "harp"}

_MIN_STREAM_NOTES = 4          # nothing in the curriculum can fire below this


def _consolidation_absorbed(annotations, note_id: str) -> bool:
    try:
        from quantization.note_consolidation import CONSOLIDATION_ANNOTATION_KIND
        value = annotations.latest_value(note_id, CONSOLIDATION_ANNOTATION_KIND)
    except Exception:
        return False
    return bool(value) and value.get("role") == "absorbed"


def _family_of(annotations, note_id: str) -> Optional[str]:
    try:
        from instrument_attribution.resolve import ANNOTATION_KIND as FAMILY_KIND
        return annotations.latest_value(note_id, FAMILY_KIND)
    except Exception:
        return None


def _departments_for(stem_value: str, family: Optional[str]) -> Sequence[str]:
    """The dialect gate: which detectors this stream may enroll in."""
    names = [n for n in UNIVERSAL_DETECTORS if n not in VERTICAL_DETECTORS]
    if stem_value in _KEYBOARD_STEMS or (family or "") in _KEYBOARD_FAMILIES:
        names.extend(KEYBOARD_DETECTORS)
    return names


def study(
        notes: List,
        annotations,
        tempo_bpm: float,
        mode: UniversityMode = UniversityMode.STUDY,
        use_consolidation: bool = True,
) -> StudyReport:
    """Run the curriculum over the frozen notes. Read-only."""
    mode = UniversityMode.parse(mode)
    report = StudyReport(mode=mode)
    if not mode.runs or not notes:
        return report

    beat_ms = 60000.0 / max(float(tempo_bpm), 1.0)

    # Drums are discrete hits with no melodic content - the curriculum
    # has nothing to say about them yet, so they are not enrolled.
    by_stem: Dict[Any, List] = defaultdict(list)
    for note in notes:
        stem_value = getattr(note.stem, "value", str(note.stem))
        if stem_value == "drums":
            continue
        if use_consolidation and _consolidation_absorbed(annotations, note.id):
            continue
        by_stem[note.stem].append(note)

    # Purpose-built melodic streaming (university/streams.py). The pipeline's
    # voice_continuity is a consolidator for instrument IDENTITY and severs a
    # line at every sustain overlap, which made >=4-note patterns nearly
    # unreachable (measured: only 29-43% of notes sat in a line long enough to
    # detect anything). It is left untouched; this is a separate reader.
    from university.streams import extract_lines

    observations: List[PatternObservation] = []
    studied = 0
    for stem, stem_notes in by_stem.items():
        if len(stem_notes) < _MIN_STREAM_NOTES:
            continue
        studied += len(stem_notes)
        stem_value = getattr(stem, "value", str(stem))
        family = _family_of(annotations, stem_notes[0].id)
        detector_names = _departments_for(stem_value, family)

        # Vertical detectors read the whole stem: simultaneity cannot exist
        # inside a monophonic line by construction.
        for name in VERTICAL_DETECTORS:
            detector = DETECTORS.get(name)
            if detector is None:
                continue
            try:
                observations.extend(detector(stem_notes, beat_ms, stem_value))
            except Exception:
                pass

        for line in extract_lines(stem_notes, beat_ms):
            stream = sorted(line.notes, key=lambda n: n.start_ms)
            if len(stream) < _MIN_STREAM_NOTES:
                continue
            for name in detector_names:
                detector = DETECTORS.get(name)
                if detector is None:
                    continue
                try:
                    observations.extend(detector(stream, beat_ms, stem_value))
                except Exception:
                    # A misbehaving detector must never take down a
                    # transcription run - it just fails to graduate.
                    continue

    report.observations = observations
    report.notes_studied = studied
    return report


def write_corpus(report: StudyReport, path: str, song: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None) -> str:
    """Publish the findings. One JSON per song, accumulating into the
    dataset that makes the NEXT question answerable (which patterns
    actually predict a good page?)."""
    payload: Dict[str, Any] = {
        "song": song,
        "summary": report.summary(),
        "observations": [
            {
                "pattern": o.pattern, "department": o.department, "stem": o.stem,
                "start_ms": round(o.start_ms, 1), "end_ms": round(o.end_ms, 1),
                "size": o.size, "confidence": round(o.confidence, 3),
                "evidence": o.evidence, "falsifier": o.falsifier,
                "note_ids": list(o.note_ids),
            }
            for o in report.observations
        ],
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


__all__ = ["study", "write_corpus"]
