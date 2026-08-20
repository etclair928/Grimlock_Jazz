# =================================================================
# MODULE: acoustic_witness/octave_stack.py
# THE OCTAVE-HAPPY FILTER, calibrated against a published edition before
# it was allowed to drop anything.
#
# WHAT IT CATCHES. A pitch detector run on a piano hears the harmonic
# series and sometimes reports the partials as notes: strike C3 and it
# emits C3, C4 and C5. The signature is a "stack" - one pitch class at
# three or more octaves inside a single struck instant.
#
# WHY THE INTERIOR AND ONLY THE INTERIOR. Octave doubling is ordinary
# piano writing: the left hand takes the octave, the right hand the top.
# Those are the OUTER members of the stack. What no pianist writes is the
# filling between them, and the published edition of the Nocturne
# Op.62 No.1 says so numerically:
#
#     octave chains per struck chord
#     chain length          2        3        4+
#     ANSWER KEY          96%       4%        0%
#     KLANGIO             95%       5%        0%
#     OURS                88%      11%        1%
#
# A four-octave stack does not occur in the edition at all, and three-
# octave stacks occur at a quarter of our rate. So the rule is: keep the
# lowest and the highest member of a chain of >= 3, flag what is between.
# Chains of exactly two are never touched - that is the writing itself.
#
# WHAT IT IS WORTH, measured by ablation against the answer key rather
# than asserted (tools/octave_ablation.py, which calls THIS function - an
# earlier version of that tool reimplemented the rule and reported a
# number this code does not produce):
#
#     variant                kept   prec   recall    F1
#     baseline               1920  0.374   0.785   0.507
#     octave stacks dropped  1881  0.379   0.779   0.510
#
# 39 notes dropped, of which 34 had no counterpart in the answer key:
# 87.2% precise, for +0.0035 F1. Small and real. It is not a headline
# number and is not offered as one - it is one witness among four, and
# the discipline that matters is that its cost in recall (0.0055) was
# measured rather than hoped for.
#
# WHAT IT DELIBERATELY LEAVES ON THE TABLE. Counting two detections of the
# SAME pitch as two rungs of the chain would flag 238 notes instead of 39,
# at 94.6% precision and +0.036 F1 - ten times the yield. Those extra
# notes are real errors, but they are RE-STRIKES, which is
# quantization/note_consolidation's problem and not an octave stack. A
# filter that swallows a neighbouring pass's job scores well and makes the
# system harder to reason about; the yield is left where it belongs.
#
# WHY NOT THE PLAYABILITY MODEL. output/playability.py can also call a
# stack impossible, and the same ablation says that route is far weaker:
# physical impossibility fires on only 15 chords, and it fires on the
# EDITION too (2.4% - a bass note under a rolled right-hand chord, which
# is played with the pedal and is perfectly real). Impossibility is a
# property of the writing; the octave stack is a property of the
# DETECTOR. Only the second one is safe to act on.
#
# ANNOTATION, NOT MUTATION (DESIGN_DECISIONS 2.2). Nothing here deletes a
# Note. It writes one typed annotation per flagged note, carrying its
# reason, and the notation builder honors it on the page. The performance
# clock is untouched, exactly as university/apply.py does it - delete the
# annotations and the page comes back.
# =================================================================

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

from core.annotation_types import Annotation, AnnotationStore
from core.source_types import Provenance

OCTAVE_STACK_ANNOTATION_KIND = "octave_stack"

# Notes struck this close together are one gesture. Deliberately tight:
# the question is what was STRUCK together, not what is SOUNDING together.
# Under the sustain pedal a wide sounding span is completely ordinary, and
# scoring on sounding-together instants inflates our apparent error rate
# from 4.4% to 13.8% while telling us nothing about detection.
STRIKE_WINDOW_MS = 50.0

# Below this many octaves of one pitch class, leave it alone - that is
# octave doubling, which is how piano music is written.
MIN_STACK = 3

# The AND-gate, mirroring university/apply.py SUPPRESS_MAX_CONFIDENCE: a
# CONFIDENT interior note stays on the page even if it looks stacked.
# Honest note on this number - it is a no-op on the Chopin measurement,
# where the most confident interior copy scored 0.82. It is a guard for
# material where the detector is surer of itself, not a tuned threshold,
# and it is set loose on purpose so it never quietly becomes the thing
# doing the work.
INTERIOR_MAX_CONFIDENCE = 0.90


def _strike_groups(notes: Sequence, window_ms: float) -> List[List[int]]:
    """Indices grouped by onset proximity - one struck gesture each.

    Anchored on the group's FIRST onset rather than the previous one, so a
    dense run cannot chain into one arbitrarily long group.
    """
    order = sorted(range(len(notes)), key=lambda i: notes[i].start_ms)
    groups: List[List[int]] = []
    current: List[int] = []
    anchor = None
    for i in order:
        start = float(notes[i].start_ms)
        if anchor is None or start - anchor <= window_ms:
            if anchor is None:
                anchor = start
            current.append(i)
        else:
            groups.append(current)
            current, anchor = [i], start
    if current:
        groups.append(current)
    return groups


def find_octave_stacks(
        notes: Sequence,
        window_ms: float = STRIKE_WINDOW_MS,
        min_stack: int = MIN_STACK,
        max_confidence: float = INTERIOR_MAX_CONFIDENCE,
) -> List[Tuple[str, Dict]]:
    """Flag the interior members of octave stacks.

    Returns (note_id, detail) pairs, one per flagged note. Pure: reads
    Notes, writes nothing, deletes nothing.
    """
    flagged: List[Tuple[str, Dict]] = []
    for group in _strike_groups(notes, window_ms):
        by_pc: Dict[int, List[int]] = {}
        for i in group:
            by_pc.setdefault(int(notes[i].pitch) % 12, []).append(i)

        for pitch_class, members in by_pc.items():
            # De-duplicate by PITCH, not by index: two detections of the
            # same pitch in one instant are a re-strike (which is
            # consolidation's problem, not this one), and must not
            # inflate the chain length into a stack that is not there.
            by_pitch: Dict[int, int] = {}
            for i in members:
                p = int(notes[i].pitch)
                if p not in by_pitch or notes[i].confidence > notes[by_pitch[p]].confidence:
                    by_pitch[p] = i
            if len(by_pitch) < min_stack:
                continue

            chain = [by_pitch[p] for p in sorted(by_pitch)]
            octaves = [int(notes[i].pitch) for i in chain]
            for i in chain[1:-1]:                       # interior only
                conf = float(notes[i].confidence)
                if conf > max_confidence:
                    continue                            # confident - leave it
                flagged.append((notes[i].id, {
                    "role": "interior",
                    "pitch": int(notes[i].pitch),
                    "pitch_class": pitch_class,
                    "stack": octaves,
                    "kept": [octaves[0], octaves[-1]],
                    "note_confidence": conf,
                    "reason": (
                        f"octave stack: pitch class {pitch_class} struck at "
                        f"{len(octaves)} octaves {octaves} within "
                        f"{window_ms:.0f}ms - the edition never writes more "
                        f"than 2, so the outer pair is kept and this interior "
                        f"copy (confidence {conf:.2f}) reads as a harmonic of "
                        f"one of them"),
                }))
    return flagged


def write_octave_annotations(
        notes: Sequence,
        annotations: AnnotationStore,
        source: Provenance,
        window_ms: float = STRIKE_WINDOW_MS,
        min_stack: int = MIN_STACK,
        max_confidence: float = INTERIOR_MAX_CONFIDENCE,
) -> Dict[str, int]:
    """Record the flags in the store. Returns counts for the MusicBox trail."""
    flagged = find_octave_stacks(notes, window_ms, min_stack, max_confidence)
    for note_id, detail in flagged:
        # The flag is most credible where the note itself is least
        # credible, so the annotation carries the complement of the
        # note's own confidence rather than a flat 1.0.
        annotations.add(Annotation(
            note_id=note_id, kind=OCTAVE_STACK_ANNOTATION_KIND,
            value=detail, source=source,
            confidence=max(0.0, min(1.0, 1.0 - float(detail["note_confidence"]))),
        ))
    stacks = len({tuple(d["stack"]) for _n, d in flagged})
    return {"interior_flagged": len(flagged), "distinct_stacks": stacks}


def octave_suppression_ids(annotations: AnnotationStore,
                           note_ids: Iterable[str]) -> Set[str]:
    """Read back what to drop from the PAGE. Mirrors
    university.apply.page_suppression_map so the notation builder can
    treat both the same way."""
    drop: Set[str] = set()
    for note_id in note_ids:
        value = annotations.latest_value(note_id, OCTAVE_STACK_ANNOTATION_KIND)
        if value and value.get("role") == "interior":
            drop.add(note_id)
    return drop


__all__ = [
    "OCTAVE_STACK_ANNOTATION_KIND", "STRIKE_WINDOW_MS", "MIN_STACK",
    "INTERIOR_MAX_CONFIDENCE", "find_octave_stacks",
    "write_octave_annotations", "octave_suppression_ids",
]
