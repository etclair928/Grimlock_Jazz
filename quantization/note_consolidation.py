# =================================================================
# MODULE: quantization/note_consolidation.py
# Consolidation pass: glue Basic Pitch's fragmented same-pitch runs back
# into single sustained notes.
#
# THE PROBLEM (measured on You Say God Says, 2026-07-28). Basic Pitch
# re-triggers a held note into a machine-gun burst of short same-pitch
# fragments: across the pitched tracks ~2000 notes are same-pitch runs
# with <50 ms gaps that a listener hears as ONE sustained note (mid_body
# alone: 641 notes that should be ~220). The playback is "jittery / all
# 32nd and 64th notes / no sustain." This is the "consolidate before you
# analyze" keystone applied at the note level.
#
# PORTED INSIGHTS from Symphony 5.6.1 agents/quantization/velocity_merge.py
# (the module the user pointed at) and the Consensus philosophy:
#   * PITCH EQUALITY IS A HARD GATE, never a tolerance. F4 and Eb4 are 2
#     semitones apart; a tolerance would silently swallow a real chromatic
#     neighbour into a wrong-pitch merge. Only identical pitch (and same
#     stem) can merge.
#   * CONNECTED-COMPONENT grouping, not pairwise: a run of five fragments
#     collapses to one sustained note, spanning earliest onset -> latest
#     offset - not four separate two-note merges.
#   * Consensus's conservative bias ("rather miss than hallucinate"): when
#     the gap is large enough to be a real repeated note, DON'T merge. The
#     gate is a small gap - a genuine re-articulation leaves at least a
#     subdivision of space; a detection fragment does not.
#
# §2 COMPLIANCE. Detected notes are frozen. This pass NEVER merges,
# rewrites, or deletes a Note - it writes a "consolidation" Annotation on
# each note (primary carries the merged end; absorbed fragments point at
# their primary), and the engraver RENDERS the consolidated span only when
# use_consolidated_timing is set. Default output is untouched - this is an
# opt-in timeline to A/B against raw, exactly like notation/groove timing.
# =================================================================

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from core import AnnotationStore, Annotation, Note, Provenance

CONSOLIDATION_ANNOTATION_KIND = "consolidation"

# Same-pitch gap below which a second onset is a Basic Pitch re-trigger, not
# a real repeated note. A genuine re-articulation leaves at least ~a 32nd of
# space (61 ms at 123 BPM); fragments touch or overlap. 60 ms is deliberately
# below a 32nd at typical tempi so we merge artifacts, not real fast repeats.
DEFAULT_MERGE_GAP_MS = 60.0


def consolidate_fragments(
        notes: List[Note],
        annotations: AnnotationStore,
        max_gap_ms: float = DEFAULT_MERGE_GAP_MS,
) -> Tuple[int, int]:
    """Finds runs of same-(stem, pitch) notes separated by <= max_gap_ms
    (overlaps included) and annotates them: the earliest note of each run
    is the "primary" (carries the merged end_ms); the rest are "absorbed"
    (point at the primary's id). Writes nothing when a note stands alone.
    Returns (runs_merged, fragments_absorbed). Never mutates a Note."""
    by_voice: Dict[Tuple[object, int], List[Note]] = defaultdict(list)
    for note in notes:
        by_voice[(note.stem, note.pitch)].append(note)

    runs_merged = 0
    fragments_absorbed = 0
    for group in by_voice.values():
        group.sort(key=lambda n: n.start_ms)
        i = 0
        while i < len(group):
            run = [group[i]]
            run_end = group[i].end_ms
            j = i + 1
            while j < len(group) and group[j].start_ms - run_end <= max_gap_ms:
                run.append(group[j])
                run_end = max(run_end, group[j].end_ms)
                j += 1
            if len(run) > 1:
                primary = run[0]
                annotations.add(Annotation(
                    note_id=primary.id, kind=CONSOLIDATION_ANNOTATION_KIND,
                    value={"role": "primary", "end_ms": run_end, "absorbed": len(run) - 1},
                    source=Provenance.CONSOLIDATION, confidence=0.8,
                ))
                for frag in run[1:]:
                    annotations.add(Annotation(
                        note_id=frag.id, kind=CONSOLIDATION_ANNOTATION_KIND,
                        value={"role": "absorbed", "into": primary.id},
                        source=Provenance.CONSOLIDATION, confidence=0.8,
                    ))
                runs_merged += 1
                fragments_absorbed += len(run) - 1
            i = j

    return runs_merged, fragments_absorbed


__all__ = ["consolidate_fragments", "CONSOLIDATION_ANNOTATION_KIND", "DEFAULT_MERGE_GAP_MS"]
