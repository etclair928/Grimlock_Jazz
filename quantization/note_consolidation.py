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

import bisect
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from core import AnnotationStore, Annotation, Note, Provenance

CONSOLIDATION_ANNOTATION_KIND = "consolidation"

# Same-pitch gap below which a second onset is a Basic Pitch re-trigger, not
# a real repeated note. A genuine re-articulation leaves at least ~a 32nd of
# space (61 ms at 123 BPM); fragments touch or overlap. 60 ms is deliberately
# below a 32nd at typical tempi so we merge artifacts, not real fast repeats.
DEFAULT_MERGE_GAP_MS = 60.0

# BELOW THIS GAP THE ATTACK GATE DOES NOT APPLY, because no player can
# re-articulate that fast and so no onset found there can be a second attack.
#
# WHY IT IS NEEDED. The attack gate is binary: any onset within 50ms of the
# boundary stops the run. That was added to fix real over-merging - measured on
# Ellington, consolidation had been deleting 31% of notes and 43% of the 4+
# note chords - and it swung the other way. Measured across the corpus, runs
# from before the gate absorbed 100% of eligible fragments and runs after it
# absorb 24-64%:
#
#     run                     eligible  absorbed  caught
#     Hopeful_UNI (pre-gate)      1376      1376    100%
#     Hopeful_TUP (post-gate)     1376       878     64%
#     HRV_GATE    (pre-gate)       857       857    100%
#     HRV_FULLRUN (post-gate)      864       310     36%
#     Chopin                       882       216     24%
#
# WHAT THE GATE IS ACTUALLY BLOCKING. Of the merges it stops, 96-99% are notes
# that end and restart within 15ms of each other - 642 of Chopin's 666, 471 of
# Burden's 478. Fifteen milliseconds is a sixty-fourth note at 240bpm. It is
# not a performed gap; it is where Basic Pitch drew a segmentation boundary
# inside one sounding event, and the onset the gate sees there is the note's
# OWN attack, found again.
#
# So the exemption is a statement about physics rather than a tuned threshold,
# and it is deliberately far below any real repeat. The 7-24 pairs per song
# that DO have 15-60ms of genuine silence stay gated - that band is where a
# fast repeated note lives, and it is what the Ellington measurement was
# protecting.
MIN_REARTICULATION_GAP_MS = 20.0


# How close a detected onset must be to the boundary between two same-pitch
# notes to count as a fresh attack there.
ATTACK_WINDOW_MS = 50.0


def _has_attack(boundary_ms: float, onsets_ms: Sequence[float]) -> bool:  # noqa: D401
    """Is there a detected attack at this boundary? This is the ONLY thing that
    separates a re-trigger from a re-articulation.

    MEASURED on Ellington (538 candidates) and Chopin (331): every single
    merge candidate has a gap of EXACTLY 0.0ms, zero overlap, and a
    second/first velocity ratio of ~0.98 with 46% louder. Basic Pitch segments a
    continuous pitch activation at frame boundaries, so a held or repeated pitch
    arrives as abutting notes regardless of whether the player struck it again.
    Gap, overlap and velocity therefore carry NO information here - the audio
    does."""
    if not onsets_ms:
        return False
    # First onset at or after the window's start; it answers the question by
    # itself, since anything later is further away. (This was written as a
    # `while` with an unconditional `return True` and no increment - correct,
    # but an `if` wearing a loop's clothes.)
    i = bisect.bisect_left(onsets_ms, boundary_ms - ATTACK_WINDOW_MS)
    return i < len(onsets_ms) and onsets_ms[i] <= boundary_ms + ATTACK_WINDOW_MS


def consolidate_fragments(
        notes: List[Note],
        annotations: AnnotationStore,
        max_gap_ms: float = DEFAULT_MERGE_GAP_MS,
        onsets_ms: Optional[Sequence[float]] = None,
        onsets_by_stem: Optional[Dict[object, Sequence[float]]] = None,
) -> Tuple[int, int]:
    """Finds runs of same-(stem, pitch) notes separated by <= max_gap_ms
    (overlaps included) and annotates them: the earliest note of each run
    is the "primary" (carries the merged end_ms); the rest are "absorbed"
    (point at the primary's id). Writes nothing when a note stands alone.
    Returns (runs_merged, fragments_absorbed). Never mutates a Note."""
    by_voice: Dict[Tuple[object, int], List[Note]] = defaultdict(list)
    for note in notes:
        by_voice[(note.stem, note.pitch)].append(note)

    # OWN-STEM ONSETS. The first version consulted the MASTER MIX's onsets, so
    # on dense material every drum hit voted on whether a held vocal note had
    # been restruck. MEASURED regression on You Say God Says: the old rule
    # absorbed ~2,300 of 2,285 same-pitch fragment pairs (4,847 -> 2,547 notes);
    # the mix-onset gate absorbed only 859 (17.8%), leaving 33.5% of the output
    # at a 32nd note or shorter at 62 BPM - the exact "jittery, all 32nd and
    # 64th notes" complaint the module was built to fix. Solo piano improved and
    # band material regressed, because onset DENSITY differs by an order of
    # magnitude between them. A piano re-articulation must be evidenced by a
    # piano attack, not a snare.
    runs_merged = 0
    fragments_absorbed = 0
    for (stem, _pitch), group in by_voice.items():
        stem_onsets = (onsets_by_stem or {}).get(stem)
        if stem_onsets is None:
            stem_onsets = onsets_ms or ()
        group.sort(key=lambda n: n.start_ms)
        i = 0
        while i < len(group):
            run = [group[i]]
            run_end = group[i].end_ms
            j = i + 1
            while (j < len(group) and group[j].start_ms - run_end <= max_gap_ms
                   # A fresh attack at the boundary means the player struck this
                   # pitch again - stop the run rather than absorbing it. Without
                   # this, planing textures (where the same pitches recur by
                   # design) lose their voicings: MEASURED on Ellington's
                   # Reflections in D, consolidation deleted 31% of notes and 43%
                   # of the 4+ note chords, taking mean chord size to 3.16 against
                   # the reference's 6.07.
                   #
                   # UNLESS the two notes are touching. Below
                   # MIN_REARTICULATION_GAP_MS no player could have struck the
                   # pitch again, so whatever onset sits at the boundary is this
                   # note's own attack seen a second time - not evidence of a
                   # repeat. See that constant for the measurement.
                   and (group[j].start_ms - run_end < MIN_REARTICULATION_GAP_MS
                        or not _has_attack(group[j].start_ms, stem_onsets))):
                run.append(group[j])
                run_end = max(run_end, group[j].end_ms)
                j += 1
            if len(run) > 1:
                primary = run[0]
                annotations.add(Annotation(
                    note_id=primary.id, kind=CONSOLIDATION_ANNOTATION_KIND,
                    # `end_ms` is a RAW performance millisecond - it is the only
                    # timeline that exists here. `last_note_id` lets a consumer
                    # that is rendering a DIFFERENT timeline (notation, groove)
                    # ask that fragment for its end on that timeline instead of
                    # pasting this raw value onto a snapped start (2026-08-17
                    # audit).
                    value={"role": "primary", "end_ms": run_end,
                           "absorbed": len(run) - 1,
                           "last_note_id": max(run, key=lambda n: n.end_ms).id},
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


__all__ = [
    "MIN_REARTICULATION_GAP_MS","consolidate_fragments", "CONSOLIDATION_ANNOTATION_KIND", "DEFAULT_MERGE_GAP_MS"]
