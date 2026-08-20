# =================================================================
# MODULE: tests/test_octave_stack.py
# Pins acoustic_witness/octave_stack.py.
#
# WHY THIS NEEDS PINNING HARDER THAN MOST. This is the one witness in the
# codebase that DELETES notes from an export, so every one of its refusals
# matters as much as its catches. The rule was chosen by ablation against a
# published edition, and the properties that made it win are the ones asserted
# here: it never touches a two-octave doubling (that IS piano writing), it
# never touches the outer members of a stack, and it never fires on notes that
# were not struck together - because under the sustain pedal a four-octave
# SOUNDING span is completely ordinary and reading it as an error inflated the
# apparent rate from 4.4% to 13.8%.
# =================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acoustic_witness.octave_stack import (  # noqa: E402
    OCTAVE_STACK_ANNOTATION_KIND, find_octave_stacks, octave_suppression_ids,
    write_octave_annotations,
)
from core.annotation_types import AnnotationStore  # noqa: E402
from core.note_types import Note  # noqa: E402
from core.source_types import Provenance  # noqa: E402
from core.separation_types import StemType  # noqa: E402


def note(pitch, start_ms, dur_ms=500.0, confidence=0.5, nid=None):
    n = Note(pitch=pitch, start_ms=start_ms, end_ms=start_ms + dur_ms,
             velocity=80, confidence=confidence, stem=StemType.OTHER,
             source=Provenance.BASIC_PITCH)
    return n


def pitches_of(flagged, notes):
    by_id = {n.id: n for n in notes}
    return sorted(by_id[nid].pitch for nid, _d in flagged)


# --- what it must NOT touch ------------------------------------------------

def test_two_octave_doubling_is_left_alone():
    """The load-bearing refusal. LH octave + RH note is how piano music is
    written, and the edition does it in 96% of its octave chains."""
    notes = [note(48, 0.0), note(60, 0.0)]
    assert find_octave_stacks(notes) == []


def test_outer_members_of_a_stack_are_never_flagged():
    notes = [note(48, 0.0), note(60, 0.0), note(72, 0.0)]
    assert pitches_of(find_octave_stacks(notes), notes) == [60]


def test_different_pitch_classes_never_stack():
    """A wide chord is not a stack. Only ONE pitch class repeating counts."""
    notes = [note(48, 0.0), note(52, 0.0), note(55, 0.0), note(59, 0.0),
             note(64, 0.0), note(67, 0.0)]
    assert find_octave_stacks(notes) == []


def test_notes_merely_sounding_together_are_not_a_stack():
    """THE PEDAL CAVEAT. Struck half a second apart and still ringing is a
    pedalled arpeggio, not a detector error."""
    notes = [note(48, 0.0, dur_ms=4000.0), note(60, 500.0, dur_ms=4000.0),
             note(72, 1000.0, dur_ms=4000.0)]
    assert find_octave_stacks(notes) == []


def test_a_confident_interior_note_survives():
    """The AND-gate, mirroring university/apply.py: pattern alone is not
    enough to delete real music."""
    notes = [note(48, 0.0), note(60, 0.0, confidence=0.99), note(72, 0.0)]
    assert find_octave_stacks(notes) == []


def test_same_pitch_twice_is_a_restrike_not_a_stack():
    """Two detections of ONE pitch in one instant is consolidation's problem.
    If they counted toward chain length, a doubled unison plus its octave
    would look like a 3-stack and lose a real note."""
    notes = [note(60, 0.0), note(60, 10.0), note(72, 0.0)]
    assert find_octave_stacks(notes) == []


# --- what it must catch ----------------------------------------------------

def test_three_octave_stack_loses_its_middle():
    notes = [note(48, 0.0), note(60, 10.0), note(72, 20.0)]
    assert pitches_of(find_octave_stacks(notes), notes) == [60]


def test_four_octave_stack_loses_both_interior_members():
    """The edition contains zero 4-stacks; we produce them at 1%."""
    notes = [note(36, 0.0), note(48, 0.0), note(60, 0.0), note(72, 0.0)]
    assert pitches_of(find_octave_stacks(notes), notes) == [48, 60]


def test_a_stack_inside_a_bigger_chord_is_still_found():
    notes = [note(48, 0.0), note(55, 0.0), note(60, 0.0), note(64, 0.0),
             note(72, 0.0)]
    assert pitches_of(find_octave_stacks(notes), notes) == [60]


def test_two_independent_stacks_in_one_chord():
    notes = [note(48, 0.0), note(60, 0.0), note(72, 0.0),
             note(50, 0.0), note(62, 0.0), note(74, 0.0)]
    assert pitches_of(find_octave_stacks(notes), notes) == [60, 62]


def test_the_reason_names_the_stack_it_kept():
    notes = [note(48, 0.0), note(60, 0.0), note(72, 0.0)]
    _nid, detail = find_octave_stacks(notes)[0]
    assert detail["stack"] == [48, 60, 72]
    assert detail["kept"] == [48, 72]
    assert "octave stack" in detail["reason"]


def test_grouping_is_anchored_so_a_run_cannot_chain():
    """Each group is measured from its own first onset. Anchoring on the
    PREVIOUS note instead would let a 40ms scale run of twenty notes collapse
    into one 'instant' and manufacture stacks out of a melodic line."""
    notes = [note(60 + i, i * 40.0) for i in range(20)]
    notes += [note(48, 0.0), note(72, 0.0)]
    flagged = find_octave_stacks(notes)
    # C5 (72) is struck at t=0 with C3 (48); the C4 in the run lands at
    # t=0 too, but its octave-mates further along the run must not join.
    assert all(d["stack"] == [48, 60, 72] for _n, d in flagged)


# --- the store round trip --------------------------------------------------

def test_annotations_round_trip_to_a_drop_set():
    notes = [note(48, 0.0), note(60, 0.0), note(72, 0.0)]
    store = AnnotationStore()
    counts = write_octave_annotations(notes, store, Provenance.OCTAVE_STACK)
    assert counts["interior_flagged"] == 1
    assert counts["distinct_stacks"] == 1

    drop = octave_suppression_ids(store, [n.id for n in notes])
    assert len(drop) == 1
    assert notes[1].id in drop
    assert notes[0].id not in drop and notes[2].id not in drop


def test_nothing_is_mutated():
    """Annotation, not mutation - asserted rather than trusted."""
    notes = [note(48, 0.0), note(60, 0.0), note(72, 0.0)]
    before = [(n.pitch, n.start_ms, n.end_ms, n.confidence) for n in notes]
    store = AnnotationStore()
    write_octave_annotations(notes, store, Provenance.OCTAVE_STACK)
    assert [(n.pitch, n.start_ms, n.end_ms, n.confidence) for n in notes] == before


def test_the_flag_is_most_confident_where_the_note_is_least():
    notes = [note(48, 0.0), note(60, 0.0, confidence=0.30), note(72, 0.0)]
    store = AnnotationStore()
    write_octave_annotations(notes, store, Provenance.OCTAVE_STACK)
    ann = store.latest(notes[1].id, OCTAVE_STACK_ANNOTATION_KIND)
    assert abs(ann.confidence - 0.70) < 1e-6


def test_an_empty_input_is_not_an_error():
    assert find_octave_stacks([]) == []
    assert octave_suppression_ids(AnnotationStore(), []) == set()
