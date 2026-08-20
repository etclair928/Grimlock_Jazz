# =================================================================
# MODULE: tests/test_key_stability.py
# Pins key_intelligence/key_stability.py.
#
# WHAT THIS IS REALLY DEFENDING. The Chopin investigation nearly produced a
# rewrite of a detector that turned out to be correct: we reported D# minor
# for a B major piece, but our audio is a 180-second excerpt, and the
# PUBLISHED EDITION reads D# minor over that same span too. What was wrong
# was the 0.84 confidence attached to a reading that is true of a passage
# and false of the work.
#
# So the property under test is not "does it find the key" - detect_key
# already does that. It is "does it refuse to sound certain when the
# evidence does not support certainty", plus the two comparison rules that
# decide what counts as disagreement at all. Both of those had bugs:
# relative pairs would otherwise mark every tonal piece unstable, and
# comparing key labels as strings scored A# against Bb minor as a
# disagreement when A# IS its tonic.
# =================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.note_types import Note  # noqa: E402
from core.separation_types import StemType  # noqa: E402
from core.source_types import Provenance  # noqa: E402
from key_intelligence.key_stability import (  # noqa: E402
    _cadence_key, _same_tonal_center, _tonic_pc, analyze_key_stability,
)

# C major and A minor scales, as pitch classes
C_MAJOR = [60, 62, 64, 65, 67, 69, 71]
FS_MAJOR = [66, 68, 70, 71, 73, 75, 77]


def scale_notes(pcs, start_ms=0.0, bars=8, step=250.0):
    """A plain diatonic run, long enough for the windowing to have material."""
    out, t = [], start_ms
    for _ in range(bars):
        for p in pcs:
            out.append(Note(pitch=p, start_ms=t, end_ms=t + step * 0.9,
                            velocity=80, confidence=0.9, stem=StemType.OTHER,
                            source=Provenance.BASIC_PITCH))
            t += step
    return out


# --- the comparison rules, both of which had bugs -------------------------

def test_relative_pairs_count_as_agreement():
    """C major and A minor share all seven notes. Counting that as a
    modulation would flag almost every tonal piece as unstable and make the
    flag useless exactly where it matters."""
    assert _same_tonal_center("C", "Am")
    assert _same_tonal_center("Am", "C")
    assert _same_tonal_center("Bbm", "Db")


def test_enharmonic_spellings_are_the_same_key():
    """The bug this module shipped with: Hopeful's cadence returned 'A#'
    against a key of 'Bbm' and was scored a disagreement, when A# is its
    tonic. Every comparison goes through pitch class for this reason."""
    assert _tonic_pc("A#") == _tonic_pc("Bb")
    assert _tonic_pc("Bbm") == _tonic_pc("A#m")
    assert _same_tonal_center("A#m", "Bbm")


def test_a_fifth_apart_is_not_agreement():
    """Tonic vs dominant is the failure mode that started all this. B and F#
    share six of seven notes and must still count as different keys."""
    assert not _same_tonal_center("B", "F#")
    assert not _same_tonal_center("C", "G")
    assert not _same_tonal_center("Am", "Em")


def test_major_and_minor_on_the_same_tonic_differ():
    """Parallel keys, not relative ones - C and Cm are genuinely different."""
    assert not _same_tonal_center("C", "Cm")


# --- stability ------------------------------------------------------------

def test_one_key_throughout_is_stable_and_keeps_its_confidence():
    st = analyze_key_stability(scale_notes(C_MAJOR, bars=24))
    assert st.stable
    assert not st.modulates
    assert st.agreement > 0.9
    assert st.confidence >= st.raw_confidence * 0.9


def test_a_piece_that_changes_key_is_flagged_and_damped():
    """THE property. Half in C, half a tritone away: the global reading is a
    summary of the span, and the confidence has to say so."""
    first = scale_notes(C_MAJOR, start_ms=0.0, bars=16)
    second = scale_notes(FS_MAJOR, start_ms=16 * 7 * 250.0, bars=16)
    st = analyze_key_stability(first + second)
    assert st.modulates
    assert st.agreement < 0.7
    assert st.confidence < st.raw_confidence


def test_the_global_reading_is_never_overridden():
    """This module reports on a key; it does not get to change it. A caller
    that wants the windows can read them."""
    from key_intelligence import analyze_key_from_notes
    notes = scale_notes(C_MAJOR, bars=24)
    assert analyze_key_stability(notes).key == analyze_key_from_notes(notes).key


def test_windows_are_reported_so_a_modulation_can_be_located():
    first = scale_notes(C_MAJOR, start_ms=0.0, bars=16)
    second = scale_notes(FS_MAJOR, start_ms=16 * 7 * 250.0, bars=16)
    st = analyze_key_stability(first + second)
    assert len(st.windows) >= 4
    keys = [k for _s, _e, k in st.windows]
    assert len(set(keys)) > 1
    for start, end, _k in st.windows:
        assert end > start


# --- the cadence witness, which is corroboration only ---------------------

def test_the_cadence_witness_reads_the_lowest_final_note():
    notes = scale_notes(C_MAJOR, bars=16)
    notes.append(Note(pitch=36, start_ms=99999.0, end_ms=100999.0, velocity=90,
                      confidence=0.9, stem=StemType.OTHER,
                      source=Provenance.BASIC_PITCH))
    assert _tonic_pc(_cadence_key(notes)) == 0          # C


def test_the_cadence_witness_never_decides_the_key():
    """It disagreed with the global reading on 3 of our 4 real songs. It is a
    second opinion, and a caller reading `cadence_agrees` must still get the
    detector's key back."""
    notes = scale_notes(C_MAJOR, bars=16)
    notes.append(Note(pitch=42, start_ms=99999.0, end_ms=100999.0, velocity=90,
                      confidence=0.9, stem=StemType.OTHER,
                      source=Provenance.BASIC_PITCH))
    st = analyze_key_stability(notes)
    assert st.cadence_agrees is False
    assert st.key == analyze_key_stability(scale_notes(C_MAJOR, bars=16)).key


def test_too_few_notes_does_not_claim_stability_it_cannot_test():
    st = analyze_key_stability(scale_notes(C_MAJOR, bars=1))
    assert "too few notes" in st.notes or st.agreement == 1.0


def test_drums_are_excluded():
    """Percussion pitch values are slot ids, not tonal content."""
    notes = scale_notes(C_MAJOR, bars=16)
    drums = [Note(pitch=36 + (i % 3), start_ms=i * 100.0, end_ms=i * 100.0 + 50,
                  velocity=90, confidence=0.9, stem=StemType.DRUMS,
                  source=Provenance.BASIC_PITCH) for i in range(400)]
    assert analyze_key_stability(notes).key == analyze_key_stability(notes + drums).key


def test_empty_input_is_not_an_error():
    st = analyze_key_stability([])
    assert st.key
    assert 0.0 <= st.confidence <= 1.0
