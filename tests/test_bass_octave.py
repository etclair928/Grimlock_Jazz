# =================================================================
# MODULE: tests/test_bass_octave.py
# Pins pitch_engine/bass_octave.py.
#
# WHAT THESE TESTS ARE REALLY DEFENDING. This module MOVES a note's pitch on
# the page, which makes its refusals as important as its catches: a wrong
# correction does not add a spurious note, it silently relabels a real one, and
# nothing downstream can tell that happened. So the assertions below are
# weighted toward what it must NOT do - not fire on a third, not fire on a
# seventh, not fire without enough voiced frames, not fire when CREPE's own
# reading is unsteady, and not fire at all when there is no CREPE evidence.
#
# The one that matters most is the seventh. Ten semitones is the closest a real
# musical interval gets to an octave, and it is common in bass lines; if the
# tolerance ever widens far enough to swallow it, this module starts rewriting
# correct notes and the tests here are the only thing that would say so.
# =================================================================

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.note_types import Note  # noqa: E402
from core.source_types import Provenance  # noqa: E402
from core.separation_types import StemType  # noqa: E402
from pitch_engine.bass_octave import arbitrate_bass_octaves  # noqa: E402


def note(pitch, start_ms=0.0, dur_ms=500.0):
    return Note(pitch=pitch, start_ms=start_ms, end_ms=start_ms + dur_ms,
                velocity=80, confidence=0.6, stem=StemType.BASS,
                source=Provenance.BASIC_PITCH)


def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def sampler_reading(midi_value, frames=40, jitter=0.0):
    """A CREPE stream that steadily reads `midi_value` over any span."""
    def sample(start_ms, end_ms):
        vals = np.full(frames, float(midi_value))
        if jitter:
            vals = vals + np.linspace(-jitter / 2.0, jitter / 2.0, frames)
        return midi_to_hz(vals)
    return sample


# ------------------------------------------------------------------ catches

def test_octave_below_is_raised():
    """Basic Pitch's classic bass failure: it locks onto a subharmonic and
    writes the note twelve semitones too low. CREPE hears the real one."""
    fixes = arbitrate_bass_octaves([note(29)], sampler_reading(41))
    assert len(fixes) == 1
    assert fixes[0].corrected_pitch == 41
    assert fixes[0].semitones == 12


def test_octave_above_is_lowered():
    fixes = arbitrate_bass_octaves([note(60)], sampler_reading(48))
    assert len(fixes) == 1
    assert fixes[0].corrected_pitch == 48


def test_two_octaves_is_corrected():
    fixes = arbitrate_bass_octaves([note(28)], sampler_reading(52))
    assert len(fixes) == 1
    assert fixes[0].corrected_pitch == 52


def test_slightly_flat_bass_still_counts_as_an_octave():
    """A real bass is not perfectly in tune and CREPE reports what it hears.
    A reading 0.6 semitones under the octave is still an octave error."""
    fixes = arbitrate_bass_octaves([note(29)], sampler_reading(40.4))
    assert len(fixes) == 1
    assert fixes[0].corrected_pitch == 41


# ----------------------------------------------------------------- refusals

def test_agreement_is_not_a_correction():
    assert arbitrate_bass_octaves([note(45)], sampler_reading(45.2)) == []


def test_a_third_is_left_alone():
    """Four semitones is a different note, not a misread octave. Correcting it
    would be re-detecting the bass, which this module does not do."""
    assert arbitrate_bass_octaves([note(45)], sampler_reading(49)) == []


def test_a_seventh_is_left_alone():
    """The tightest real interval near an octave. If this ever fires, the
    tolerance has been widened past the point where the module is safe."""
    assert arbitrate_bass_octaves([note(45)], sampler_reading(55)) == []


def test_three_octaves_is_left_alone():
    assert arbitrate_bass_octaves([note(24)], sampler_reading(60)) == []


def test_too_few_voiced_frames_is_no_evidence():
    assert arbitrate_bass_octaves([note(29)], sampler_reading(41, frames=2)) == []


def test_unsteady_crepe_reading_is_no_evidence():
    """CREPE sliding across three semitones inside one note is not confidently
    in any octave, so it must not be used to move one."""
    assert arbitrate_bass_octaves([note(29)], sampler_reading(41, jitter=3.0)) == []


def test_no_sampler_corrects_nothing():
    """Without CREPE there is no evidence, and a structural guess about
    octaves is worse than leaving the detector's answer alone."""
    assert arbitrate_bass_octaves([note(29)], None) == []


def test_correction_never_leaves_midi_range():
    fixes = arbitrate_bass_octaves([note(2)], sampler_reading(-10))
    assert all(0 <= f.corrected_pitch <= 127 for f in fixes)


def test_notes_are_not_mutated():
    """DESIGN_DECISIONS §2.2 - the frozen Note is evidence, not a scratchpad."""
    n = note(29)
    arbitrate_bass_octaves([n], sampler_reading(41))
    assert n.pitch == 29
