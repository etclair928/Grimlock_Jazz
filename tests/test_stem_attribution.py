# =================================================================
# MODULE: tests/test_stem_attribution.py
# Pins separation_engine/stem_attribution.py.
#
# Synthetic audio with known content, because the point of the module is that
# a note's energy tells you which stem carries it - and that claim is only
# testable when you built the stems and know the answer. The real-corpus
# validation lives in the commit message and the audit; these pin the
# mechanism.
# =================================================================

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from separation_engine.stem_attribution import (  # noqa: E402
    ATTRIBUTION_SAMPLE_RATE, Attribution, StemAttributor, attribute_notes,
    _midi_to_hz,
)

SR = ATTRIBUTION_SAMPLE_RATE


class FakeNote:
    def __init__(self, pitch, start_ms, end_ms, nid="n"):
        self.pitch = pitch
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.id = nid


def tone(pitch, seconds=1.0, sr=SR, partials=(1, 2, 3)):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    f0 = _midi_to_hz(pitch)
    return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in partials).astype(np.float32)


def silence(seconds=1.0, sr=SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


# --- the mechanism --------------------------------------------------------

def test_midi_to_hz_is_right():
    assert abs(_midi_to_hz(69) - 440.0) < 1e-9
    assert abs(_midi_to_hz(81) - 880.0) < 1e-6


def test_a_note_is_attributed_to_the_stem_that_contains_it():
    audio = {"bass": tone(40), "vocals": silence(), "other": silence()}
    result = attribute_notes([FakeNote(40, 0, 900)], audio)
    assert len(result) == 1
    assert result[0].stem == "bass"
    assert result[0].confidence > 0.9


def test_the_loudest_stem_wins_when_several_contain_it():
    audio = {"quiet": tone(60) * 0.1, "loud": tone(60), "empty": silence()}
    result = attribute_notes([FakeNote(60, 0, 900)], audio)
    assert result[0].stem == "loud"


def test_a_close_second_is_reported_as_contested():
    """The whole value over a hard split: a note whose energy is shared says
    so, instead of being silently duplicated into two stems."""
    audio = {"a": tone(60), "b": tone(60) * 0.95}
    result = attribute_notes([FakeNote(60, 0, 900)], audio)
    assert result[0].is_contested
    assert result[0].contested_by in ("a", "b")
    assert result[0].contested_by != result[0].stem


def test_an_uncontested_note_says_so():
    audio = {"a": tone(60), "b": silence()}
    result = attribute_notes([FakeNote(60, 0, 900)], audio)
    assert not result[0].is_contested
    assert result[0].contested_by is None


def test_different_pitches_are_told_apart():
    """The band is narrow enough to distinguish notes, which is the property
    that makes per-note attribution possible at all."""
    audio = {"low": tone(48), "high": tone(72)}
    low = attribute_notes([FakeNote(48, 0, 900)], audio)[0]
    high = attribute_notes([FakeNote(72, 0, 900)], audio)[0]
    assert low.stem == "low"
    assert high.stem == "high"


def test_a_very_short_note_is_still_measurable():
    """Notes below the window floor are widened rather than dropped - a 20ms
    note is exactly the kind this has to label."""
    audio = {"a": tone(60), "b": silence()}
    result = attribute_notes([FakeNote(60, 100, 120)], audio)
    assert len(result) == 1 and result[0].stem == "a"


def test_silence_everywhere_yields_no_attribution():
    """No energy is not a vote for the first stem alphabetically."""
    audio = {"a": silence(), "b": silence()}
    assert attribute_notes([FakeNote(60, 0, 900)], audio) == []


def test_empty_inputs_are_not_an_error():
    assert attribute_notes([], {"a": tone(60)}) == []
    assert attribute_notes([FakeNote(60, 0, 900)], {}) == []


def test_energy_by_stem_is_reported_for_every_stem_measured():
    """A verdict nobody can audit is a verdict nobody should act on - the
    numbers behind the choice travel with it."""
    audio = {"a": tone(60), "b": tone(60) * 0.2, "c": silence()}
    result = attribute_notes([FakeNote(60, 0, 900)], audio)[0]
    assert set(result.energy_by_stem) == {"a", "b", "c"}
    assert result.energy_by_stem["a"] > result.energy_by_stem["b"]


def test_the_spectrogram_is_computed_once_per_stem():
    """A note-at-a-time STFT would dominate the cost and recompute the same
    frames thousands of times."""
    audio = {"a": tone(60, seconds=2.0), "b": silence(2.0)}
    attributor = StemAttributor(audio)
    assert set(attributor.spec) == {"a", "b"}
    notes = [FakeNote(60, i * 100, i * 100 + 90, f"n{i}") for i in range(10)]
    assert all(attributor.attribute(n).stem == "a" for n in notes)
