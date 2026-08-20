# =================================================================
# MODULE: tests/test_solo_detector.py
# Pins separation_engine/solo_detector.py.
#
# THE ASYMMETRY THIS DEFENDS. A wrong "solo" verdict silently discards real
# instruments and no downstream stage can notice; a wrong "ensemble" costs
# only time. So these tests are weighted accordingly - the ones that matter
# assert that things which ARE ensembles are never called solo, and that each
# physical witness fires on the signal it was built for.
#
# Synthetic audio, deliberately. The corpus has four real recordings and no
# clean solo guitar at all, so behaviour is pinned against signals whose
# content is known by construction rather than against three files.
# =================================================================

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from separation_engine.solo_detector import (  # noqa: E402
    ENSEMBLE, SOLO_GUITAR, SOLO_PIANO, UNKNOWN, analyze_solo,
    _rising_sustain_fraction,
)

SR = 22050


def decaying_tone(freq, seconds=0.6, sr=SR):
    """A struck string: instant attack, monotone decay. What a piano does."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    env = np.exp(-4.0 * t)
    wave = sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3, 4))
    return (wave * env).astype(np.float32)


def swelling_tone(freq, seconds=0.6, sr=SR):
    """A voice or a bow: energy RISES while sounding. A piano cannot."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    env = np.linspace(0.05, 1.0, t.size) ** 1.5
    wave = sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3))
    return (wave * env).astype(np.float32)


def cymbal(seconds=0.35, sr=SR):
    """Broadband noise with a decay - the thing pitched instruments can't make."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (rng.normal(0, 1, t.size) * np.exp(-6.0 * t)).astype(np.float32)


def sequence(builder, freqs, gap=0.5, sr=SR, seconds=14.0):
    out = np.zeros(int(seconds * sr), dtype=np.float32)
    pos = 0
    i = 0
    while pos < out.size - sr:
        seg = builder(freqs[i % len(freqs)])
        end = min(out.size, pos + seg.size)
        out[pos:end] += seg[:end - pos]
        pos += int(gap * sr)
        i += 1
    return out


# --- the piano-cannot-crescendo witness -----------------------------------

def test_rising_sustain_separates_decay_from_swell():
    """The witness in isolation, so a failure elsewhere is not blamed on it."""
    import librosa
    dec = np.abs(librosa.stft(sequence(decaying_tone, [220, 330, 440]), n_fft=2048, hop_length=512))
    swl = np.abs(librosa.stft(sequence(swelling_tone, [220, 330, 440]), n_fft=2048, hop_length=512))
    assert _rising_sustain_fraction(swl) > _rising_sustain_fraction(dec)


def test_something_that_swells_is_never_called_solo():
    """A singer over a piano is the case a percussion test cannot see. This is
    the witness that exists for it, and calling this 'solo' would silently
    delete the voice."""
    piano = sequence(decaying_tone, [110, 165, 220])
    voice = sequence(swelling_tone, [330, 392, 440])
    v = analyze_solo(piano + voice, SR)
    assert not v.is_solo, v.reason


# --- the kit witness -------------------------------------------------------

def test_a_kit_is_never_called_solo():
    piano = sequence(decaying_tone, [110, 165, 220])
    kit = np.zeros_like(piano)
    step = int(0.25 * SR)
    for pos in range(0, kit.size - SR, step):
        seg = cymbal()
        end = min(kit.size, pos + seg.size)
        kit[pos:end] += seg[:end - pos] * 0.5
    v = analyze_solo(piano + kit, SR)
    assert v.verdict == ENSEMBLE, v.reason


# --- solo verdicts ---------------------------------------------------------

def test_decaying_strings_with_low_notes_read_as_solo_piano():
    v = analyze_solo(sequence(decaying_tone, [55, 65, 82, 110]), SR)
    assert v.verdict == SOLO_PIANO, v.reason
    assert v.is_solo


def test_a_solo_verdict_is_surer_of_being_solo_than_of_which_instrument():
    """The instrument boundary could not be calibrated - there is no clean
    solo guitar in the corpus - and the numbers have to admit that."""
    v = analyze_solo(sequence(decaying_tone, [55, 65, 82, 110]), SR)
    assert v.is_solo
    assert v.instrument_confidence < v.confidence


def test_silence_and_empty_input_do_not_claim_a_solo():
    assert analyze_solo(np.array([], dtype=np.float32), SR).verdict == UNKNOWN
    quiet = analyze_solo(np.zeros(SR * 5, dtype=np.float32), SR)
    assert quiet.verdict in (UNKNOWN, SOLO_GUITAR, SOLO_PIANO, ENSEMBLE)


def test_the_verdict_carries_the_numbers_that_decided_it():
    """A verdict nobody can audit is a verdict nobody should act on, and this
    one skips a whole stage of the pipeline."""
    v = analyze_solo(sequence(decaying_tone, [55, 110]), SR)
    assert v.reason
    for field in (v.kit_hf_fraction, v.crescendo_fraction, v.low_energy_fraction):
        assert 0.0 <= field <= 1.0
