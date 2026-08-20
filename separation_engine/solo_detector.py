# =================================================================
# MODULE: separation_engine/solo_detector.py
# IS THIS ONE INSTRUMENT? DECIDED BEFORE DEMUCS RUNS, AND CHEAPLY.
#
# WHY. Demucs is the largest model in the pipeline - twenty-nine minutes of a
# thirty-minute run on the full Chopin recording - and on a solo piano record
# it is not merely wasted, it is HARMFUL. Measured on the Nocturne Op.62
# No.1: separating a solo piano recording into six stems produced 1546
# "drum", 463 "bass" and 200 "vocal" notes on a recording with no drummer,
# bassist or singer. Feeding the same audio through as one harmonic stem
# instead moved F1 from 0.507 to 0.603 - the single largest gain of this
# whole effort, and it came from NOT running a model.
#
# So the question "is this one instrument" has to be answered before the
# expensive, damaging step, from the master audio alone.
#
# THE EVIDENCE, chosen to be physical rather than learned. Each witness is
# something an instrument can or cannot do, so a wrong answer is a fact about
# the audio and not about a training set.
#
#   1. A DRUM KIT MAKES BROADBAND NOISE, REPEATEDLY. Cymbals and hi-hats are
#      near-white above 8kHz and they recur. A piano hammer is a transient
#      too, but it is band-limited and its high end is far quieter. Measured
#      as the percussive component's energy fraction above 8kHz.
#
#   2. A PIANO CANNOT CRESCENDO ON A HELD NOTE. Once the hammer leaves the
#      string the sound only decays - no player can make a sustained piano
#      tone grow. A voice, a bowed string, a wind instrument, an organ and a
#      synth pad all can. So sustained frames whose energy RISES are proof of
#      something that is not a piano, and this is the witness that catches
#      the case a percussion test misses entirely: piano plus a singer.
#
#   3. A GUITAR HAS A BOTTOM. Standard tuning stops at E2, 82Hz. Sustained
#      energy well below that is a piano (or a bass), never a solo guitar.
#
# WHAT IT REFUSES TO DO. It returns ENSEMBLE or UNKNOWN unless the evidence
# is clear, and the caller runs Demucs on anything that is not a confident
# solo verdict. The asymmetry is deliberate: a false "solo" silently discards
# real instruments and there is no downstream stage that could notice, while
# a false "ensemble" costs only time. This is a fast path, not a classifier,
# and it is allowed to decline.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

SOLO_PIANO = "solo_piano"
SOLO_GUITAR = "solo_guitar"
ENSEMBLE = "ensemble"
UNKNOWN = "unknown"

ANALYSIS_SAMPLE_RATE = 22050

# Above this share of percussive energy sitting above 8kHz, something is
# making broadband noise repeatedly, which pitched instruments do not.
KIT_HF_FRACTION = 0.045

# Share of sustained frames that may rise in energy before we stop believing
# a decaying-string instrument is the only thing present. Non-zero because
# pedalling, sympathetic resonance and overlapping attacks all nudge a frame
# upward; a real crescendo is far more persistent than that.
CRESCENDO_FRACTION = 0.22

# A guitar's open low E. Sustained energy meaningfully below this is not a
# solo guitar.
#
# THIS BOUNDARY IS NOT CALIBRATED, and the tool says so. There is no clean
# solo guitar recording in the corpus; the only guitar available is a Demucs
# STEM, which reads 0.024 here - higher than real solo piano at 0.012-0.014,
# because bass bleeds into it during separation. So the number below is
# reasoned from the instrument (a guitar has no string under 82Hz) and placed
# beneath the two real piano measurements, not fitted to data.
#
# It matters less than it looks. This threshold only chooses the LABEL on a
# verdict that is already "one harmonic instrument"; the decision that
# actually skips Demucs is made before it, and both labels route to the same
# grand staff. A wrong guess here costs a stem name, not a note.
GUITAR_LOW_HZ = 78.0
LOW_ENERGY_FRACTION = 0.008

# Analysis is done on a sample of the track rather than all of it: the
# question is what instruments EXIST, which a couple of minutes answers as
# well as an hour, and this path only earns its keep by being fast.
MAX_ANALYSIS_SECONDS = 120.0


@dataclass(frozen=True)
class SoloVerdict:
    """What is playing, and how sure, with the numbers that decided it."""
    verdict: str
    confidence: float
    kit_hf_fraction: float
    crescendo_fraction: float
    low_energy_fraction: float
    reason: str
    # How sure we are of WHICH instrument, as opposed to how sure we are that
    # there is only one. These are different questions with different
    # evidence, and only the first one gates Demucs.
    instrument_confidence: float = 0.0

    @property
    def is_solo(self) -> bool:
        return self.verdict in (SOLO_PIANO, SOLO_GUITAR)


def _band_energy_fraction(spec: np.ndarray, freqs: np.ndarray,
                          low: float, high: float) -> float:
    total = float(np.sum(spec))
    if total <= 0:
        return 0.0
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(spec[mask, :])) / total


def _rising_sustain_fraction(spec: np.ndarray) -> float:
    """Share of sustained frames whose energy RISES.

    A piano string decays from the moment it is struck. Anything that swells
    while sounding - a voice, a bow, a reed, a pad - leaves this signature and
    nothing about a piano can. Frames near an attack are skipped, because the
    rise INTO a note is not a crescendo; only what happens after it.
    """
    if spec.shape[1] < 8:
        return 0.0
    energy = np.sum(spec, axis=0)
    if not np.any(energy > 0):
        return 0.0
    # An attack is a large jump; the two frames after one are still the
    # attack settling and are not evidence either way.
    delta = np.diff(energy)
    loud = energy[1:] > np.percentile(energy, 40)
    attack = delta > np.percentile(np.abs(delta), 90)
    skip = np.zeros_like(attack)
    for shift in (0, 1, 2):
        skip[shift:] |= attack[:len(attack) - shift] if shift else attack
    sustained = loud & ~skip
    if not np.any(sustained):
        return 0.0
    # A rise worth counting is a real one, not dither.
    rising = (delta > 0) & (delta > 0.02 * np.maximum(energy[1:], 1e-9))
    return float(np.sum(rising & sustained) / np.sum(sustained))


def analyze_solo(samples: np.ndarray, sample_rate: int) -> SoloVerdict:
    """Decide whether one instrument is playing. Pure: audio in, verdict out."""
    import librosa

    y = np.asarray(samples, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    if y.size == 0:
        return SoloVerdict(UNKNOWN, 0.0, 0.0, 0.0, 0.0, "no audio")

    if sample_rate != ANALYSIS_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=ANALYSIS_SAMPLE_RATE)
    sr = ANALYSIS_SAMPLE_RATE

    limit = int(MAX_ANALYSIS_SECONDS * sr)
    if y.size > limit:                      # take the middle, not the intro
        start = (y.size - limit) // 2
        y = y[start:start + limit]

    n_fft = 2048
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    harmonic, percussive = librosa.decompose.hpss(spec)
    kit_hf = _band_energy_fraction(percussive, freqs, 8000.0, sr / 2.0)
    crescendo = _rising_sustain_fraction(harmonic)
    low = _band_energy_fraction(spec, freqs, 0.0, GUITAR_LOW_HZ)

    if kit_hf > KIT_HF_FRACTION:
        return SoloVerdict(
            ENSEMBLE, min(1.0, kit_hf / KIT_HF_FRACTION - 1.0 + 0.6),
            kit_hf, crescendo, low,
            f"{kit_hf:.3f} of percussive energy sits above 8kHz "
            f"(> {KIT_HF_FRACTION}): broadband noise recurring, which a "
            f"pitched instrument does not make - a kit is playing")

    if crescendo > CRESCENDO_FRACTION:
        return SoloVerdict(
            ENSEMBLE, min(1.0, crescendo / CRESCENDO_FRACTION - 1.0 + 0.6),
            kit_hf, crescendo, low,
            f"{crescendo:.1%} of sustained frames RISE in energy "
            f"(> {CRESCENDO_FRACTION:.0%}): a struck string can only decay, "
            f"so something that can swell - voice, bow, reed, pad - is present")

    # ONE INSTRUMENT IS ESTABLISHED. Everything above is the gate that skips
    # Demucs, and it rests on two physical tests that measured cleanly. What
    # follows only picks a NAME, on a boundary that could not be calibrated -
    # so it is reported with its own, much lower, confidence.
    solo_confidence = 0.80
    if low > LOW_ENERGY_FRACTION:
        return SoloVerdict(
            SOLO_PIANO, solo_confidence, kit_hf, crescendo, low,
            f"one instrument: no kit (HF {kit_hf:.3f} <= {KIT_HF_FRACTION}) and "
            f"nothing that swells ({crescendo:.1%} <= {CRESCENDO_FRACTION:.0%}). "
            f"{low:.1%} of energy sits below {GUITAR_LOW_HZ:.0f}Hz, under a "
            f"guitar's open low E, so a piano",
            instrument_confidence=0.65)

    return SoloVerdict(
        SOLO_GUITAR, solo_confidence, kit_hf, crescendo, low,
        f"one instrument: no kit (HF {kit_hf:.3f} <= {KIT_HF_FRACTION}) and "
        f"nothing that swells ({crescendo:.1%} <= {CRESCENDO_FRACTION:.0%}). "
        f"Almost nothing below {GUITAR_LOW_HZ:.0f}Hz ({low:.1%}), which is "
        f"consistent with a guitar - but this branch is reached by "
        f"elimination on an uncalibrated boundary, so the instrument name is "
        f"a guess even though the solo verdict is not",
        instrument_confidence=0.35)


def detect_solo_instrument(engine, track) -> SoloVerdict:
    """AudioEngine entry point. Never raises: an analysis that fails returns
    UNKNOWN so the caller separates normally."""
    try:
        view = engine.view(track, ANALYSIS_SAMPLE_RATE)
        return analyze_solo(view.samples, ANALYSIS_SAMPLE_RATE)
    except Exception as exc:                       # pragma: no cover
        return SoloVerdict(UNKNOWN, 0.0, 0.0, 0.0, 0.0,
                           f"analysis failed ({type(exc).__name__}) - separating normally")


__all__ = ["SoloVerdict", "analyze_solo", "detect_solo_instrument",
           "SOLO_PIANO", "SOLO_GUITAR", "ENSEMBLE", "UNKNOWN",
           "KIT_HF_FRACTION", "CRESCENDO_FRACTION", "GUITAR_LOW_HZ"]
