# =================================================================
# MODULE: rhythm_engine/tempo_witness.py
# Rhythm Engine (GRIMLOCK_6.0_DESIGN_DECISIONS.md §7): tempo/meter/
# groove analysis. Two INDEPENDENT tempo witnesses (librosa's onset-
# envelope beat tracker, madmom's RNN+DBN beat tracker) rather than one
# - not because 6.0 wants a council (the Pitch layer explicitly
# rejected that), but because Epistemic's whole reason to exist is
# arbitrating genuine scalar disputes (§7 Epistemic), and tempo/octave
# disagreement between two real trackers is the canonical case for it.
# Each witness is independent and honest about its own confidence;
# reconciling them is epistemic/referee.py's job, not this module's.
# =================================================================

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf

from audio_engine import AudioEngine, AudioTrack
from core import Provenance
from rhythm_engine.octave_correction import correct as correct_octave

LIBROSA_TEMPO_SAMPLE_RATE = 22050
MADMOM_TEMPO_SAMPLE_RATE = 44100
TEMPO_HOP_LENGTH = 512


@dataclass(frozen=True)
class TempoWitness:
    tempo_bpm: float
    confidence: float
    beat_times_ms: Tuple[float, ...]
    source: Provenance
    # Per-witness-TYPE trust weight (distinct from this reading's own
    # confidence) - epistemic.resolve_tempo uses confidence*weight to
    # pick the anchor witness and weight alone as the vote denominator,
    # matching Symphony's tempo_intelligence.py witness_weights table
    # (madmom 1.0 > librosa 0.8 > note-onset-pattern 0.7 > lattice 0.6).
    # Provenance alone can't carry this - several witnesses share the
    # same Provenance bucket (TEMPO_INTELLIGENCE, RHYTHM_ENGINE).
    weight: float = 1.0


def _ioi_regularity_confidence(beat_times_ms: np.ndarray) -> float:
    """A tracker whose beats land at near-constant intervals is more
    trustworthy than one with erratic spacing - coefficient of
    variation of inter-onset-intervals, inverted and clamped to [0, 1]."""
    if len(beat_times_ms) < 3:
        return 0.3
    iois = np.diff(beat_times_ms)
    if np.mean(iois) <= 0:
        return 0.3
    cv = np.std(iois) / np.mean(iois)
    return float(np.clip(1.0 - cv, 0.1, 0.95))


def _re_anchor_beat_grid(beat_times_ms: np.ndarray, corrected_bpm: float) -> np.ndarray:
    """After octave correction changes the tempo, the ORIGINAL beat
    times only have the right period for the uncorrected reading (e.g.
    half as many beats as a corrected-2x tempo implies) - regenerates a
    grid at the corrected period, anchored to the first detected beat's
    phase so the correction doesn't also throw away where the beats
    actually fell."""
    if len(beat_times_ms) == 0 or corrected_bpm <= 0:
        return beat_times_ms
    if len(beat_times_ms) < 2:
        return beat_times_ms

    # RE-INDEX, DO NOT REGENERATE (2026-08-11).
    #
    # This used to return `start + arange(n) * period_ms` - a perfectly straight
    # line, keeping only the first tracked beat and discarding every other
    # observation. It solved a COUNT problem (a corrected 2x tempo implies twice
    # as many beats as the tracker found) by destroying all the TIMING
    # information, which is the one thing the tracker was for.
    #
    # MEASURED cost of that trade: an isochronous grid sits >0.35s from the real
    # beats for 93% of Rubinstein's Chopin, drifting up to 6.99s. And the error
    # it was protecting against is far cheaper - an octave mistake is a LABELLING
    # error (we write eighths where quarters belong; every note is still in the
    # right bar and a global halving fixes it), whereas a flattened curve is
    # unrecoverable information loss that puts notes in the wrong bar.
    #
    # So: interpolate to subdivide, decimate to coarsen. Both keep the observed
    # rubato. Non-integer ratios fall back to nearest-neighbour resampling in
    # beat space, which still follows the curve.
    ratio = corrected_bpm * float(np.median(np.diff(beat_times_ms))) / 60000.0
    if ratio <= 0:
        return beat_times_ms
    idx = np.arange(0, len(beat_times_ms) - 1 + 1e-9, 1.0 / ratio)
    return np.interp(idx, np.arange(len(beat_times_ms)), beat_times_ms)


def run_librosa_tempo(engine: AudioEngine, track: AudioTrack) -> TempoWitness:
    onset_env = engine.onset_envelope(track, LIBROSA_TEMPO_SAMPLE_RATE, hop_length=TEMPO_HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=LIBROSA_TEMPO_SAMPLE_RATE, hop_length=TEMPO_HOP_LENGTH, units="frames",
    )
    beat_times_ms = librosa.frames_to_time(
        beat_frames, sr=LIBROSA_TEMPO_SAMPLE_RATE, hop_length=TEMPO_HOP_LENGTH,
    ) * 1000.0
    tempo_val = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    confidence = _ioi_regularity_confidence(beat_times_ms)

    corrected_bpm, reason = correct_octave(
        tempo_val, onset_env, LIBROSA_TEMPO_SAMPLE_RATE, TEMPO_HOP_LENGTH, anchor_confidence=confidence,
    )
    if reason != "no_correction_needed" and corrected_bpm != tempo_val:
        beat_times_ms = _re_anchor_beat_grid(beat_times_ms, corrected_bpm)
        tempo_val = corrected_bpm

    return TempoWitness(
        tempo_bpm=tempo_val,
        confidence=confidence,
        beat_times_ms=tuple(float(t) for t in beat_times_ms),
        source=Provenance.TEMPO_INTELLIGENCE,
        weight=0.8,
    )


def run_madmom_tempo(engine: AudioEngine, track: AudioTrack) -> TempoWitness:
    """madmom's processors load audio from a file themselves rather
    than accepting a raw array - same temp-WAV pattern basic_pitch_
    engine.py uses for the same reason."""
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

    view = engine.view(track, MADMOM_TEMPO_SAMPLE_RATE)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    try:
        sf.write(temp_path, view.samples, MADMOM_TEMPO_SAMPLE_RATE)
        activations = RNNBeatProcessor()(temp_path)
        beat_times_s = DBNBeatTrackingProcessor(fps=100)(activations)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    beat_times_ms = np.asarray(beat_times_s, dtype=np.float64) * 1000.0
    if len(beat_times_ms) >= 2:
        median_ioi_ms = float(np.median(np.diff(beat_times_ms)))
        tempo_bpm = 60000.0 / median_ioi_ms if median_ioi_ms > 0 else 0.0
    else:
        tempo_bpm = 0.0

    confidence = _ioi_regularity_confidence(beat_times_ms) if tempo_bpm > 0 else 0.1

    if tempo_bpm > 0:
        onset_env = engine.onset_envelope(track, LIBROSA_TEMPO_SAMPLE_RATE, hop_length=TEMPO_HOP_LENGTH)
        corrected_bpm, reason = correct_octave(
            tempo_bpm, onset_env, LIBROSA_TEMPO_SAMPLE_RATE, TEMPO_HOP_LENGTH, anchor_confidence=confidence,
        )
        if reason != "no_correction_needed" and corrected_bpm != tempo_bpm:
            beat_times_ms = _re_anchor_beat_grid(beat_times_ms, corrected_bpm)
            tempo_bpm = corrected_bpm

    return TempoWitness(
        tempo_bpm=tempo_bpm,
        confidence=confidence,
        beat_times_ms=tuple(float(t) for t in beat_times_ms),
        source=Provenance.RHYTHM_ENGINE,
        weight=1.0,
    )


__all__ = ["TempoWitness", "run_librosa_tempo", "run_madmom_tempo"]
