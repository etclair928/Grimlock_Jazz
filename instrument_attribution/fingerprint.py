# =================================================================
# MODULE: instrument_attribution/fingerprint.py
# Step 1 of "fingerprint -> stream -> resolve" (GRIMLOCK_6.0_DESIGN_
# DECISIONS.md §5). Per-NOTE timbre features - genuinely computed DSP
# (MFCC, spectral centroid/bandwidth, zero-crossing rate) over each
# note's own sounding window. These are individually noisy (a single
# note is a short, sometimes-overlapping slice of audio) - that noise
# is exactly why §5 couples this to voice_continuity.py's line-level
# aggregation rather than trusting any one note's fingerprint alone.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import librosa

from audio_engine import AudioEngine, AudioTrack
from core import Note

FINGERPRINT_SAMPLE_RATE = 22050
_MIN_WINDOW_MS = 40.0  # librosa's default frame/hop needs a minimum span to produce any frames


@dataclass(frozen=True)
class TimbreFingerprint:
    note_id: str
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    zero_crossing_rate: float
    mfcc_mean: Tuple[float, ...]


def _features_for_window(samples: np.ndarray, sr: int) -> Tuple[float, float, float, Tuple[float, ...]]:
    n_fft = min(2048, max(64, 1 << (len(samples).bit_length() - 1)))
    hop_length = max(1, n_fft // 4)

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=samples, frame_length=n_fft, hop_length=hop_length)))
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    mfcc_mean = tuple(float(v) for v in np.mean(mfcc, axis=1))
    return centroid, bandwidth, zcr, mfcc_mean


def fingerprint_notes(
        engine: AudioEngine,
        track: AudioTrack,
        notes: List[Note],
        sample_rate: int = FINGERPRINT_SAMPLE_RATE,
) -> Dict[str, TimbreFingerprint]:
    """One TimbreFingerprint per note, keyed by Note.id. Notes shorter
    than `_MIN_WINDOW_MS` get their window widened (centered on the
    note) rather than skipped - a real, if brief, note still deserves a
    fingerprint; it just draws on a little surrounding audio context."""
    view = engine.view(track, sample_rate)
    fingerprints: Dict[str, TimbreFingerprint] = {}

    for note in notes:
        start_ms, end_ms = note.start_ms, note.end_ms
        if end_ms - start_ms < _MIN_WINDOW_MS:
            center = (start_ms + end_ms) / 2.0
            start_ms = center - _MIN_WINDOW_MS / 2.0
            end_ms = center + _MIN_WINDOW_MS / 2.0

        window = view.slice_ms(max(0.0, start_ms), end_ms)
        if len(window.samples) < 32:
            continue

        centroid, bandwidth, zcr, mfcc_mean = _features_for_window(window.samples, sample_rate)
        fingerprints[note.id] = TimbreFingerprint(
            note_id=note.id,
            spectral_centroid_hz=centroid,
            spectral_bandwidth_hz=bandwidth,
            zero_crossing_rate=zcr,
            mfcc_mean=mfcc_mean,
        )

    return fingerprints


__all__ = ["TimbreFingerprint", "fingerprint_notes", "FINGERPRINT_SAMPLE_RATE"]
