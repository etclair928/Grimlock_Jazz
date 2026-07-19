# =================================================================
# MODULE: rhythm_engine/groove.py
# Rhythm Engine: swing/groove estimation. Measures how far real onsets
# that land between two beats (the "and") sit from the exact halfway
# point - straight time sits at ~50%, swung/shuffled time pushes that
# later (commonly ~60-67% for a triplet-feel swing). A real, if simple,
# measurement from actual onset positions relative to the beat grid -
# not a fixed guess.
# =================================================================

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import librosa

from audio_engine import AudioEngine, AudioTrack

GROOVE_SAMPLE_RATE = 22050
GROOVE_HOP_LENGTH = 512


def estimate_groove(
        engine: AudioEngine,
        track: AudioTrack,
        beat_times_ms: Sequence[float],
) -> Tuple[float, float]:
    """Returns (swing_ratio, confidence). swing_ratio is the mean
    fractional position (0.5 = straight, higher = more swung) of onsets
    falling in the gap between consecutive beats. Requires at least a
    few beat intervals with a detected onset in them to report anything
    above baseline confidence."""
    if len(beat_times_ms) < 2:
        return 0.5, 0.1

    onset_env = engine.onset_envelope(track, GROOVE_SAMPLE_RATE, hop_length=GROOVE_HOP_LENGTH)
    onset_times_ms = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=GROOVE_SAMPLE_RATE, hop_length=GROOVE_HOP_LENGTH, units="time",
    ) * 1000.0

    beats = np.asarray(beat_times_ms, dtype=np.float64)
    fractions = []
    for i in range(len(beats) - 1):
        interval_start, interval_end = beats[i], beats[i + 1]
        interval_len = interval_end - interval_start
        if interval_len <= 0:
            continue
        in_gap = onset_times_ms[(onset_times_ms > interval_start + 0.15 * interval_len) &
                                 (onset_times_ms < interval_end - 0.05 * interval_len)]
        if len(in_gap) == 0:
            continue
        # Nearest-to-midpoint onset represents this gap's "and" position
        midpoint = interval_start + 0.5 * interval_len
        nearest = in_gap[np.argmin(np.abs(in_gap - midpoint))]
        fractions.append((nearest - interval_start) / interval_len)

    if not fractions:
        return 0.5, 0.2

    swing_ratio = float(np.clip(np.mean(fractions), 0.5, 0.75))
    confidence = float(np.clip(len(fractions) / max(1, len(beats) - 1), 0.2, 0.9))
    return swing_ratio, confidence


__all__ = ["estimate_groove"]
