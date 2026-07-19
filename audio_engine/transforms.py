# =================================================================
# MODULE: audio_engine/transforms.py
# Cached transform layer (§7 Audio Engine): STFT / CQT / Mel /
# onset-envelope computed once per (track, rate, params) and served from
# a byte-budgeted LRU cache. This is where FeatureBundle's contents live
# in 6.0 - they are cached derived audio data, not a separate stage's
# private state.
#
# Build-order note (§7): this ships the plain-LRU API boundary first.
# Zero-allocation mmap/virtual-array optimization is deferred until
# profiling proves memory pressure is real - it slots in behind this
# same interface later without callers changing.
# =================================================================

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Tuple

import numpy as np
import librosa

from audio_engine.track import AudioTrack
from audio_engine.resampler import get_view

_DEFAULT_BUDGET_BYTES = 512 * 1024 * 1024  # 512MB


class TransformCache:
    """Byte-budgeted LRU cache for derived audio features. Keyed by
    (track content-hash, sample_rate, transform kind, params) so two
    different tracks - or the same track resampled two ways - never
    collide, and re-decoding the same file in a new process starts cold
    (content_hash is a decode-time value, not persisted)."""

    def __init__(self, max_bytes: int = _DEFAULT_BUDGET_BYTES):
        self._max_bytes = max_bytes
        self._store: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()
        self._total_bytes = 0

    def _get_or_compute(self, key: Tuple[Any, ...], compute) -> np.ndarray:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]

        result = compute()
        result = np.asarray(result)
        self._store[key] = result
        self._total_bytes += result.nbytes
        self._evict_if_needed()
        return result

    def _evict_if_needed(self) -> None:
        while self._total_bytes > self._max_bytes and self._store:
            _, evicted = self._store.popitem(last=False)
            self._total_bytes -= evicted.nbytes

    def stft(self, track: AudioTrack, sample_rate: int, n_fft: int = 2048,
             hop_length: int = 512) -> np.ndarray:
        key = (track.content_hash, sample_rate, "stft", n_fft, hop_length)
        view = get_view(track, sample_rate)
        return self._get_or_compute(
            key, lambda: librosa.stft(view.samples, n_fft=n_fft, hop_length=hop_length)
        )

    def mel(self, track: AudioTrack, sample_rate: int, n_fft: int = 2048,
            hop_length: int = 512, n_mels: int = 128) -> np.ndarray:
        key = (track.content_hash, sample_rate, "mel", n_fft, hop_length, n_mels)
        view = get_view(track, sample_rate)
        return self._get_or_compute(
            key,
            lambda: librosa.feature.melspectrogram(
                y=view.samples, sr=sample_rate, n_fft=n_fft,
                hop_length=hop_length, n_mels=n_mels,
            ),
        )

    def cqt(self, track: AudioTrack, sample_rate: int, hop_length: int = 512,
            fmin: Optional[float] = None, n_bins: int = 84,
            bins_per_octave: int = 12) -> np.ndarray:
        key = (track.content_hash, sample_rate, "cqt", hop_length, fmin, n_bins, bins_per_octave)
        view = get_view(track, sample_rate)
        return self._get_or_compute(
            key,
            lambda: librosa.cqt(
                view.samples, sr=sample_rate, hop_length=hop_length, fmin=fmin,
                n_bins=n_bins, bins_per_octave=bins_per_octave,
            ),
        )

    def onset_envelope(self, track: AudioTrack, sample_rate: int,
                        hop_length: int = 512) -> np.ndarray:
        key = (track.content_hash, sample_rate, "onset_envelope", hop_length)
        view = get_view(track, sample_rate)
        return self._get_or_compute(
            key,
            lambda: librosa.onset.onset_strength(y=view.samples, sr=sample_rate, hop_length=hop_length),
        )

    @property
    def cached_bytes(self) -> int:
        return self._total_bytes

    @property
    def cached_keys(self):
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()
        self._total_bytes = 0


__all__ = ["TransformCache"]
