# =================================================================
# MODULE: audio_engine/__init__.py
# Public API surface for Grimlock 6.0's Audio Engine (GRIMLOCK_6.0_
# DESIGN_DECISIONS.md §7). Highly optimized DSP domain - no ML models
# live here (that's Model Registry / Pitch Engine's job).
#
# AudioEngine is the single entry point: decode a file once, then ask
# it for views (resampled, mono, cached) or transforms (STFT/CQT/Mel/
# onset-envelope, cached). Every other layer talks to audio through
# this object - nothing downstream calls soundfile/librosa directly.
# =================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from audio_engine.track import AudioTrack, AudioView
from audio_engine.decoder import decode_file
from audio_engine.resampler import get_view, get_stereo, resample
from audio_engine.transforms import TransformCache


class AudioEngine:
    """Ask this for tracks, views, and transforms. Never decode, resample,
    or run librosa/soundfile calls outside this module."""

    def __init__(self, transform_cache_budget_bytes: Optional[int] = None):
        self._transforms = (
            TransformCache(transform_cache_budget_bytes)
            if transform_cache_budget_bytes is not None
            else TransformCache()
        )

    def decode(self, path: Union[str, Path]) -> AudioTrack:
        """Decode a source file into its one canonical AudioTrack."""
        return decode_file(path)

    def view(self, track: AudioTrack, sample_rate: int) -> AudioView:
        """The mono view of `track` at `sample_rate` - resampled once,
        cached on the track for subsequent requests."""
        return get_view(track, sample_rate)

    def stereo(self, track: AudioTrack, sample_rate: int):
        """Full original-channel-layout audio at `sample_rate`, read-only,
        cached on the track. For consumers (Separation Engine) that need
        real stereo, not Pitch Engine's mono downmix."""
        return get_stereo(track, sample_rate)

    def stft(self, track: AudioTrack, sample_rate: int, n_fft: int = 2048,
             hop_length: int = 512):
        return self._transforms.stft(track, sample_rate, n_fft, hop_length)

    def mel(self, track: AudioTrack, sample_rate: int, n_fft: int = 2048,
            hop_length: int = 512, n_mels: int = 128):
        return self._transforms.mel(track, sample_rate, n_fft, hop_length, n_mels)

    def cqt(self, track: AudioTrack, sample_rate: int, hop_length: int = 512,
            fmin: Optional[float] = None, n_bins: int = 84, bins_per_octave: int = 12):
        return self._transforms.cqt(track, sample_rate, hop_length, fmin, n_bins, bins_per_octave)

    def onset_envelope(self, track: AudioTrack, sample_rate: int, hop_length: int = 512):
        return self._transforms.onset_envelope(track, sample_rate, hop_length)

    @property
    def transform_cache_bytes(self) -> int:
        return self._transforms.cached_bytes

    def clear_transform_cache(self) -> None:
        self._transforms.clear()


__all__ = ["AudioEngine", "AudioTrack", "AudioView", "resample"]
