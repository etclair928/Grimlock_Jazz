# =================================================================
# MODULE: audio_engine/resampler.py
# Unified resampler (§7 Audio Engine): the ONE place librosa.resample (or
# any future resampling backend) gets called. Model wrappers ask for
# `engine.view(track, sample_rate)`, never resample audio themselves -
# this is what kills the scattered per-stage resampling and temp-WAV
# round-trips Symphony's 5.x line accumulated.
# =================================================================

from __future__ import annotations

import numpy as np
import librosa

from audio_engine.track import AudioTrack, AudioView


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """The one authoritative resample call. samples must be mono 1-D."""
    if orig_sr == target_sr:
        return samples
    return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def get_view(track: AudioTrack, sample_rate: int) -> AudioView:
    """Mono view of `track` at `sample_rate`, cached on the track so a
    second request for the same rate is free. This is the only function
    that should ever populate AudioTrack._view_cache."""
    cached = track._view_cache.get(sample_rate)
    if cached is not None:
        return cached

    mono = track.mono()
    resampled = resample(mono, track.sample_rate, sample_rate)
    view = AudioView(samples=resampled, sample_rate=sample_rate, start_ms=0.0)
    track._view_cache[sample_rate] = view
    return view


def get_stereo(track: AudioTrack, sample_rate: int) -> np.ndarray:
    """Full original-channel-layout audio at `sample_rate`, read-only,
    cached on the track. Separation (Demucs et al.) needs the real
    inter-channel panning/phase cues a mono downmix discards - this is
    the one place that audio gets served, resampled per-channel through
    the same unified `resample()` every mono view uses."""
    cached = track._stereo_cache.get(sample_rate)
    if cached is not None:
        return cached

    if sample_rate == track.sample_rate:
        result = track.samples
    else:
        result = np.stack([
            resample(track.samples[ch], track.sample_rate, sample_rate)
            for ch in range(track.num_channels)
        ])
        result = result.astype(np.float32)
        result.setflags(write=False)

    track._stereo_cache[sample_rate] = result
    return result


__all__ = ["resample", "get_view", "get_stereo"]
