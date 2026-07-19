# =================================================================
# MODULE: audio_engine/track.py
# The canonical audio proxy (GRIMLOCK_6.0_DESIGN_DECISIONS.md §7 Audio
# Engine): AudioTrack is the ONE decoded array per source file - nothing
# downstream re-decodes or re-loads. AudioView is what agents actually
# receive: an immutable, read-only slice at a specific sample rate. Views
# are requested from the engine (audio_engine/__init__.py's AudioEngine),
# never constructed by callers, so resampling stays in exactly one place.
#
# AudioTrack itself is not frozen - it lazily grows a view/transform
# cache as the engine services requests - but `samples` is marked
# non-writeable at construction, so the one thing that must never change
# (the decoded master) is protected the same way Note protects pitch:
# the container can grow bookkeeping, the fact it holds cannot mutate.
# =================================================================

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


def _content_hash(samples: np.ndarray, source_path: str) -> str:
    """Cheap cache-key identity for a decoded track - NOT a substitute for
    Ingestion's forensic file-integrity hash. Sampling a slice rather than
    hashing the full array keeps decode-time overhead flat regardless of
    track length."""
    sample_bytes = samples.astype(np.float32).tobytes()[:65536]
    return hashlib.sha256(f"{source_path}:{samples.shape}:{sample_bytes}".encode()).hexdigest()


@dataclass
class AudioTrack:
    """The canonical decoded master for one source file. One AudioTrack
    per file, ever - everything else downstream is a view into it.

    samples keeps its ORIGINAL channel layout, shape (channels, n) -
    always 2-D, mono files included (channels=1). A real bug in Symphony
    (§7a of the audit) came from forcing mono unconditionally at
    ingestion, which fed Demucs two duplicated-mono channels disguised as
    stereo and discarded the panning/phase cues separation needs. Mono
    downmix happens on demand via `.mono()`, never at decode time - the
    one consumer group that actually wants mono (Pitch Engine's views)
    asks for it explicitly through the engine.
    """
    samples: np.ndarray          # (channels, n) float32, the canonical decode
    sample_rate: int
    num_channels: int
    duration_seconds: float
    source_path: str
    content_hash: str = field(default="", repr=False, compare=False)
    _view_cache: Dict[int, "AudioView"] = field(default_factory=dict, repr=False, compare=False)
    _mono_cache: "np.ndarray | None" = field(default=None, repr=False, compare=False)
    _stereo_cache: Dict[int, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError(
                f"AudioTrack.samples must be (channels, n), got shape {self.samples.shape}"
            )
        self.samples.setflags(write=False)
        if not self.content_hash:
            self.content_hash = _content_hash(self.samples, self.source_path)

    def mono(self) -> np.ndarray:
        """Power-preserving mono downmix, cached, read-only. (L+R)/sqrt(2)
        for stereo so RMS energy is preserved rather than halved by a
        plain average."""
        if self._mono_cache is None:
            if self.num_channels == 1:
                mono = self.samples[0]
            else:
                mono = np.sum(self.samples, axis=0) / np.sqrt(self.num_channels)
            mono = mono.astype(np.float32)
            mono.setflags(write=False)
            self._mono_cache = mono
        return self._mono_cache

    def __repr__(self) -> str:
        return (
            f"AudioTrack({self.duration_seconds:.1f}s, {self.sample_rate}Hz, "
            f"channels={self.num_channels}, '{self.source_path}', "
            f"hash={self.content_hash[:12]}...)"
        )


@dataclass(frozen=True)
class AudioView:
    """An immutable, read-only slice of a track at a specific sample
    rate. This is what Pitch Engine / Rhythm Engine / anything else
    actually consumes - never AudioTrack.samples directly.

    start_ms anchors this view's sample 0 within the ORIGINAL track's
    timeline, so a caller that sliced a sub-segment can still report
    onset/offset times relative to the whole track.
    """
    samples: np.ndarray   # mono float32, read-only, 1-D
    sample_rate: int
    start_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"AudioView.samples must be mono 1-D, got shape {self.samples.shape}")
        self.samples.setflags(write=False)

    @property
    def duration_ms(self) -> float:
        return len(self.samples) / self.sample_rate * 1000.0

    def slice_ms(self, start_ms: float, end_ms: float) -> "AudioView":
        """A read-only sub-view over [start_ms, end_ms) of THIS view.
        Slicing a non-writeable numpy array yields a non-writeable view
        over the same buffer - no copy, immutability holds transitively."""
        start_idx = max(0, int(start_ms / 1000.0 * self.sample_rate))
        end_idx = min(len(self.samples), int(end_ms / 1000.0 * self.sample_rate))
        if end_idx < start_idx:
            end_idx = start_idx
        return AudioView(
            samples=self.samples[start_idx:end_idx],
            sample_rate=self.sample_rate,
            start_ms=self.start_ms + start_ms,
        )

    def __repr__(self) -> str:
        return f"AudioView({self.duration_ms:.0f}ms @ {self.sample_rate}Hz, start_ms={self.start_ms:.0f})"


__all__ = ["AudioTrack", "AudioView"]
