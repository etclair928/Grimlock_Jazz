# =================================================================
# MODULE: audio_engine/decoder.py
# Decoder boundary (§7 Audio Engine): wraps SoundFile/librosa, decodes a
# source file exactly ONCE into a canonical AudioTrack. Nothing else in
# the codebase should call soundfile.read/librosa.load directly - ask the
# engine for a track instead, so there is exactly one place decode
# failures, format quirks, and normalization live.
# =================================================================

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np

from audio_engine.track import AudioTrack

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def _read_raw(path: Path) -> Tuple[np.ndarray, int]:
    """Returns (samples, sample_rate) with samples shape (channels, n),
    float32. soundfile is tried first (it preserves original channel
    layout without librosa's resampling side effects); librosa is the
    fallback for formats soundfile can't open."""
    # Fallback ladder: soundfile first, librosa second. A backend's failure
    # falls through to the next, but the WHY is captured (not swallowed) so
    # that if BOTH fail the raise below reports each backend's actual error
    # rather than a generic "no working backend" - a real decode failure
    # should say why it failed, per the surface-loudly law (DESIGN_DECISIONS
    # §10.1), not hand back an uninformative message.
    errors: list = []
    if _HAS_SOUNDFILE:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return data.T, sr  # sf gives (n, channels) -> (channels, n)
        except Exception as exc:
            errors.append(f"soundfile: {type(exc).__name__}: {exc}")

    if _HAS_LIBROSA:
        try:
            data, sr = librosa.load(str(path), mono=False, sr=None, dtype=np.float32)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            return data, sr
        except Exception as exc:
            errors.append(f"librosa: {type(exc).__name__}: {exc}")

    detail = "; ".join(errors) if errors else "no decode backend installed"
    raise RuntimeError(
        f"Could not decode {path} (soundfile={_HAS_SOUNDFILE}, "
        f"librosa={_HAS_LIBROSA}): {detail}"
    )


def decode_file(path: Union[str, Path]) -> AudioTrack:
    """Decode a source file into the one canonical AudioTrack for it.
    Peak-normalizes only if the file exceeds [-1, 1] (some encoders clip
    slightly above digital full-scale); never rescales audio that's
    already in range, since that would silently change relative levels
    across a whole batch of otherwise-compliant files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    samples, sample_rate = _read_raw(path)

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak

    num_channels = samples.shape[0]
    duration_seconds = samples.shape[1] / sample_rate if sample_rate else 0.0

    return AudioTrack(
        samples=samples.astype(np.float32),
        sample_rate=sample_rate,
        num_channels=num_channels,
        duration_seconds=duration_seconds,
        source_path=str(path),
    )


__all__ = ["decode_file"]
