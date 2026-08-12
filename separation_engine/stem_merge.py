# =================================================================
# MODULE: separation_engine/stem_merge.py
# Post-separation stem consolidation.
#
# WHY THIS EXISTS (measured, not assumed): htdemucs_6s's guitar and
# piano heads are weakly trained relative to drums/bass/vocals, and they
# reassign the SAME content between guitar, piano and other over time -
# audible as an instrument fading arbitrarily between stems. Because the
# pipeline transcribes each stem independently and unions the results,
# an unstable assignment gets counted more than once:
#
#     |U T(stem_i)|  !=  |T(sum stem_i)|
#
# Transcription does not distribute over an unstable partition. Measured
# on Hopeful (GRIMLOCK_6.0_OPEN_PROBLEMS.md §XII.1):
#
#     guitar alone 1549 + piano alone 1564 + other alone 1532 = 4645
#     the three summed and transcribed ONCE               = 2183
#     -> 53% of those notes were the same music counted twice or thrice
#
# Merging restores what a 4-stem model's "other" would have been -
# drums/bass/vocals stay separated (those heads ARE reliable), and
# everything harmonic becomes one coherent stem. This is de-duplication,
# NOT filtering: no note is judged or deleted, the same audio is simply
# transcribed once instead of three times.
# =================================================================

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from audio_engine import AudioTrack
from core import Separation, StemType

# Merged INTO StemType.OTHER, which is already defined as the 4-stem
# catch-all ("everything that isn't drums/bass/vocals") - exactly what
# this sum reconstructs. No new stem identity is invented.
HARMONIC_STEMS: Tuple[StemType, ...] = (StemType.GUITAR, StemType.PIANO, StemType.OTHER)
MERGED_MODEL_SUFFIX = "+merged_harmonic"


def merge_harmonic_stems(separation: Separation) -> Separation:
    """Sums GUITAR + PIANO + OTHER into a single OTHER stem.

    Returns the separation unchanged if fewer than two of them are
    present (nothing to merge - e.g. a plain 4-stem model already).

    Stereo is PRESERVED: the sum runs over each stem's own canonical
    (channels, n) samples, not over a mono view. Peak-normalizes only if
    the sum exceeds full scale, so relative levels are otherwise
    untouched.
    """
    present = [s for s in HARMONIC_STEMS if separation.has_stem(s)]
    if len(present) < 2:
        return separation

    tracks = [separation.get_stem(s) for s in present]
    base = tracks[0]

    merged = np.zeros_like(np.asarray(base.samples), dtype=np.float32)
    for track in tracks:
        merged = merged + np.asarray(track.samples, dtype=np.float32)

    peak = float(np.max(np.abs(merged))) if merged.size else 0.0
    if peak > 1.0:
        merged = merged / peak
    merged = np.ascontiguousarray(merged, dtype=np.float32)

    merged_track = AudioTrack(
        samples=merged,
        sample_rate=base.sample_rate,
        num_channels=merged.shape[0],
        duration_seconds=merged.shape[-1] / base.sample_rate,
        source_path=f"{base.source_path}::merged_harmonic",
    )

    stems = {st: tr for st, tr in separation.stems.items() if st not in present}
    stems[StemType.OTHER] = merged_track

    # A merged member can no longer be individually "hallucinated".
    hallucinated = tuple(s for s in separation.stems_hallucinated if s not in present)

    return Separation(
        stems=stems,
        model_used=f"{separation.model_used}{MERGED_MODEL_SUFFIX}",
        confidence=separation.confidence,
        separation_time_seconds=separation.separation_time_seconds,
        stems_hallucinated=hallucinated,
    )


def load_cached_separation(stems_dir, sample_rate: int = 44100) -> Optional[Separation]:
    """Rebuild a Separation from stems already on disk (Input/<song>/stems/,
    as written by tools/build_stem_cache.py using the SAME model+seed the
    pipeline uses: htdemucs_6s, seed 0).

    WHY: Demucs is the single largest model in a run and re-separating audio we
    have already separated buys nothing. Loading the cache makes iteration
    cheaper AND makes runs bit-identical in their separation, removing one
    source of run-to-run variation when comparing notation changes.

    Returns None when the directory is missing or holds no recognised stems, so
    the caller can fall back to real separation rather than failing.
    """
    from pathlib import Path
    import numpy as np
    import soundfile as sf

    directory = Path(stems_dir)
    if not directory.is_dir():
        return None

    stems: Dict[StemType, AudioTrack] = {}
    for stem in StemType:
        path = directory / f"{getattr(stem, 'value', str(stem)).lower()}.wav"
        if not path.exists():
            continue
        samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
        samples = np.ascontiguousarray(samples.T, dtype=np.float32)   # (channels, n)
        stems[stem] = AudioTrack(
            samples=samples, sample_rate=int(sr), num_channels=samples.shape[0],
            duration_seconds=samples.shape[-1] / float(sr),
            source_path=str(path),
        )
    if not stems:
        return None
    return Separation(
        stems=stems, model_used="htdemucs_6s+cached",
        confidence=1.0, separation_time_seconds=0.0,
    )


__all__ = ["merge_harmonic_stems", "load_cached_separation",
           "HARMONIC_STEMS", "MERGED_MODEL_SUFFIX"]
