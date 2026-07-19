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

from typing import Tuple

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


__all__ = ["merge_harmonic_stems", "HARMONIC_STEMS", "MERGED_MODEL_SUFFIX"]
