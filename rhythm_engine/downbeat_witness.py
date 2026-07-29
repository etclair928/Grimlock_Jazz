# =================================================================
# MODULE: rhythm_engine/downbeat_witness.py
# The "distributed downbeat" witness (open problem #7): where is beat 1,
# and how many beats to a bar?
#
# WHY A TRAINED MODEL, NOT A HAND-ROLLED FUSION. #7 says the macro pulse
# is felt through harmony/bass/accents, not onset loudness, and proposes
# fusing bass-roots + drum-accents + harmonic-change. We MEASURED each of
# those on the test library (2026-07-28): every single source is a WEAK
# downbeat cue - beat-synchronous harmonic change concentrates at only
# 1.04-1.23x chance (flat even on a clean 4/4), and the "kick on 1"
# heuristic tops out at 1.10-1.38x and is WEAKEST on the syncopated
# hip-hop where you'd most want it. Hand-fusing weak votes gives a weak,
# fragile result. madmom's DBNDownBeatTrackingProcessor (an RNN over the
# spectrogram feeding a dynamic Bayesian network bar-pointer model) fuses
# exactly these cues implicitly and is the literature SOTA - and madmom is
# ALREADY a Jazz dependency (rhythm_engine.tempo_witness.run_madmom_tempo).
# On the three known-answer songs it matched the ear where the onset-only
# detector failed: No Pasaran 130 BPM 4/4 (ear: "132"; old pipeline: 65),
# Hopeful 146 BPM 6/4 (ear: "6/4 ~147"; old pipeline: 152 4/4), You Say
# stable 4/4 - all with downbeat-spacing CV < 0.07.
#
# This is a WITNESS (helper, not fighter): it returns its reading with a
# confidence; the Epistemic layer's resolve_meter votes it against the
# existing onset-salience and FFT-periodicity estimates rather than letting
# it hard-override them.
# =================================================================

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from audio_engine import AudioEngine, AudioTrack
from core import Provenance

DOWNBEAT_SAMPLE_RATE = 44100          # madmom's RNN was trained at 44.1k
# Bar lengths madmom is allowed to choose between. 3 (waltz), 4 (common),
# 6 (6/4 - Hopeful needs this). Kept small: every extra option is another
# hypothesis the DBN can wander into on ambiguous material.
DEFAULT_BEATS_PER_BAR = (3, 4, 6)


@dataclass(frozen=True)
class DownbeatWitness:
    """One reading of the metrical top level. `beats_per_bar` is the
    numerator; `downbeat_times_ms` are the beat-1 instants; `confidence`
    is driven by how regular the bar length is (a stable meter is a
    trustworthy reading, a wandering one is not)."""
    beats_per_bar: int
    downbeat_times_ms: Tuple[float, ...]
    beat_times_ms: Tuple[float, ...]
    confidence: float
    source: Provenance = Provenance.RHYTHM_ENGINE


def run_madmom_downbeat(
        engine: AudioEngine,
        track: AudioTrack,
        beats_per_bar: Sequence[int] = DEFAULT_BEATS_PER_BAR,
) -> Optional[DownbeatWitness]:
    """Runs madmom's RNN+DBN downbeat tracker on `track`. Returns a
    DownbeatWitness, or None if the tracker produced too little to judge.
    Mirrors run_madmom_tempo's temp-WAV pattern (madmom's processors read
    a file, not a raw array)."""
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

    view = engine.view(track, DOWNBEAT_SAMPLE_RATE)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    try:
        sf.write(temp_path, view.samples, DOWNBEAT_SAMPLE_RATE)
        activations = RNNDownBeatProcessor()(temp_path)
        result = DBNDownBeatTrackingProcessor(beats_per_bar=list(beats_per_bar), fps=100)(activations)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    if result is None or len(result) < 4:
        return None

    beat_times_ms = np.asarray(result[:, 0], dtype=np.float64) * 1000.0
    positions = np.asarray(result[:, 1], dtype=int)
    downbeat_times_ms = beat_times_ms[positions == 1]
    if len(downbeat_times_ms) < 3:
        return None

    # Numerator = the most common bar length in beats (mode of the gaps
    # between consecutive downbeats), robust to an occasional dropped/added
    # bar; NOT max(position) which any single stray reading inflates.
    beat_ioi_ms = float(np.median(np.diff(beat_times_ms)))
    bar_beats = np.round(np.diff(downbeat_times_ms) / beat_ioi_ms).astype(int)
    bar_beats = bar_beats[bar_beats > 0]
    if len(bar_beats) == 0:
        return None
    values, counts = np.unique(bar_beats, return_counts=True)
    beats_per_bar_est = int(values[np.argmax(counts)])
    mode_share = float(counts.max()) / float(counts.sum())

    # Confidence: a stable meter (low bar-length CV) read consistently
    # (high mode share) is trustworthy; a wandering one is not.
    downbeat_iois = np.diff(downbeat_times_ms)
    cv = float(np.std(downbeat_iois) / (np.mean(downbeat_iois) + 1e-9))
    confidence = float(np.clip((1.0 - 4.0 * cv) * mode_share, 0.3, 0.92))

    return DownbeatWitness(
        beats_per_bar=beats_per_bar_est,
        downbeat_times_ms=tuple(float(t) for t in downbeat_times_ms),
        beat_times_ms=tuple(float(t) for t in beat_times_ms),
        confidence=confidence,
    )


__all__ = ["DownbeatWitness", "run_madmom_downbeat", "DEFAULT_BEATS_PER_BAR"]
