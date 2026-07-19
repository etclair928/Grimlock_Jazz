# =================================================================
# MODULE: rhythm_engine/onsets.py
# Shared onset-candidate detection (GRIMLOCK_6.0_DESIGN_DECISIONS.md
# §7 Rhythm Engine). Ports Symphony's OnsetPrecision dual-witness
# pattern (agents/quantization/ritornello.py) - librosa's spectral-flux
# onset-strength + backtrack recipe, corroborated by madmom's
# RNNOnsetProcessor (a real, pretrained onset-specific model,
# independent of the DBN beat tracker already used for tempo).
#
# ONE place computes onset candidates; every other module in
# rhythm_engine/ and quantization/ (tempo witnesses, meter detection,
# groove, the lattice quantizer) asks this instead of each running its
# own librosa.onset.onset_detect() call - consolidates duplicate work
# and gives every consumer the same corroborated candidates instead of
# silently drifting apart on "how onsets get detected here."
# =================================================================

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

from audio_engine import AudioEngine, AudioTrack

ONSET_SAMPLE_RATE = 22050
ONSET_HOP_LENGTH = 256
MADMOM_ONSET_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class OnsetCandidates:
    """Sorted onset timestamps (ms) from each witness. `madmom_ms` is
    empty when madmom isn't installed or fails - callers treat that as
    "absent," not "disagreement" (see refine_onset_to_nearest)."""
    librosa_ms: Tuple[float, ...]
    madmom_ms: Tuple[float, ...]

    @property
    def combined_ms(self) -> Tuple[float, ...]:
        """Every candidate from either witness, deduplicated by
        rounding to the nearest ms and sorted - useful for consumers
        (e.g. the lattice witness) that want ALL plausible events
        rather than a per-note refinement."""
        merged = sorted(set(round(t) for t in self.librosa_ms) | set(round(t) for t in self.madmom_ms))
        return tuple(float(t) for t in merged)


def _detect_librosa_candidates_ms(audio: np.ndarray, sample_rate: int) -> Tuple[float, ...]:
    """Spectral-flux onset-strength -> peak-pick -> backtrack to the
    nearest true energy transient - the same recipe already validated
    in this project's history for placing onsets more precisely than
    a bare peak-pick."""
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=ONSET_HOP_LENGTH)
    except Exception:
        return ()
    if len(onset_env) == 0:
        return ()

    try:
        peak_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sample_rate, hop_length=ONSET_HOP_LENGTH,
            units="frames", backtrack=False,
        )
    except Exception:
        return ()
    if len(peak_frames) == 0:
        return ()

    backtracked_frames = librosa.onset.onset_backtrack(peak_frames, onset_env)
    candidates = sorted(float(f) * ONSET_HOP_LENGTH / sample_rate * 1000.0 for f in backtracked_frames)
    return tuple(candidates)


def _detect_madmom_candidates_ms(audio: np.ndarray, sample_rate: int) -> Tuple[float, ...]:
    """madmom's RNNOnsetProcessor + OnsetPeakPickingProcessor - a real
    pretrained onset-specific model, independent of the librosa
    spectral-flux recipe above. Optional dependency; returns empty on
    any failure rather than raising."""
    try:
        from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor
    except ImportError:
        return ()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    try:
        sf.write(temp_path, audio, sample_rate)
        activations = RNNOnsetProcessor()(temp_path)
        if len(activations) == 0:
            return ()
        peaks_sec = OnsetPeakPickingProcessor(fps=100)(activations)
        return tuple(sorted(float(p) * 1000.0 for p in peaks_sec))
    except Exception:
        return ()
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def detect_onset_candidates(engine: AudioEngine, track: AudioTrack) -> OnsetCandidates:
    """The one place onset candidates get computed for a track. Runs
    both witnesses at their own preferred sample rates via the shared
    Audio Engine views (no independent resampling)."""
    librosa_view = engine.view(track, ONSET_SAMPLE_RATE)
    madmom_view = engine.view(track, MADMOM_ONSET_SAMPLE_RATE)

    return OnsetCandidates(
        librosa_ms=_detect_librosa_candidates_ms(librosa_view.samples, ONSET_SAMPLE_RATE),
        madmom_ms=_detect_madmom_candidates_ms(madmom_view.samples, MADMOM_ONSET_SAMPLE_RATE),
    )


def _nearest_within(candidates: Tuple[float, ...], target_ms: float, max_shift_ms: float) -> Optional[float]:
    if not candidates:
        return None
    nearby = [c for c in candidates if abs(c - target_ms) <= max_shift_ms]
    if not nearby:
        return None
    return min(nearby, key=lambda c: abs(c - target_ms))


def refine_onset_to_nearest(
        candidates: OnsetCandidates, target_ms: float, max_shift_ms: float,
) -> Tuple[Optional[float], float]:
    """Resolves `target_ms` (a coarse onset estimate) against both
    witnesses. Returns (resolved_ms, disagreement) where disagreement
    is in [0, 1] - 0 when only one witness has a nearby candidate
    (absence isn't disagreement), otherwise the normalized distance
    between the two witnesses' candidates. Returns (None, 0.0) when
    neither witness has anything nearby."""
    librosa_best = _nearest_within(candidates.librosa_ms, target_ms, max_shift_ms)
    madmom_best = _nearest_within(candidates.madmom_ms, target_ms, max_shift_ms)

    if librosa_best is None and madmom_best is None:
        return None, 0.0
    if librosa_best is not None and madmom_best is not None:
        resolved = 0.5 * librosa_best + 0.5 * madmom_best
        disagreement = abs(librosa_best - madmom_best) / max(max_shift_ms, 1.0)
        return resolved, min(1.0, disagreement)
    return (librosa_best if librosa_best is not None else madmom_best), 0.0


__all__ = ["OnsetCandidates", "detect_onset_candidates", "refine_onset_to_nearest"]
