# =================================================================
# MODULE: rhythm_engine/pulse_field.py
# Ports Symphony's PulseField in full (agents/analysis/pulse_field.py) -
# probabilistic MULTI-hypothesis pulse-grid tracking rather than a
# single tempo value. "A note's value is defined by its distance to its
# neighbor, not its distance to the grid" - the pulse field provides
# relational context (several simultaneously live candidate periods,
# each with its own probability), which is what real polymeter/rubato/
# tempo-drift material needs: a single best-guess grid can't represent
# "this could plausibly be read at either the quarter-note or the
# dotted-quarter level" the way a probability distribution over
# harmonically-related periods can.
#
# Three real pieces, each doing a distinct job:
# - PulseHypothesisTracker: maintains several candidate periods
#   (harmonics/subdivisions of an initial tempo) and updates each one's
#   probability from how well real onsets align to it.
# - PhaseCoherenceDetector: circular-statistics phase detection - where
#   in each period's cycle onsets actually cluster, and how tightly.
# - PulseStrengthAnnealer: exponential-decay smoothing so a hypothesis's
#   strength doesn't jump around from one noisy window to the next.
# =================================================================

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import circmean, circvar

from audio_engine import AudioEngine, AudioTrack
from rhythm_engine.onsets import detect_onset_candidates

MIN_TEMPO_BPM = 20.0
MAX_TEMPO_BPM = 300.0
MIN_PULSE_STRENGTH = 0.1
PHASE_BINS = 36
HARMONIC_LEVELS = (1.0, 2.0, 3.0, 4.0, 0.5, 0.25)
REVERSE_GEO_WEIGHT = 0.3
BEAT_REFINEMENT_WINDOW_MS = 50.0
ANNEALING_RATE = 0.05


@dataclass(frozen=True)
class PulseHypothesis:
    """A candidate pulse period with probability weight."""
    period_ms: float
    probability: float
    harmonic_level: float


class PulseHypothesisTracker:
    """Tracks multiple pulse hypotheses simultaneously - each a
    possible pulse period (tempo) with its own strength."""

    def __init__(self) -> None:
        self.hypotheses: Dict[float, PulseHypothesis] = {}

    def initialize_hypotheses(self, tempo_bpm: float, confidence: float = 0.5) -> None:
        if tempo_bpm <= 0:
            return
        base_period_ms = 60000.0 / tempo_bpm
        for level in HARMONIC_LEVELS:
            period_ms = base_period_ms / level if level > 0 else base_period_ms * 2
            tempo = 60000.0 / period_ms if period_ms > 0 else tempo_bpm
            if MIN_TEMPO_BPM <= tempo <= MAX_TEMPO_BPM:
                self.hypotheses[period_ms] = PulseHypothesis(
                    period_ms=period_ms,
                    probability=confidence * (1.0 / abs(level) if level != 0 else 1.0),
                    harmonic_level=level,
                )

    def update_from_onsets(self, onset_times_ms: List[float]) -> None:
        if len(onset_times_ms) < 4:
            return
        for period_ms, hypothesis in list(self.hypotheses.items()):
            alignment_score = _compute_alignment_score(onset_times_ms, period_ms)
            new_prob = hypothesis.probability * 0.7 + alignment_score * 0.3
            self.hypotheses[period_ms] = replace(
                hypothesis, probability=max(MIN_PULSE_STRENGTH, min(1.0, new_prob)),
            )

    def update_from_lattice(self, lattice_period_ms: float, lattice_confidence: float) -> None:
        if lattice_period_ms <= 0 or not self.hypotheses:
            return
        closest_period = min(self.hypotheses.keys(), key=lambda p: abs(p - lattice_period_ms))
        current = self.hypotheses[closest_period]
        weighted_prob = current.probability * (1 - REVERSE_GEO_WEIGHT) + lattice_confidence * REVERSE_GEO_WEIGHT
        self.hypotheses[closest_period] = replace(current, probability=weighted_prob)

    def get_best_hypothesis(self) -> Optional[PulseHypothesis]:
        if not self.hypotheses:
            return None
        return max(self.hypotheses.values(), key=lambda h: h.probability)

    def get_all_hypotheses_sorted(self) -> List[PulseHypothesis]:
        return sorted(self.hypotheses.values(), key=lambda h: h.probability, reverse=True)


def _compute_alignment_score(onset_times_ms: List[float], period_ms: float) -> float:
    if len(onset_times_ms) < 2 or period_ms <= 0:
        return 0.0
    phases = [(t % period_ms) / period_ms for t in onset_times_ms]
    rad_phases = [2 * np.pi * p for p in phases]
    try:
        variance = circvar(rad_phases)
        return float(1.0 - variance)
    except Exception:
        hist, _ = np.histogram(phases, bins=PHASE_BINS, range=(0, 1))
        return float(hist[np.argmax(hist)] / len(phases))


class PhaseCoherenceDetector:
    """Circular-statistics phase detection: where in each period's
    cycle onsets actually cluster, and how tightly (coherence)."""

    @staticmethod
    def detect_phase(onset_times_ms: List[float], period_ms: float) -> Tuple[float, float]:
        """Returns (phase_offset_ms, coherence)."""
        if len(onset_times_ms) < 3 or period_ms <= 0:
            return 0.0, 0.0
        phases = [t % period_ms for t in onset_times_ms]
        rad_phases = [2 * np.pi * p / period_ms for p in phases]
        try:
            mean_rad = circmean(rad_phases)
            mean_phase_ms = (mean_rad / (2 * np.pi)) * period_ms
            coherence = float(1.0 - circvar(rad_phases))
        except Exception:
            hist, bin_edges = np.histogram(phases, bins=PHASE_BINS)
            peak_idx = np.argmax(hist)
            mean_phase_ms = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
            coherence = float(hist[peak_idx] / len(phases))
        return float(mean_phase_ms), coherence

    @staticmethod
    def refine_phase_with_onsets(phase_ms: float, period_ms: float, onset_times_ms: List[float]) -> float:
        """Nudges phase to align with nearby real onsets (median
        refinement across several predicted beats, for robustness)."""
        if not onset_times_ms or period_ms <= 0:
            return phase_ms
        predicted_beats = [phase_ms + i * period_ms for i in range(-3, 4)]
        refinements = []
        onset_arr = np.asarray(onset_times_ms)
        for beat in predicted_beats:
            distances = np.abs(onset_arr - beat)
            min_idx = int(np.argmin(distances))
            if distances[min_idx] < BEAT_REFINEMENT_WINDOW_MS:
                refinements.append(onset_arr[min_idx] - beat)
        if refinements:
            return phase_ms + float(np.median(refinements))
        return phase_ms


class PulseStrengthAnnealer:
    """Exponential-decay smoothing so a hypothesis's strength doesn't
    jump around from one noisy measurement to the next."""

    def __init__(self) -> None:
        self._strength_history: Dict[float, Deque[float]] = {}
        self._last_time_ms: float = 0.0

    def anneal_strength(self, period_ms: float, current_strength: float, time_ms: float) -> float:
        if period_ms not in self._strength_history:
            self._strength_history[period_ms] = deque(maxlen=10)
            self._strength_history[period_ms].append(current_strength)
            return current_strength

        delta_sec = (time_ms - self._last_time_ms) / 1000.0 if self._last_time_ms > 0 else 0.0
        self._last_time_ms = time_ms
        decay = float(np.exp(-delta_sec * ANNEALING_RATE))
        last_strength = self._strength_history[period_ms][-1]
        annealed = last_strength * decay + current_strength * (1 - decay)
        self._strength_history[period_ms].append(annealed)
        return annealed


@dataclass(frozen=True)
class PulseFieldResult:
    best_period_ms: float
    best_tempo_bpm: float
    best_probability: float
    phase_ms: float
    coherence: float
    all_hypotheses: Tuple[PulseHypothesis, ...] = field(default_factory=tuple)


def run_pulse_field(
        engine: AudioEngine,
        track: AudioTrack,
        initial_tempo_bpm: float,
        initial_confidence: float = 0.5,
        lattice_period_ms: Optional[float] = None,
        lattice_confidence: float = 0.0,
) -> Optional[PulseFieldResult]:
    """Runs the full multi-hypothesis pulse tracking pipeline: seed
    hypotheses at harmonic levels of `initial_tempo_bpm`, reinforce
    them against real onset candidates, optionally fold in a
    ReverseGeoCrypt lattice reading, then detect and refine phase for
    the winning hypothesis. Returns None if there's no usable tempo to
    seed from."""
    if initial_tempo_bpm <= 0:
        return None

    tracker = PulseHypothesisTracker()
    tracker.initialize_hypotheses(initial_tempo_bpm, initial_confidence)
    if not tracker.hypotheses:
        return None

    candidates = detect_onset_candidates(engine, track)
    onset_times_ms = list(candidates.combined_ms)
    tracker.update_from_onsets(onset_times_ms)

    if lattice_period_ms is not None:
        tracker.update_from_lattice(lattice_period_ms, lattice_confidence)

    best = tracker.get_best_hypothesis()
    if best is None:
        return None

    phase_ms, coherence = PhaseCoherenceDetector.detect_phase(onset_times_ms, best.period_ms)
    phase_ms = PhaseCoherenceDetector.refine_phase_with_onsets(phase_ms, best.period_ms, onset_times_ms)

    return PulseFieldResult(
        best_period_ms=best.period_ms,
        best_tempo_bpm=60000.0 / best.period_ms if best.period_ms > 0 else 0.0,
        best_probability=best.probability,
        phase_ms=phase_ms,
        coherence=coherence,
        all_hypotheses=tuple(tracker.get_all_hypotheses_sorted()),
    )


__all__ = [
    "PulseHypothesis", "PulseHypothesisTracker", "PhaseCoherenceDetector",
    "PulseStrengthAnnealer", "PulseFieldResult", "run_pulse_field",
]
