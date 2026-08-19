# =================================================================
# MODULE: rhythm_engine/meter.py
# Ports Symphony's TimeSignatureDetector (agents/analysis/
# tempo_intelligence.py) - the real "ensemble downbeat finder" - plus
# two fixes converged on independently by three separate reviews
# (this session's own diagnosis, and two external AI reviews) of why
# larger bar lengths (6/4 in particular) were never surfacing:
#
# 1. PHASE-LOCKED GRID (build_phase_locked_grid): every witness's own
#    beat grid carries that witness's own tempo error. At a ~5% tempo
#    error, accumulated phase drift exceeds this codebase's own 15%
#    alignment tolerance within about 3 beats (0.15/0.05 ≈ 3) - a bar
#    of 4 mostly fits inside that window, a bar of 6 does not. Testing
#    meter against three separately-drifting witness grids was
#    measuring which grid stayed phase-locked long enough to sample
#    the accent correctly, not measuring meter. Fix: build ONE grid
#    anchored to the already-resolved tempo (the best cross-witness
#    estimate), with phase found via PulseField's own tested circular-
#    statistics code - not a fresh, cruder implementation.
#
# 2. PER-N SIGNIFICANCE (percentile threshold): a flat 0.3 cutoff on
#    the chance-corrected score is applied identically to every
#    candidate N, but larger N means fewer accent samples per position-
#    class, hence a wider, noisier shuffle (chance) distribution for
#    exactly the same reason smaller samples always have more variance.
#    A flat threshold ignores that and is systematically harder for
#    larger N to clear even when a real signal is present. Fix: gate on
#    whether the real salience clears THIS candidate's own shuffle
#    distribution's 95th percentile, not one constant for every N -
#    this auto-adjusts for N because the distribution itself is wider
#    for larger N.
#
# Method (unchanged core): sample the onset-strength envelope at each
# beat to get a genuine per-beat accent value, then for each candidate
# bar length test every possible phase (which beat is "beat 1") for how
# much stronger that phase's average accent is than the other in-bar
# positions.
#
# Returns `downbeat_phase` - which beat_times index is the real
# downbeat - so a pickup/anacrusis measure (where index 0 is NOT beat 1)
# doesn't get silently mislabeled.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from audio_engine import AudioEngine, AudioTrack
from rhythm_engine.pulse_field import PhaseCoherenceDetector

METER_SAMPLE_RATE = 22050
METER_HOP_LENGTH = 512
CANDIDATE_NUMERATORS = (2, 3, 4, 5, 6, 7, 9, 12)
NUMERATORS_WITH_COMPOUND_READING = (6, 9, 12)
N_PERMUTATIONS = 100
# A pickup is only declared when the off-beat-0 downbeat clears BOTH the
# shuffle significance test AND this salience floor. Marginal significance
# at tiny salience is accent-fold noise, not a real anacrusis.
PICKUP_MIN_CONFIDENCE = 0.30
SIGNIFICANCE_PERCENTILE = 95.0


def build_phase_locked_grid(
        resolved_tempo_bpm: float,
        onset_times_ms: Sequence[float],
        duration_ms: float,
) -> Tuple[float, ...]:
    """One canonical beat grid anchored to the already-resolved tempo -
    not any single witness's own independently-tracked grid. Phase-
    locking to a fixed, already-correct tempo and searching only for
    phase is a far smaller, better-conditioned problem than the joint
    tempo+phase search each individual beat tracker runs, and it
    removes the accumulated-drift failure mode described above before
    a downbeat-salience test ever sees the data."""
    if resolved_tempo_bpm <= 0:
        return ()
    period_ms = 60000.0 / resolved_tempo_bpm
    onset_times_ms = list(onset_times_ms)
    phase_ms, _coherence = PhaseCoherenceDetector.detect_phase(onset_times_ms, period_ms)
    phase_ms = PhaseCoherenceDetector.refine_phase_with_onsets(phase_ms, period_ms, onset_times_ms)
    n_beats = max(2, int(duration_ms / period_ms) + 1)
    return tuple(float(phase_ms + i * period_ms) for i in range(n_beats))


def _sample_accents_at_beats(beat_times_ms: np.ndarray, onset_env: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Interpolates the real onset-strength envelope at each beat time."""
    frame_times_ms = np.arange(len(onset_env)) * hop_length / sr * 1000.0
    return np.interp(beat_times_ms, frame_times_ms, onset_env)


def _downbeat_salience(beat_accents: np.ndarray, numerator: int) -> Tuple[float, int]:
    """Best-phase downbeat salience: (loudest position - rest) / (loudest + rest).
    Returns (salience, phase) where phase is the beat_accents index
    (0..numerator-1) whose position-class scored loudest."""
    n = len(beat_accents)
    if n < numerator:
        return 0.0, 0
    # ONE partition, not `numerator` rotations of it. The position CLASSES a
    # phase induces are just a relabelling of the phase-0 classes - the old
    # loop rebuilt all of them from scratch for every phase and then took the
    # max, which is the same answer for numerator times the work, inside 100
    # permutations x 8 candidate numerators (2026-08-17 audit).
    #
    # (The old loop also started each phase's scan at index `phase`, so each
    # phase silently scored a different subset of the beats - a phase could win
    # by dropping noisy leading beats rather than by finding a real downbeat.
    # Using the whole sequence for every class removes that artifact.)
    accents = np.asarray(beat_accents, dtype=np.float64)
    usable = (len(accents) // numerator) * numerator
    folded = accents[:usable].reshape(-1, numerator)
    position_means = folded.mean(axis=0)

    downbeat_idx = int(np.argmax(position_means))
    downbeat_mean = float(position_means[downbeat_idx])
    others = np.delete(position_means, downbeat_idx)
    other_mean = float(others.mean()) if others.size else 0.0

    denom = downbeat_mean + other_mean
    salience = (downbeat_mean - other_mean) / denom if denom > 0 else 0.0
    return max(0.0, salience), downbeat_idx


def resolve_denominator(numerator: int, ratio_family: Optional[str]) -> int:
    """Simple (4) unless real ternary evidence justifies compound (8)
    for a 6/9/12-beat bar - shared by every caller that needs to turn a
    numerator into a full time signature, so this decision is made in
    exactly one place (see estimate_time_signature's docstring for why
    numerator alone can't decide it)."""
    if numerator in NUMERATORS_WITH_COMPOUND_READING and ratio_family == "ternary":
        return 8
    return 4


def _score_candidate(beat_accents: np.ndarray, numerator: int) -> Tuple[float, int, bool]:
    """Chance-corrected downbeat salience for a candidate bar length,
    gated by THIS candidate's own shuffle distribution rather than one
    flat constant applied to every N (see module docstring, fix 2).
    Returns (score, phase, is_significant)."""
    real_salience, winning_phase = _downbeat_salience(beat_accents, numerator)

    rng = np.random.default_rng(seed=numerator)
    shuffled = beat_accents.copy()
    chance_saliences = np.empty(N_PERMUTATIONS, dtype=np.float64)
    for i in range(N_PERMUTATIONS):
        rng.shuffle(shuffled)
        chance_saliences[i] = _downbeat_salience(shuffled, numerator)[0]

    chance_baseline = float(np.mean(chance_saliences))
    chance_threshold = float(np.percentile(chance_saliences, SIGNIFICANCE_PERCENTILE))

    if chance_baseline >= 1.0:
        return 0.0, winning_phase, False

    score = max(0.0, min(1.0, (real_salience - chance_baseline) / (1.0 - chance_baseline)))
    is_significant = real_salience > chance_threshold
    return score, winning_phase, is_significant


@dataclass(frozen=True)
class PickupResult:
    """Where the first true downbeat falls. `pickup_beats` is how many
    beats precede it - the anacrusis. 0 means the song starts ON beat 1
    (no pickup)."""
    pickup_beats: int
    downbeat_phase: int        # beat index (mod numerator) carrying the downbeat
    numerator: int
    confidence: float
    is_significant: bool
    detail: str = ""


def detect_pickup(
        engine: AudioEngine,
        track: AudioTrack,
        beat_times_ms: Sequence[float],
        numerator: int,
) -> PickupResult:
    """Detects a pickup/anacrusis: does beat 0 carry the downbeat, or do
    the first N beats precede the first full measure?

    Reuses the same chance-corrected downbeat-salience machinery the meter
    detector already trusts (_score_candidate): fold the per-beat accents
    by the bar length, find which beat-in-bar is the loudest (the
    downbeat), and read the pickup off its phase.

    BURDEN OF PROOF is on 'there is a pickup', never on 'there isn't': a
    non-zero pickup is only reported when the off-beat-0 downbeat is
    STATISTICALLY significant against shuffled accents. A song that starts
    on the downbeat, and a song whose first-beat evidence is weak, both
    return pickup_beats = 0. We do not invent anacruses."""
    if numerator < 2 or len(beat_times_ms) < numerator * 2:
        return PickupResult(0, 0, numerator, 0.0, False, "too few beats to judge")

    accents = sample_beat_accents(engine, track, beat_times_ms)
    score, phase, is_significant = _score_candidate(accents, numerator)

    # Burden of proof: a non-zero pickup needs BOTH statistical significance
    # AND a real salience margin. A marginally-significant weak downbeat
    # (e.g. a 5-beat "pickup" at score 0.12) is noise in the accent fold, not
    # an anacrusis - a song that starts on the downbeat reads pickup 0.
    if phase == 0 or not is_significant or score < PICKUP_MIN_CONFIDENCE:
        reason = ("starts on the downbeat" if phase == 0
                  else f"downbeat-at-beat-{phase} not significant vs chance" if not is_significant
                  else f"downbeat-at-beat-{phase} salience {score:.2f} below floor {PICKUP_MIN_CONFIDENCE} - too weak to call")
        return PickupResult(
            pickup_beats=0, downbeat_phase=phase, numerator=numerator,
            confidence=score, is_significant=is_significant,
            detail=reason + " - no pickup declared",
        )

    return PickupResult(
        pickup_beats=phase, downbeat_phase=phase, numerator=numerator,
        confidence=score, is_significant=True,
        detail=f"first downbeat at beat index {phase} -> {phase}-beat pickup",
    )


def sample_beat_accents(engine: AudioEngine, track: AudioTrack, beat_times_ms: Sequence[float]) -> np.ndarray:
    """Public wrapper so callers (e.g. the Conductor, feeding
    fft_meter_candidate) can get the same per-beat accent values
    estimate_time_signature uses internally, without duplicating the
    onset-envelope sampling logic. engine.onset_envelope() is itself
    cached, so calling this alongside estimate_time_signature is cheap."""
    onset_env = engine.onset_envelope(track, METER_SAMPLE_RATE, hop_length=METER_HOP_LENGTH)
    return _sample_accents_at_beats(
        np.asarray(beat_times_ms, dtype=np.float64), onset_env, METER_SAMPLE_RATE, METER_HOP_LENGTH,
    )


def fft_meter_candidate(
        beat_accents: np.ndarray,
        candidate_numerators: Sequence[int] = CANDIDATE_NUMERATORS,
) -> Optional[Tuple[int, float]]:
    """Independent second method: look for periodicity in the beat-
    accent sequence's own power spectrum instead of testing each
    candidate bar length one at a time against a downbeat-salience
    heuristic. A real N-beat cycle should show spectral energy at
    frequency 1/N (cycles per beat) regardless of whether the salience
    test's own assumptions (one loudest position, chance-shuffling)
    happen to hold for this material. Returns (numerator, relative
    power) for whichever candidate has the strongest spectral peak, or
    None if there's too little data to say anything real."""
    n = len(beat_accents)
    if n < 16:
        return None

    detrended = beat_accents - np.mean(beat_accents)
    spectrum = np.abs(np.fft.rfft(detrended)) ** 2
    total_power = float(np.sum(spectrum)) + 1e-8

    best_numerator, best_relative_power = None, 0.0
    for numerator in candidate_numerators:
        if n < numerator * 3:
            continue
        target_bin = int(round(n / numerator))
        if target_bin <= 0 or target_bin >= len(spectrum):
            continue
        window = spectrum[max(0, target_bin - 1):target_bin + 2]
        relative_power = float(np.sum(window) / total_power)
        if relative_power > best_relative_power:
            best_relative_power = relative_power
            best_numerator = numerator

    if best_numerator is None:
        return None
    return best_numerator, best_relative_power


def estimate_time_signature(
        engine: AudioEngine,
        track: AudioTrack,
        beat_times_ms: Sequence[float],
        ratio_family: Optional[str] = None,
) -> Tuple[int, int, float]:
    """Returns (numerator, denominator, confidence) for the best-
    scoring candidate bar length, chance-corrected and gated per-N
    against its own shuffle distribution (see module docstring). Falls
    back to the 4/4 prior (not a hardcoded unconditional answer - just
    the most common meter) when fewer than 8 beats are available or no
    candidate clears its own significance threshold.

    `ratio_family` (from lattice_witness.find_ratio_clusters, the same
    real per-track evidence the tempo lattice search already computes)
    decides simple vs compound for a 6/9/12-beat bar - NOT the numerator
    alone. A bar of 6 is genuinely ambiguous on numerator count alone:
    6/4 (six quarter-note beats, simple) and 6/8 (two dotted-quarter
    beats each subdividing into three eighth notes, compound) can
    describe the same audio two different ways, and only real evidence
    of ternary sub-beat subdivision distinguishes them. Absent real
    ternary evidence (the common case - most bars-of-6 in practice ARE
    6/8, but not all), this reports simple (denominator=4) rather than
    assuming compound.
    """
    if len(beat_times_ms) < 8:
        return 4, 4, 0.3

    beat_accents = sample_beat_accents(engine, track, beat_times_ms)

    candidates = []
    for numerator in CANDIDATE_NUMERATORS:
        if len(beat_accents) < numerator * 3:
            continue  # need at least 3 complete bars to say anything real
        score, phase, significant = _score_candidate(beat_accents, numerator)
        if significant:
            denominator = resolve_denominator(numerator, ratio_family)
            candidates.append((numerator, denominator, score, phase))

    if not candidates:
        return 4, 4, 0.3

    candidates.sort(key=lambda c: -c[2])
    numerator, denominator, score, _phase = candidates[0]
    return numerator, denominator, score


def estimate_time_signature_with_phase(
        engine: AudioEngine,
        track: AudioTrack,
        beat_times_ms: Sequence[float],
        ratio_family: Optional[str] = None,
) -> Tuple[int, int, float, int]:
    """estimate_time_signature, plus the DOWNBEAT PHASE it already computes.

    _downbeat_salience returns which beat-in-bar carries the downbeat,
    _score_candidate passes it up, and estimate_time_signature unpacked it into
    `_phase` and threw it away - so the barline phase was never decided
    anywhere, and two downstream modules each invented their own
    (musicxml_exporter anchored bars to the earliest note in the score,
    piano_reduction._smooth_slots to absolute zero). Returns
    (numerator, denominator, confidence, downbeat_phase), where the phase is
    the index into beat_times_ms of the first beat that carries a downbeat.
    Phase 0 on the 4/4 fallback paths, which is the honest reading when there
    was not enough evidence to say otherwise (2026-08-17 audit)."""
    if len(beat_times_ms) < 8:
        return 4, 4, 0.3, 0

    beat_accents = sample_beat_accents(engine, track, beat_times_ms)

    candidates = []
    for numerator in CANDIDATE_NUMERATORS:
        if len(beat_accents) < numerator * 3:
            continue
        score, phase, significant = _score_candidate(beat_accents, numerator)
        if significant:
            denominator = resolve_denominator(numerator, ratio_family)
            candidates.append((numerator, denominator, score, phase))

    if not candidates:
        return 4, 4, 0.3, 0

    candidates.sort(key=lambda c: -c[2])
    numerator, denominator, score, phase = candidates[0]
    # Same burden of proof detect_pickup applies: a weak downbeat is noise in
    # the accent fold, not evidence of an anacrusis. Default to phase 0.
    if score < PICKUP_MIN_CONFIDENCE:
        phase = 0
    return numerator, denominator, score, phase


__all__ = [
    "build_phase_locked_grid", "estimate_time_signature",
    "estimate_time_signature_with_phase",
    "sample_beat_accents", "fft_meter_candidate", "resolve_denominator",
    "PickupResult", "detect_pickup",
]
