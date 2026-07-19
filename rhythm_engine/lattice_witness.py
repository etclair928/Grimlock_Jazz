# =================================================================
# MODULE: rhythm_engine/lattice_witness.py
# Ports Symphony's ReverseGeoCrypt (agents/analysis/reverse_geo_crypt.py)
# - a genuinely different paradigm from the other tempo witnesses.
# Instead of beat-tracking, it takes arbitrary musical events (onset
# candidates here) and searches for the rational-ratio LATTICE that
# best explains their relative spacing: "don't ask what the tempo is,
# ask what geometric lattice explains these events - tempo is the
# OUTPUT of finding the lattice that minimizes structural entropy."
#
# Valuable as an independent witness for sparse/quiet passages where
# beat-tracking struggles - it votes on ratio relationships between
# events, not transient density, so it doesn't need a strong rhythmic
# pulse to work with.
#
# `Anchor` ("a confirmed structural downbeat... never mutated") is the
# phase-lock concept: when the caller has one (e.g. a high-confidence
# downbeat from meter.py), passing it in lets the lattice search prefer
# candidates whose grid actually lands on it.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde

from audio_engine import AudioEngine, AudioTrack
from core import Provenance
from rhythm_engine.onsets import detect_onset_candidates
from rhythm_engine.tempo_witness import TempoWitness

MIN_BPM = 20.0
MAX_BPM = 300.0
MIN_IOI_SEC = 0.05
MAX_IOI_SEC = 2.0
MIN_IOIS_FOR_ANALYSIS = 4
MIN_RATIOS_FOR_CLUSTERING = 5
RATIO_MATCH_TOLERANCE = 0.1
MIN_RATIO = 0.4
MAX_RATIO = 2.6
KDE_BANDWIDTH = 0.15
MIN_IOIS_FOR_SEARCH = 3
MIN_PERIOD_SEC = 0.05
MAX_CANDIDATES = 20
MULTIPLIERS_TO_TEST = (1, 2, 3, 4, 6, 8)
SYSTEMATIC_BPM_STEP = 0.5
PHASE_BINS = 20
MAX_ANCHORS_TO_CHECK = 10
ANCHOR_TOLERANCE_RATIO = 0.1

RATIO_FAMILY_IDEALS: Dict[str, List[float]] = {
    "binary": [1.0, 2.0, 0.5, 4.0, 0.25],
    "ternary": [1.3333, 0.6667, 0.3333, 2.6667],
    "swing": [1.5, 0.6667, 3.0],
    "quintuplet": [1.25, 0.8, 2.5],
    "septuplet": [1.1429, 0.875, 2.2857],
}
# Symmetric in both directions - the winning "subdivision" from
# search_lattice isn't reliably finer than the true tactus. Confirmed
# directly on real audio: the search's winner was a bar-length period
# (~2.76s), coarser than the true beat, so recovering the tactus needed
# dividing by ~6, not multiplying - the original multiply-only set
# (0.5, 1, 2, 3, 4, 6, 8) had no way to reach that at all.
TACTUS_MULTIPLIERS_TO_TEST = (
    1.0 / 8, 1.0 / 6, 1.0 / 4, 1.0 / 3, 0.5,
    1.0, 2.0, 3.0, 4.0, 6.0, 8.0,
)
TACTUS_ALIGNMENT_TOLERANCE_RATIO = 0.15
TACTUS_PRIOR_CENTER_BPM = 110.0
TACTUS_PRIOR_OCTAVE_STD = 0.9


@dataclass(frozen=True)
class Anchor:
    """A confirmed structural downbeat. Sacred - never mutated, only
    ever produced fresh by whatever detector confirmed it."""
    time_ms: float
    confidence: float


@dataclass(frozen=True)
class GeometricLattice:
    subdivision_ms: float
    ratio_family: str
    error_score: float       # 0=perfect alignment, 1=random
    entropy: float           # rubato/unpredictability, 0-1
    confidence: float
    derived_bpm: float       # BPM of the subdivision itself (can be fine-grained)
    phase: float             # 0-1, where the grid starts relative to t=0
    anchor_alignment: float
    candidate_periods_tested: int
    events_analyzed: int



def _compute_ratios(iois_sec: np.ndarray) -> List[float]:
    ratios = []
    eps = 1e-9
    for i in range(len(iois_sec) - 1):
        if iois_sec[i] > eps:
            ratios.append(iois_sec[i + 1] / iois_sec[i])
        if i < len(iois_sec) - 2 and iois_sec[i] > eps:
            ratios.append(iois_sec[i + 2] / iois_sec[i])
    return [r for r in ratios if MIN_RATIO < r < MAX_RATIO]


def find_ratio_clusters(iois_sec: np.ndarray) -> Dict[str, float]:
    """Finds recurring interval proportions via KDE peak-picking over
    IOI ratios, scored against each rhythm family's ideal ratios."""
    if len(iois_sec) < MIN_IOIS_FOR_ANALYSIS:
        return {family: 0.5 for family in RATIO_FAMILY_IDEALS}

    ratios = _compute_ratios(iois_sec)
    if len(ratios) < MIN_RATIOS_FOR_CLUSTERING:
        return {family: 0.5 for family in RATIO_FAMILY_IDEALS}

    try:
        kde = gaussian_kde(ratios, bw_method=KDE_BANDWIDTH)
        x_range = np.linspace(0.4, 2.6, 200)
        densities = kde(x_range)
        peaks_idx = np.where((densities[1:-1] > densities[:-2]) & (densities[1:-1] > densities[2:]))[0]
        peak_ratios = x_range[peaks_idx + 1]
        peak_heights = densities[peaks_idx + 1]

        scores = {family: 0.0 for family in RATIO_FAMILY_IDEALS}
        for ratio, height in zip(peak_ratios, peak_heights):
            for family, ideals in RATIO_FAMILY_IDEALS.items():
                if any(abs(ratio - ideal) < RATIO_MATCH_TOLERANCE for ideal in ideals):
                    scores[family] += height
                    break

        total = sum(scores.values())
        if total > 0:
            scores = {family: min(1.0, s / total) for family, s in scores.items()}
        return scores
    except Exception:
        return {family: 0.5 for family in RATIO_FAMILY_IDEALS}


def _alignment_error(onset_times_sec: np.ndarray, period: float) -> float:
    """0 = onsets land exactly on this period's grid, 1 = maximally off
    (half a period from the nearest grid line).

    Computed from absolute onset-TIME remainders modulo the period, NOT
    consecutive-IOI ratios. An IOI-ratio test degenerates for periods
    much larger than the typical event spacing: any small IOI divided
    by a large period rounds trivially toward 0, reporting a spuriously
    LOW error regardless of whether real structure exists at that
    scale - confirmed directly on synthetic pure-noise IOIs, where the
    old formula's "error" monotonically fell toward 0 as period grew,
    with zero real periodicity present. This is exactly what let this
    module's own systematic wide-range period scan (added to close a
    different blind spot) make the search's overall winner WORSE - it
    handed the old, scale-biased formula easy-looking large-period
    candidates it couldn't previously reach. Onset-time-modulo distance
    carries no such bias: verified flat (~0.5, no preference for any
    period) on the same pure-noise data across every scale tested."""
    if len(onset_times_sec) == 0 or period <= 0:
        return 1.0
    remainders = onset_times_sec % period
    distances = np.minimum(remainders, period - remainders)
    return float(np.clip(np.mean(distances) / (period / 2.0), 0.0, 1.0))


def _find_phase(times_sec: np.ndarray, period: float) -> float:
    if len(times_sec) < 2:
        return 0.0
    remainders = times_sec % period
    hist, bins = np.histogram(remainders, bins=PHASE_BINS)
    peak_bin = np.argmax(hist)
    phase = (bins[peak_bin] + bins[peak_bin + 1]) / 2
    return phase / period


def _anchor_alignment(anchors: List[Anchor], period: float, phase: float) -> float:
    if not anchors:
        return 0.5
    phase_offset = phase * period
    matches, total_weight = 0.0, 0.0
    for anchor in anchors[:MAX_ANCHORS_TO_CHECK]:
        anchor_time_sec = anchor.time_ms / 1000.0
        adjusted = anchor_time_sec - phase_offset
        nearest_int = round(adjusted / period)
        grid_time = phase_offset + nearest_int * period
        distance = abs(anchor_time_sec - grid_time)
        tolerance = period * ANCHOR_TOLERANCE_RATIO
        if distance < tolerance:
            matches += anchor.confidence * (1.0 - distance / tolerance)
        total_weight += anchor.confidence
    return min(1.0, matches / total_weight) if total_weight > 0 else 0.5


def _entropy(iois_sec: np.ndarray, period: float) -> float:
    if len(iois_sec) == 0 or period <= 0:
        return 0.5
    normalized = (iois_sec % period) / period
    return min(1.0, float(np.std(normalized)) * 2) if len(normalized) > 0 else 0.5


def _systematic_bpm_candidates() -> np.ndarray:
    """Direct period candidates spanning the whole valid tempo range,
    independent of what onset spacings happen to be observed.

    The IOI-derived candidates below are real events, not noise - but
    they're a discrete, data-driven set: a real periodicity that no two
    onsets happen to be (near-)exactly one period apart at is simply
    absent from that set and can never win, no matter how well it would
    have scored. Confirmed directly on real audio: a verified-correct
    tempo (from an external reference) scored competitively with the
    search's actual winner when evaluated directly, but was never once
    tested, because it wasn't within tolerance of any observed IOI or
    its small-integer multiples. This closes that blind spot."""
    bpms = np.arange(MIN_BPM, MAX_BPM, SYSTEMATIC_BPM_STEP)
    return np.round(60.0 / bpms, 4)


def _tactus_alignment_score(onset_times_sec: np.ndarray, period_sec: float, phase_sec: float) -> float:
    """Fraction of real onsets landing near a predicted beat at this
    candidate tactus period/phase - a direct test of "does THIS level
    explain the felt beat," not the fine subdivision's own structural-
    entropy fit to itself (a different question: does the fine grid
    explain the fine events)."""
    if period_sec <= 0 or len(onset_times_sec) == 0:
        return 0.0
    tolerance = period_sec * TACTUS_ALIGNMENT_TOLERANCE_RATIO
    remainders = (onset_times_sec - phase_sec) % period_sec
    distances = np.minimum(remainders, period_sec - remainders)
    return float(np.mean(distances < tolerance))


def resolve_tactus_bpm(lattice: GeometricLattice, onset_times_sec: np.ndarray) -> Tuple[float, float]:
    """Chooses the felt-beat (tactus) level from the winning subdivision
    by directly scoring how well each candidate multiple's own predicted
    beat grid aligns with the real onsets - never a fixed per-ratio-
    family divisor.

    A "best-fit subdivision" only answers "what's the fastest common
    note value that explains these events" - the tactus can legitimately
    be that subdivision itself, or several of its slower multiples
    (tactus selection is a real, hard, well-known beat-tracking problem:
    Dixon's BeatRoot and the Tactus Hypothesis Tracker both solve it by
    scoring multiple beat-level hypotheses against the actual onsets,
    not by a fixed rule). A constant divisor keyed only on ratio_family
    can't know which level a given piece actually uses - confirmed
    directly on real audio: the winning subdivision (107 BPM) folded to
    26.8 BPM via the old "binary family -> divide by 4" rule, while an
    externally-verified reference for the same track measured 145 BPM -
    not reachable by dividing 107 by any of the fixed per-family
    divisors this replaces.

    Returns (tactus_bpm, alignment_score) so callers can see how
    confidently the winning hypothesis actually explains the onsets,
    not just that some hypothesis was mechanically produced.
    """
    base_period_sec = lattice.subdivision_ms / 1000.0
    phase_sec = lattice.phase * base_period_sec

    best_period, best_combined, best_alignment = base_period_sec, -1.0, 0.0
    for multiplier in TACTUS_MULTIPLIERS_TO_TEST:
        period = base_period_sec * multiplier
        bpm = 60.0 / period if period > 0 else 0.0
        if not (MIN_BPM <= bpm <= MAX_BPM):
            continue

        alignment = _tactus_alignment_score(onset_times_sec, period, phase_sec)
        # Mild log-normal tempo prior (same convention octave_correction.py
        # uses) breaks ties between hypotheses that explain the onsets
        # almost equally well - a real ambiguity in some material, not
        # something alignment score alone always resolves.
        octaves_from_center = np.log2(bpm / TACTUS_PRIOR_CENTER_BPM) if bpm > 0 else 0.0
        prior = float(np.exp(-0.5 * (octaves_from_center / TACTUS_PRIOR_OCTAVE_STD) ** 2))
        combined = alignment * 0.8 + prior * 0.2

        if combined > best_combined:
            best_combined = combined
            best_period = period
            best_alignment = alignment

    tactus_bpm = 60.0 / best_period if best_period > 0 else 0.0
    return tactus_bpm, best_alignment


def search_lattice(
        iois_sec: np.ndarray,
        onset_times_sec: np.ndarray,
        ratio_scores: Dict[str, float],
        anchors: List[Anchor],
) -> Optional[GeometricLattice]:
    """Searches for the subdivision that minimizes structural entropy -
    the core ReverseGeoCrypt algorithm."""
    if len(iois_sec) < MIN_IOIS_FOR_SEARCH:
        return None

    best_lattice, best_score = None, -np.inf
    ioi_candidates = np.unique(np.round(iois_sec, 4))
    ioi_candidates = ioi_candidates[ioi_candidates > MIN_PERIOD_SEC][:MAX_CANDIDATES]
    candidates = np.unique(np.concatenate([ioi_candidates, _systematic_bpm_candidates()]))
    candidates = candidates[candidates > MIN_PERIOD_SEC]
    periods_tested = 0

    for base_period in candidates:
        for multiplier in MULTIPLIERS_TO_TEST:
            period = base_period * multiplier
            bpm = 60.0 / period
            if not (MIN_BPM <= bpm <= MAX_BPM):
                continue
            periods_tested += 1

            error = _alignment_error(onset_times_sec, period)
            phase = _find_phase(onset_times_sec, period)
            alignment = _anchor_alignment(anchors, period, phase)
            entropy = _entropy(iois_sec, period)
            ratio_family = max(ratio_scores, key=ratio_scores.get)
            best_ratio_score = max(ratio_scores.values())

            confidence = min(1.0, max(0.0,
                (1.0 - error) * 0.3 + alignment * 0.3 + (1.0 - entropy) * 0.2 + best_ratio_score * 0.2))
            score = (1.0 - error) * 0.4 + alignment * 0.4 + (1.0 - entropy) * 0.1 + best_ratio_score * 0.1

            if score > best_score:
                best_score = score
                best_lattice = GeometricLattice(
                    subdivision_ms=period * 1000.0, ratio_family=ratio_family,
                    error_score=error, entropy=entropy, confidence=confidence,
                    derived_bpm=bpm, phase=phase, anchor_alignment=alignment,
                    candidate_periods_tested=periods_tested, events_analyzed=len(onset_times_sec),
                )

    return best_lattice


def run_lattice_witness(
        engine: AudioEngine,
        track: AudioTrack,
        anchors: Optional[List[Anchor]] = None,
) -> Tuple[Optional[TempoWitness], Optional[GeometricLattice]]:
    """Runs the full event-extraction -> ratio-cluster -> lattice-search
    pipeline against `track`'s onset candidates (from the shared
    onsets.py witnesses, not stem-specific audio). Returns (witness,
    lattice) - witness is None when there's too little event data to
    say anything real, matching decrypt()'s own graceful degradation."""
    anchors = anchors or []
    candidates = detect_onset_candidates(engine, track)
    onset_times_sec = np.array(sorted(t / 1000.0 for t in candidates.combined_ms))

    if len(onset_times_sec) < MIN_IOIS_FOR_ANALYSIS:
        return None, None

    iois_sec = np.diff(onset_times_sec)
    iois_sec = iois_sec[(iois_sec > MIN_IOI_SEC) & (iois_sec < MAX_IOI_SEC)]
    if len(iois_sec) < MIN_IOIS_FOR_SEARCH:
        return None, None

    ratio_scores = find_ratio_clusters(iois_sec)
    lattice = search_lattice(iois_sec, onset_times_sec, ratio_scores, anchors)
    if lattice is None:
        return None, None

    tactus_bpm, tactus_alignment = resolve_tactus_bpm(lattice, onset_times_sec)
    if tactus_bpm <= 0:
        return None, lattice

    period_ms = 60000.0 / tactus_bpm
    phase_ms = lattice.phase * lattice.subdivision_ms
    span_ms = (onset_times_sec[-1] - onset_times_sec[0]) * 1000.0
    n_beats = max(2, int(span_ms / period_ms) + 1)
    beat_times_ms = tuple(float(phase_ms + i * period_ms) for i in range(n_beats))

    # Blend the subdivision's own fit (lattice.confidence) with how well
    # THIS tactus hypothesis specifically aligns to the real onsets - a
    # well-fit subdivision can still have its tactus level chosen badly;
    # the witness's confidence should reflect both questions, not just
    # the first one.
    confidence = float(np.clip(lattice.confidence * 0.5 + tactus_alignment * 0.5, 0.0, 1.0))

    witness = TempoWitness(
        tempo_bpm=tactus_bpm,
        confidence=confidence,
        beat_times_ms=beat_times_ms,
        source=Provenance.RHYTHM_ENGINE,
        weight=0.6,
    )
    return witness, lattice


__all__ = [
    "Anchor", "GeometricLattice", "find_ratio_clusters", "search_lattice",
    "resolve_tactus_bpm", "run_lattice_witness",
]
