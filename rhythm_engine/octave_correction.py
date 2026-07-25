# =================================================================
# MODULE: rhythm_engine/octave_correction.py
# Ports Symphony's TempoOctaveCorrector (agents/analysis/
# tempo_intelligence.py) - tests whether a tempo estimate is off by a
# musically-common rational factor (the classic "tempo octave error":
# a beat tracker reports double/half/triplet/dotted-etc. of the tempo a
# listener actually feels).
#
# Uses onset-envelope autocorrelation (always available here via Audio
# Engine's cached transforms) rather than the onset-to-grid-alignment
# fallback Symphony's version needed when no envelope was passed - per
# that module's own analysis, autocorrelation is the more robust test
# for swung/syncopated material, so this port only implements that path.
#
# Two things make this more than "does another value correlate too":
# 1. SUBHARMONIC-ALIAS PENALTY - raw autocorrelation can't distinguish a
#    true tempo from a too-slow reading, because a clean pulse train's
#    autocorrelation peaks at its own period AND every multiple of it.
#    Penalizing each candidate by its strength at double/triple its own
#    rate (where a true fundamental has none, but a slow alias always
#    does - that period IS the real one) breaks the tie a naive
#    comparison cannot.
# 2. TEMPO PRIOR - a mild log-normal preference for musically typical
#    tempos (centered 110 BPM), which is what disambiguates the
#    remaining case autocorrelation genuinely can't: a tactus vs. its
#    own subdivision, both of which are truly periodic.
# 3. COMPOUND-TACTUS OVERRIDE - the prior in (2) still mis-picks when the
#    real tactus sits BELOW ~110 BPM and its double sits nearer the prior
#    centre (a 68 BPM triplet-feel tune read as 136). For that specific
#    case, a stronger piece of evidence decides: a genuine tactus divides
#    into BOTH 2 and 3 populated sub-pulses (duple + triplet feel), while a
#    subdivision-level misread has neither. When a slower related candidate
#    shows that compound signature clearly, it overrides (2). Straight
#    material has ~zero triple salience, so this only fires on real
#    compound/shuffle feels - it is not a blanket "prefer slower" rule.
# =================================================================

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# (numerator, denominator, label, description)
RATIOS: List[Tuple[int, int, str, str]] = [
    (1, 1, "exact", "no correction"),
    (1, 2, "1/2", "half tempo"),
    (2, 1, "2x", "double tempo"),
    (1, 3, "1/3", "compound meter (12/8, 6/4)"),
    (3, 1, "3x", "triple tempo"),
    (2, 3, "2/3", "triplet pair"),
    (3, 2, "3/2", "dotted quarter"),
    (3, 4, "3/4", "dotted eighth"),
    (4, 3, "4/3", "inverse dotted"),
    (5, 6, "5/6", "compound correction"),
    (4, 5, "4/5", "quintuplet expansion"),
    (5, 4, "5/4", "quintuplet compression"),
]

SUBHARMONIC_PENALTY_WEIGHT = 0.3
TEMPO_PRIOR_CENTER_BPM = 110.0
TEMPO_PRIOR_OCTAVE_STD = 0.9
MIN_TEMPO_BPM = 20.0
MAX_TEMPO_BPM = 300.0
DEFAULT_MIN_IMPROVEMENT = 0.15

# Compound-tactus override (see correct()'s tail and _compound_coherence).
# The floor is the minimum duple+triple subdivision coherence a slower
# candidate must show to be trusted as the real (compound/triplet-feel)
# tactus; the min-prior keeps the override inside the tappable range so a
# too-slow measure-level (whose x3 also hits a strong pulse) can't win it;
# the margin is how much cleaner the slower candidate's compound signature
# must be than the generically-chosen tempo's. Tuned on real audio: a 68 BPM
# triplet-feel gospel mash-up (coherence 0.11 at 68, ~0 at the 136 misread)
# corrects; straight material (triple salience ~0) never trips it.
COMPOUND_COHERENCE_FLOOR = 0.05
COMPOUND_TACTUS_MIN_PRIOR = 0.4
COMPOUND_COHERENCE_MARGIN = 0.04


def _tempo_prior(bpm: float) -> float:
    if bpm <= 0:
        return 0.0
    octaves_from_center = np.log2(bpm / TEMPO_PRIOR_CENTER_BPM)
    return float(np.exp(-0.5 * (octaves_from_center / TEMPO_PRIOR_OCTAVE_STD) ** 2))


def _autocorrelation_score(candidate_bpm: float, autocorr: np.ndarray, hop_sec: float) -> float:
    """Normalized autocorrelation strength at the candidate's period,
    linearly interpolated between frames."""
    if candidate_bpm <= 0 or hop_sec <= 0:
        return 0.0
    period_frames = (60.0 / candidate_bpm) / hop_sec
    lo = int(np.floor(period_frames))
    hi = lo + 1
    if lo < 1 or hi >= len(autocorr):
        return 0.0
    frac = period_frames - lo
    return float(max(0.0, autocorr[lo] * (1 - frac) + autocorr[hi] * frac))


def _subdivision_salience(candidate_bpm: float, div: int, autocorr: np.ndarray, hop_sec: float) -> float:
    """Contrast strength of the candidate beat's 1/`div` subdivision: the
    autocorrelation at that faster sub-lag minus a nearby off-grid baseline
    (the mean at div+/-0.5), so it measures a real, phase-locked sub-pulse
    rather than broadband correlation. >0 only when the beat genuinely
    divides into `div` populated parts."""
    sub = _autocorrelation_score(candidate_bpm * div, autocorr, hop_sec)
    base = 0.5 * (
        _autocorrelation_score(candidate_bpm * (div + 0.5), autocorr, hop_sec)
        + _autocorrelation_score(candidate_bpm * (div - 0.5), autocorr, hop_sec)
    )
    return max(0.0, sub - base)


def _compound_coherence(candidate_bpm: float, autocorr: np.ndarray, hop_sec: float) -> float:
    """min(duple salience, triple salience): high ONLY when the beat divides
    cleanly into BOTH 2 and 3 populated sub-pulses - the acoustic signature
    of a compound/triplet-feel tactus. A subdivision-level misread has
    neither strong; a too-slow measure-level has only the triple (its x3
    lands on the real beat); a straight tune has ~zero triple. Requiring
    both is what makes this a targeted compound-feel detector instead of a
    blanket 'halve the tempo' rule."""
    return min(
        _subdivision_salience(candidate_bpm, 2, autocorr, hop_sec),
        _subdivision_salience(candidate_bpm, 3, autocorr, hop_sec),
    )


def correct(
        raw_tempo_bpm: float,
        onset_env: np.ndarray,
        onset_env_sr: int,
        onset_env_hop_length: int,
        anchor_confidence: Optional[float] = None,
        min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
) -> Tuple[float, str]:
    """Returns (corrected_tempo_bpm, reason). `reason` is
    "no_correction_needed" when raw_tempo_bpm is kept as-is."""
    if raw_tempo_bpm <= 0 or len(onset_env) < 8:
        return raw_tempo_bpm, "insufficient_evidence"

    hop_sec = onset_env_hop_length / onset_env_sr
    raw_autocorr = np.correlate(onset_env, onset_env, mode="full")
    raw_autocorr = raw_autocorr[len(raw_autocorr) // 2:]
    peak = raw_autocorr[0]
    autocorr = raw_autocorr / peak if peak > 0 else raw_autocorr

    candidates = []
    for n, d, label, desc in RATIOS:
        candidate_bpm = raw_tempo_bpm * n / d
        if not (MIN_TEMPO_BPM <= candidate_bpm <= MAX_TEMPO_BPM):
            continue

        score = _autocorrelation_score(candidate_bpm, autocorr, hop_sec)
        sub_strength = max(
            _autocorrelation_score(candidate_bpm * 2, autocorr, hop_sec),
            _autocorrelation_score(candidate_bpm * 3, autocorr, hop_sec),
        )
        effective = max(0.0, score - SUBHARMONIC_PENALTY_WEIGHT * sub_strength)
        effective *= _tempo_prior(candidate_bpm)

        candidates.append({"bpm": candidate_bpm, "label": label, "desc": desc,
                            "score": score, "effective_score": effective})

    if not candidates:
        return raw_tempo_bpm, "no_candidates"

    candidates.sort(key=lambda c: -c["effective_score"])
    exact = next((c for c in candidates if c["label"] == "exact"), candidates[0])
    best = candidates[0]

    required_margin = min_improvement
    if anchor_confidence is not None:
        required_margin = min_improvement * (1.0 + max(0.0, min(1.0, anchor_confidence)))

    if best["label"] == "exact" or (best["effective_score"] - exact["effective_score"]) < required_margin:
        result_bpm, result_reason = raw_tempo_bpm, "no_correction_needed"
    else:
        result_bpm, result_reason = best["bpm"], f"{best['label']} ({best['desc']})"

    # Compound-tactus override. The one case the scoring above genuinely
    # cannot resolve is a true tactus vs. its own subdivision: both are
    # periodic, so autocorrelation ties, and the 110 BPM prior then mis-picks
    # the FASTER level whenever the real tactus sits below the prior centre
    # (a 68 BPM triplet-feel tune read as 136 - the prior scores 136 higher).
    # The subharmonic penalty makes it worse here: it penalises the true slow
    # tactus precisely because its double is strong. The tell that the slower
    # level is the real beat is a clean COMPOUND subdivision - the beat divides
    # into BOTH 2 and 3 populated sub-pulses (duple + triplet feel). When a
    # slower related candidate shows that signature above a floor, within a
    # tappable range (prior gate excludes too-slow measure-levels), and clearly
    # more than the generically-chosen tempo does, it IS the compound tactus.
    # Straight material has ~zero triple salience, so this never fires on it.
    result_coherence = _compound_coherence(result_bpm, autocorr, hop_sec)
    compound_best = None
    for c in candidates:
        if c["bpm"] >= result_bpm:
            continue  # only a SLOWER related level can be the compound tactus
        coherence = _compound_coherence(c["bpm"], autocorr, hop_sec)
        if (coherence >= COMPOUND_COHERENCE_FLOOR
                and _tempo_prior(c["bpm"]) >= COMPOUND_TACTUS_MIN_PRIOR
                and coherence >= result_coherence + COMPOUND_COHERENCE_MARGIN):
            if compound_best is None or coherence > compound_best[1]:
                compound_best = (c["bpm"], coherence)
    if compound_best is not None:
        return compound_best[0], (
            f"compound_tactus (duple+triple subdivision, coherence={compound_best[1]:.2f})"
        )

    return result_bpm, result_reason


__all__ = ["correct", "RATIOS"]
