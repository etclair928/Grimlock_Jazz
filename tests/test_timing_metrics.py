# =================================================================
# MODULE: tests/test_timing_metrics.py
# Pins the threshold-free timing metrics (tools/score_vs_answer_key.py).
#
# WHY THEY NEEDED PINNING. These exist because a hard tolerance turned one
# transcription into two stories - recall 0.645 at tau=0.25s and 0.903 at
# tau=0.75s. A metric built to stop that happening is worth exactly as much as
# its own correctness, so the properties it claims are asserted here:
#   * scored over EVERY reference note, so a miss costs a real zero and no
#     threshold can quietly exclude it;
#   * decays smoothly, so there is no cliff for a conclusion to hide behind;
#   * reports delta/sigma, the ratio that decides whether calibrating the bias
#     is worth anything at all.
# =================================================================

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))

from score_vs_answer_key import gaussian_timing_score, timing_moments  # noqa: E402


def _ours(pairs):
    """(time, pitch) pairs -> the by_pitch index the scorer uses."""
    by_pitch = {}
    for i, (t, p) in enumerate(pairs):
        by_pitch.setdefault(p, []).append((t, i))
    return by_pitch


# --- the kernel -----------------------------------------------------------

def test_perfect_timing_scores_one():
    ref = [(1.0, 60), (2.0, 62), (3.0, 64)]
    score, errs = gaussian_timing_score(ref, _ours([(1.0, 60), (2.0, 62), (3.0, 64)]))
    assert abs(score - 1.0) < 1e-9
    assert all(e < 1e-9 for e in errs)


def test_a_missing_note_costs_a_real_zero():
    """The property that makes this a soft RECALL rather than a conditional
    average: a reference note with no candidate contributes 0, it is not
    silently dropped from the denominator."""
    ref = [(1.0, 60), (2.0, 62)]
    score, _e = gaussian_timing_score(ref, _ours([(1.0, 60)]))
    assert abs(score - 0.5) < 1e-9


def test_no_candidates_at_all_scores_zero():
    score, _e = gaussian_timing_score([(1.0, 60), (2.0, 62)], _ours([]))
    assert score == 0.0


def test_decay_is_smooth_and_matches_the_kernel():
    """No cliff: the score falls continuously with error, and by the stated
    formula rather than some other curve."""
    sigma = 0.100
    prev = 1.1
    for dt in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40):
        score, _e = gaussian_timing_score([(1.0, 60)], _ours([(1.0 + dt, 60)]),
                                          sigma=sigma)
        expected = math.exp(-(dt ** 2) / (2 * sigma ** 2))
        assert abs(score - expected) < 1e-9, (dt, score, expected)
        assert score < prev            # strictly decreasing, never a step
        prev = score


def test_each_candidate_is_used_once():
    """Two reference notes must not both claim the same emitted note - that
    would inflate the score exactly where we over-emit."""
    ref = [(1.0, 60), (1.05, 60)]
    score, errs = gaussian_timing_score(ref, _ours([(1.0, 60)]))
    assert len(errs) == 1
    assert score < 1.0


def test_pitch_must_match():
    score, _e = gaussian_timing_score([(1.0, 60)], _ours([(1.0, 61)]))
    assert score == 0.0


# --- the moments ----------------------------------------------------------

def test_moments_on_a_known_distribution():
    errs = [0.10, 0.10, -0.10, -0.10]      # zero mean, 100ms spread
    m = timing_moments(errs)
    assert abs(m["delta_ms"]) < 1e-6
    assert abs(m["sigma_ms"] - 100.0) < 1e-6
    assert abs(m["rmse_ms"] - 100.0) < 1e-6


def test_delta_over_sigma_separates_the_two_regimes():
    """The number that decides whether phase calibration is worth doing.
    Their worked example (delta 25ms, sigma 5ms) is bias-dominated; our
    measured Chopin distribution is not, and the ratio has to say so."""
    bias_dominated = timing_moments([0.025, 0.030, 0.020, 0.025])
    assert bias_dominated["delta_over_sigma"] > 1.0

    variance_dominated = timing_moments([-0.024, 0.300, -0.280, 0.150, -0.170])
    assert variance_dominated["delta_over_sigma"] < 1.0


def test_moments_survive_an_empty_input():
    m = timing_moments([])
    assert math.isnan(m["delta_ms"]) and math.isnan(m["sigma_ms"])


def test_rmse_combines_bias_and_spread():
    """RMSE^2 = delta^2 + sigma^2 - the identity that makes RMSE the honest
    single number when both terms matter."""
    errs = [0.05, 0.15, 0.05, 0.15]
    m = timing_moments(errs)
    lhs = m["rmse_ms"] ** 2
    rhs = m["delta_ms"] ** 2 + m["sigma_ms"] ** 2
    assert abs(lhs - rhs) < 1e-6
