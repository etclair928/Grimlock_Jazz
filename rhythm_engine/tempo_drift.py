# =================================================================
# MODULE: rhythm_engine/tempo_drift.py
# Detects and CLASSIFIES how tempo moves over time - the thing a single
# scalar tempo (and a metronomic guided grid) cannot represent:
# accelerando, ritardando, a discrete section change, or a shift into
# double-time feel.
#
# THE ONE LESSON THAT SHAPES THIS MODULE (learned on Hopeful, 2026-07-20):
# a naive local-tempo curve HALLUCINATES drift. librosa.feature.tempo and
# any beat tracker pick a discrete autocorrelation LAG. At hop 512 /
# 22050 Hz the lags near 145 BPM are integers 20,19,18,17,16 -> 129, 136,
# 144, 152, 161 BPM, each ~5.5% apart. As a song's texture thickens the
# peak hops one lag over, and a steady 145 song reads as a smooth 30-BPM
# "accelerando" that is pure quantization. The user's ear caught it when
# two independent methods did not.
#
# So this module does NOT pick lag bins. It interpolates the
# autocorrelation peak PARABOLICALLY for sub-bin (continuous) tempo, then
# octave-folds, then demands the residual trend be SMOOTH and SUSTAINED
# before it will call anything drift. Burden of proof is on "the tempo
# moved," never on "the tempo held" - a steady song must read steady.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from audio_engine import AudioEngine, AudioTrack

DRIFT_SAMPLE_RATE = 22050
DRIFT_HOP_LENGTH = 512

# Windowed local-tempo analysis. A window must be long enough to hold
# several bars (so autocorrelation has a real peak) but short enough to
# localize a change.
WINDOW_S = 12.0
HOP_S = 3.0
MIN_LAG_BPM = 60.0
MAX_LAG_BPM = 220.0

# Classification thresholds - deliberately conservative (see module note).
STEADY_TOLERANCE_FRAC = 0.06      # within +/-6% of the reference over the whole song = STEADY
MONOTONIC_MIN_FRAC = 0.85         # this fraction of steps moving one direction = a real trend
STEP_MIN_FRAC = 0.10              # a section change must jump at least this fraction of the reference
DOUBLE_TIME_RATIO_LO = 1.8        # a fold event near 2x (or 0.5x) is metric, not drift
DOUBLE_TIME_RATIO_HI = 2.2
# A real double-time PASSAGE is a sustained, contiguous run of 2x readings -
# not scattered single windows, which are autocorrelation octave ambiguity
# (the same failure family as adjacent-lag drift). Require a contiguous run
# of at least this many windows before calling double-time.
DOUBLE_TIME_MIN_RUN = 4
SMOOTHING_WINDOW = 5              # windows; kills residual per-window jitter before classifying


class DriftKind(str, Enum):
    STEADY = "steady"
    ACCELERANDO = "accelerando"
    RITARDANDO = "ritardando"
    STEP_CHANGE = "step_change"
    DOUBLE_TIME = "double_time"


@dataclass(frozen=True)
class TempoDriftResult:
    kind: DriftKind
    reference_bpm: float                      # the octave the song is folded to
    times_s: Tuple[float, ...] = field(default_factory=tuple)
    bpm_curve: Tuple[float, ...] = field(default_factory=tuple)  # octave-folded, smoothed
    span_bpm: float = 0.0                      # max-min of the folded curve
    confidence: float = 0.0
    detail: str = ""


# How far the ANCHORED tempo search may stray from the reference before it
# would risk locking onto a different metrical level. A third either way
# comfortably covers any real ritard/accel while making a half/double jump
# impossible - that jump is then handled by the explicit double-time probe.
_ANCHOR_BAND_LO = 0.75
_ANCHOR_BAND_HI = 1.33


def _autocorr(onset_env: np.ndarray) -> Optional[np.ndarray]:
    if onset_env.size < 8:
        return None
    env = onset_env - onset_env.mean()
    ac = np.correlate(env, env, mode="full")[env.size - 1:]
    if ac.size < 4 or ac[0] <= 0:
        return None
    return ac / ac[0]


def _peak_tempo_in_band(ac: np.ndarray, sr: int, hop: int,
                        bpm_lo: float, bpm_hi: float) -> Optional[Tuple[float, float]]:
    """Sub-bin tempo of the strongest autocorrelation peak WITHIN a BPM
    band, plus that peak's strength. Parabolic interpolation removes the
    integer-lag quantization that fabricates drift; the band constraint
    stops the peak jumping to a different metrical level (the thing that
    made a sparse section read half-time)."""
    lag_hi = int(60.0 * sr / (hop * bpm_lo))
    lag_lo = max(2, int(60.0 * sr / (hop * bpm_hi)))
    lag_hi = min(lag_hi, ac.size - 2)
    if lag_hi <= lag_lo:
        return None
    band = ac[lag_lo:lag_hi]
    peak = int(np.argmax(band)) + lag_lo
    if peak <= 1 or peak >= ac.size - 1:
        return None
    y0, y1, y2 = ac[peak - 1], ac[peak], ac[peak + 1]
    denom = (y0 - 2 * y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    frac_lag = peak + float(np.clip(delta, -1.0, 1.0))
    if frac_lag <= 0:
        return None
    return 60.0 * sr / (hop * frac_lag), float(y1)


def _octave_fold(bpm: float, reference: float) -> float:
    """Folds a tempo onto the reference's octave, so a double-time or
    half-time READING doesn't masquerade as drift. Tries the musically
    real ratios, keeps whichever lands closest to the reference."""
    if bpm <= 0 or reference <= 0:
        return bpm
    best = bpm
    for r in (1.0, 2.0, 0.5, 1.5, 1.0 / 1.5, 3.0, 1.0 / 3.0):
        cand = bpm * r
        if abs(cand - reference) < abs(best - reference):
            best = cand
    return best


def local_tempo_curve(
        engine: AudioEngine, track: AudioTrack,
        reference_bpm: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    """Returns (times_s, sub_bin_bpm). When `reference_bpm` is given the
    per-window search is ANCHORED to its octave (a sparse section reads a
    weak ~reference, never a phantom half/double). Unanchored, it searches
    the full tempo band - only for exploratory use."""
    onset_env = np.asarray(engine.onset_envelope(track, DRIFT_SAMPLE_RATE, hop_length=DRIFT_HOP_LENGTH),
                           dtype=np.float64)
    fps = DRIFT_SAMPLE_RATE / DRIFT_HOP_LENGTH
    win_f = int(WINDOW_S * fps)
    hop_f = int(HOP_S * fps)
    if reference_bpm is not None:
        lo, hi = reference_bpm * _ANCHOR_BAND_LO, reference_bpm * _ANCHOR_BAND_HI
    else:
        lo, hi = MIN_LAG_BPM, MAX_LAG_BPM
    times: List[float] = []
    bpms: List[float] = []
    for f0 in range(0, max(1, onset_env.size - win_f), hop_f):
        ac = _autocorr(onset_env[f0:f0 + win_f])
        if ac is None:
            continue
        got = _peak_tempo_in_band(ac, DRIFT_SAMPLE_RATE, DRIFT_HOP_LENGTH, lo, hi)
        if got is not None:
            times.append((f0 + win_f / 2) / fps)
            bpms.append(got[0])
    return times, bpms


def _double_time_runs(engine: AudioEngine, track: AudioTrack, reference_bpm: float) -> Tuple[int, float]:
    """Double-time is where the FELT beat halves/doubles for a sustained
    passage. It CANNOT be detected from autocorrelation peak strength:
    the autocorrelation of any periodic signal is naturally strong at
    every subharmonic (a steady song repeats every 2 beats too), so the
    half-time peak looks 'dominant' for a perfectly steady song. Testing
    that way fabricated a double-time call on steady Hopeful - the exact
    subharmonic-alias trap TempoOctaveCorrector carries a penalty for.

    Real double-time detection needs evidence autocorrelation cannot fake:
    a sustained change in ONSET DENSITY (double-time is genuinely busier;
    half-time genuinely sparser) and/or which metrical level the DRUMS
    articulate. That is deferred until this reads the drum stem's hits, not
    the mix's autocorrelation. Until then this returns 'no run' rather than
    cry double-time on steady material.
    """
    return 0, 0.0


def classify_drift(
        engine: AudioEngine,
        track: AudioTrack,
        reference_bpm: float,
) -> TempoDriftResult:
    """Anchored sub-bin local tempo -> smooth -> classify. Conservative by
    construction: STEADY unless movement is smooth and sustained WITHIN the
    reference octave, or the reference pulse genuinely collapses into a
    sustained double-time run."""
    times, raw = local_tempo_curve(engine, track, reference_bpm=reference_bpm)
    if len(raw) < 4:
        return TempoDriftResult(kind=DriftKind.STEADY, reference_bpm=reference_bpm,
                                confidence=0.0, detail="too little data")

    # The anchored curve cannot octave-jump, so it needs no folding.
    folded = np.asarray(raw)
    if folded.size >= SMOOTHING_WINDOW:
        kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
        sm = np.convolve(folded, kernel, mode="valid")
        sm_times = times[SMOOTHING_WINDOW // 2: SMOOTHING_WINDOW // 2 + sm.size]
    else:
        sm = folded
        sm_times = times

    span = float(sm.max() - sm.min())
    span_frac = span / max(reference_bpm, 1.0)
    result_curve = (tuple(sm_times), tuple(round(float(x), 1) for x in sm))

    # DOUBLE-TIME: only when the reference pulse genuinely collapses for a
    # sustained contiguous run (explicit probe, not "a 2x peak exists").
    dt_run, dt_frac = _double_time_runs(engine, track, reference_bpm)
    if dt_run >= DOUBLE_TIME_MIN_RUN:
        return TempoDriftResult(
            kind=DriftKind.DOUBLE_TIME, reference_bpm=reference_bpm,
            times_s=result_curve[0], bpm_curve=result_curve[1],
            span_bpm=span, confidence=min(0.9, 0.4 + dt_frac),
            detail=f"reference pulse collapses for a contiguous run of {dt_run} windows "
                   f"({dt_frac*100:.0f}% total) - genuine double/half-time feel",
        )

    # STEADY: the folded curve barely moves.
    if span_frac <= STEADY_TOLERANCE_FRAC:
        return TempoDriftResult(
            kind=DriftKind.STEADY, reference_bpm=reference_bpm,
            times_s=result_curve[0], bpm_curve=result_curve[1],
            span_bpm=span, confidence=0.9,
            detail=f"folded curve spans {span:.1f} BPM ({span_frac*100:.1f}%) - within steady tolerance",
        )

    # Movement is real. Is it a smooth trend, or a discrete step?
    diffs = np.diff(sm)
    up = float(np.mean(diffs > 0))
    down = float(np.mean(diffs < 0))
    monotonic = max(up, down)

    # STEP: one dominant jump much larger than the rest.
    biggest = float(np.max(np.abs(diffs))) if diffs.size else 0.0
    if biggest >= STEP_MIN_FRAC * reference_bpm and biggest > 2.5 * float(np.median(np.abs(diffs)) + 1e-9):
        return TempoDriftResult(
            kind=DriftKind.STEP_CHANGE, reference_bpm=reference_bpm,
            times_s=result_curve[0], bpm_curve=result_curve[1],
            span_bpm=span, confidence=0.7,
            detail=f"discrete jump of {biggest:.1f} BPM against small surrounding changes",
        )

    # TREND: sustained one-directional movement.
    if monotonic >= MONOTONIC_MIN_FRAC:
        kind = DriftKind.ACCELERANDO if up > down else DriftKind.RITARDANDO
        return TempoDriftResult(
            kind=kind, reference_bpm=reference_bpm,
            times_s=result_curve[0], bpm_curve=result_curve[1],
            span_bpm=span, confidence=min(0.85, monotonic),
            detail=f"{monotonic*100:.0f}% of steps move one way; {span:.1f} BPM over the span",
        )

    # Moves, but neither smooth nor a clean step - untrustworthy. Report
    # STEADY rather than invent a drift shape from noise.
    return TempoDriftResult(
        kind=DriftKind.STEADY, reference_bpm=reference_bpm,
        times_s=result_curve[0], bpm_curve=result_curve[1],
        span_bpm=span, confidence=0.5,
        detail=f"folded curve spans {span:.1f} BPM but neither monotonic nor a clean step - not called drift",
    )


__all__ = [
    "DriftKind",
    "TempoDriftResult",
    "classify_drift",
    "local_tempo_curve",
]
