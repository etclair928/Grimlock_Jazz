#!/usr/bin/env python3
# =================================================================
# TOOL: tools/validate_musical_time.py
# PHASE 0. Is our beat curve RIGHT, or merely non-flat?
#
# The whole rhythm redesign rests on one assumption: that the tracked beat
# times are a usable map between performed time and musical time. We have
# established they are not a straight line (222 beats, CV 0.168, local tempo
# 53-133 BPM on the Chopin excerpt). We have NOT established they are correct.
#
# That distinction decides everything. A rigid grid fails DIFFUSELY - notes land
# near the wrong place everywhere. A locally wrong tempo map fails CONFIDENTLY -
# notes land squarely in the wrong bar. The second is worse, and it is the
# failure mode to check for before building on top of it.
#
# HOW WE GET GROUND TRUTH. A published edition plus a recording of it gives the
# true mapping: the score-to-performance alignment IS musical time. Align the
# edition's chroma to the transcription's, and the DTW path evaluated at each
# integer quarter tells us when that beat actually happened. Then ask how far
# our tracked beats sit from those instants.
#
# WHAT IS COMPARED. Raw madmom beats, straight from the tracker, BEFORE
# octave correction and before _re_anchor_beat_grid flattens them - because that
# is the signal the redesign proposes to preserve.
#
# The metric that matters is error in BEATS, not seconds: half a beat of error
# puts a note on the wrong side of a subdivision, which is what actually breaks
# a page, and it is comparable across tempi.
#
#   python tools/validate_musical_time.py <ours.musicxml> <answer_key.mxl> <audio.wav>
# =================================================================

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)
sys.path.insert(0, os.path.join(JAZZ, "tools"))


def tracked_beats(audio_path: str, cache: str) -> np.ndarray:
    """Raw madmom beat instants, in seconds. Cached - the RNN is slow."""
    if os.path.exists(cache):
        return np.load(cache)["beats"]
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    print("[madmom] tracking beats (slow, cached afterwards) ...", flush=True)
    act = RNNBeatProcessor()(audio_path)
    beats = DBNBeatTrackingProcessor(fps=100)(act)
    beats = np.asarray(beats, dtype=float)
    np.savez_compressed(cache, beats=beats)
    return beats


def true_beat_times(ours_xml: str, key_path: str, steps: int):
    """Ground-truth mapping: musical quarter -> performed second, from the
    score/performance alignment. Returns (quarters, seconds)."""
    from score_vs_answer_key import load_answer_key, load_ours, chroma_seq, dtw_path
    key, endq = load_answer_key(key_path)
    ours = load_ours(ours_xml)
    ours_end = max(e for _s, e, _p in ours)
    A = chroma_seq(key, endq, steps)
    B = chroma_seq(ours, ours_end, steps)
    path, cost = dtw_path(A, B, open_end=True)
    kq = np.array([i for i, _j in path], float) / steps * endq
    osec = np.array([j for _i, j in path], float) / steps * ours_end
    # collapse to a single-valued function
    uq = np.unique(kq)
    f = np.array([osec[kq == q].mean() for q in uq])
    return uq, f, cost, ours_end


def _monotonic_errors(true_t: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """One-to-one, order-preserving assignment of candidates to true beats.

    Plain nearest-neighbour matching is invalid here: it lets one candidate
    serve many targets and rewards whichever hypothesis simply has MORE
    candidates (measured - doubling the candidate count halved the reported
    error, which is exactly the s/4 you expect from chance at that density).
    A real alignment is monotonic and one-to-one, so score that instead."""
    n, m = len(true_t), len(cand)
    if n == 0 or m == 0:
        return np.array([])
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    D[0, :] = 0.0                      # candidates may be skipped freely
    for i in range(1, n + 1):
        cost = np.abs(cand - true_t[i - 1])
        for j in range(1, m + 1):
            take = D[i - 1, j - 1] + cost[j - 1]
            skip = D[i, j - 1]
            D[i, j] = take if take < skip else skip
    errs, i, j = [], n, m
    while i > 0 and j > 0:
        take = D[i - 1, j - 1] + abs(cand[j - 1] - true_t[i - 1])
        if take <= D[i, j - 1]:
            errs.append(abs(cand[j - 1] - true_t[i - 1]))
            i -= 1
        j -= 1
    return np.array(errs[::-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ours")
    ap.add_argument("answer_key")
    ap.add_argument("audio")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    cache = args.cache or os.path.splitext(args.audio)[0] + "_beats.npz"
    beats = tracked_beats(args.audio, cache)
    uq, f, cost, ours_end = true_beat_times(args.ours, args.answer_key, args.steps)
    print(f"tracked beats: {len(beats)}  span {beats.min():.1f}-{beats.max():.1f}s")
    print(f"alignment: DTW cost {cost:.4f}, resolution {ours_end/args.steps*1000:.0f}ms/step")

    d = np.diff(beats)
    print(f"tracked beat interval: median {np.median(d):.3f}s  CV {d.std()/d.mean():.3f}  "
          f"local tempo {60/d.max():.0f}-{60/d.min():.0f} BPM")

    # The edition is in 4/4, so integer quarters ARE beats. Where did each fall?
    q_int = np.arange(0, int(uq.max()) + 1)
    q_int = q_int[q_int >= uq.min()]
    true_t = np.interp(q_int, uq, f)
    true_d = np.diff(true_t)
    ok = (true_d > 0.05) & (true_d < 5.0)
    print(f"ground-truth beats in window: {len(true_t)}  "
          f"interval median {np.median(true_d[ok]):.3f}s  "
          f"CV {true_d[ok].std()/true_d[ok].mean():.3f}  "
          f"local tempo {60/np.percentile(true_d[ok],95):.0f}-"
          f"{60/np.percentile(true_d[ok],5):.0f} BPM")

    # ---- guard the reference itself -------------------------------------
    # DTW paths degenerate: flat and vertical runs produce wild implied beat
    # intervals. If the reference's own CV is far above the tracker's, the
    # reference is noise at this resolution and nothing below it means anything.
    ref_cv = float(true_d[ok].std() / true_d[ok].mean()) if ok.any() else float("nan")
    trk_cv = float(d.std() / d.mean())
    if ref_cv > 3 * trk_cv:
        print(f"\n!! REFERENCE UNRELIABLE: ground-truth CV {ref_cv:.3f} vs tracked "
              f"{trk_cv:.3f}. The DTW path is degenerate at this resolution, so the "
              f"'true' beat times are not trustworthy per-beat. Treat what follows "
              f"as indicative only.")

    print("\n=== DO OUR BEATS LAND ON THE REAL ONES? ===")
    print("(monotonic one-to-one matching, scored against a density-matched null -")
    print(" nearest-neighbour alone rewards whichever hypothesis has more candidates)")
    print(f"{'hypothesis':22s} {'matched':>8s} {'med err(s)':>11s} "
          f"{'med err(beats)':>15s} {'>0.25 beat':>11s}  null / lift")
    print("-" * 96)
    best = None
    for label, ratio in (("1:1 (beat = beat)", 1.0),
                         ("2:1 (we track 8ths)", 2.0),
                         ("1:2 (we track halves)", 0.5)):
        # our beats, re-expressed at the edition's beat level
        if ratio == 1.0:
            ours_b = beats
        elif ratio == 2.0:
            ours_b = beats[::2]
        else:
            ours_b = np.interp(np.arange(0, 2 * len(beats) - 1) / 2.0,
                               np.arange(len(beats)), beats)
        if len(ours_b) < 4:
            continue
        errs = _monotonic_errors(true_t, ours_b)
        if errs.size == 0:
            continue
        local = np.median(true_d[ok]) if ok.any() else 1.0
        med_beats = float(np.median(errs)) / local
        frac_bad = float(np.mean(errs / local > 0.25))
        # NULL: same count, same mean spacing, random phase. This is what a
        # sequence that knows NOTHING about the music scores at this density.
        rng = np.random.default_rng(0)
        nulls = []
        for _ in range(20):
            span = ours_b.max() - ours_b.min()
            step = span / max(len(ours_b) - 1, 1)
            fake = ours_b.min() + rng.uniform(0, step) + np.arange(len(ours_b)) * step
            ne = _monotonic_errors(true_t, fake)
            if ne.size:
                nulls.append(float(np.median(ne)))
        null_med = float(np.median(nulls)) / local if nulls else float("nan")
        lift = null_med / med_beats if med_beats > 0 else float("nan")
        print(f"{label:22s} {len(errs):8d} {np.median(errs):11.3f} "
              f"{med_beats:15.3f} {100*frac_bad:10.0f}%  null={null_med:.3f} "
              f"lift={lift:.2f}x")
        if best is None or lift > best[4]:
            best = (label, float(np.median(errs)), med_beats, frac_bad, lift)

    print("\n=== VERDICT ===")
    if best is None:
        print("  no usable correspondence - the tracker and the edition disagree")
        return
    label, med_s, med_b, frac_bad, lift = best
    print(f"  best correspondence: {label}")
    print(f"  median error {med_s:.3f}s = {med_b:.3f} beats; "
          f"{100*frac_bad:.0f}% of beats are more than a quarter-beat off")
    print(f"  lift over a density-matched null: {lift:.2f}x  "
          f"(1.0x = no better than evenly spaced beats that know nothing)")
    if lift < 1.15:
        print("  -> NO REAL INFORMATION at this resolution. The tracker scores what")
        print("     a regular sequence of the same density scores. Do not build on it.")
        return
    if med_b < 0.10 and frac_bad < 0.20:
        print("  -> the curve is TRUSTWORTHY. Build Musical Time on it.")
    elif med_b < 0.25:
        print("  -> the curve is USABLE but needs per-region confidence; expect")
        print("     local failures where it is wrong, and gate quantisation on them.")
    else:
        print("  -> the curve is NOT reliable enough to place notes in bars.")
        print("     A rigid grid fails diffusely; this would fail confidently.")
        print("     Phase 1 should not proceed on this evidence alone.")


if __name__ == "__main__":
    main()
