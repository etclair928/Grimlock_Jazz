#!/usr/bin/env python3
# =================================================================
# TOOL: tools/scorer_selftest.py
# CAN THE SCORER SEE A TIMING BIAS AT ALL?
#
# Inject a KNOWN offset into our note times and ask the scorer to recover it.
# If a +100ms shift comes back as +100ms, the instrument measures bias honestly.
# If it comes back as +20ms, chroma-DTW is ABSORBING the shift by sliding its
# own path - which would mean the tool is structurally partly blind to exactly
# the quantity anyone proposing phase calibration wants to calibrate.
#
# WHY SUSPECT THIS. The alignment is a DTW over chroma with an open end. A
# global translation of the performance is very close to a free parameter for
# such a path: shifting every note by a constant costs the path almost nothing,
# because it can simply start one step later. Two independent measurements of
# our Chopin bias disagreed in SIGN (-24ms and +46.6ms) depending on the match
# window, which is what a partly-unobservable parameter looks like.
#
# READ IT AS A SLOPE. Recovered-vs-injected should be a line of slope 1.0
# through the origin. The measured slope IS the fraction of a real timing bias
# this metric can see; 1 - slope is what it silently eats.
#
#   python tools/scorer_selftest.py <ours.mid|.musicxml> <answer_key> [--steps N]
# =================================================================

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_vs_answer_key import (  # noqa: E402
    load_answer_key, load_ours, chroma_seq, dtw_path, timing_moments,
)

INJECTIONS_MS = (-400, -200, -100, -50, 0, 50, 100, 200, 400)


def measure_bias(ours, key, key_end_q, steps, search_cap=2.0):
    """Run the scorer's own alignment and report the bias it recovers."""
    ours_end = max(e for _s, e, _p in ours)
    A = chroma_seq(key, key_end_q, steps)
    B = chroma_seq(ours, ours_end, steps)
    path, cost = dtw_path(A, B, open_end=True)
    kq = np.array([i for i, _j in path], float) / steps * key_end_q
    osec = np.array([j for _i, j in path], float) / steps * ours_end
    covered = float(kq.max())

    by_pitch = {}
    for i, (s, _e, p) in enumerate(sorted(ours)):
        by_pitch.setdefault(p, []).append((s, i))
    srt = sorted(ours)

    used, signed = set(), []
    for s, _e, p in key:
        if s > covered:
            continue
        t = float(np.interp(s, kq, osec))
        best, bd = None, search_cap
        for st, i in by_pitch.get(p, ()):
            if i in used:
                continue
            if abs(st - t) < bd:
                best, bd = i, abs(st - t)
        if best is not None:
            used.add(best)
            signed.append(srt[best][0] - t)
    return timing_moments(signed), len(signed), cost


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ours")
    ap.add_argument("answer_key")
    ap.add_argument("--steps", type=int, default=1800)
    args = ap.parse_args()

    key, key_end_q = load_answer_key(args.answer_key)
    base = load_ours(args.ours)
    print(f"{os.path.basename(args.ours)}  {len(base)} notes, "
          f"alignment {args.steps} steps\n")
    print(f"{'injected':>9} {'recovered':>10} {'delta of':>10} {'sigma':>8} "
          f"{'matched':>8} {'dtw cost':>9}")
    print(f"{'(ms)':>9} {'bias (ms)':>10} {'deltas':>10} {'(ms)':>8}")
    print("-" * 60)

    rows = []
    ref = None
    for inj in INJECTIONS_MS:
        shifted = [(s + inj / 1000.0, e + inj / 1000.0, p) for s, e, p in base]
        m, n, cost = measure_bias(shifted, key, key_end_q, args.steps)
        if inj == 0:
            ref = m["delta_ms"]
        rows.append((inj, m["delta_ms"]))
        d_of = m["delta_ms"] - (ref if ref is not None else 0.0)
        print(f"{inj:>9} {m['delta_ms']:>10.1f} {d_of:>10.1f} "
              f"{m['sigma_ms']:>8.1f} {n:>8} {cost:>9.4f}")

    x = np.array([r[0] for r in rows], float)
    y = np.array([r[1] for r in rows], float)
    slope, intercept = np.polyfit(x, y, 1)
    print("-" * 60)
    print(f"  recovered = {slope:.3f} x injected {intercept:+.1f} ms")
    print(f"  the scorer sees {100*slope:.1f}% of a real timing bias; "
          f"it absorbs {100*(1-slope):.1f}%")
    if slope > 0.9:
        print("  VERDICT: bias is observable - phase calibration is measurable.")
    elif slope > 0.5:
        print("  VERDICT: bias is PARTLY absorbed by the alignment. Any delta "
              "measured here understates the truth by roughly 1/slope.")
    else:
        print("  VERDICT: bias is largely INVISIBLE to this metric. Do not "
              "calibrate against it - the number being fitted is mostly the "
              "alignment's freedom to slide, not the pipeline's latency.")


if __name__ == "__main__":
    main()
