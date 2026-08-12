#!/usr/bin/env python3
# =================================================================
# TOOL: tools/musicianship_report.py
# Grades a transcription the way a TEACHER would, not the way a grid would.
#
# WHY A SECOND GRADER. score_vs_answer_key.py asks "is this note in the right
# place", which is the right question for Chopin and the WRONG one for
# Ellington's Reflections in D. The reference transcription of that piece
# carries eleven meter changes (4/2, 4/4, 5/4, 6/4, 3/2) because a human
# interpreted freely-played rubato into readable bars. Barline placement there
# is an editorial opinion, not a fact, so scoring against it measures agreement
# with one interpreter rather than musical correctness.
#
# What DOES survive that ambiguity - and what the user asked for:
#
#   PROPORTION  a triplet is a triplet because of its RATIO to what surrounds
#               it, not its absolute duration. Compare the distribution of
#               consecutive inter-onset ratios. Alignment-free by construction:
#               it does not care where bar one is, or that the performance
#               breathes.
#   CHORDS      this is orchestral piano writing - block voicings moving in
#               parallel (planing). Measure whether simultaneities survive AS
#               chords instead of being shredded into voices, and whether
#               parallel motion is preserved.
#   PHRASES     Ellington lets chords ring and starts the next phrase when the
#               decay allows. Measure phrase segmentation from silence, and
#               what the motif detectors find.
#   CONTENT     pitch-class profile and register span. No alignment needed.
#
# Every measure here compares OUR distribution against the ANSWER KEY's. None
# of them requires the two to be time-aligned.
#
#   python tools/musicianship_report.py <ours.musicxml> <answer_key.mxl>
# =================================================================

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

CHORD_WINDOW = 0.06        # seconds (ours) / quarters (key) for "struck together"
# Ratios a musician actually reads. Everything else is either a rounding
# artifact or a rhythm nobody would notate.
NAMED_RATIOS = {0.0: "1:1", 1.0: "2:1", -1.0: "1:2", 2.0: "4:1", -2.0: "1:4",
                math.log2(1.5): "3:2", -math.log2(1.5): "2:3",
                math.log2(3.0): "3:1", -math.log2(3.0): "1:3"}


def load_key(path: str):
    from music21 import converter
    s = converter.parse(path)
    out = []
    for el in s.flatten().notes:
        off = float(el.offset)
        for p in (el.pitches if el.isChord else [el.pitch]):
            out.append((off, off + float(el.quarterLength), int(p.midi)))
    return sorted(out)


def load_ours(path: str):
    sys.path.insert(0, os.path.join(JAZZ, "tools"))
    from score_vs_answer_key import load_ours as _lo
    return _lo(path)


def _events(notes: Sequence[Tuple[float, float, int]], window: float):
    """Group notes struck together into chord events: [(onset, [pitches])]."""
    out = []
    for s, _e, p in sorted(notes):
        if out and s - out[-1][0] <= window:
            out[-1][1].append(p)
        else:
            out.append((s, [p]))
    return [(t, sorted(ps)) for t, ps in out]


def proportion_profile(events) -> np.ndarray:
    """Histogram of log2(IOI[i+1] / IOI[i]) - how each duration relates to the
    one before it. This is what makes a triplet legible: not its length, but
    its ratio to its neighbours. Rubato scales both sides of the ratio and
    cancels out, which is exactly why this works on freely-played music."""
    t = np.array([e[0] for e in events], float)
    if len(t) < 3:
        return np.zeros(24)
    iois = np.diff(t)
    iois = iois[iois > 1e-6]
    if len(iois) < 2:
        return np.zeros(24)
    r = np.log2(iois[1:] / iois[:-1])
    r = r[np.abs(r) <= 3.0]
    hist, _ = np.histogram(r, bins=24, range=(-3.0, 3.0))
    return hist / max(hist.sum(), 1)


def named_ratio_share(events, tol=0.08):
    t = np.array([e[0] for e in events], float)
    if len(t) < 3:
        return {}, 0.0
    iois = np.diff(t)
    iois = iois[iois > 1e-6]
    if len(iois) < 2:
        return {}, 0.0
    r = np.log2(iois[1:] / iois[:-1])
    hits = Counter()
    for v in r:
        best = min(NAMED_RATIOS, key=lambda k: abs(k - v))
        if abs(best - v) <= tol:
            hits[NAMED_RATIOS[best]] += 1
    return hits, sum(hits.values()) / max(len(r), 1)


def chord_stats(events):
    sizes = [len(ps) for _t, ps in events]
    n = len(sizes)
    multi = [s for s in sizes if s >= 2]
    # PLANING: consecutive chords of equal size whose voices all move by the
    # same interval. This is the texture the piece is built from.
    parallel = comparable = 0
    for (t1, a), (t2, b) in zip(events, events[1:]):
        if len(a) >= 3 and len(a) == len(b):
            comparable += 1
            steps = {y - x for x, y in zip(a, b)}
            if len(steps) == 1 and 0 not in steps:
                parallel += 1
    return {
        "events": n,
        "chord_share": len(multi) / max(n, 1),
        "mean_chord_size": float(np.mean(multi)) if multi else 0.0,
        "size_hist": Counter(sizes),
        "parallel_share": parallel / max(comparable, 1),
        "comparable_pairs": comparable,
    }


def phrase_stats(events, gap_factor=3.0):
    """Phrases from silence: Ellington lets a chord ring and enters when the
    decay allows, so a gap far longer than the local norm is a phrase end."""
    t = np.array([e[0] for e in events], float)
    if len(t) < 4:
        return {"phrases": 0, "mean_len": 0.0}
    iois = np.diff(t)
    med = float(np.median(iois))
    breaks = int(np.sum(iois > gap_factor * med))
    return {"phrases": breaks + 1,
            "mean_len": len(t) / max(breaks + 1, 1),
            "median_ioi": med,
            "longest_gap_x": float(iois.max() / med) if med > 0 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ours")
    ap.add_argument("answer_key")
    args = ap.parse_args()

    ours = load_ours(args.ours)
    key = load_key(args.answer_key)
    oe = _events(ours, CHORD_WINDOW)
    ke = _events(key, CHORD_WINDOW)
    print(f"ours: {len(ours)} noteheads in {len(oe)} events")
    print(f"key : {len(key)} noteheads in {len(ke)} events")

    print("\n=== 1. RHYTHMIC PROPORTION (alignment-free) ===")
    po, pk = proportion_profile(oe), proportion_profile(ke)
    r = float(np.corrcoef(po, pk)[0, 1]) if po.any() and pk.any() else float("nan")
    print(f"  IOI-ratio distribution correlation: r = {r:.4f}")
    ho, so = named_ratio_share(oe)
    hk, sk = named_ratio_share(ke)
    print(f"  share of transitions landing on a READABLE ratio:")
    print(f"     ours {100*so:5.1f}%     key {100*sk:5.1f}%")
    print(f"     ours: " + "  ".join(f"{k}:{v}" for k, v in ho.most_common(6)))
    print(f"     key : " + "  ".join(f"{k}:{v}" for k, v in hk.most_common(6)))

    print("\n=== 2. BLOCK CHORDS AND PLANING ===")
    co, ck = chord_stats(oe), chord_stats(ke)
    print(f"{'':22s} {'ours':>10s} {'key':>10s}")
    print(f"  {'chord share':20s} {co['chord_share']:9.3f} {ck['chord_share']:10.3f}")
    print(f"  {'mean chord size':20s} {co['mean_chord_size']:9.2f} {ck['mean_chord_size']:10.2f}")
    print(f"  {'parallel-motion share':20s} {co['parallel_share']:9.3f} "
          f"{ck['parallel_share']:10.3f}")
    print(f"  {'(comparable pairs)':20s} {co['comparable_pairs']:9d} "
          f"{ck['comparable_pairs']:10d}")
    print("  chord-size histogram:")
    for n in range(1, 7):
        print(f"     {n} note{'s' if n > 1 else ' '}: ours {co['size_hist'].get(n,0):5d}"
              f"   key {ck['size_hist'].get(n,0):5d}")

    print("\n=== 3. PHRASING (silence as structure) ===")
    fo, fk = phrase_stats(oe), phrase_stats(ke)
    print(f"  {'phrases found':22s} ours {fo['phrases']:5d}   key {fk['phrases']:5d}")
    print(f"  {'mean events/phrase':22s} ours {fo['mean_len']:5.1f}   key {fk['mean_len']:5.1f}")
    print(f"  {'longest gap (x median)':22s} ours {fo['longest_gap_x']:5.1f}   "
          f"key {fk['longest_gap_x']:5.1f}")

    print("\n=== 4. CONTENT (alignment-free) ===")
    def pcp(ns):
        v = np.zeros(12)
        for _s, _e, p in ns:
            v[p % 12] += 1
        return v / max(v.sum(), 1)
    a, b = pcp(ours), pcp(key)
    PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    print(f"  pitch-class correlation r = {np.corrcoef(a, b)[0,1]:.4f}")
    print("    ours: " + " ".join(f"{PC[i]}:{a[i]*100:.0f}" for i in np.argsort(a)[::-1][:6]))
    print("    key : " + " ".join(f"{PC[i]}:{b[i]*100:.0f}" for i in np.argsort(b)[::-1][:6]))
    op = [p for _s, _e, p in ours]; kp = [p for _s, _e, p in key]
    print(f"  range  ours {min(op)}-{max(op)}   key {min(kp)}-{max(kp)}")
    print(f"  median ours {int(np.median(op))}      key {int(np.median(kp))}")


if __name__ == "__main__":
    main()
