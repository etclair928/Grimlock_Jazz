#!/usr/bin/env python3
# =================================================================
# TOOL: tools/event_survival_audit.py
# ERROR LOCALIZATION. Answer-key recall measured at EVERY pipeline stage, so we
# can see where known-correct musical information stops surviving.
#
# WHY THIS AND NOT MORE ARGUING. We have spent weeks reasoning about which
# module "feels" responsible for the gap, and been wrong repeatedly - the
# fragmented melody was blamed on voice assignment (it was note durations), the
# thin chords on a narrow grouping window (it was consolidation), the meter on
# tempo (the tempo was right to 0.2%). With an answer key, causality is
# measurable: reconstruct the note set as it exists after each transformation
# and ask how much of the reference is still recoverable.
#
# READ IT AS A WATERFALL. A stage that drops recall by 15 points is the
# bottleneck regardless of how reasonable its docstring is. A stage that drops
# nothing is not worth optimising however bad its aggregate numbers look.
#
# NB recall here is PITCH-AND-TIME at a generous tolerance, because the point is
# to find where notes VANISH, not to re-measure timing precision (that is
# score_vs_answer_key's job). A note moved 300ms still counts as surviving.
#
#   python tools/event_survival_audit.py <intermediate.pkl> <answer_key> [--audio-seconds N]
# =================================================================

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)
sys.path.insert(0, os.path.join(JAZZ, "tools"))

TOLERANCE_S = 0.75      # generous: we are locating disappearance, not jitter


def recoverable(ours_s: Sequence[Tuple[float, int]],
                key_t: Sequence[Tuple[float, int]], tol: float) -> float:
    """Fraction of answer-key notes with a same-pitch partner within tol."""
    if not key_t:
        return float("nan")
    idx: Dict[int, List[float]] = {}
    for s, p in ours_s:
        idx.setdefault(p, []).append(s)
    for v in idx.values():
        v.sort()
    hit = 0
    for t, p in key_t:
        arr = idx.get(p)
        if not arr:
            continue
        i = np.searchsorted(arr, t)
        for j in (i - 1, i):
            if 0 <= j < len(arr) and abs(arr[j] - t) <= tol:
                hit += 1
                break
    return hit / len(key_t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl")
    ap.add_argument("answer_key")
    ap.add_argument("--audio-seconds", type=float, default=180.0)
    ap.add_argument("--steps", type=int, default=900)
    args = ap.parse_args()

    from score_vs_answer_key import load_answer_key, chroma_seq, dtw_path
    from core import StemType

    with open(args.pkl, "rb") as fh:
        d = pickle.load(fh)
    ann = d["annotations"]
    notes = [n for n in d["all_notes"] if n.stem != StemType.DRUMS]
    key, key_end_q = load_answer_key(args.answer_key)

    # ---- build each stage's note set ------------------------------------
    def get(kind, nid):
        return ann.latest_value(nid, kind)

    stages: List[Tuple[str, List[Tuple[float, float, int]]]] = []

    raw = [(n.start_ms / 1000.0, n.end_ms / 1000.0, n.pitch) for n in notes]
    stages.append(("1. Basic Pitch (raw)", raw))

    ref = []
    for n in notes:
        r = get("onset_refinement", n.id) or {}
        ref.append((r.get("start_ms", n.start_ms) / 1000.0, n.end_ms / 1000.0, n.pitch))
    stages.append(("2. + onset refinement", ref))

    sus = []
    for (s, _e, p), n in zip(ref, notes):
        v = get("sustain_extension", n.id) or {}
        sus.append((s, max(n.end_ms, v.get("end_ms", n.end_ms)) / 1000.0, p))
    stages.append(("3. + sustain recovery", sus))

    cons = []
    for (s, e, p), n in zip(sus, notes):
        c = get("consolidation", n.id) or {}
        if c.get("role") == "absorbed":
            continue
        cons.append((s, max(e, c.get("end_ms", n.end_ms) / 1000.0), p))
    stages.append(("4. + consolidation", cons))

    leg = []
    for (s, e, p), n in zip(sus, notes):
        c = get("consolidation", n.id) or {}
        if c.get("role") == "absorbed":
            continue
        v = get("legitimacy_verdict", n.id) or {}
        if v.get("verdict") not in (None, "legitimate"):
            continue
        leg.append((s, e, p))
    stages.append(("5. + legitimacy filter", leg))

    nt = []
    for n in notes:
        c = get("consolidation", n.id) or {}
        if c.get("role") == "absorbed":
            continue
        v = get("notation_timing", n.id) or get("quantization", n.id) or {}
        nt.append((v.get("start_ms", n.start_ms) / 1000.0,
                   v.get("end_ms", n.end_ms) / 1000.0, n.pitch))
    stages.append(("6. + notation quantization", nt))

    # ---- align ONCE, using the final stage, and score every stage on it ---
    ours_end = max(e for _s, e, _p in nt) if nt else args.audio_seconds
    A = chroma_seq(key, key_end_q, args.steps)
    B = chroma_seq(nt, ours_end, args.steps)
    path, cost = dtw_path(A, B, open_end=True)
    kq = np.array([i for i, _j in path], float) / args.steps * key_end_q
    osec = np.array([j for _i, j in path], float) / args.steps * ours_end
    covered = float(kq.max())
    key_t = [(float(np.interp(s, kq, osec)), p) for s, _e, p in key if s <= covered]

    print(f"{os.path.basename(args.pkl)}   answer-key notes in window: {len(key_t)}"
          f"   (alignment cost {cost:.4f}, covers {100*covered/key_end_q:.0f}% of the key)")
    print(f"\n{'stage':30s} {'notes':>7s} {'recoverable':>12s} {'delta':>8s}")
    print("-" * 62)
    prev = None
    for label, ns in stages:
        r = recoverable([(s, p) for s, _e, p in ns], key_t, TOLERANCE_S)
        delta = "" if prev is None else f"{100*(r - prev):+7.1f}"
        flag = ""
        if prev is not None and (r - prev) <= -0.05:
            flag = "   <-- LOSS"
        print(f"{label:30s} {len(ns):7d} {100*r:11.1f}% {delta:>8s}{flag}")
        prev = r

    print(f"\ntolerance {TOLERANCE_S}s - generous on purpose. This locates where notes")
    print("VANISH; score_vs_answer_key measures whether the survivors are placed well.")


if __name__ == "__main__":
    main()
