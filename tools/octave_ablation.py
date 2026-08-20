#!/usr/bin/env python3
# =================================================================
# TOOL: tools/octave_ablation.py
# IS THE OCTAVE FILTER DELETING ERRORS, OR DELETING MUSIC?
#
# acoustic_witness/octave_stack.py is the only witness in this codebase that
# takes notes OUT of an export, so it does not get to justify itself with a
# plausible story. It gets an ablation: remove exactly what the shipped
# function flags, re-score against the answer key, and read two numbers.
#
#   precision up, recall FLAT  -> those were false positives. The filter is real.
#   precision up, recall DOWN  -> we are deleting real music, and no amount of
#                                 precision buys that back.
#
# IT CALLS THE SHIPPED FUNCTION, deliberately. An earlier version of this
# script reimplemented the rule and reported +0.036 F1 for 238 notes; the
# shipped module flags a different set, because it refuses to count two
# detections of the SAME pitch as two rungs of an octave chain. A tool that
# re-derives the rule it is auditing measures the tool, not the code.
#
#   python tools/octave_ablation.py <ours.pkl> <answer_key.musicxml>
# =================================================================

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acoustic_witness.octave_stack import find_octave_stacks  # noqa: E402
from score_vs_answer_key import chroma_seq, dtw_path, load_answer_key  # noqa: E402

ONSET_TOL_S = 0.25


def score(notes, keep_ids, ref):
    """Greedy pitch-matched hit count, each emitted note claimable once."""
    by_pitch = {}
    for n in sorted((n for n in notes if n.id in keep_ids), key=lambda n: n.start_ms):
        by_pitch.setdefault(int(n.pitch), []).append(n.start_ms / 1000.0)
    used, hits = set(), 0
    for t, p in ref:
        best, bd = None, ONSET_TOL_S
        for j, st in enumerate(by_pitch.get(p, ())):
            if (p, j) in used:
                continue
            if abs(st - t) < bd:
                best, bd = j, abs(st - t)
        if best is not None:
            used.add((p, best))
            hits += 1
    prec = hits / max(len(keep_ids), 1)
    rec = hits / max(len(ref), 1)
    return hits, prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ours_pkl")
    ap.add_argument("answer_key")
    ap.add_argument("--steps", type=int, default=1800)
    args = ap.parse_args()

    with open(args.ours_pkl, "rb") as fh:
        notes = [n for n in pickle.load(fh)["all_notes"]
                 if n.stem.value != "drums"]
    key, key_end_q = load_answer_key(args.answer_key)

    # ONE alignment, fitted on the full set and reused, so the comparison
    # isolates the deletion instead of silently re-warping around it.
    ours = [(n.start_ms / 1000.0, n.end_ms / 1000.0, int(n.pitch)) for n in notes]
    ours_end = max(e for _s, e, _p in ours)
    path, _cost = dtw_path(chroma_seq(key, key_end_q, args.steps),
                           chroma_seq(ours, ours_end, args.steps), open_end=True)
    kq = np.array([i for i, _j in path], float) / args.steps * key_end_q
    osec = np.array([j for _i, j in path], float) / args.steps * ours_end
    covered = float(kq.max())
    ref = [(float(np.interp(s, kq, osec)), p) for s, _e, p in key if s <= covered]

    flagged = {nid for nid, _d in find_octave_stacks(notes)}
    all_ids = {n.id for n in notes}

    print(f"{os.path.basename(args.ours_pkl)}: {len(notes)} pitched notes")
    print(f"answer key notes inside the aligned span: {len(ref)}")
    print(f"flagged as octave-stack interior: {len(flagged)} "
          f"({100.0 * len(flagged) / max(len(notes), 1):.1f}%)\n")

    print(f"{'variant':22} {'kept':>6} {'hits':>6} {'prec':>7} {'recall':>7} {'F1':>7}")
    print("-" * 60)
    b = score(notes, all_ids, ref)
    print(f"{'baseline (all notes)':22} {len(all_ids):>6} {b[0]:>6} "
          f"{b[1]:>7.3f} {b[2]:>7.3f} {b[3]:>7.3f}")
    a = score(notes, all_ids - flagged, ref)
    print(f"{'octave stacks dropped':22} {len(all_ids) - len(flagged):>6} {a[0]:>6} "
          f"{a[1]:>7.3f} {a[2]:>7.3f} {a[3]:>7.3f}")
    print("-" * 60)
    lost = b[0] - a[0]
    kept_wrong = len(flagged) - lost
    print(f"  recall {a[2] - b[2]:+.4f}   precision {a[1] - b[1]:+.4f}   "
          f"F1 {a[3] - b[3]:+.4f}")
    print(f"  of {len(flagged)} dropped notes, {lost} had an answer-key "
          f"counterpart and {kept_wrong} did not")
    if flagged:
        print(f"  filter precision: {100.0 * kept_wrong / len(flagged):.1f}% "
              f"of what it removes is unmatched by the reference")


if __name__ == "__main__":
    main()
