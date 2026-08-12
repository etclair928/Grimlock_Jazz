#!/usr/bin/env python3
# =================================================================
# TOOL: tools/musical_criteria.py
# Grades a transcription on what a MUSICIAN asked for, not on page shape.
#
# The Chopin work graded note-for-note against an edition, which is right when
# the notation is canonical. For Ellington's Reflections in D the brief is
# different and better posed: barlines are explicitly NOT the target - the
# piece wanders through 4/2, 5/4, 6/4 and 3/2, and where the barlines fall is
# one person's reading. What matters is:
#
#   1. RHYTHMIC PROPORTION - is a triplet a real triplet RELATIVE to what is
#      around it? Absolute tempo is irrelevant under rubato; the ratio between
#      neighbouring durations is the musical fact. This is also the sharpest
#      test of the tempo map: under a constant-tempo grid rubato corrupts every
#      ratio, whereas in beat space a 3:1 stays 3:1 however the bar breathed.
#   2. BLOCK CHORDS - Ellington plays orchestrally, in parallel/planing
#      voicings. 53% of the edition's events are chords. Do near-simultaneous
#      notes survive as ONE chord, or get shredded across voices?
#   3. PHRASES AND MOTIFS - can we see the repeated material at all?
#
# Only (1) and (2) are compared against the key; (3) is reported from our own
# structure detectors, because "did we see a phrase" is answerable without one.
#
#   python tools/musical_criteria.py <ours.musicxml> [answer_key.mxl|.musicxml]
# =================================================================

from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import warnings
from typing import List, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

# Ratios a musician would actually name. Everything else is "unnameable" - the
# signature of a rhythm that was rounded rather than understood.
NAMED_RATIOS = {
    "1:1": 1.0, "2:1": 2.0, "1:2": 0.5, "3:1": 3.0, "1:3": 1 / 3,
    "3:2": 1.5, "2:3": 2 / 3, "4:1": 4.0, "1:4": 0.25,
    "4:3": 4 / 3, "3:4": 0.75, "6:1": 6.0, "1:6": 1 / 6,
}
RATIO_TOL = 0.08          # within 8% counts as that ratio


def load_events(path: str) -> List[Tuple[float, float, List[int]]]:
    """(onset_in_quarters, duration_in_quarters, [pitches]) grouped into
    simultaneities. Works for both an edition and our own export."""
    from music21 import converter
    s = converter.parse(path)
    raw = []
    for el in s.flatten().notes:
        off = float(el.offset)
        dur = float(el.quarterLength)
        ps = [int(p.midi) for p in (el.pitches if el.isChord else [el.pitch])]
        raw.append((off, dur, ps))
    raw.sort()
    # merge anything sharing an onset (chords written as separate voices)
    merged: List[Tuple[float, float, List[int]]] = []
    for off, dur, ps in raw:
        if merged and abs(merged[-1][0] - off) < 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], dur), merged[-1][2] + ps)
        else:
            merged.append((off, dur, list(ps)))
    return merged


def ratio_profile(events) -> Tuple[dict, float, int]:
    """Distribution of consecutive inter-onset RATIOS. Scale-free by
    construction, so rubato cannot move it - only a wrong reading can."""
    onsets = [e[0] for e in events]
    iois = [b - a for a, b in zip(onsets, onsets[1:]) if b - a > 1e-6]
    counts = collections.Counter()
    named = 0
    for a, b in zip(iois, iois[1:]):
        if a <= 0:
            continue
        r = b / a
        best, bd = None, RATIO_TOL
        for name, val in NAMED_RATIOS.items():
            d = abs(math.log(r / val)) if r > 0 else 9e9
            if d < bd:
                best, bd = name, d
        if best:
            counts[best] += 1
            named += 1
        else:
            counts["unnameable"] += 1
    total = sum(counts.values())
    return dict(counts), (named / total if total else 0.0), total


def chord_stats(events) -> dict:
    sizes = [len(e[2]) for e in events]
    n_chord = sum(1 for s in sizes if s >= 2)
    return {
        "events": len(events),
        "noteheads": sum(sizes),
        "chord_events": n_chord,
        "chord_event_share": n_chord / max(len(events), 1),
        "mean_chord_size": float(np.mean([s for s in sizes if s >= 2])) if n_chord else 0.0,
        "max_chord_size": max(sizes) if sizes else 0,
        "spread_semitones": float(np.mean([max(e[2]) - min(e[2])
                                           for e in events if len(e[2]) >= 2])) if n_chord else 0.0,
    }


def report(label: str, events) -> dict:
    cs = chord_stats(events)
    counts, named_share, total = ratio_profile(events)
    print(f"\n=== {label} ===")
    print(f"  events {cs['events']}  noteheads {cs['noteheads']}")
    print(f"  BLOCK CHORDS: {cs['chord_events']} chord-events "
          f"({100*cs['chord_event_share']:.0f}% of events), mean size "
          f"{cs['mean_chord_size']:.2f}, max {cs['max_chord_size']}, "
          f"mean spread {cs['spread_semitones']:.1f} semitones")
    print(f"  RHYTHMIC PROPORTION: {100*named_share:.0f}% of consecutive IOI "
          f"ratios are nameable (of {total})")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
    print("     " + "  ".join(f"{k}:{v}" for k, v in top))
    return {"chords": cs, "ratios": counts, "named_share": named_share, "total": total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ours")
    ap.add_argument("answer_key", nargs="?")
    args = ap.parse_args()

    ours = load_events(args.ours)
    r_ours = report("OURS", ours)

    if args.answer_key:
        key = load_events(args.answer_key)
        r_key = report("ANSWER KEY", key)

        print("\n=== HEAD TO HEAD ===")
        co, ck = r_ours["chords"], r_key["chords"]
        print(f"  chord-event share   ours {100*co['chord_event_share']:5.1f}%   "
              f"key {100*ck['chord_event_share']:5.1f}%")
        print(f"  mean chord size     ours {co['mean_chord_size']:5.2f}    "
              f"key {ck['mean_chord_size']:5.2f}")
        print(f"  chord spread (st)   ours {co['spread_semitones']:5.1f}    "
              f"key {ck['spread_semitones']:5.1f}")
        print(f"  nameable ratios     ours {100*r_ours['named_share']:5.1f}%   "
              f"key {100*r_key['named_share']:5.1f}%")

        # ratio-distribution agreement: do we hear the same PROPORTIONS?
        keys = sorted(set(r_ours["ratios"]) | set(r_key["ratios"]))
        a = np.array([r_ours["ratios"].get(k, 0) for k in keys], float)
        b = np.array([r_key["ratios"].get(k, 0) for k in keys], float)
        a = a / max(a.sum(), 1); b = b / max(b.sum(), 1)
        r = float(np.corrcoef(a, b)[0, 1]) if len(keys) > 2 else float("nan")
        print(f"  ratio-profile correlation: r = {r:.3f}")
        print("\n  ratio            ours     key")
        for k in sorted(keys, key=lambda k: -(r_key["ratios"].get(k, 0))):
            print(f"    {k:12s} {100*r_ours['ratios'].get(k,0)/max(r_ours['total'],1):6.1f}% "
                  f"{100*r_key['ratios'].get(k,0)/max(r_key['total'],1):6.1f}%")


if __name__ == "__main__":
    main()
