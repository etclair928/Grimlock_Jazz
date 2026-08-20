#!/usr/bin/env python3
# =================================================================
# TOOL: tools/rhythmic_proportion.py
# TEMPO-INVARIANT RELATIVE DURATION ACCURACY - "IOI ratio fidelity".
#
# THE QUESTION. Every other instrument here grades ABSOLUTE placement: does this
# note land on the beat the reference names. That is unanswerable while meter
# and barline phase are wrong, and it is not the only question worth asking.
# The reader's other question is RELATIONAL - "do eighths follow this quarter",
# "is this chord held for a half note" - and it survives being in the wrong bar
# entirely. This tool asks only that:
#
#   we do not care whether the global meter or tempo estimate is off;
#   we care whether the inter-note rhythmic proportions are topologically right.
#
# MULTIPLICATIVE TIME, NOT ADDITIVE. Grid-based evaluation says "this note sits
# at offset 480 of a 960-TPQN bar" and is destroyed by a tempo or meter error.
# Proportional evaluation says "this note is half the duration of the one
# before" and is robust to both. Everything below is computed as RATIOS and
# compared in LOG SPACE, so 2:1 and 1:2 are symmetric and a uniform time-scaling
# is exactly zero change.
#
# WHAT IS MEASURED
#   1. IOI RATIO TOPOLOGY  - consecutive inter-onset intervals as ratios,
#      classified into the proportions a reader can name. This is the piece's
#      rhythmic vocabulary. Isochrony (1:1) is reported separately because on
#      this repertoire it dominates.
#   2. DURATION RATIO      - the same for written note VALUES rather than onset
#      spacing. IOI answers "when does the next note start"; duration answers
#      "how long is this one held", and they differ wherever there are rests.
#   3. PROPORTIONAL COHERENCE - continuous, not bucketed: how far each ratio
#      sits from the NEAREST nameable proportion, in log2. A rhythm whose ratios
#      are all near-nameable has proportional structure even if the particular
#      proportions are wrong; one that is uniformly off-lattice does not. The
#      old "% other" bucket hid this behind a threshold.
#   4. FILL FACTOR         - duration / gap to the next onset. 1.0 = notes tile
#      the time. Below = the page fills with rests; above = notes overrun.
#   5. SEQUENCE AGREEMENT  - DTW over the ratio stream, reported BOTH as a class
#      hit-rate and as a continuous log-space distance. Distribution can be
#      right while the ORDER is wrong; this is what catches that.
#
#   python tools/rhythmic_proportion.py <reference.musicxml> <candidate...>
#   python tools/rhythmic_proportion.py --selftest <any.musicxml>
# =================================================================

from __future__ import annotations

import argparse
import math
import os
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

# Proportions a reader can name, in log2 so 2:1 and 1:2 are symmetric.
_RATIO_CLASSES = {
    "1:4": -2.0, "1:3": -math.log2(3), "1:2": -1.0, "2:3": -math.log2(1.5),
    "1:1": 0.0,
    "3:2": math.log2(1.5), "2:1": 1.0, "3:1": math.log2(3), "4:1": 2.0,
}
_CLASS_ORDER = ["1:1", "1:2", "2:1", "1:3", "3:1", "2:3", "3:2", "1:4", "4:1", "other"]
_TOL_LOG2 = 0.18          # ~13% in ratio terms; separates 3:2 from 2:1 cleanly

# Two notes this close in NOTATED time are one rhythmic event. A fixed epsilon
# was wrong: different engravers quantise chords differently, so an exact-offset
# match splits a chord in one file and not in another, and the two files then
# have different event counts for identical music. A 32nd is below any real
# rhythmic distinction on this repertoire.
_CHORD_TOL_QL = 0.125


def skeleton(path: str):
    """Chord-collapsed (onset, duration) in quarter-lengths, all staves and
    voices. Pitch is irrelevant - this measures the SHAPE of the rhythm."""
    from music21 import converter
    score = converter.parse(path)
    raw = sorted((round(float(n.offset), 6), float(n.quarterLength))
                 for n in score.flatten().notes)
    onsets, durs = [], []
    for off, dur in raw:
        if onsets and off - onsets[-1] <= _CHORD_TOL_QL:
            durs[-1] = max(durs[-1], dur)      # a chord holds as long as its longest tone
        else:
            onsets.append(off)
            durs.append(dur)
    return onsets, durs


def voice_fills(path: str):
    """Fill factor computed WITHIN each voice, never across staves.

    Measuring it on the chord-collapsed skeleton was wrong: a left hand holding
    a half note under a moving right hand reads as a 4x "overrun" when it is
    simply polyphony, and that alone put the published edition at 38% apparently
    overrunning. The question "is this chord held for a half note" is about one
    voice's own succession, so that is the only place it can be asked."""
    from music21 import converter
    score = converter.parse(path)
    fills = []
    for part in score.parts:
        streams = []
        voices = part.recurse().getElementsByClass("Voice")
        if voices:
            streams = [list(v.notes) for v in voices]
        else:
            streams = [list(part.recurse().notes)]
        for st in streams:
            evs = {}
            for n in st:
                off = round(float(n.offset), 6)
                evs[off] = max(evs.get(off, 0.0), float(n.quarterLength))
            ons = sorted(evs)
            for i in range(len(ons) - 1):
                gap = ons[i + 1] - ons[i]
                if gap > 1e-9:
                    fills.append(evs[ons[i]] / gap)
    return np.array(fills) if fills else np.array([0.0])


def _deviation(ratio: float):
    """(class name, distance in log2 to the nearest nameable proportion)."""
    if ratio <= 0:
        return "other", float("inf")
    lg = math.log2(ratio)
    name, dist = min(((n, abs(lg - t)) for n, t in _RATIO_CLASSES.items()),
                     key=lambda x: x[1])
    return (name if dist <= _TOL_LOG2 else "other"), dist


def _ratio_stream(values):
    """Consecutive ratios of a positive sequence, with their log2."""
    out = []
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if a > 1e-9 and b > 1e-9:
            out.append(b / a)
    return out


def profile(path: str):
    onsets, durs = skeleton(path)
    iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    iois = [d for d in iois if d > 1e-9]

    ioi_ratios = _ratio_stream(iois)
    dur_ratios = _ratio_stream([d for d in durs if d > 1e-9])

    ioi_cls, ioi_dev = zip(*(_deviation(r) for r in ioi_ratios)) if ioi_ratios else ((), ())
    dur_cls = [c for c, _d in (_deviation(r) for r in dur_ratios)]

    fills = []
    for i in range(len(onsets) - 1):
        gap = onsets[i + 1] - onsets[i]
        if gap > 1e-9:
            fills.append(durs[i] / gap)

    finite = [d for d in ioi_dev if math.isfinite(d)]
    return {
        "file": os.path.basename(path),
        "events": len(onsets),
        "ioi_classes": Counter(ioi_cls),
        "dur_classes": Counter(dur_cls),
        "ioi_seq": list(ioi_cls),
        "log_seq": [math.log2(r) for r in ioi_ratios],
        "deviation": np.array(finite) if finite else np.array([0.0]),
        "fill": np.array(fills) if fills else np.array([0.0]),
        "vfill": voice_fills(path),
    }


def dtw(ref, cand, cost_fn):
    """Generic DTW; returns (mean cost per aligned pair, hit fraction)."""
    cap = 2200
    a, b = ref[:cap], cand[:cap]
    n, m = len(a), len(b)
    if not n or not m:
        return float("nan"), 0.0
    INF = float("inf")
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    prev_cost = np.zeros(m + 1)
    prev_hits = np.zeros(m + 1)
    prev_len = np.zeros(m + 1)
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        cur_cost = np.zeros(m + 1)
        cur_hits = np.zeros(m + 1)
        cur_len = np.zeros(m + 1)
        for j in range(1, m + 1):
            c, hit = cost_fn(a[i - 1], b[j - 1])
            opts = ((prev[j - 1], prev_cost[j - 1], prev_hits[j - 1], prev_len[j - 1]),
                    (prev[j] + 1.0, prev_cost[j], prev_hits[j], prev_len[j]),
                    (cur[j - 1] + 1.0, cur_cost[j - 1], cur_hits[j - 1], cur_len[j - 1]))
            base, bc, bh, bl = min(opts, key=lambda o: o[0])
            cur[j] = base + c
            cur_cost[j] = bc + c
            cur_hits[j] = bh + hit
            cur_len[j] = bl + 1
        prev, prev_cost, prev_hits, prev_len = cur, cur_cost, cur_hits, cur_len
    ln = max(prev_len[m], 1)
    return float(prev_cost[m] / ln), float(prev_hits[m] / ln)


def report(rows):
    w = max(len(r["file"]) for r in rows) + 1

    print("1. IOI RATIO TOPOLOGY - the rhythmic vocabulary (tempo- and bar-invariant)")
    print(f"{'file':{w}} {'events':>7} " + "".join(f"{c:>7}" for c in _CLASS_ORDER))
    print("-" * (w + 8 + 7 * len(_CLASS_ORDER)))
    for r in rows:
        tot = max(sum(r["ioi_classes"].values()), 1)
        print(f"{r['file']:{w}} {r['events']:>7} " +
              "".join(f"{100.0*r['ioi_classes'][c]/tot:>6.1f} " for c in _CLASS_ORDER))

    print("\n2. DURATION RATIO - written note VALUES rather than onset spacing")
    print(f"{'file':{w}} " + "".join(f"{c:>7}" for c in _CLASS_ORDER))
    print("-" * (w + 7 * len(_CLASS_ORDER)))
    for r in rows:
        tot = max(sum(r["dur_classes"].values()), 1)
        print(f"{r['file']:{w}} " +
              "".join(f"{100.0*r['dur_classes'][c]/tot:>6.1f} " for c in _CLASS_ORDER))

    print("\n3. PROPORTIONAL COHERENCE - distance to the nearest nameable ratio (log2)")
    print(f"{'file':{w}} {'median':>8} {'mean':>8} {'isochrony':>10} {'on-lattice':>11}")
    print("-" * (w + 40))
    for r in rows:
        d = r["deviation"]
        tot = max(sum(r["ioi_classes"].values()), 1)
        iso = 100.0 * r["ioi_classes"]["1:1"] / tot
        on = 100.0 * (1.0 - r["ioi_classes"]["other"] / tot)
        print(f"{r['file']:{w}} {np.median(d):>8.4f} {d.mean():>8.4f} "
              f"{iso:>9.1f}% {on:>10.1f}%")

    print("\n4. FILL FACTOR - duration / gap to next onset, WITHIN ONE VOICE")
    print(f"{'file':{w}} {'median':>8} {'<0.6':>7} {'0.9-1.1':>8} {'>1.1':>7}"
          f"     collapsed across staves (confounded by polyphony)")
    print("-" * (w + 78))
    for r in rows:
        f, v = r["fill"], r["vfill"]
        print(f"{r['file']:{w}} {np.median(v):>8.3f} {100*(v<0.6).mean():>6.1f}% "
              f"{100*((v>=0.9)&(v<=1.1)).mean():>7.1f}% {100*(v>1.1).mean():>6.1f}%"
              f"     median {np.median(f):.3f}, >1.1 {100*(f>1.1).mean():.1f}%")

    print("\n5. SEQUENCE AGREEMENT with the reference (order, not just vocabulary)")
    ref = rows[0]

    def cls_cost(x, y):
        same = (x == y)
        return (0.0 if same else 1.0), (1.0 if same else 0.0)

    def log_cost(x, y):
        d = abs(x - y)
        return d, (1.0 if d <= _TOL_LOG2 else 0.0)

    print(f"{'file':{w}} {'class hit':>10} {'log2 dist':>11}")
    print("-" * (w + 23))
    for r in rows[1:]:
        _c, hit = dtw(ref["ioi_seq"], r["ioi_seq"], cls_cost)
        dist, _h = dtw(ref["log_seq"], r["log_seq"], log_cost)
        print(f"{r['file']:{w}} {100*hit:>9.1f}% {dist:>11.4f}")

    print("\n  isochrony = share of 1:1 (even runs). on-lattice = share within "
          f"{_TOL_LOG2} log2 of a nameable ratio.")
    print("  fill 1.0 = notes tile the time. log2 dist 0 = identical proportions.")


def selftest(path: str) -> None:
    """Is this tool ACTUALLY tempo-invariant? Claiming invariance is cheap;
    a uniform time-scaling must move nothing, and a rubato warp tells us the
    noise floor below which two files cannot be told apart."""
    print(f"SELF-TEST on {os.path.basename(path)}\n")
    onsets, durs = skeleton(path)

    def metrics(on, du):
        iois = [on[i + 1] - on[i] for i in range(len(on) - 1)]
        iois = [d for d in iois if d > 1e-9]
        rs = _ratio_stream(iois)
        cls = Counter(_deviation(r)[0] for r in rs)
        tot = max(sum(cls.values()), 1)
        return 100.0 * cls["1:1"] / tot, 100.0 * (1 - cls["other"] / tot)

    base_iso, base_lat = metrics(onsets, durs)
    print(f"{'transform':<34} {'isochrony':>10} {'on-lattice':>11} {'drift':>8}")
    print("-" * 66)
    print(f"{'unchanged':<34} {base_iso:>9.2f}% {base_lat:>10.2f}% {0.0:>8.3f}")

    for k in (0.5, 2.0, 3.7):
        iso, lat = metrics([o * k for o in onsets], [d * k for d in durs])
        print(f"{'uniform x' + str(k) + ' (tempo change)':<34} {iso:>9.2f}% "
              f"{lat:>10.2f}% {abs(iso-base_iso):>8.3f}")

    # RUBATO, SIZED AGAINST A BEAT - not against the whole piece. The first
    # version of this test warped by a fraction of the total SPAN, so "5%" moved
    # notes by 19 quarter-notes and the rows measured a scramble rather than a
    # performance. A player pushes and pulls by a fraction of a BEAT over a
    # phrase, so that is what has to be injected.
    base_iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    med_ioi = float(np.median([d for d in base_iois if d > 1e-9])) if base_iois else 1.0
    period = 8.0                                   # a phrase-length push-pull
    for amp in (0.05, 0.10, 0.25, 0.50):
        warped = [o + amp * med_ioi * math.sin(2 * math.pi * o / period)
                  for o in onsets]
        monotonic = all(warped[i + 1] > warped[i] for i in range(len(warped) - 1))
        iso, lat = metrics(warped, durs)
        note = "" if monotonic else "   (REORDERED - warp too large to be rubato)"
        print(f"{'rubato +/-' + str(int(amp*100)) + '% of a beat':<34} "
              f"{iso:>9.2f}% {lat:>10.2f}% {abs(iso-base_iso):>8.3f}{note}")

    print("\n  uniform scaling MUST show drift 0.000 - ratios are scale-invariant "
          "by construction.")
    print("  the rubato rows are the NOISE FLOOR: two transcriptions of the same")
    print("  performance cannot be distinguished by less than this.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidates", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.reference)
        return

    rows = [profile(args.reference)]
    for c in args.candidates:
        try:
            rows.append(profile(c))
        except Exception as exc:
            print(f"{os.path.basename(c)}: ERROR {type(exc).__name__}: {exc}")
    report(rows)


if __name__ == "__main__":
    main()
