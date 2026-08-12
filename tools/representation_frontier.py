#!/usr/bin/env python3
# =================================================================
# TOOL: tools/representation_frontier.py
# RESEARCH ONLY - influences nothing in production.
#
# THE QUESTION. Every readability metric we have is ONE-SIDED: fewer voices
# means fewer voices to fill with rests, so rest/note falls monotonically as the
# cap falls and the "optimum" is always cap 1. Measured, on all 23 regions of
# Federal Blvd: rest/note 0.059 -> 0.117 -> 0.185 -> 0.215 for caps 1..4, with
# cap 1 winning 23 out of 23. That is arithmetic, not a finding. A one-sided
# objective cannot choose a representation.
#
# So this tool measures the OPPOSING force: what musical information does a
# representation destroy? Then we look for a frontier instead of declaring a
# winner. If cap 2 buys a large readability gain for tiny information loss,
# that is interesting. If cap 2 is dominated by cap 3 everywhere, cap 2 is not
# justified on this axis and only survives as a global regularizer.
#
# WHY MULTIDIMENSIONAL. "Distortion = 0.17" is useless - it cannot distinguish
# harmless simplification from mangling the melody. Five separate channels:
#
#   onset     notes moved in time
#   duration  notes shortened/lengthened relative to the consolidated source
#   overlap   simultaneity that existed in the source and does not survive
#             (a sustained note cut short because its voice was needed)
#   ordering  pairs whose temporal succession flipped
#   dropped   source events with no representative at all
#
# THE REFERENCE is the CONSOLIDATED event set - notes after note_consolidation
# merged Basic Pitch's fragments, before any notation decision. That is the best
# available statement of "what was played", and it is what the frozen-note law
# protects. Comparing notation against raw Basic Pitch output instead would
# score us against fragmentation we already know is an artifact.
#
#   python tools/representation_frontier.py <pkl> [--caps 1 2 3 4]
# =================================================================

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

ONSET_MATCH_MS = 250.0        # generous: we are measuring displacement, not identity
DIMS = ("onset", "duration", "overlap", "ordering", "dropped")


def consolidated_source(d) -> List[Tuple[float, float, int]]:
    """The reference event set: consolidation applied, notation not yet.
    'absorbed' fragments are folded into their primary, which is exactly what
    note_consolidation concluded actually sounded."""
    from core import StemType
    ann = d["annotations"]
    out = []
    for n in d["all_notes"]:
        if n.stem == StemType.DRUMS:
            continue
        c = ann.latest_value(n.id, "consolidation") or {}
        if c.get("role") == "absorbed":
            continue
        end = n.end_ms
        if c.get("role") == "primary" and c.get("end_ms") is not None:
            end = max(end, c["end_ms"])
        out.append((n.start_ms, end, n.pitch))
    return sorted(out)


def notated_events(score) -> List[Tuple[float, float, int]]:
    out = []
    for part in score.parts:
        if getattr(part, "is_drum", False):
            continue
        for nn in part.notes:
            out.append((nn.start_ms, nn.end_ms, nn.pitch))
    return sorted(out)


def match(src, cand) -> List[Tuple[int, Optional[int]]]:
    """Nearest same-pitch candidate for each source event."""
    by_pitch: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for j, (s, _e, p) in enumerate(cand):
        by_pitch[p].append((s, j))
    for v in by_pitch.values():
        v.sort()
    pairs = []
    used = set()
    for i, (s, _e, p) in enumerate(src):
        best, bestd = None, ONSET_MATCH_MS
        for st, j in by_pitch.get(p, ()):
            if j in used:
                continue
            dd = abs(st - s)
            if dd <= bestd:
                best, bestd = j, dd
            elif st > s + ONSET_MATCH_MS:
                break
        if best is not None:
            used.add(best)
        pairs.append((i, best))
    return pairs


def distortion(src, cand) -> Dict[str, float]:
    if not src:
        return {k: float("nan") for k in DIMS}
    pairs = match(src, cand)
    matched = [(i, j) for i, j in pairs if j is not None]
    if not matched:
        return {"onset": float("nan"), "duration": float("nan"),
                "overlap": float("nan"), "ordering": float("nan"), "dropped": 1.0}

    onset = float(np.mean([abs(cand[j][0] - src[i][0]) for i, j in matched]))
    dur = float(np.mean([
        abs((cand[j][1] - cand[j][0]) - (src[i][1] - src[i][0])) /
        max(src[i][1] - src[i][0], 1.0) for i, j in matched]))

    # OVERLAP: source pairs that sounded together - does the notation keep them?
    idx = {i: j for i, j in matched}
    kept = lost = 0
    order = sorted(idx)
    for a_pos in range(len(order)):
        i1 = order[a_pos]
        for i2 in order[a_pos + 1:a_pos + 12]:
            s1, e1, _ = src[i1]; s2, e2, _ = src[i2]
            if s2 >= e1:
                break
            c1, c2 = cand[idx[i1]], cand[idx[i2]]
            if c2[0] < c1[1]:
                kept += 1
            else:
                lost += 1
    overlap = lost / max(kept + lost, 1)

    # ORDERING: did any matched pair swap temporal order?
    flips = tot = 0
    for a_pos in range(len(order) - 1):
        i1, i2 = order[a_pos], order[a_pos + 1]
        if src[i1][0] == src[i2][0]:
            continue
        tot += 1
        if (cand[idx[i1]][0] - cand[idx[i2]][0]) * (src[i1][0] - src[i2][0]) < 0:
            flips += 1
    ordering = flips / max(tot, 1)

    dropped = 1.0 - len(matched) / len(src)
    return {"onset": onset, "duration": dur, "overlap": overlap,
            "ordering": ordering, "dropped": dropped}


def pareto_front(points: Dict[int, Dict[str, float]], axes: Sequence[str]) -> List[int]:
    """Caps not dominated on every axis (lower is better on all of them)."""
    keys = list(points)
    front = []
    for a in keys:
        dominated = False
        for b in keys:
            if a == b:
                continue
            if all(points[b][k] <= points[a][k] for k in axes) and \
               any(points[b][k] < points[a][k] for k in axes):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl")
    ap.add_argument("--caps", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--min-notes", type=int, default=12)
    args = ap.parse_args()

    from output.notation_score import build_routed_score

    with open(args.pkl, "rb") as fh:
        d = pickle.load(fh)
    name = os.path.splitext(os.path.basename(args.pkl))[0]
    ann = d["annotations"]

    src_all = consolidated_source(d)
    print(f"{name}: {len(src_all)} consolidated source events")

    # regions from `section`, same definition the rebuilt probe uses
    seen = {}
    for n in d["all_notes"]:
        s = ann.latest_value(n.id, "section")
        if isinstance(s, dict):
            seen.setdefault((s["label"], s["instance"]), s)
    regions = sorted(seen.values(), key=lambda x: x["start_ms"])
    if not regions:
        raise SystemExit("no modern `section` annotations - re-run this song")
    print(f"regions: {len(regions)}")

    cand_by_cap = {}
    for cap in args.caps:
        score = build_routed_score(
            d["all_notes"], d["annotations"], tempo_bpm=d["tempo_bpm"],
            time_signature=d["time_signature"], key=d["key"],
            use_consolidation=True, ratio_family=d.get("ratio_family"),
            max_voices=cap)
        cand_by_cap[cap] = notated_events(score)
        print(f"  cap {cap}: {len(cand_by_cap[cap])} notated events", flush=True)

    print(f"\n{'region':10s} {'cap':>3s} {'n':>5s} " +
          " ".join(f"{k:>9s}" for k in DIMS) + "   front")
    print("-" * 86)
    front_counts = defaultdict(int)
    scored = 0
    for r in regions:
        s0, s1 = r["start_ms"], r["end_ms"]
        src = [x for x in src_all if s0 <= x[0] < s1]
        if len(src) < args.min_notes:
            continue
        pts = {}
        for cap in args.caps:
            cand = [x for x in cand_by_cap[cap] if s0 <= x[0] < s1]
            pts[cap] = distortion(src, cand)
        front = pareto_front(pts, ("duration", "overlap", "dropped"))
        for c in front:
            front_counts[c] += 1
        scored += 1
        tag = f"{r['label']}#{r['instance']}"
        for cap in args.caps:
            mark = "  <-" if cap in front else ""
            print(f"{tag:10s} {cap:3d} {len(src):5d} " +
                  " ".join(f"{pts[cap][k]:9.3f}" for k in DIMS) + mark)
        print()

    print(f"=== PARETO FRONT MEMBERSHIP over {scored} regions "
          f"(axes: duration, overlap, dropped) ===")
    for cap in args.caps:
        print(f"  cap {cap}: on the frontier in {front_counts[cap]:3d} regions "
              f"({100*front_counts[cap]/max(scored,1):5.1f}%)")
    print("\nREAD THIS AS: a cap that is NEVER on the frontier is dominated - it")
    print("costs information without buying anything. A frontier that always")
    print("contains exactly one cap means the representation choice is global,")
    print("not regional, and the local-budget direction is dead.")


if __name__ == "__main__":
    main()
