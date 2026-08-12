#!/usr/bin/env python3
# =================================================================
# TOOL: tools/region_voice_budget.py
# EXPERIMENT 4 (read-only). Is the best voice cap a property of the SONG, or
# of the REGION?
#
# WHY THIS BEFORE ANY CLASSIFIER. The texture probe measured that local
# organisation is temporally persistent (Federal Blvd churn 0.244 vs a 0.518
# random-order baseline) but that its labels were generated from onset density
# alone - pitch-shuffling changed nothing. The tempting next move is to add
# pitch features and make the classifier smarter. That skips a question:
#
#   Does a per-region voice budget have anything to predict in the first place?
#
# If cap 2 wins in every region of every song, then `voice_budget(t)` is a
# solution to a problem that does not exist, and no amount of feature
# engineering will change that. So: DO NOT predict the answer. Export the same
# notes at caps 1-4, score each region on the emitted page, and look at whether
# the winner actually varies.
#
# This mirrors the discipline that has repeatedly paid: measure the effect
# before building the mechanism (§XIX).
#
#   python tools/region_voice_budget.py <pkl> [--region-measures 4] [--caps 1 2 3 4]
# =================================================================

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

_STEP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def per_measure_stats(path: str) -> Dict[int, Dict[str, float]]:
    """notes / rests / top-line voice changes, per measure index, summed over
    every non-drum part. Same definitions as output/readability.py, bucketed by
    measure so regions can be compared."""
    root = ET.parse(path).getroot()
    names = {sp.get("id"): (sp.find("part-name").text or "?")
             for sp in root.findall(".//score-part")}
    acc: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"notes": 0.0, "rests": 0.0, "top_onsets": 0.0, "top_stable": 0.0,
                 "chord": 0.0})
    for part in root.findall(".//part"):
        nm = (names.get(part.get("id"), "") or "").lower()
        if "drum" in nm or "perc" in nm:
            continue
        for mi, m in enumerate(part.findall("measure")):
            cursor: Dict[str, int] = {}
            last_onset: Dict[str, int] = {}
            by_onset: Dict[int, Tuple[int, str]] = {}
            for n in m.findall("note"):
                if n.find("grace") is not None:
                    continue
                v_el = n.find("voice")
                key = v_el.text if v_el is not None else "1"
                dur = int(n.find("duration").text) if n.find("duration") is not None else 0
                if n.find("chord") is not None:
                    onset = last_onset.get(key, cursor.get(key, 0))
                    acc[mi]["chord"] += 1
                else:
                    onset = cursor.get(key, 0)
                    last_onset[key] = onset
                    cursor[key] = onset + dur
                if n.find("rest") is not None:
                    acc[mi]["rests"] += 1
                    continue
                p_el = n.find("pitch")
                if p_el is None:
                    continue
                acc[mi]["notes"] += 1
                octave = int(p_el.find("octave").text)
                alter = int(p_el.find("alter").text) if p_el.find("alter") is not None else 0
                midi = (octave + 1) * 12 + _STEP[p_el.find("step").text] + alter
                cur = by_onset.get(onset)
                if cur is None or midi > cur[0]:
                    by_onset[onset] = (midi, key)
            if len(set(v.text for v in m.iter("voice") if v.text)) > 1:
                prev = None
                for onset in sorted(by_onset):
                    _, key = by_onset[onset]
                    acc[mi]["top_onsets"] += 1
                    if prev is not None and key == prev:
                        acc[mi]["top_stable"] += 1
                    prev = key
    return acc


def regionise(stats: Dict[int, Dict[str, float]], size: int) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"notes": 0.0, "rests": 0.0, "top_onsets": 0.0, "top_stable": 0.0, "chord": 0.0})
    for mi, s in stats.items():
        r = mi // size
        for k, v in s.items():
            out[r][k] += v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl")
    ap.add_argument("--caps", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--region-measures", type=int, default=4)
    ap.add_argument("--min-notes", type=int, default=12,
                    help="regions thinner than this are skipped as too noisy to rank")
    args = ap.parse_args()

    from output.notation_score import build_routed_score
    from output.musicxml_exporter import export_musicxml

    with open(args.pkl, "rb") as fh:
        d = pickle.load(fh)
    name = os.path.splitext(os.path.basename(args.pkl))[0]
    out_dir = os.path.join(JAZZ, "transcriptions", "region_budget")
    os.makedirs(out_dir, exist_ok=True)

    per_cap: Dict[int, Dict[int, Dict[str, float]]] = {}
    for cap in args.caps:
        p = os.path.join(out_dir, f"{name}_cap{cap}.musicxml")
        if not os.path.exists(p):
            score = build_routed_score(
                d["all_notes"], d["annotations"], tempo_bpm=d["tempo_bpm"],
                time_signature=d["time_signature"], key=d["key"],
                use_consolidation=True, ratio_family=d.get("ratio_family"),
                max_voices=cap)
            export_musicxml(score, p)
        per_cap[cap] = regionise(per_measure_stats(p), args.region_measures)
        print(f"  cap {cap}: exported/scored", flush=True)

    regions = sorted(set.intersection(*[set(v) for v in per_cap.values()]))
    print(f"\n{name}   {len(regions)} regions of {args.region_measures} measures, "
          f"caps {args.caps}")

    hdr = f"{'region':>7s} {'notes':>6s} " + " ".join(f"{'rest@'+str(c):>9s}" for c in args.caps) \
          + " " + " ".join(f"{'stab@'+str(c):>9s}" for c in args.caps) + "  winner"
    print(hdr); print("-" * len(hdr))

    win_rest, win_stab, rows = Counter(), Counter(), 0
    for r in regions:
        notes = per_cap[args.caps[0]][r]["notes"]
        if notes < args.min_notes:
            continue
        rest, stab = {}, {}
        for c in args.caps:
            s = per_cap[c][r]
            rest[c] = s["rests"] / s["notes"] if s["notes"] else float("nan")
            stab[c] = (s["top_stable"] / (s["top_onsets"] - 1)
                       if s["top_onsets"] > 1 else float("nan"))
        br = min((c for c in args.caps if not np.isnan(rest[c])), key=lambda c: rest[c])
        valid = [c for c in args.caps if not np.isnan(stab[c])]
        bs = max(valid, key=lambda c: stab[c]) if valid else None
        win_rest[br] += 1
        if bs is not None:
            win_stab[bs] += 1
        rows += 1
        if rows <= 30:
            print(f"{r:7d} {notes:6.0f} "
                  + " ".join(f"{rest[c]:9.3f}" for c in args.caps) + " "
                  + " ".join(f"{stab[c]:9.3f}" for c in args.caps)
                  + f"  rest->{br} stab->{bs}")

    print(f"\n=== WHICH CAP WINS, PER REGION ({rows} scored regions) ===")
    print("  by rest/note:        " + "  ".join(
        f"cap{c}: {win_rest[c]:3d} ({100*win_rest[c]/max(rows,1):4.1f}%)" for c in args.caps))
    tot = sum(win_stab.values())
    print("  by top-line stability:" + "  ".join(
        f"cap{c}: {win_stab[c]:3d} ({100*win_stab[c]/max(tot,1):4.1f}%)" for c in args.caps))
    print("\nREAD THIS AS: if one cap wins nearly every region on both axes, a")
    print("per-region voice budget has nothing to predict and the local-texture")
    print("direction should be dropped. Variation is what justifies continuing.")


if __name__ == "__main__":
    main()
