#!/usr/bin/env python3
# =================================================================
# TOOL: tools/voice_cap_sweep.py
# Sweeps max_voices per staff across every saved pipeline intermediate and
# scores the emitted page, so a layout default is chosen on the whole test
# library rather than on the one song that happened to be open.
#
# WHY THIS EXISTS. Cap 2 was REJECTED once already, on `mean_voice_jump`
# alone - the metric later proven blind to melodic coherence (it read 4.49,
# "passing", while the top line changed voice on 52% of onsets). Re-deciding
# it needs the coherence axes AND more than two songs.
#
#   python tools/voice_cap_sweep.py                 # every pkl under transcriptions/
#   python tools/voice_cap_sweep.py --caps 2 3 4
# =================================================================

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caps", nargs="*", type=int, default=[2, 3])
    ap.add_argument("--out", default=os.path.join(JAZZ, "transcriptions", "cap_sweep"))
    ap.add_argument("--pkl-root", default=os.path.join(JAZZ, "transcriptions"))
    args = ap.parse_args()

    from output.notation_score import build_routed_score
    from output.musicxml_exporter import export_musicxml
    from output.readability import measure_readability

    os.makedirs(args.out, exist_ok=True)
    pkls = []
    for root, _dirs, files in os.walk(args.pkl_root):
        if os.path.abspath(root).startswith(os.path.abspath(args.out)):
            continue
        for f in files:
            if f.endswith(".pkl"):
                pkls.append(os.path.join(root, f))
    pkls.sort()

    hdr = (f"{'song':26s} {'cap':>3s} {'notes':>6s} {'rest':>6s} {'jump':>6s} "
           f"{'topstab':>8s} {'v1top':>6s} {'chord':>6s} {'ties':>5s} {'ovf':>4s}")
    print(hdr)
    print("-" * len(hdr))

    totals = {c: {"rest": [], "jump": [], "top": [], "v1": [], "chord": [], "ovf": 0}
              for c in args.caps}
    for p in pkls:
        name = os.path.splitext(os.path.basename(p))[0][:26]
        try:
            with open(p, "rb") as fh:
                d = pickle.load(fh)
        except Exception as e:
            print(f"{name:26s}   load failed: {e}")
            continue
        for cap in args.caps:
            try:
                t0 = time.time()
                score = build_routed_score(
                    d["all_notes"], d["annotations"], tempo_bpm=d["tempo_bpm"],
                    time_signature=d["time_signature"], key=d["key"],
                    use_consolidation=True, ratio_family=d.get("ratio_family"),
                    max_voices=cap)
                out = os.path.join(args.out, f"{name}_cap{cap}.musicxml")
                export_musicxml(score, out)
                r = measure_readability(out)
                totals[cap]["rest"].append(r.rest_ratio)
                totals[cap]["jump"].append(r.mean_voice_jump)
                totals[cap]["top"].append(r.top_line_stability)
                totals[cap]["v1"].append(r.voice1_is_top)
                totals[cap]["chord"].append(r.chord_share)
                totals[cap]["ovf"] += r.overfull_voice_measures
                print(f"{name:26s} {cap:3d} {r.notes:6d} {r.rest_ratio:6.3f} "
                      f"{r.mean_voice_jump:6.2f} {r.top_line_stability:8.3f} "
                      f"{r.voice1_is_top:6.3f} {r.chord_share:6.3f} {r.ties:5d} "
                      f"{r.overfull_voice_measures:4d}   ({time.time()-t0:.0f}s)",
                      flush=True)
            except Exception as e:
                print(f"{name:26s} {cap:3d}   FAILED: {e}", flush=True)
        print(flush=True)

    print("=== LIBRARY MEANS ===")
    print(f"{'cap':>3s} {'rest':>7s} {'jump':>7s} {'topstab':>8s} {'v1top':>7s} "
          f"{'chord':>7s} {'overfull':>9s}")
    for cap in args.caps:
        t = totals[cap]
        if not t["rest"]:
            continue
        m = lambda k: sum(t[k]) / len(t[k])
        print(f"{cap:3d} {m('rest'):7.3f} {m('jump'):7.2f} {m('top'):8.3f} "
              f"{m('v1'):7.3f} {m('chord'):7.3f} {t['ovf']:9d}")


if __name__ == "__main__":
    main()
