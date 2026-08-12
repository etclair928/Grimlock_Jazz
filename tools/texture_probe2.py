#!/usr/bin/env python3
# =================================================================
# TOOL: tools/voice_edge_probe (tools/texture_probe2.py)
# RESEARCH ONLY - influences nothing in production.
#
# THE QUESTION. Voice assignment is currently a greedy monophonic streamer:
# take the nearest free voice by recent pitch, commit, never reconsider. The
# proposal on the table is to replace that with a PAIRWISE RELATIONSHIP model -
# score every plausible (note_i -> note_j) continuation with several witnesses,
# then find coherent paths through the resulting graph.
#
# Before building any of that, one thing has to be true: the witnesses must
# actually DISCRIMINATE. So this probe scores candidate edges and asks whether
# pairs that are genuinely part of the same musical line score higher than
# pairs that are not - per witness, separately, with no hand-chosen weights.
# Weights come after discrimination is demonstrated, not before.
#
# THE GROUND TRUTH PROBLEM, AND THE ANSWER ALREADY IN THE PICKLE. "Same voice"
# has no labels here. But `motif` annotations do exist (235 on Federal Blvd):
# repeated interval n-grams found AFTER consolidation. Consecutive notes inside
# one motif occurrence are, by construction, part of one musical figure - that
# is what made them a motif. So:
#
#   POSITIVE pairs  consecutive notes within a single motif occurrence
#   NEGATIVE pairs  notes equally close in time that are NOT in that motif
#
# It is imperfect (a motif is not a voice) but it is REAL, already computed, and
# independent of pitch proximity - which matters, because pitch proximity is the
# hypothesis under test and cannot be allowed to define its own success.
#
# WHAT WOULD FALSIFY THE GRAPH IDEA: if pitch/register witnesses separate the
# two sets and nothing else does, the graph is just the greedy streamer with
# extra steps. The interesting outcome is a witness that discriminates and that
# pitch proximity CANNOT see - rhythm, motif continuation, duration
# compatibility - because that is the information the current voicer throws away.
#
#   python tools/texture_probe2.py <pkl> [--max-gap-beats 2] [--verbose]
# =================================================================

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

WITNESSES = ("pitch_prox", "register", "ioi_regular", "metric_pos",
             "duration_compat", "overlap", "family_same", "stem_same",
             "motif_cont", "section_same")


@dataclass
class Ev:
    id: str
    start_ms: float
    end_ms: float
    pitch: int
    family: str = "unknown"
    stem: str = "?"
    motif: Optional[int] = None
    motif_occ: Optional[int] = None
    section: Optional[str] = None


def load(pkl_path: str) -> Tuple[List[Ev], float]:
    from core import StemType
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    ann = d["annotations"]
    beat_ms = 60000.0 / float(d["tempo_bpm"])

    evs: List[Ev] = []
    for n in d["all_notes"]:
        if n.stem == StemType.DRUMS:
            continue
        c = ann.latest_value(n.id, "consolidation") or {}
        if c.get("role") == "absorbed":
            continue          # fragments are not events (36% of notes on Federal)
        end = n.end_ms
        if c.get("role") == "primary" and c.get("end_ms") is not None:
            end = max(end, c["end_ms"])
        sec = ann.latest_value(n.id, "section")
        evs.append(Ev(
            id=n.id, start_ms=n.start_ms, end_ms=end, pitch=n.pitch,
            family=ann.latest_value(n.id, "instrument_family") or "unknown",
            stem=getattr(n.stem, "value", str(n.stem)),
            motif=ann.latest_value(n.id, "motif"),
            section=(sec.get("label") if isinstance(sec, dict) else sec)))
    evs.sort(key=lambda e: (e.start_ms, e.pitch))

    # split each motif id into OCCURRENCES: notes sharing a motif id but
    # separated by a long gap are different statements of the same figure.
    by_motif: Dict[int, List[Ev]] = defaultdict(list)
    for e in evs:
        if e.motif is not None:
            by_motif[e.motif].append(e)
    for mid, group in by_motif.items():
        group.sort(key=lambda e: e.start_ms)
        occ = 0
        for prev, cur in zip([None] + group[:-1], group):
            if prev is not None and cur.start_ms - prev.start_ms > 4 * beat_ms:
                occ += 1
            cur.motif_occ = occ
    return evs, beat_ms


# ---------------------------------------------------------------------------
# witnesses - each returns [0,1], higher = "these two belong together"
# ---------------------------------------------------------------------------

def edge_scores(a: Ev, b: Ev, beat_ms: float, ctx: Dict) -> Dict[str, float]:
    dt = b.start_ms - a.start_ms
    dp = abs(b.pitch - a.pitch)

    s = {}
    s["pitch_prox"] = float(np.exp(-dp / 5.0))
    s["register"] = float(np.exp(-abs(b.pitch - a.pitch) / 12.0))

    # rhythmic: is this IOI typical of the local pulse, and on a metric position?
    ioi_beats = dt / beat_ms if beat_ms else 0.0
    nearest = min((1, 0.5, 0.25, 2, 3, 4, 1.5, 0.75), key=lambda g: abs(ioi_beats - g))
    s["ioi_regular"] = float(np.exp(-abs(ioi_beats - nearest) * 4.0))
    phase_a = (a.start_ms % beat_ms) / beat_ms if beat_ms else 0.0
    phase_b = (b.start_ms % beat_ms) / beat_ms if beat_ms else 0.0
    d_phase = min(abs(phase_a - phase_b), 1 - abs(phase_a - phase_b))
    s["metric_pos"] = float(1.0 - 2 * d_phase)

    da, db = max(a.end_ms - a.start_ms, 1.0), max(b.end_ms - b.start_ms, 1.0)
    s["duration_compat"] = float(min(da, db) / max(da, db))

    # overlap: a monophonic line's notes should NOT sound over each other much
    ov = max(0.0, a.end_ms - b.start_ms)
    s["overlap"] = float(np.clip(1.0 - ov / da, 0, 1))

    s["family_same"] = 1.0 if a.family == b.family else 0.0
    s["stem_same"] = 1.0 if a.stem == b.stem else 0.0
    s["motif_cont"] = 1.0 if (a.motif is not None and a.motif == b.motif
                              and a.motif_occ == b.motif_occ) else 0.0
    s["section_same"] = 1.0 if a.section == b.section else 0.0
    return s


def build_pairs(evs: List[Ev], beat_ms: float, max_gap_beats: float,
                candidates: int = 8):
    """POSITIVE: consecutive notes inside one motif occurrence.
    NEGATIVE: other notes starting within the same time window."""
    max_gap = max_gap_beats * beat_ms
    starts = np.array([e.start_ms for e in evs])
    pos, neg = [], []
    by_occ: Dict[Tuple[int, int], List[Ev]] = defaultdict(list)
    for e in evs:
        if e.motif is not None and e.motif_occ is not None:
            by_occ[(e.motif, e.motif_occ)].append(e)

    positive_ids = set()
    for group in by_occ.values():
        group.sort(key=lambda e: e.start_ms)
        for a, b in zip(group, group[1:]):
            if 0 < b.start_ms - a.start_ms <= max_gap:
                pos.append((a, b))
                positive_ids.add((a.id, b.id))

    for a, b in pos:
        i = int(np.searchsorted(starts, a.start_ms))
        taken = 0
        for j in range(i + 1, min(i + 1 + candidates * 3, len(evs))):
            c = evs[j]
            if c.id == b.id or c.id == a.id:
                continue
            if not (0 < c.start_ms - a.start_ms <= max_gap):
                continue
            if (a.id, c.id) in positive_ids:
                continue
            neg.append((a, c))
            taken += 1
            if taken >= 2:
                break
    return pos, neg


def auc(pos_vals: Sequence[float], neg_vals: Sequence[float]) -> float:
    """P(random positive scores above random negative). 0.5 = no information."""
    if not pos_vals or not neg_vals:
        return float("nan")
    p = np.asarray(pos_vals); n = np.asarray(neg_vals)
    allv = np.concatenate([p, n])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties, else discrete witnesses look artificially strong
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:len(p)].sum()
    return float((r_pos - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl")
    ap.add_argument("--max-gap-beats", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    evs, beat_ms = load(args.pkl)
    name = os.path.basename(args.pkl)
    n_motif = len({(e.motif, e.motif_occ) for e in evs if e.motif is not None})
    print(f"{name}   events={len(evs)}  beat={beat_ms:.0f}ms  "
          f"motif occurrences={n_motif}")

    pos, neg = build_pairs(evs, beat_ms, args.max_gap_beats)
    print(f"pairs: {len(pos)} positive (within a motif occurrence), "
          f"{len(neg)} negative (same time window, different figure)")
    if len(pos) < 15 or len(neg) < 15:
        raise SystemExit("too few pairs to measure - this song has little motif "
                         "coverage; try another.")

    ctx: Dict = {}
    P = [edge_scores(a, b, beat_ms, ctx) for a, b in pos]
    N = [edge_scores(a, b, beat_ms, ctx) for a, b in neg]

    print(f"\n=== PER-WITNESS DISCRIMINATION (AUC; 0.50 = no information) ===")
    print(f"{'witness':18s} {'pos mean':>9s} {'neg mean':>9s} {'AUC':>7s}   verdict")
    print("-" * 62)
    results = {}
    for w in WITNESSES:
        pv = [x[w] for x in P]; nv = [x[w] for x in N]
        a = auc(pv, nv)
        results[w] = a
        if np.isnan(a):
            verdict = "n/a"
        elif a >= 0.65:
            verdict = "DISCRIMINATES"
        elif a >= 0.55:
            verdict = "weak"
        elif a <= 0.45:
            verdict = "INVERTED"
        else:
            verdict = "no information"
        print(f"{w:18s} {np.mean(pv):9.3f} {np.mean(nv):9.3f} {a:7.3f}   {verdict}")

    print("\n=== WHAT THIS DECIDES ===")
    strong = [w for w, a in results.items() if not np.isnan(a) and a >= 0.65]
    beyond_pitch = [w for w in strong if w not in ("pitch_prox", "register")]
    print(f"  discriminating witnesses: {strong or 'NONE'}")
    if not strong:
        print("  -> no witness separates same-figure pairs from neighbours. The")
        print("     graph would have nothing to reason over; do not build it.")
    elif not beyond_pitch:
        print("  -> ONLY pitch/register discriminate. A pairwise graph would be")
        print("     the greedy streamer with extra steps - the current voicer")
        print("     already uses exactly this information. Not worth building.")
    else:
        print(f"  -> {beyond_pitch} discriminate WITHOUT being pitch proximity.")
        print("     That is information the current voicer discards, and is the")
        print("     case for a relationship model rather than a nearest-pitch one.")

    if args.verbose:
        print("\n  sample positive edges:")
        for (a, b), s in list(zip(pos, P))[:6]:
            print(f"    {a.pitch:3d}->{b.pitch:3d} dt={b.start_ms-a.start_ms:7.0f}ms  " +
                  " ".join(f"{k}={s[k]:.2f}" for k in ("pitch_prox", "ioi_regular",
                                                       "duration_compat", "overlap")))
        print("  sample negative edges:")
        for (a, b), s in list(zip(neg, N))[:6]:
            print(f"    {a.pitch:3d}->{b.pitch:3d} dt={b.start_ms-a.start_ms:7.0f}ms  " +
                  " ".join(f"{k}={s[k]:.2f}" for k in ("pitch_prox", "ioi_regular",
                                                       "duration_compat", "overlap")))


if __name__ == "__main__":
    main()
