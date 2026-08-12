#!/usr/bin/env python3
# =================================================================
# TOOL: tools/texture_probe.py
# READ-ONLY PROBE. Asks whether the local organisation of a transcription is
# a real, persistent, PREDICTABLE property - the precondition for ever letting
# a per-region voice budget replace the global max_voices. Changes nothing,
# wired into nothing.
#
# VERSION 2 (2026-08-10). Version 1 was written from scratch: it invented
# fixed-size windows and classified them from onset density alone. Two things
# were wrong with that, and both were found by measurement:
#
#   1. Pitch-shuffling the input changed NOTHING - not one label, not one
#      percentage point. Its features (`mean_stack`, `sync_share`) were
#      computed purely from onsets, so it was a simultaneity-density detector
#      wearing the word "texture". It could not tell a C-major triad from three
#      unrelated pitches struck together.
#   2. It read `start_ms`, `end_ms` and `pitch` and NOTHING ELSE - while the
#      same pickle carries TWENTY annotation streams and ~78,000 annotations
#      produced by the pipeline it was analysing. Most damning: it invented
#      4-measure windows when `section` (9,713 annotations) already holds real
#      form boundaries from detect_form, with labels AND instance identity.
#      That is §XVIII.2's "we compute more evidence than we consume" with this
#      tool as the offender.
#
# So version 2 is a CONSUMER. Every input below already exists in the pickle:
#
#   section            -> regions are real form boundaries, not arbitrary windows
#   repeat_group       -> repeats give REPLICATION: the same material appears
#                         several times, so a texture reading can be checked
#                         against itself rather than only against a null
#   instrument_family  -> the semantic layer density alone cannot see;
#                         "melody + accompaniment" is largely bright_lead over
#                         warm_sustained, and this is what should make a
#                         pitch/timbre shuffle finally bite
#   notation_timing    -> simultaneity measured on SYMBOLIC time. Version 1 used
#                         raw performance time, where 63.6% of consecutive notes
#                         overlap by ~104ms, so two notes of one chord could
#                         miss the window while two unrelated notes fell inside
#   consolidation      -> 'absorbed' notes are fragments of another note. Density
#                         inflated by fragmentation is not musical density, and
#                         version 1 could not tell the two apart
#   legitimacy_verdict -> exclude notes the harmonic audit calls illegitimate
#                         before computing anything
#   acoustic_activity  -> separates "sparse because silent" from "sparse because
#                         sustained", which version 1 collapsed into one label
#
# THE BAR IT MUST CLEAR, unchanged from version 1 and still not cleared:
#   * churn on real music must be clearly below the random-order baseline
#     (1 - sum p^2), AND
#   * a PITCH/FAMILY shuffle must change the answer. If it does not, the probe
#     is still measuring density and should be deleted rather than tuned (§XIX).
#
#   python tools/texture_probe.py <pkl> [--verbose]
# =================================================================

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

SYNC_WINDOW_MS = 60.0
MIN_NOTES_PER_REGION = 12
CHORD_INTERVALS = {3, 4, 5, 7, 8, 9}     # thirds/fourths/fifths/sixths mod 12
CLUSTER_INTERVALS = {1, 2}               # seconds - a cluster, not a chord

LABELS = ("CHORDAL", "MELODY_ACCOMP", "COUNTERPOINT", "ARPEGGIO",
          "PEDAL_PLUS_ACTIVITY", "MONOPHONIC", "SPARSE")


@dataclass
class PNote:
    """A note plus the evidence the pipeline already recorded about it."""
    start_ms: float
    end_ms: float
    pitch: int
    family: str = "unknown"
    absorbed: bool = False          # consolidation says this is a fragment
    void: float = float("nan")      # acoustic_activity rhythmic_void_probability
    resonance: float = float("nan")


@dataclass
class Region:
    label: str            # section label from detect_form
    instance: int
    start_ms: float
    end_ms: float
    notes: List[PNote] = field(default_factory=list)
    texture: str = "SPARSE"
    confidence: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# loading - everything here comes from annotations the pipeline already writes
# ---------------------------------------------------------------------------

def load(pkl_path: str, use_notation_timing: bool = True,
         drop_absorbed: bool = True, drop_illegitimate: bool = True):
    from core import StemType
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    ann = d["annotations"]

    notes: List[PNote] = []
    n_absorbed = n_illegit = 0
    for n in d["all_notes"]:
        if n.stem == StemType.DRUMS:
            continue
        cons = ann.latest_value(n.id, "consolidation") or {}
        absorbed = cons.get("role") == "absorbed"
        if absorbed:
            n_absorbed += 1
            if drop_absorbed:
                continue
        leg = ann.latest_value(n.id, "legitimacy_verdict") or {}
        if leg.get("verdict") not in (None, "legitimate"):
            n_illegit += 1
            if drop_illegitimate:
                continue
        start, end = n.start_ms, n.end_ms
        if use_notation_timing:
            nt = ann.latest_value(n.id, "notation_timing") or ann.latest_value(n.id, "quantization")
            if nt and nt.get("start_ms") is not None:
                start, end = nt["start_ms"], nt.get("end_ms", n.end_ms)
        act = ann.latest_value(n.id, "acoustic_activity") or {}
        notes.append(PNote(
            start_ms=start, end_ms=end, pitch=n.pitch,
            family=ann.latest_value(n.id, "instrument_family") or "unknown",
            absorbed=absorbed,
            void=act.get("rhythmic_void_probability", float("nan")),
            resonance=act.get("resonance_probability", float("nan"))))

    # regions from detect_form's section annotations
    # `section` gained {label,index,instance,start_ms,end_ms} when check/check.py
    # was fixed to carry instance identity. Pickles written before that store a
    # bare label string, which has no boundaries and no way to tell one
    # occurrence of "A" from another - so those runs cannot supply regions and
    # must be re-run rather than guessed at.
    seen: Dict[Tuple[str, int], Dict] = {}
    legacy = 0
    for n in d["all_notes"]:
        s = ann.latest_value(n.id, "section")
        if not s:
            continue
        if not isinstance(s, dict):
            legacy += 1
            continue
        seen.setdefault((s["label"], s["instance"]), s)
    if legacy and not seen:
        raise SystemExit(
            f"{os.path.basename(pkl_path)}: `section` annotations are the LEGACY "
            f"bare-label format ({legacy} notes) with no boundaries or instance "
            f"identity. Re-run the pipeline on this song to use it here.")
    regions = [Region(label=s["label"], instance=s["instance"],
                      start_ms=s["start_ms"], end_ms=s["end_ms"])
               for s in sorted(seen.values(), key=lambda x: x["start_ms"])]
    for r in regions:
        r.notes = [x for x in notes if r.start_ms <= x.start_ms < r.end_ms]
    return d, notes, regions, {"absorbed": n_absorbed, "illegitimate": n_illegit}


# ---------------------------------------------------------------------------
# features - now pitch- and family-aware, so a shuffle can actually bite
# ---------------------------------------------------------------------------

def _stacks(notes: Sequence[PNote]) -> List[List[PNote]]:
    out: List[List[PNote]] = []
    for n in sorted(notes, key=lambda x: x.start_ms):
        if out and n.start_ms - out[-1][0].start_ms <= SYNC_WINDOW_MS:
            out[-1].append(n)
        else:
            out.append([n])
    return out


def chord_likeness(stack: Sequence[PNote]) -> float:
    """Do the pitches in this simultaneity form a CHORD (thirds/fourths/fifths)
    or a CLUSTER? Version 1 could not ask this at all - it only counted how many
    notes arrived together, which is why shuffling pitch changed nothing."""
    if len(stack) < 2:
        return 0.0
    ivs = []
    ps = sorted(n.pitch for n in stack)
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            ivs.append((ps[j] - ps[i]) % 12)
    if not ivs:
        return 0.0
    good = sum(1 for v in ivs if v in CHORD_INTERVALS)
    bad = sum(1 for v in ivs if v in CLUSTER_INTERVALS)
    return (good - bad) / len(ivs)


VECTOR_DIMS = ("chordality", "melodic_motion", "accomp_regularity",
               "independence", "sustain", "ostinato")


def texture_vector(region: Region) -> Optional[np.ndarray]:
    """Describe a region as SIX CONTINUOUS QUANTITIES rather than one label.

    WHY NOT A LABEL. Real music is not exclusively chordal or exclusively
    contrapuntal, and a hard argmax over near-equal categories flips on noise.
    Measured symptom: Nov19's section A - the SAME material, four occurrences -
    received four different labels, which reads as "no structure" when it may
    only mean "a mixed region near a category boundary". A vector cannot flip;
    two mixed regions simply land near each other.

    Each dimension is in [0,1] and measures one of the four organisational
    axes: vertical (chordality), horizontal (melodic_motion), rhythmic-role
    (accomp_regularity, independence, ostinato), temporal (sustain)."""
    ns = region.notes
    if len(ns) < MIN_NOTES_PER_REGION:
        return None
    span = max(region.end_ms - region.start_ms, 1.0)
    stacks = _stacks(ns)
    multi = [s for s in stacks if len(s) > 1]

    # VERTICAL: do simultaneities spell chords, and how often do they occur?
    chordness = float(np.mean([chord_likeness(s) for s in multi])) if multi else 0.0
    sync_share = sum(len(s) for s in multi) / len(ns)
    chordality = float(np.clip(sync_share * (0.5 + 0.5 * max(chordness, 0.0)), 0, 1))

    pitches = np.array([n.pitch for n in ns], dtype=float)
    split = float(np.median(pitches))
    top = sorted([n for n in ns if n.pitch > split], key=lambda x: x.start_ms)
    bot = sorted([n for n in ns if n.pitch <= split], key=lambda x: x.start_ms)

    # HORIZONTAL: does the upper band move like a line (stepwise, connected)?
    if len(top) > 2:
        d = np.abs(np.diff([n.pitch for n in top]))
        melodic_motion = float(np.mean((d > 0) & (d <= 2)))
    else:
        melodic_motion = 0.0

    def iois(seq):
        return np.diff([n.start_ms for n in seq]) if len(seq) > 2 else np.array([])

    def regularity(seq) -> float:
        """Low spread of inter-onset intervals = a steady figure."""
        v = iois(seq)
        v = v[v > 1.0]
        if len(v) < 3:
            return 0.0
        return float(np.clip(1.0 - (v.std() / (v.mean() + 1e-9)), 0, 1))

    accomp_regularity = regularity(bot)

    # RHYTHMIC ROLE: do the two bands strike at DIFFERENT times? Independent
    # layers rarely coincide; a homophonic texture almost always does.
    if top and bot:
        bs = np.sort(np.array([n.start_ms for n in bot]))
        coinc = 0
        for n in top:
            i = int(np.searchsorted(bs, n.start_ms))
            near = [abs(bs[j] - n.start_ms) for j in (i - 1, i) if 0 <= j < len(bs)]
            if near and min(near) <= SYNC_WINDOW_MS:
                coinc += 1
        independence = float(1.0 - coinc / len(top))
    else:
        independence = 0.0

    # TEMPORAL: how much of each note's life overlaps LATER attacks? A pedal
    # under activity scores high; a chain of separate notes scores near zero.
    starts = np.sort(np.array([n.start_ms for n in ns]))
    ov = []
    for n in ns:
        dur = max(n.end_ms - n.start_ms, 1.0)
        later = starts[(starts > n.start_ms + SYNC_WINDOW_MS) & (starts < n.end_ms)]
        ov.append(min(1.0, len(later) * 0.25) if len(later) else 0.0)
    sustain = float(np.mean(ov)) if ov else 0.0

    # OSTINATO: does a rhythmic cell recur? Autocorrelation of the onset train.
    ost = 0.0
    if len(ns) > 8:
        grid = np.zeros(64)
        for n in ns:
            idx = int(((n.start_ms - region.start_ms) / span) * 63)
            if 0 <= idx < 64:
                grid[idx] += 1
        g = grid - grid.mean()
        if g.std() > 0:
            ac = np.correlate(g, g, mode="full")[64:]
            denom = float(np.dot(g, g)) + 1e-9
            ost = float(np.clip(ac.max() / denom, 0, 1)) if len(ac) else 0.0

    return np.array([chordality, melodic_motion, accomp_regularity,
                     independence, sustain, ost], dtype=float)


def classify(region: Region) -> None:
    ns = region.notes
    if len(ns) < MIN_NOTES_PER_REGION:
        region.texture, region.confidence = "SPARSE", 1.0
        region.features = {"n": float(len(ns))}
        return

    span = max(region.end_ms - region.start_ms, 1.0)
    stacks = _stacks(ns)
    mean_stack = float(np.mean([len(s) for s in stacks]))
    sync_share = float(sum(len(s) for s in stacks if len(s) > 1) / len(ns))
    chordness = float(np.mean([chord_likeness(s) for s in stacks if len(s) > 1])
                      if any(len(s) > 1 for s in stacks) else 0.0)

    pitches = np.array([n.pitch for n in ns], dtype=float)
    durs = np.array([n.end_ms - n.start_ms for n in ns], dtype=float)

    # FAMILY STRATIFICATION - the semantic layer density cannot see.
    fam = Counter(n.family for n in ns)
    n_fam = len([f for f, c in fam.items() if c >= 3])
    fam_top = fam.most_common(1)[0][1] / len(ns)

    # register bands, and whether the top band actually MOVES (a line) or
    # repeats (an accompaniment figure)
    split = float(np.median(pitches))
    top = [n for n in ns if n.pitch > split]
    bot = [n for n in ns if n.pitch <= split]
    top_stacks = len(_stacks(top)) if top else 0
    bot_stacks = len(_stacks(bot)) if bot else 0
    activity_ratio = top_stacks / max(bot_stacks, 1)
    top_seq = [n.pitch for n in sorted(top, key=lambda x: x.start_ms)]
    top_mobility = (float(np.mean(np.abs(np.diff(top_seq)))) if len(top_seq) > 2 else 0.0)
    stepwise = (float(np.mean([1.0 for a, b in zip(top_seq, top_seq[1:])
                               if 0 < abs(b - a) <= 2]) if len(top_seq) > 2 else 0.0))

    # SUSTAIN vs SILENCE, from acoustic_activity - version 1 called both "sparse"
    voids = np.array([n.void for n in ns], dtype=float)
    mean_void = float(np.nanmean(voids)) if not np.all(np.isnan(voids)) else float("nan")

    pedal = bool((durs >= 0.60 * span).any()) and len(stacks) >= 4

    region.features = {
        "mean_stack": mean_stack, "sync_share": sync_share, "chordness": chordness,
        "n_families": float(n_fam), "family_dominance": fam_top,
        "activity_ratio": activity_ratio, "top_mobility": top_mobility,
        "stepwise": stepwise, "mean_void": mean_void, "pedal": float(pedal),
        "n": float(len(ns)),
    }

    if mean_stack < 1.2:
        region.texture, region.confidence = "MONOPHONIC", 0.6
        return
    # A real chord: notes arrive together AND spell chord intervals.
    if sync_share >= 0.55 and mean_stack >= 2.2 and chordness > 0.15:
        region.texture, region.confidence = "CHORDAL", min(1.0, sync_share * (0.5 + chordness))
        return
    if pedal:
        region.texture, region.confidence = "PEDAL_PLUS_ACTIVITY", 0.7
        return
    # Melody + accompaniment: distinct families OR a mobile stepwise top band
    # over a less active bottom band.
    if (n_fam >= 2 or stepwise > 0.35) and activity_ratio >= 1.3:
        region.texture = "MELODY_ACCOMP"
        region.confidence = min(1.0, 0.4 + 0.3 * (n_fam >= 2) + 0.3 * stepwise)
        return
    if sync_share <= 0.35 and mean_stack >= 1.2:
        region.texture, region.confidence = "COUNTERPOINT", 1.0 - sync_share
        return
    region.texture, region.confidence = "MELODY_ACCOMP", 0.35


# ---------------------------------------------------------------------------
# nulls
# ---------------------------------------------------------------------------

def expected_churn(labels: Sequence[str]) -> float:
    real = [l for l in labels if l != "SPARSE"]
    if len(real) < 2:
        return float("nan")
    c = Counter(real); n = len(real)
    return 1.0 - sum((v / n) ** 2 for v in c.values())


def churn(regions: Sequence[Region]) -> float:
    real = [r for r in regions if r.texture != "SPARSE"]
    if len(real) < 2:
        return float("nan")
    return sum(1 for a, b in zip(real, real[1:]) if a.texture != b.texture) / (len(real) - 1)


def shuffled_copy(regions: Sequence[Region], mode: str, rng) -> List[Region]:
    """Rebuild regions with one property destroyed. `pitch` and `family` are the
    ones version 1 could not respond to - if they still do nothing, the probe is
    STILL a density detector."""
    alln = [n for r in regions for n in r.notes]
    if mode == "pitch":
        vals = [n.pitch for n in alln]; rng.shuffle(vals)
    elif mode == "family":
        vals = [n.family for n in alln]; rng.shuffle(vals)
    elif mode == "time":
        lo = min(n.start_ms for n in alln); hi = max(n.start_ms for n in alln)
        vals = [rng.uniform(lo, hi) for _ in alln]
    out, i = [], 0
    for r in regions:
        r2 = Region(r.label, r.instance, r.start_ms, r.end_ms)
        for n in r.notes:
            if mode == "pitch":
                r2.notes.append(PNote(n.start_ms, n.end_ms, vals[i], n.family, n.absorbed, n.void, n.resonance))
            elif mode == "family":
                r2.notes.append(PNote(n.start_ms, n.end_ms, n.pitch, vals[i], n.absorbed, n.void, n.resonance))
            else:
                s = vals[i]
                r2.notes.append(PNote(s, s + (n.end_ms - n.start_ms), n.pitch, n.family, n.absorbed, n.void, n.resonance))
            i += 1
        classify(r2)
        out.append(r2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-absorbed", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    d, notes, regions, dropped = load(args.pkl, drop_absorbed=not args.keep_absorbed)
    name = os.path.basename(args.pkl)
    print(f"{name}   tempo={d['tempo_bpm']:.1f}  meter={d['time_signature']}")
    print(f"pitched notes kept: {len(notes)}   "
          f"(dropped {dropped['absorbed']} consolidation-absorbed fragments, "
          f"{dropped['illegitimate']} illegitimate)")
    print(f"regions from `section` annotations: {len(regions)}  "
          f"labels={sorted({r.label for r in regions})}")

    for r in regions:
        classify(r)

    counts = Counter(r.texture for r in regions)
    print(f"\n=== TEXTURE BY SECTION ===")
    for lab in LABELS:
        if counts.get(lab):
            print(f"  {lab:22s} {counts[lab]:3d}  ({100*counts[lab]/len(regions):5.1f}%)")
    if args.verbose:
        for r in regions:
            print(f"    {r.start_ms/1000:7.1f}-{r.end_ms/1000:7.1f}s  {r.label}#{r.instance}  "
                  f"{r.texture:22s} conf={r.confidence:.2f}  n={len(r.notes):4d}  "
                  f"stack={r.features.get('mean_stack',0):.2f} "
                  f"chord={r.features.get('chordness',0):+.2f} "
                  f"fam={r.features.get('n_families',0):.0f}")

    # ---- VECTOR REPLICATION: the label-free test -------------------------
    # A hard label flips on noise when a region sits near a category boundary.
    # MEASURED symptom: Nov19's section A - the SAME material, four occurrences -
    # got four different labels, which reads as "no structure" but may only mean
    # "a mixed region". A vector cannot flip; two mixed regions land near each
    # other. So the real replication test is DISTANCE between instances of the
    # same section, against different-section pairs as the baseline.
    vecs = {id(r): texture_vector(r) for r in regions}
    have = [r for r in regions if vecs[id(r)] is not None]
    if len(have) >= 4:
        print("\n=== TEXTURE VECTORS (label-free) ===")
        print("  " + "  ".join(f"{d[:9]:>9s}" for d in VECTOR_DIMS) + "     section")
        for r in have:
            print("  " + "  ".join(f"{x:9.3f}" for x in vecs[id(r)]) +
                  f"     {r.label}#{r.instance} ({len(r.notes)}n)")
        same, diff = [], []
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                dist = float(np.linalg.norm(vecs[id(a)] - vecs[id(b)]))
                (same if a.label == b.label else diff).append(dist)
        if same and diff:
            s_m, d_m = float(np.mean(same)), float(np.mean(diff))
            pooled = float(np.std(np.array(same + diff))) + 1e-9
            print(f"\n  mean distance SAME section:      {s_m:.3f}  (n={len(same)} pairs)")
            print(f"  mean distance DIFFERENT section: {d_m:.3f}  (n={len(diff)} pairs)")
            print(f"  separation: {d_m - s_m:+.3f}  ({(d_m - s_m)/pooled:+.2f} pooled SD)")
            print("  -> positive means the vector describes the MATERIAL, not the clock.")

        # PER-DIMENSION. The 6-vector separates far more weakly than the
        # categorical label does (0.12 SD vs 92.9%-against-52%), which is what
        # happens when only a couple of dimensions carry signal and Euclidean
        # distance lets the rest add noise. So ask each dimension separately:
        # do same-section repeats agree on THIS quantity? A dimension whose
        # separation is <= 0 is actively harming the vector and should be cut,
        # not kept for completeness.
        print("\n  per-dimension separation (different - same; >0 = informative):")
        for k, dim in enumerate(VECTOR_DIMS):
            s_d, d_d = [], []
            for i, a in enumerate(have):
                for b in have[i + 1:]:
                    gap = abs(vecs[id(a)][k] - vecs[id(b)][k])
                    (s_d if a.label == b.label else d_d).append(gap)
            if not s_d or not d_d:
                continue
            sd = float(np.std(np.array(s_d + d_d))) + 1e-9
            sep = float(np.mean(d_d) - np.mean(s_d))
            flag = "  <- informative" if sep / sd > 0.25 else (
                "  <- NOISE" if sep <= 0 else "")
            print(f"    {dim:20s} same={np.mean(s_d):.3f}  diff={np.mean(d_d):.3f}  "
                  f"sep={sep:+.3f} ({sep/sd:+.2f} SD){flag}")

        # REDUCED VECTOR. Measured per-dimension on two songs, `chordality` is
        # the ONLY dimension that separates same-section repeats on BOTH.
        # melodic_motion FLIPS SIGN between them (+0.60 SD / -0.48 SD), and
        # accomp_regularity, independence and sustain do nothing on at least
        # one. Carrying five uninformative dimensions dilutes the one that
        # works: chordality alone separates at 1.11 SD where the full six-vector
        # manages 0.12. More dimensions is not more information.
        for label_txt, dims in (("chordality only", ("chordality",)),
                                ("chordality+melodic", ("chordality", "melodic_motion"))):
            sel = [VECTOR_DIMS.index(x) for x in dims]
            s_r, d_r = [], []
            for i, a in enumerate(have):
                for b in have[i + 1:]:
                    dist = float(np.linalg.norm(vecs[id(a)][sel] - vecs[id(b)][sel]))
                    (s_r if a.label == b.label else d_r).append(dist)
            if s_r and d_r:
                sd = float(np.std(np.array(s_r + d_r))) + 1e-9
                sep = float(np.mean(d_r) - np.mean(s_r))
                print(f"  REDUCED [{label_txt:20s}] separation {sep:+.3f} "
                      f"({sep/sd:+.2f} pooled SD)")

    # REPLICATION: repeats of the same section label should agree with each other
    print(f"\n=== REPLICATION - categorical (kept for comparison) ===")
    by_label: Dict[str, List[Region]] = defaultdict(list)
    for r in regions:
        by_label[r.label].append(r)
    agree = total = 0
    for lab, rs in sorted(by_label.items()):
        if len(rs) < 2:
            continue
        c = Counter(r.texture for r in rs)
        top, n = c.most_common(1)[0]
        agree += n; total += len(rs)
        print(f"  {lab}: {len(rs)} instances -> {dict(c)}   "
              f"consistency {100*n/len(rs):5.1f}%")
    if total:
        print(f"  OVERALL section-instance consistency: {100*agree/total:.1f}%")
        print(f"  (chance for this label mix ~ {100*(1-expected_churn([r.texture for r in regions])):.1f}%)")

    # ---- NULL MODELS, scored on the VECTOR ------------------------------
    # Churn was the right statistic for fixed windows and is the WRONG one for
    # form-derived regions: a section boundary is BY DEFINITION where the music
    # changes, so high churn between adjacent sections is expected rather than a
    # defect. (On Federal Blvd real churn 0.571 sat ABOVE its 0.480 random
    # baseline for exactly this reason.) The statistic that survives the change
    # of region definition is same-vs-different SEPARATION - and if the vector
    # describes musical material, destroying pitch or family should collapse it.
    print(f"\n=== NULL MODELS (scored on vector separation) ===")
    rng = random.Random(args.seed)

    def separation(rs) -> float:
        vv = {id(r): texture_vector(r) for r in rs}
        hv = [r for r in rs if vv[id(r)] is not None]
        s, dd = [], []
        for i, a in enumerate(hv):
            for b in hv[i + 1:]:
                dist = float(np.linalg.norm(vv[id(a)] - vv[id(b)]))
                (s if a.label == b.label else dd).append(dist)
        if not s or not dd:
            return float("nan")
        return float(np.mean(dd) - np.mean(s))

    print(f"{'condition':16s} {'separation':>11s}   {'churn':>7s}  labels")
    print(f"{'real':16s} {separation(regions):11.3f}   {churn(regions):7.3f}  " +
          " ".join(f"{k}:{v}" for k, v in counts.most_common(3)))
    for mode in ("pitch", "family", "time"):
        seps, ch, dist = [], [], Counter()
        for _ in range(5):
            sh = shuffled_copy(regions, mode, rng)
            seps.append(separation(sh)); ch.append(churn(sh))
            dist.update(r.texture for r in sh)
        print(f"{mode+'-shuffled':16s} {np.nanmean(seps):11.3f}   "
              f"{np.nanmean(ch):7.3f}  " +
              " ".join(f"{k}:{v//5}" for k, v in dist.most_common(3)))

    print("\nBAR TO CLEAR: real separation must be POSITIVE (same-section repeats")
    print("describe each other better than unrelated sections do) AND must fall")
    print("toward zero under the pitch/family shuffles. If shuffling pitch leaves")
    print("separation intact, this is still a density detector - delete it rather")
    print("than tune it (SS XIX).")


if __name__ == "__main__":
    main()
