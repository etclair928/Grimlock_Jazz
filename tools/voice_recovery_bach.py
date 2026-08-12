#!/usr/bin/env python3
# =================================================================
# TOOL: tools/voice_recovery_bach.py
# RESEARCH ONLY. Ground truth for "same voice", and a controlled degradation
# study that asks which of our measured data defects actually breaks voicing.
#
# THE HOLE THIS FILLS. Voice assignment has never been evaluated, because we
# have no labels: nothing in a transcription says which notes belong to one
# line. `motif` looked like a candidate and is not - measured, its annotations
# sit a median 14.6 BEATS apart, because it marks one anchor per OCCURRENCE
# rather than the notes comprising a figure. So every voicing change all session
# was judged only by page-shape proxies (rest ratio, top-line stability), never
# by whether the voices were RIGHT.
#
# Symbolic music has the labels by construction. music21 ships 433 Bach works
# with named Soprano/Alto/Tenor/Bass parts - the literature-standard benchmark
# for voice separation. Flatten a chorale into a bare note list, run our voicer,
# and measure how much of the original grouping comes back.
#
# THE QUESTION THAT MATTERS MORE THAN THE BASELINE. We cannot currently tell
# whether our voice problems come from a bad ALGORITHM or from bad DATA. So
# after the clean baseline, inject OUR OWN measured defects into Bach, one at a
# time, at the rates we actually measured:
#
#   overlap    63.6% of consecutive notes overlap the next (median 104ms)
#   duration   139% mean duration distortion vs consolidated source
#   onset      grid alignment R = 0.03 (essentially none)
#
# If overlap alone collapses recovery on Bach, then the fragmented melody we
# chased all session is a DURATION bug wearing a voice-assignment costume, and
# the texture/voice-budget direction is treating a symptom.
#
# HONEST LIMITS. Chorales are homophonic 4-part writing with clean voice
# leading; pop and jazz are not. A witness that fails here will certainly fail
# on our material (necessary, not sufficient). Recovery rates are a CEILING.
# Voice crossing is common in Bach and rare in our repertoire.
#
#   python tools/voice_recovery_bach.py [--works 12] [--caps 2 4]
# =================================================================

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

# Rates measured on our own output - see module header.
OVERLAP_RATE = 0.636
OVERLAP_MEDIAN_MS = 104.0
DURATION_DISTORTION = 1.39
ONSET_JITTER_MS = 60.0

QUARTER_MS = 500.0        # 120bpm; chorale tempo is arbitrary for this test


def load_chorales(n: int) -> List[Tuple[str, List[Tuple[float, float, int, int]]]]:
    """(work_id, [(start_ms, end_ms, pitch, true_voice), ...])"""
    from music21 import corpus
    out = []
    paths = corpus.getComposer('bach')
    for p in paths:
        if len(out) >= n:
            break
        try:
            s = corpus.parse(p)
        except Exception:
            continue
        parts = list(s.parts)
        if len(parts) != 4:
            continue
        notes = []
        ok = True
        for vi, part in enumerate(parts):
            for el in part.flatten().notes:
                if el.isChord:            # chorales should be monophonic per part
                    ok = False
                    break
                notes.append((float(el.offset) * QUARTER_MS,
                              float(el.offset + el.quarterLength) * QUARTER_MS,
                              int(el.pitch.midi), vi))
            if not ok:
                break
        if ok and len(notes) > 80:
            out.append((str(p), sorted(notes)))
    return out


def to_notation_notes(notes):
    from output.notation_score import NotationNote
    return [NotationNote(pitch=p, start_ms=s, end_ms=e, velocity=80,
                         source_note_id=f"n{i}")
            for i, (s, e, p, _v) in enumerate(notes)]


def pairwise_prf(true_voice: Sequence[int], pred_voice: Sequence[int]):
    """Standard clustering agreement on the SAME-VOICE relation.
    precision = of pairs we put together, how many truly belong together
    recall    = of pairs that truly belong together, how many we put together"""
    n = len(true_voice)
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_t = true_voice[i] == true_voice[j]
            same_p = pred_voice[i] == pred_voice[j]
            if same_t and same_p:
                tp += 1
            elif same_p and not same_t:
                fp += 1
            elif same_t and not same_p:
                fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def transition_accuracy(notes, true_voice, pred_voice) -> float:
    """Of consecutive notes WITHIN a true voice, how often does the voicer keep
    them together? This is what a melody line actually needs."""
    by_true = defaultdict(list)
    for idx, (s, _e, _p, v) in enumerate(notes):
        by_true[v].append((s, idx))
    good = tot = 0
    for v, lst in by_true.items():
        lst.sort()
        for (_s1, i1), (_s2, i2) in zip(lst, lst[1:]):
            tot += 1
            if pred_voice[i1] == pred_voice[i2]:
                good += 1
    return good / tot if tot else float("nan")


def run_voicer(notes, max_voices: int, chords: bool = True):
    import output.piano_reduction as pr
    if not chords:
        # Disable chord grouping so we read the ASSOCIATION algorithm rather
        # than the layout policy in front of it. A chorale is HOMOPHONIC - all
        # four parts attack together - so `_chord_events` collapses them into
        # one slot which then takes one voice, pinning pairwise precision at
        # ~1/4 (measured 0.257 in every condition: the fingerprint of four true
        # voices landing in one predicted voice).
        #
        # NB the obvious lever does NOT work: CHORD_ONSET_TOL_MS /
        # CHORD_OFFSET_TOL_MS are DEAD CONSTANTS. `_chord_events` was rewritten
        # to grid-aligned grouping (grid_division=4) and no longer reads them,
        # so patching them changed nothing - the first attempt produced numbers
        # identical to the last digit. Replace the function instead.
        if not getattr(pr, "_orig_chord_events", None):
            pr._orig_chord_events = pr._chord_events

        def _one_slot_each(notes, ms_per_beat=0.0, grid_division=4,
                           end_tolerance_cells=1):
            return sorted([pr._Slot(n) for n in notes], key=lambda s: s.start)

        pr._chord_events = _one_slot_each
    elif getattr(pr, "_orig_chord_events", None):
        pr._chord_events = pr._orig_chord_events
    nns = to_notation_notes(notes)
    voiced = pr.assign_voices(nns, max_voices=max_voices, smooth=False,
                              ms_per_beat=QUARTER_MS)
    by_id = {nn.source_note_id: nn.voice_index for nn in voiced}
    return [by_id.get(f"n{i}", -1) for i in range(len(notes))]


# ---------------------------------------------------------------------------
# degradations - inject OUR measured defects into clean data
# ---------------------------------------------------------------------------

def degrade_overlap(notes, rng):
    """Extend note ends so ~OVERLAP_RATE of consecutive notes overlap the next,
    reproducing what Basic Pitch does by ending notes on energy decay."""
    out = list(notes)
    starts = sorted(n[0] for n in out)
    res = []
    for (s, e, p, v) in out:
        nxt = None
        for t in starts:
            if t > s + 1:
                nxt = t
                break
        if nxt is not None and rng.random() < OVERLAP_RATE:
            e = max(e, nxt + abs(rng.gauss(OVERLAP_MEDIAN_MS, OVERLAP_MEDIAN_MS * 0.4)))
        res.append((s, e, p, v))
    return sorted(res)


def degrade_duration(notes, rng):
    res = []
    for (s, e, p, v) in notes:
        d = max(e - s, 1.0)
        factor = max(0.15, 1.0 + rng.gauss(0.0, DURATION_DISTORTION))
        res.append((s, s + d * factor, p, v))
    return sorted(res)


def degrade_onset(notes, rng):
    res = []
    for (s, e, p, v) in notes:
        j = rng.gauss(0.0, ONSET_JITTER_MS)
        res.append((max(0.0, s + j), max(s + j + 1, e + j), p, v))
    return sorted(res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--works", type=int, default=12)
    ap.add_argument("--caps", nargs="*", type=int, default=[4, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-chords", action="store_true",
                    help="disable chord grouping to read the association "
                         "algorithm rather than the layout policy in front of it")
    args = ap.parse_args()

    print(f"loading up to {args.works} four-part Bach works ...", flush=True)
    works = load_chorales(args.works)
    print(f"loaded {len(works)} works, "
          f"{sum(len(n) for _w, n in works)} notes total\n", flush=True)
    if not works:
        raise SystemExit("no usable chorales")

    rng = random.Random(args.seed)
    conditions = [
        ("CLEAN (ground truth)", lambda ns: ns),
        ("+ overlap 63.6%", lambda ns: degrade_overlap(ns, rng)),
        ("+ duration 139%", lambda ns: degrade_duration(ns, rng)),
        ("+ onset jitter 60ms", lambda ns: degrade_onset(ns, rng)),
        ("+ ALL THREE", lambda ns: degrade_onset(degrade_duration(
            degrade_overlap(ns, rng), rng), rng)),
    ]

    tag = "   [CHORD GROUPING OFF]" if args.no_chords else ""
    for cap in args.caps:
        print(f"=== max_voices = {cap}   (Bach has 4 true voices){tag} ===")
        print(f"{'condition':22s} {'pair_P':>7s} {'pair_R':>7s} {'pair_F1':>8s} "
              f"{'transitions':>12s} {'voices_used':>12s}")
        print("-" * 74)
        for label, fn in conditions:
            P, R, F, T, V = [], [], [], [], []
            for _w, notes in works:
                deg = fn(notes)
                true_v = [n[3] for n in deg]
                pred_v = run_voicer(deg, cap, chords=not args.no_chords)
                p, r, f = pairwise_prf(true_v, pred_v)
                P.append(p); R.append(r); F.append(f)
                T.append(transition_accuracy(deg, true_v, pred_v))
                V.append(len(set(pred_v)))
            print(f"{label:22s} {np.mean(P):7.3f} {np.mean(R):7.3f} "
                  f"{np.mean(F):8.3f} {np.mean(T):12.3f} {np.mean(V):12.2f}",
                  flush=True)
        print()

    print("READ THIS AS:")
    print("  CLEAN is the algorithm's ceiling - what it can do on perfect data.")
    print("  A degradation that collapses recovery is a DATA defect worth fixing")
    print("  upstream; one that barely moves it is not our voicing problem.")


if __name__ == "__main__":
    main()
