#!/usr/bin/env python3
# =================================================================
# TOOL: tools/proportional_export.py
# UNMETERED, PROPORTIONAL notation: the event sequence with correct RELATIVE
# durations, and no barlines, meter, or measure-filling at all.
#
# WHY. On Ellington's Reflections in D the reference transcription carries
# eleven meter changes (4/2, 4/4, 5/4, 6/4, 3/2) because a human interpreted
# freely-played rubato into readable bars. Barline placement there is an
# editorial opinion, so grading against it measures agreement with one
# interpreter. What is NOT opinion is proportion: a triplet is a triplet
# because of its ratio to what surrounds it.
#
# Dropping bars removes every error source that comes from meter - wrong time
# signature, misplaced downbeat, measures that must be filled with rests and
# ties - and leaves exactly the thing worth getting right.
#
# THE DEFECT THIS ROUTES AROUND. notation_quantizer._subdivisions_per_beat
# returns 3 only for 6/8, 9/8 and 12/8, and 4 for everything else. This piece
# is in 4/4, 5/4, 6/4 and 3/2, so the lattice is a 16th grid and a triplet
# CANNOT BE EXPRESSED - every one rounds to the nearest 16th. MEASURED against
# the reference: it has 135 tuplet events and 31 transitions at 3:2 or 2:3; our
# barred output has effectively none, while over-producing 1:1 (288 vs 193).
#
# HOW. Quantize RATIOS, not positions. Each inter-onset interval is expressed
# as a multiple of a LOCAL reference unit (a running median, so rubato scales
# both sides and cancels) and snapped to a small set of ratios a musician
# actually reads. No global grid, no accumulating drift, no bar to fill.
#
#   python tools/proportional_export.py <intermediate.pkl> <out.musicxml>
#                                       [--raw] [--answer-key K.mxl]
# =================================================================

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from typing import List, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

# Ratios a reader can actually parse, as multiples of the local unit. Includes
# the triple relations the 16th grid cannot express - which is the point.
READABLE = np.array([1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 1.0, 4/3, 3/2, 2.0,
                     8/3, 3.0, 4.0, 6.0, 8.0])
CHORD_WINDOW_MS = 90.0     # raw proximity: what counts as struck together
LOCAL_WINDOW = 9           # events either side used for the running unit


def load_events(pkl_path: str, use_consolidation: bool) -> List[Tuple[float, List[int], float]]:
    """(onset_ms, [pitches], end_ms) - chord events from RAW performance time.

    Consolidation is OFF by default here: measured on this piece it deletes 31%
    of notes and 43% of the 4+ note chords, because in a planing texture the
    same pitches recur constantly and it reads recurrence as fragmentation."""
    from core import StemType
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    ann = d["annotations"]
    notes = []
    for n in d["all_notes"]:
        if n.stem == StemType.DRUMS:
            continue
        if use_consolidation:
            c = ann.latest_value(n.id, "consolidation") or {}
            if c.get("role") == "absorbed":
                continue
        notes.append((n.start_ms, n.end_ms, n.pitch))
    notes.sort()

    events: List[Tuple[float, List[int], float]] = []
    for s, e, p in notes:
        if events and s - events[-1][0] <= CHORD_WINDOW_MS:
            events[-1][1].append(p)
            events[-1] = (events[-1][0], events[-1][1], max(events[-1][2], e))
        else:
            events.append((s, [p], e))
    return events


def proportional_durations(onsets: Sequence[float]) -> np.ndarray:
    """Each IOI as a snapped multiple of a LOCAL unit.

    The unit is a running median over neighbouring IOIs rather than a global
    tempo, so a ritardando moves the unit with the music and the RATIOS stay
    intact. That is what makes this work on playing with no steady pulse."""
    t = np.asarray(onsets, float)
    if len(t) < 3:
        return np.ones(max(len(t), 1))
    iois = np.diff(t)
    iois = np.clip(iois, 1.0, None)
    out = np.empty(len(iois))
    for i, v in enumerate(iois):
        lo = max(0, i - LOCAL_WINDOW)
        hi = min(len(iois), i + LOCAL_WINDOW + 1)
        unit = float(np.median(iois[lo:hi]))
        ratio = v / max(unit, 1.0)
        out[i] = READABLE[int(np.argmin(np.abs(READABLE - ratio)))]
    return out


def write_musicxml(events, durs, path: str, key_fifths: int = 2) -> None:
    """Unmetered proportional score: one part, no time signature, barlines
    suppressed. Durations are the snapped ratios in quarter-length units."""
    from music21 import stream, note as m21note, chord as m21chord, key as m21key, bar
    sc = stream.Score()
    part = stream.Part()
    part.append(m21key.KeySignature(key_fifths))
    for i, (onset, pitches, end) in enumerate(events):
        ql = float(durs[i]) if i < len(durs) else 1.0
        ql = max(0.125, round(ql * 12) / 12.0)     # 12ths express both 16ths and triplets
        ps = sorted(set(pitches))
        el = m21chord.Chord(ps) if len(ps) > 1 else m21note.Note(ps[0])
        el.quarterLength = ql
        part.append(el)
    # one continuous stream - no measures imposed
    for b in part.getElementsByClass(bar.Barline):
        part.remove(b)
    sc.append(part)
    sc.write("musicxml", fp=path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl")
    ap.add_argument("out")
    ap.add_argument("--consolidated", action="store_true",
                    help="use consolidation (default OFF - it deletes 31%% of notes here)")
    ap.add_argument("--answer-key", default=None)
    args = ap.parse_args()

    events = load_events(args.pkl, args.consolidated)
    onsets = [e[0] for e in events]
    durs = proportional_durations(onsets)
    sizes = [len(e[1]) for e in events]
    multi = [s for s in sizes if s >= 2]
    print(f"events {len(events)}   noteheads {sum(sizes)}   "
          f"mean chord size {np.mean(multi) if multi else 0:.2f}   "
          f"4+ note events {sum(1 for s in sizes if s >= 4)}")

    from collections import Counter
    c = Counter(round(float(d), 4) for d in durs)
    print("duration ratios emitted (as multiples of the local unit):")
    for v, n in c.most_common(10):
        print(f"   {v:6.3f} x{n}")

    write_musicxml(events, durs, args.out)
    print(f"wrote {args.out}")

    if args.answer_key:
        import subprocess
        subprocess.run([sys.executable, os.path.join(JAZZ, "tools", "musicianship_report.py"),
                        args.out, args.answer_key])


if __name__ == "__main__":
    main()
