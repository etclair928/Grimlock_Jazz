# =================================================================
# TOOL: tools/playability_report.py
# Slice 0 diagnostic (GRIMLOCK_6.0_PERFORMANCE_ENGRAVING.md §12.6).
#
# Loads produced MusicXML pages and answers the one question that gates
# the whole reduction project: if we tried to put the harmonic content
# onto a piano grand staff, HOW MUCH OF IT IS PHYSICALLY UNPLAYABLE?
#
# It changes nothing. It parses the page, unions the selected parts back
# into one instrument's worth of simultaneity, and runs
# output.playability over it under two brackets (comfortable octave-hand,
# stretch tenth-hand) so the answer is a range, not a false-precise
# single number.
#
#   python tools/playability_report.py transcriptions/Hopeful_notation.musicxml
#   python tools/playability_report.py "transcriptions/Hopeful_*.musicxml"
#   python tools/playability_report.py <file> --scope all_pitched
#
# Part scopes (which parts to treat as the piano-reduction candidate):
#   harmonic       (default) - everything except drums, bass, vocals
#   harmonic_bass            - everything except drums, vocals (LH often has the bass)
#   all_pitched              - everything except drums
# =================================================================

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Tuple

# Run from the repo root so `output` imports cleanly whether or not the
# tool is invoked as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from output.playability import PlayabilityConfig, assess  # noqa: E402

# Parts excluded from each scope, matched as case-insensitive substrings
# of the music21 partName (which the exporter builds as "<family> <voice>").
_SCOPES = {
    "harmonic": ("drums", "bass", "vocals"),
    "harmonic_bass": ("drums", "vocals"),
    "all_pitched": ("drums",),
}

_BRACKETS = [
    ("comfortable(octave)", PlayabilityConfig(max_span_semitones=12, max_fingers_per_hand=5)),
    ("stretch(tenth)", PlayabilityConfig(max_span_semitones=16, max_fingers_per_hand=5)),
]


def _excluded(part_name: str, exclude: Tuple[str, ...]) -> bool:
    pn = (part_name or "").lower()
    return any(tok in pn for tok in exclude)


def _load_notes(path: str, exclude: Tuple[str, ...]) -> Tuple[List[Tuple[float, float, int]], List[str]]:
    """Return (notes, kept_part_names). notes are (start_beat, end_beat,
    midi) unioned across all non-excluded parts, in absolute offsets."""
    from music21 import converter
    score = converter.parse(path)
    notes: List[Tuple[float, float, int]] = []
    kept: List[str] = []
    for part in score.parts:
        if _excluded(part.partName, exclude):
            continue
        kept.append(part.partName)
        for el in part.flatten().notes:
            start = float(el.offset)
            ql = float(el.quarterLength)
            if ql <= 0:
                continue
            end = start + ql
            pitches = el.pitches if hasattr(el, "pitches") else [el.pitch]
            for p in pitches:
                notes.append((start, end, int(p.midi)))
    return notes, kept


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def report_file(path: str, scope: str) -> None:
    exclude = _SCOPES[scope]
    notes, kept = _load_notes(path, exclude)
    print("=" * 78)
    print(f"{os.path.basename(path)}   scope={scope}")
    print(f"  parts included ({len(kept)}): {', '.join(kept) if kept else '(none)'}")
    print(f"  pitched note-events unioned: {len(notes)}")
    if not notes:
        print("  (nothing to score)")
        return

    for label, cfg in _BRACKETS:
        rep = assess(notes, cfg)
        print(f"  --- {label}: span<={cfg.max_span_semitones}st, "
              f"<={cfg.max_fingers_per_hand}/hand, <={cfg.max_total_notes} total ---")
        print(f"      PLAYABLE  time={_fmt_pct(rep.time_pass_rate)}  "
              f"instants={_fmt_pct(rep.instant_pass_rate)}   "
              f"(so unplayable time={_fmt_pct(1 - rep.time_pass_rate)})")
        print(f"      simultaneity: max={rep.max_simultaneity}, "
              f"mean~{rep.mean_simultaneity:.1f}, max span={rep.max_span}st")
        unplayable = rep.sounding_beats - rep.playable_beats
        if unplayable > 0:
            fc = rep.unplayable_beats_finger_count
            ss = rep.unplayable_beats_span_or_split
            print(f"      unplayable cause (by time): >10 fingers={_fmt_pct(fc / rep.sounding_beats)}, "
                  f"span/split={_fmt_pct(ss / rep.sounding_beats)}")
        # histogram
        hist = rep.simultaneity_beats
        order = ["1-5", "6-8", "9-10", "11-15", "16+"]
        total = rep.sounding_beats or 1.0
        bars = "  ".join(f"{k}:{_fmt_pct(hist.get(k, 0.0) / total)}" for k in order)
        print(f"      time by #notes-at-once:  {bars}")
        print(f"      reduction floor: {rep.excess_notehead_beats:.0f} notehead-beats over 10 fingers")
        if rep.worst:
            w = rep.worst[0]
            print(f"      worst instant: {w.n} notes, span {w.span}st, "
                  f"at beat {w.start:.1f} for {w.duration:.2f} beats")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Playability diagnostic for produced MusicXML pages.")
    ap.add_argument("paths", nargs="+", help="MusicXML file(s) or glob(s).")
    ap.add_argument("--scope", choices=list(_SCOPES), default="harmonic",
                    help="Which parts form the piano-reduction candidate (default: harmonic).")
    args = ap.parse_args()

    files: List[str] = []
    for pat in args.paths:
        matched = glob.glob(pat)
        files.extend(matched if matched else [pat])
    files = [f for f in files if f.lower().endswith((".musicxml", ".xml", ".mxl"))]
    if not files:
        print("No MusicXML files matched.", file=sys.stderr)
        sys.exit(1)

    for f in sorted(set(files)):
        if not os.path.exists(f):
            print(f"(missing) {f}", file=sys.stderr)
            continue
        report_file(f, args.scope)


if __name__ == "__main__":
    main()
