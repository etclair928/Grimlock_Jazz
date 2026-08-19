#!/usr/bin/env python3
# =================================================================
# TOOL: tools/compare_versions.py
# Puts several exports of the SAME song side by side on the axes that decide
# whether a page is readable, so a change can be judged against every previous
# attempt at once instead of against whichever one was open at the time.
#
# WHY THESE AXES. Note count alone says nothing - §XVII.4 measured that clutter
# was never a note-count problem, since Klangio carries 1.19-1.85x our notes
# with 6-9x fewer rests. What separates a readable page from ours is the SHAPE:
# how much of it is rests, how fragmented the durations are, whether tuplets
# are real, and whether anything crosses a barline. A Klangio export in the
# list is the benchmark; an answer key, where one exists, is the truth.
#
#   python tools/compare_versions.py <page.musicxml> [more...]
# =================================================================

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# A ratio written against one of these is a real tuplet; anything else is
# arithmetic (see tools/tuplet_audit.py, which learned this from the edition).
_LEGAL_NORMALS = frozenset({1, 2, 3, 4, 6, 8, 12, 16})


def profile(path: str) -> dict:
    """Everything measurable straight off the XML, no music21 parse needed for
    the cheap fields - a full parse on a 3MB page costs a minute and most of
    what matters is countable."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        x = fh.read()

    notes = len(re.findall(r"<note[ >]", x))
    rests = len(re.findall(r"<rest\s*/?>", x))
    ties = len(re.findall(r'<tie type="start"', x))
    chords = len(re.findall(r"<chord\s*/?>", x))
    ratios = Counter(
        f"{a}:{b}" for a, b in
        re.findall(r"<actual-notes>(\d+)</actual-notes>\s*<normal-notes>(\d+)</normal-notes>", x))
    types = Counter(re.findall(r"<type>(\w+)</type>", x))
    ts = re.search(r"<beats>(\d+)</beats>\s*<beat-type>(\d+)</beat-type>", x)
    fifths = re.search(r"<fifths>(-?\d+)</fifths>", x)
    mode = re.search(r"<mode>(\w+)</mode>", x)
    parts = len(re.findall(r"<score-part\b", x))
    measures = len(re.findall(r"<measure\b", x))

    junk = sum(v for k, v in ratios.items()
               if int(k.split(":")[1]) not in _LEGAL_NORMALS)
    fine = sum(v for k, v in types.items() if k in ("32nd", "64th", "128th"))

    return {
        "file": os.path.basename(path),
        "notes": notes,
        "rest/note": rests / max(notes, 1),
        "tie/note": ties / max(notes, 1),
        "chord/note": chords / max(notes, 1),
        "tuplet%": 100.0 * sum(ratios.values()) / max(notes, 1),
        "junk": junk,
        "sub32%": 100.0 * fine / max(notes, 1),
        "meter": f"{ts.group(1)}/{ts.group(2)}" if ts else "?",
        "key": (f"{fifths.group(1)}{'m' if mode and mode.group(1) == 'minor' else ''}"
                if fifths else "?"),
        "parts": parts,
        "measures": measures,
        "ratios": ratios,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pages", nargs="+")
    ap.add_argument("--ratios", action="store_true", help="also print the ratio census")
    args = ap.parse_args()

    rows = []
    for page in args.pages:
        try:
            rows.append(profile(page))
        except Exception as exc:
            print(f"{os.path.basename(page)}: ERROR {type(exc).__name__}: {exc}")

    if not rows:
        return

    name_w = max(len(r["file"]) for r in rows) + 1
    hdr = (f"{'file':{name_w}} {'meter':>7} {'key':>5} {'notes':>7} {'rest/n':>7} "
           f"{'tie/n':>6} {'chord/n':>8} {'tuplet%':>8} {'junk':>5} {'sub32%':>7} {'bars':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['file']:{name_w}} {r['meter']:>7} {r['key']:>5} {r['notes']:>7} "
              f"{r['rest/note']:>7.3f} {r['tie/note']:>6.3f} {r['chord/note']:>8.3f} "
              f"{r['tuplet%']:>8.2f} {r['junk']:>5} {r['sub32%']:>7.2f} {r['measures']:>5}")

    if args.ratios:
        print()
        for r in rows:
            top = ", ".join(f"{k}x{v}" for k, v in r["ratios"].most_common(6))
            print(f"  {r['file']}: {top or 'none'}")

    print()
    print("  rest/n  fraction of noteheads that are rests - the clutter axis (§XVII.4)")
    print("  junk    tuplets written against a nonsensical normal count; should be 0")
    print("  sub32%  share of noteheads finer than a 16th - the 'confetti' axis")


if __name__ == "__main__":
    main()
