#!/usr/bin/env python3
# =================================================================
# TOOL: tools/tuplet_audit.py
# Checks the two hard rules a tuplet has to obey on a finished page.
#
#   1. A TUPLET NEVER CROSSES A BARLINE. A tuplet bracket is a statement about
#      how one metric unit is divided; a group that spans a barline is not a
#      tuplet, and a serializer handed one splits it into tied fragments whose
#      ratios (24:13, 48:43) no engraver would write and no reader can parse.
#   2. A TUPLET IS ANCHORED TO A METRIC UNIT. It replaces that unit's normal
#      division, so it starts where the unit starts. The unit is NOT always the
#      beat: read off the Chopin edition, 4:3 and 8:2 groups anchor to the
#      second EIGHTH of a beat, which is standard - the operative unit is
#      whatever the tuplet subdivides. So the test is that the frame begins on a
#      binary subdivision (offset denominator a power of two), which admits
#      beats, half-beats and quarter-beats and still refuses a frame floating at
#      a third of a beat, which no reader can locate.
#
# CALIBRATED AGAINST THE PUBLISHED EDITION. The first version of this tool
# FAILED Chopin_ANSWER_KEY.musicxml, which is a published edition and therefore
# correct by definition - so the tool was wrong, twice:
#   * it called 14:2 un-notatable on the grounds that 14 > 12. Chopin writes
#     14-tuplets in ornamental runs and the edition contains them. What makes a
#     ratio junk is not a large ACTUAL count but a nonsensical NORMAL one: real
#     tuplets are written against 1, 2, 3, 4, 6, 8 or 12 - never against 13, 19
#     or 43. 24:13 is float error being reconciled; 14:2 is music.
#   * it counted every note INSIDE a tuplet as a group start, since music21
#     leaves Tuplet.type None on the middle notes. That reported 415 off-beat
#     "groups" in an edition that has none. Only type == "start" begins a group,
#     and rests count - a frame held on the beat by a tuplet rest has its first
#     NOTE later, and scanning only notes cannot see the frame at all.
#   * it demanded a whole-beat anchor, which the edition disproves (above).
#
# WHAT THE EDITION STILL TRIPS, HONESTLY. After all three corrections the answer
# key reports 2 tuplet notes crossing a barline in bars 50 and 52, where a 6:4
# run tiles from beat 3.083 and the last group ends one twelfth past the bar.
# That is an encoding artifact of this particular MusicXML file, not something
# Chopin wrote: the file is a transcription too, and it is ground truth for
# CONTENT rather than for byte-perfect encoding. Worth remembering before
# treating any single number off it as absolute.
# Running an auditor against known-good ground truth before trusting it on our
# own output is the cheapest way to find out which of the two is broken.
#
# Also reports the ratio census, because the shape of that list is the fastest
# read on whether a page's tuplets are real: a clean page shows a handful of
# named ratios (3:2, 5:4, 6:4) and nothing else. Klangio's exports are 100%
# 3:2. Ours used to carry a tail of 12:11, 24:13, 48:43 - not tuplets anybody
# detected, just accumulated float error being reconciled.
#
#   python tools/tuplet_audit.py <page.musicxml> [more...]
# =================================================================

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import Counter
from fractions import Fraction

warnings.filterwarnings("ignore")


# What a tuplet may legitimately be written AGAINST. A ratio whose normal
# count is outside this set is not a tuplet anybody notated - it is arithmetic.
_LEGAL_NORMALS = frozenset({1, 2, 3, 4, 6, 8, 12, 16})


def measure_tuplets(path: str = None, score=None) -> dict:
    """The audit's FINDINGS, as data. Printing lives in `audit` below.

    Split out because app/diagnostics.py needs these numbers and the first
    version of that module reimplemented the rules instead - reporting 44
    off-beat groups on a page that has 9, because it demanded a whole-beat
    anchor and used offset-within-part rather than beat-within-measure. The
    edition disproves the whole-beat rule (see the header), so the naive
    version was measuring its own mistake. One implementation, two presenters.
    """
    from music21 import converter

    if score is None:
        score = converter.parse(path)
    ratios: Counter = Counter()
    crossings = []
    unanchored = []
    total_notes = 0
    tuplet_notes = 0

    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            bar_start = float(measure.offset)
            bar_len = float(measure.barDuration.quarterLength)
            for el in measure.recurse().notesAndRests:
                if not el.isRest:
                    total_notes += 1
                tuplets = el.duration.tuplets
                if not tuplets:
                    continue
                if not el.isRest:
                    tuplet_notes += 1
                t = tuplets[0]
                ratios[f"{t.numberNotesActual}:{t.numberNotesNormal}"] += 1
                is_group_start = (t.type == "start")

                # rule 1 - the note must lie wholly inside its measure
                start = float(el.offset)
                end = start + float(el.duration.quarterLength)
                if end > bar_len + 1e-6:
                    crossings.append((measure.number, start, end, bar_len))

                # rule 2 - the group must begin on a beat of the measure
                try:
                    beat = Fraction(el.beat).limit_denominator(64)
                except Exception:
                    continue
                # A frame may anchor to any BINARY subdivision - beat,
                # half-beat, quarter-beat. Only a start on a non-binary
                # position is genuinely floating.
                if is_group_start and (beat.denominator & (beat.denominator - 1)):
                    unanchored.append((measure.number, float(el.beat)))

    junk = {r: c for r, c in ratios.items()
            if int(r.split(":")[1]) not in _LEGAL_NORMALS}
    return {
        "total_notes": total_notes, "tuplet_notes": tuplet_notes,
        "ratios": dict(ratios), "junk_ratios": junk,
        "junk_count": sum(junk.values()),
        "crossings": crossings, "unanchored": unanchored,
        "ok": not junk and not crossings,
    }


def audit(path: str) -> bool:
    """The CLI presenter. Delegates every rule to measure_tuplets."""
    m = measure_tuplets(path)
    total_notes, tuplet_notes = m["total_notes"], m["tuplet_notes"]
    ratios, crossings, unanchored = m["ratios"], m["crossings"], m["unanchored"]
    junk = list(m["junk_ratios"])

    name = os.path.basename(path)
    pct = 100.0 * tuplet_notes / max(total_notes, 1)
    print(f"\n{name}")
    print(f"  notes {total_notes}   tuplet notes {tuplet_notes} ({pct:.1f}%)")
    print(f"  ratios: {ratios if ratios else 'none'}")

    ok = True
    if junk:
        print(f"  FAIL  un-notatable ratios present: {junk}")
        ok = False
    if crossings:
        print(f"  FAIL  {len(crossings)} tuplet note(s) cross a barline, e.g. "
              f"measure {crossings[0][0]} ends at {crossings[0][2]:.3f} "
              f"in a bar of {crossings[0][3]:.3f}")
        ok = False
    if unanchored:
        print(f"  WARN  {len(unanchored)} tuplet group(s) start off the beat, e.g. "
              f"measure {unanchored[0][0]} at beat {unanchored[0][1]:.3f}")
    if ok and not unanchored:
        print("  PASS  every tuplet is notatable, inside its bar, and on a beat")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pages", nargs="+")
    args = ap.parse_args()
    failures = 0
    for page in args.pages:
        try:
            if not audit(page):
                failures += 1
        except Exception as exc:
            print(f"\n{os.path.basename(page)}\n  ERROR {type(exc).__name__}: {exc}")
            failures += 1
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
