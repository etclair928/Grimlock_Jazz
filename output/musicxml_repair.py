# =================================================================
# MODULE: output/musicxml_repair.py
# THE LAST WORD ON THE BARLINE, and the only place it can be said.
#
# WHY A POST-WRITE PASS. music21 runs its OWN notation pass at serialization,
# after anything done to the score in memory. That was established by
# measurement, not assumption: an in-memory clamp reported zero elements
# touched while the written file still carried thirteen barline crossings, and
# a score with 2326 clean onsets came back from a write/read round trip with
# 2327, one of them off the grid. Suppressing that pass with
# `makeNotation=False` raises "Cannot convert complex durations to MusicXML",
# and calling `splitAtDurations()` first does not clear it.
#
# So the bytes on disk are the only place left, and this is the pass that
# operates on them.
#
# WHAT IT REPAIRS. THE AMBIGUOUS WHOLE REST, and the investigation that found
# it is worth repeating because it began with a wrong diagnosis.
#
# The health panel reported four barline-crossing rests on Hopeful. Reading
# the raw bytes, there is no such thing: measure 83's voice 1 is
# 2520+5040+2520+40320+10080 = 60480 divisions, which is exactly one 6/4 bar.
# The file's arithmetic is correct.
#
# What happens is that music21's READER treats any rest with `<type>whole</type>`
# as a full-measure rest and hands back quarterLength 6.0 - the bar - ignoring
# the written duration of 40320, which is four quarters. So a correct file
# reads back as a six-beat rest starting on beat two, and the diagnostic
# believed it.
#
# The reader is not wrong to do that: a whole rest IS the conventional bar
# rest, in any meter. Which makes writing one for four beats of a six-beat bar
# genuinely ambiguous - MuseScore will draw it as filling the bar. music21's
# writer emits exactly that. So this pass rewrites any whole rest that is not
# actually a full bar into unambiguous pieces, which is the same rule as
# everywhere else in this codebase: a value that cannot mean what it says must
# be written as something that can.
#
# IT SHORTENS, NEVER LENGTHENS (user directive, 2026-08-21: "do not force
# rhythmic values into spaces they won't or can't possibly fit"). A rest that
# starts at beat two of a six-beat bar becomes a four-beat rest, not a six-beat
# rest moved to beat one - moving it would silently relocate music, while
# trimming it only stops it claiming time that is not there. The same rule
# that made notatable_at_most return 0.0 rather than something too long.
#
# WHAT IT DOES NOT TOUCH. Notes. If a NOTE ever crosses a barline here, that
# is a defect upstream in the engraver and must be fixed there, where it can
# be split and tied properly - silently trimming a note would hide it. This
# pass only removes an artifact the serializer itself invented.
# =================================================================

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

# quarter-lengths per whole note, for reading <time> into a bar length
_BEAT_TYPE_TO_QUARTERS = 4.0


def _int_text(element: Optional[ET.Element], default: int = 0) -> int:
    if element is None or not (element.text or "").strip():
        return default
    try:
        return int(float(element.text))
    except ValueError:
        return default


def _bar_divisions(divisions: int, beats: int, beat_type: int) -> int:
    """One bar's length, in the same division units durations are written in."""
    if beat_type <= 0:
        return 0
    return int(round(divisions * _BEAT_TYPE_TO_QUARTERS * beats / beat_type))


def repair_measures(path: str) -> Dict[str, int]:
    """Trim whole-measure rests that overrun their bar. Returns counts.

    Walks each part in document order, tracking `divisions` and `<time>` as
    they are declared (both are inherited by later measures, so they cannot be
    read per-measure), and a per-voice cursor advanced by `<note>` durations
    and moved by `<backup>` / `<forward>`.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    stats = {"rests_trimmed": 0, "rests_removed": 0, "notes_overrunning": 0,
             "whole_rests_disambiguated": 0}

    for part in root.findall(".//part"):
        divisions = 0
        beats, beat_type = 4, 4
        for measure in part.findall("measure"):
            attributes = measure.find("attributes")
            if attributes is not None:
                divisions = _int_text(attributes.find("divisions"), divisions)
                time_el = attributes.find("time")
                if time_el is not None:
                    beats = _int_text(time_el.find("beats"), beats)
                    beat_type = _int_text(time_el.find("beat-type"), beat_type)
            if divisions <= 0:
                continue
            bar = _bar_divisions(divisions, beats, beat_type)
            if bar <= 0:
                continue

            cursor = 0
            for element in list(measure):
                tag = element.tag
                if tag == "backup":
                    cursor = max(0, cursor - _int_text(element.find("duration")))
                    continue
                if tag == "forward":
                    cursor += _int_text(element.find("duration"))
                    continue
                if tag != "note":
                    continue

                duration_el = element.find("duration")
                duration = _int_text(duration_el)
                # A chord member sounds WITH the previous note and does not
                # advance the cursor; treating it as sequential would make
                # every chord look like an overrun.
                if element.find("chord") is not None:
                    continue

                rest_el = element.find("rest")
                if rest_el is not None and duration > 0:
                    type_el = element.find("type")
                    is_whole = (type_el is not None
                                and (type_el.text or "").strip() == "whole")
                    marked_bar = rest_el.get("measure") is not None
                    # A whole rest that is not a whole bar reads as a bar rest
                    # to any conventional reader. Rewrite it as pieces that can
                    # only mean their own length.
                    if is_whole and not marked_bar and duration != bar:
                        pieces = _unambiguous_rest_pieces(duration, divisions)
                        if pieces:
                            _replace_rest(measure, element, pieces)
                            stats["whole_rests_disambiguated"] += 1
                            cursor += duration
                            continue

                overrun = cursor + duration - bar
                if overrun > 0:
                    if element.find("rest") is not None:
                        room = bar - cursor
                        if room <= 0:
                            measure.remove(element)
                            stats["rests_removed"] += 1
                            continue
                        duration_el.text = str(int(room))
                        # It is no longer a whole-measure rest, and saying so
                        # matters: `measure="yes"` tells a reader to draw a bar
                        # rest whatever the duration says.
                        rest_el = element.find("rest")
                        if rest_el.get("measure") is not None:
                            del rest_el.attrib["measure"]
                        duration = int(room)
                        stats["rests_trimmed"] += 1
                    else:
                        # Not ours to silently fix - see the header.
                        stats["notes_overrunning"] += 1
                cursor += duration

    if any(stats[k] for k in ("rests_trimmed", "rests_removed",
                              "whole_rests_disambiguated")):
        _write_preserving_doctype(tree, path)
    return stats


# Rest types that can only mean their own length. `whole` is deliberately
# absent - it doubles as the bar rest, which is the ambiguity being removed -
# and `breve` with it.


def _unambiguous_rest_pieces(duration: int, divisions: int
                             ) -> List[Tuple[str, int, int]]:
    """Decompose a duration into (type, dots, duration) rests that are not
    whole rests. Returns [] if it cannot be done exactly - in which case the
    original is left alone rather than approximated, because a rest of the
    wrong length is worse than an ambiguous one."""
    shapes: List[Tuple[str, float, int]] = []
    # No 32nd rests either - same rule as notes.
    for name, quarters in (("half", 2.0), ("quarter", 1.0), ("eighth", 0.5),
                           ("16th", 0.25)):
        shapes.append((name, quarters * 1.5, 1))     # dotted
        shapes.append((name, quarters, 0))
    shapes.sort(key=lambda s: -s[1])

    out: List[Tuple[str, int, int]] = []
    remaining = duration
    guard = 0
    while remaining > 0 and guard < 12:
        guard += 1
        for name, quarters, dots in shapes:
            ticks = int(round(quarters * divisions))
            if ticks and ticks <= remaining:
                out.append((name, dots, ticks))
                remaining -= ticks
                break
        else:
            return []                                 # nothing fits exactly
    return out if remaining == 0 else []


def _replace_rest(measure: ET.Element, element: ET.Element,
                  pieces: List[Tuple[str, int, int]]) -> None:
    """Swap one rest element for a sequence of unambiguous ones, in place."""
    index = list(measure).index(element)
    voice = element.findtext("voice")
    staff = element.findtext("staff")
    measure.remove(element)
    for offset, (name, dots, ticks) in enumerate(pieces):
        note = ET.Element("note")
        ET.SubElement(note, "rest")
        ET.SubElement(note, "duration").text = str(int(ticks))
        if voice is not None:
            ET.SubElement(note, "voice").text = voice
        ET.SubElement(note, "type").text = name
        for _ in range(dots):
            ET.SubElement(note, "dot")
        if staff is not None:
            ET.SubElement(note, "staff").text = staff
        measure.insert(index + offset, note)


def _write_preserving_doctype(tree: ET.ElementTree, path: str) -> None:
    """ElementTree drops the DOCTYPE; MuseScore tolerates that and validating
    parsers do not. Captured and restored, exactly as _normalize_voice_numbers
    already has to do."""
    import re

    with open(path, "r", encoding="utf-8") as fh:
        original = fh.read()
    match = re.search(r"^<!DOCTYPE[^>]*>", original, re.MULTILINE)
    doctype = match.group(0) if match else None

    tree.write(path, encoding="UTF-8", xml_declaration=True)
    if doctype is None:
        return
    with open(path, "r", encoding="utf-8") as fh:
        rewritten = fh.read()
    if "<!DOCTYPE" in rewritten:
        return
    lines = rewritten.split("\n", 1)
    if len(lines) == 2 and lines[0].lstrip().startswith("<?xml"):
        rewritten = f"{lines[0]}\n{doctype}\n{lines[1]}"
    else:
        rewritten = f"{doctype}\n{rewritten}"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(rewritten)


__all__ = ["repair_measures"]
