# =================================================================
# MODULE: output/beat_hierarchy.py
# THE METRIC HIERARCHY A PAGE MUST RESPECT (user specification, 2026-08-18).
#
# "Musical notation relies on a fundamental rule: every note, rest, and tuplet
# must visually respect the primary subdivisions of the measure's time
# signature. When rhythm aligns with this hierarchy, the performer's eye can
# instantly locate the primary beats without having to calculate duration on
# the fly."
#
# The test, stated as a rule of thumb: A PERFORMER SHOULD BE ABLE TO DRAW
# VERTICAL LINES THROUGH BEATS 1, 2, 3 AND 4 OF A MEASURE WITHOUT THOSE LINES
# CUTTING THROUGH THE MIDDLE OF AN UN-TIED DURATION OR A TUPLET BRACKET.
#
# THE FOUR LEVELS
#   0  BARLINE          Absolute. Nothing crosses it, ever. Terminate at the
#                       barline and tie into the next measure.
#   1  PRIMARY PULSE    The half-measure. In 4/4 beat 3 is an invisible
#                       barline; in compound meters each dotted-quarter group
#                       anchors. A half note starting on beat 2 hides beat 3
#                       and must be written as two tied quarters.
#   2  BEAT UNIT        Where tuplets live. A tuplet replaces a beat's normal
#                       division, so it starts where that beat starts and ends
#                       where it ends.
#   3  SUB-BEAT         Durations inside a beat complete that beat before
#                       moving on. A duration that starts off-beat and bleeds
#                       into the next beat is split and tied so the boundary
#                       stays visible.
#
# THE ONE RULE THAT GENERATES ALL FOUR. A note may cross a boundary only if it
# STARTS on a position at least as strong as that boundary. Everything above
# falls out of it:
#   * a dotted half on beat 1 crosses the half-measure - allowed, it starts on
#     the barline, which is stronger;
#   * a half note on beat 2 crosses the half-measure - refused, beat 2 is
#     weaker, so it becomes quarter-tied-to-quarter and beat 3 reappears;
#   * a quarter on the "and" of 2 crosses beat 3 - refused twice over;
#   * nothing crosses a barline, because no position is stronger than one.
#
# WHY THIS IS A VIEW CONCERN AND LIVES IN output/. Splitting one sounding note
# into two tied noteheads changes nothing about what was heard - it changes how
# the page SAYS it. The performance clock and the frozen Notes are untouched
# (§2.2); this is the same class of edit as the legato fill, applied for
# legibility rather than for sustain.
#
# PER VOICE, NOT PER STAFF. This is also the answer to what voice separation is
# for: two independent rhythms sharing one staff force the reader to
# disentangle them before they can find the beat, so each voice is spelled
# against the hierarchy on its own and the beats stay visible in both.
# =================================================================

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

# Levels, strongest first. Lower number = stronger boundary.
LEVEL_BARLINE = 0
LEVEL_PRIMARY = 1
LEVEL_BEAT = 2
LEVEL_SUBBEAT = 3

# Compound meters group in threes; the group is the primary pulse, not the
# written beat. 6/8 is two dotted quarters, 9/8 three, 12/8 four.
_COMPOUND_NUMERATORS = (6, 9, 12)


def primary_pulse_offsets(numerator: int, denominator: int) -> Tuple[Fraction, ...]:
    """Offsets WITHIN THE BAR, in beat units, that carry the primary pulse -
    the invisible barlines of Level 1.

    Compound meters group in threes. Simple duple/quadruple meters have a half
    measure. Meters with an odd numerator (3/4, 5/4, 7/8) have no unambiguous
    half, and inventing one would put a boundary where no reader expects it -
    so they get none, and Level 0/2/3 still apply."""
    if denominator == 8 and numerator in _COMPOUND_NUMERATORS:
        return tuple(Fraction(g) for g in range(3, numerator, 3))
    if numerator % 2 == 0 and numerator >= 4:
        return (Fraction(numerator, 2),)
    return ()


def position_level(offset_in_bar: Fraction, numerator: int, denominator: int) -> int:
    """How strong the metric position at `offset_in_bar` (in beats) is."""
    offset_in_bar = Fraction(offset_in_bar)
    if offset_in_bar % numerator == 0:
        return LEVEL_BARLINE
    if offset_in_bar in primary_pulse_offsets(numerator, denominator):
        return LEVEL_PRIMARY
    if offset_in_bar.denominator == 1:
        return LEVEL_BEAT
    return LEVEL_SUBBEAT


def _boundaries_between(start: Fraction, end: Fraction,
                        numerator: int, denominator: int) -> List[Tuple[Fraction, int]]:
    """Every metric boundary strictly inside (start, end), with its level."""
    out: List[Tuple[Fraction, int]] = []
    bar = Fraction(numerator)
    first_bar = (start // bar + 1) * bar
    pos = first_bar
    while pos < end:
        out.append((pos, LEVEL_BARLINE))
        pos += bar
    pulses = primary_pulse_offsets(numerator, denominator)
    bar_index = start // bar
    for extra in range(int((end - start) // bar) + 2):
        base = (bar_index + extra) * bar
        for p in pulses:
            here = base + p
            if start < here < end:
                out.append((here, LEVEL_PRIMARY))
    beat = Fraction(1)
    pos = (start // beat + 1) * beat
    while pos < end:
        if all(pos != b for b, _ in out):
            out.append((pos, LEVEL_BEAT))
        pos += beat
    out.sort()
    return out


def split_for_hierarchy(start: Fraction, duration: Fraction,
                        numerator: int = 4, denominator: int = 4,
                        is_tuplet: bool = False) -> List[Tuple[Fraction, Fraction]]:
    """Split one sounding note into the tied chain the page should show.

    Returns [(start, duration), ...] in beat units, contiguous and summing to
    the original duration. A single-element result means the note already
    respects the hierarchy and is written as-is.

    A note may cross a boundary only if it starts on a position at least as
    strong as that boundary (see the module header). Tuplets are never split:
    they are anchored to one beat upstream, so they cannot reach a boundary -
    and a bracket that got split would be exactly the un-notatable fragment
    this whole module exists to prevent."""
    start = Fraction(start)
    duration = Fraction(duration)
    if duration <= 0:
        return [(start, duration)]
    if is_tuplet:
        return [(start, duration)]

    segments: List[Tuple[Fraction, Fraction]] = []
    cursor = start
    remaining = duration
    guard = 0
    while remaining > 0 and guard < 64:
        guard += 1
        end = cursor + remaining
        start_level = position_level(cursor, numerator, denominator)
        cut: Optional[Fraction] = None
        for pos, level in _boundaries_between(cursor, end, numerator, denominator):
            # The barline is ABSOLUTE - it is not merely the strongest
            # boundary, it is one nothing crosses, including a note that
            # started on a barline itself. Every other boundary may be crossed
            # from an equally or more strongly accented position.
            if level == LEVEL_BARLINE or level < start_level:
                cut = pos
                break
        if cut is None:
            segments.append((cursor, remaining))
            break
        segments.append((cursor, cut - cursor))
        remaining = end - cut
        cursor = cut
    return segments


def respects_hierarchy(start: Fraction, duration: Fraction,
                       numerator: int = 4, denominator: int = 4,
                       is_tuplet: bool = False) -> bool:
    """True when the note needs no splitting - the audit form of the rule."""
    return len(split_for_hierarchy(start, duration, numerator, denominator, is_tuplet)) == 1


__all__ = [
    "LEVEL_BARLINE", "LEVEL_PRIMARY", "LEVEL_BEAT", "LEVEL_SUBBEAT",
    "primary_pulse_offsets", "position_level",
    "split_for_hierarchy", "respects_hierarchy",
]
