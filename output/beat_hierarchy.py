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


# Durations a single notehead can carry, in beats: powers of two, their dotted
# forms, and the same series scaled into the thirds family for notes that live
# on a ternary grid. Anything not here needs a tie, whatever the metric
# hierarchy thinks.
# The plain note values, in beats: 8 down to a SIXTEENTH, and no further.
# Nothing below 1/4 of a beat may be written (user directive, 2026-08-22) -
# the ternary family below scales these by 2/3, so its floor is the sixteenth
# triplet at 1/6, which still reads as a sixteenth notehead.
_BASE_VALUES: Tuple[Fraction, ...] = (
    Fraction(8), Fraction(4), Fraction(2), Fraction(1),
    Fraction(1, 2), Fraction(1, 4),
)

# What a SINGLE notehead can carry: those values, their dotted forms (x3/2),
# and the same series inside a triplet (x2/3, which is how 1/3, 2/3, 1/6 and
# 1/12 arise). Anything outside this needs a tie, whatever the metric
# hierarchy would like.
# Kept as two FAMILIES rather than one pool, because which family a remainder
# is decomposed in decides whether the result reads. Five twelfths taken from
# the combined pool greedily gives 3/8 + 1/24 - a binary note tied to a
# twenty-fourth, which is both unreadable and the exact k/24 scrap this
# codebase spent a session removing. Taken in its own family it gives
# 1/3 + 1/12, a triplet quarter tied to a triplet thirty-second, which is
# what a copyist writes.
_BINARY_SHAPES: Tuple[Fraction, ...] = tuple(sorted(
    {v for v in _BASE_VALUES} | {v * Fraction(3, 2) for v in _BASE_VALUES},
    reverse=True))
_TERNARY_SHAPES: Tuple[Fraction, ...] = tuple(sorted(
    {v * Fraction(2, 3) for v in _BASE_VALUES}, reverse=True))
_NOTE_SHAPES: Tuple[Fraction, ...] = tuple(sorted(
    set(_BINARY_SHAPES) | set(_TERNARY_SHAPES), reverse=True))


def _shapes_for(value: Fraction) -> Tuple[Fraction, ...]:
    """A remainder is written in the family it belongs to - thirds with
    thirds, binary with binary."""
    return _TERNARY_SHAPES if value.denominator % 3 == 0 else _BINARY_SHAPES


def _notatable_chain(start: Fraction, duration: Fraction
                     ) -> List[Tuple[Fraction, Fraction]]:
    """Break one segment into pieces a notehead can actually carry.

    WHY THIS IS NEEDED HERE. This module's contract is that it returns the tied
    chain the page should show - so every link has to be writable, and it was
    returning links that were not. Measured on Chopin: a note of 3/4 starting at
    560/3 (a ternary onset carrying a binary duration) was cut at the beat line
    into 1/3 + 5/12, and no notehead expresses five twelfths. music21 then
    invented a 6:5 bracket to reconcile it, which is where fourteen of the
    page's junk ratios came from.

    Greedy largest-first WITHIN ONE FAMILY: the longest writable value that
    FITS, then the remainder, tied. Staying in the family is what makes the
    result readable - see _shapes_for.

    NOTHING IS EVER LENGTHENED TO MAKE IT FIT (user directive, 2026-08-21: "do
    not force rhythmic values into spaces they won't or can't possibly fit. If
    it doesn't fit then it must be something else that can actually fit"). A
    first draft of this function folded a sub-notatable residue into the
    previous piece, which is the same error as notatable_at_most returning a
    value longer than its own ceiling - the bug that produced k/24 in the first
    place. Growing a note to absorb a leftover pushes the next onset, and a
    ceiling that can be exceeded is not a ceiling.

    So a residue smaller than the shortest writable value is DROPPED and the
    chain ends fractionally early. Shortening can never overlap the next note,
    cross a barline, or displace anything; at the sizes involved - under a
    twenty-fourth of a beat, 36ms at 70bpm - it is inaudible and unwritable
    either way. The chain therefore sums to AT MOST the original duration,
    never more.
    """
    out: List[Tuple[Fraction, Fraction]] = []
    cursor, remaining = start, duration
    shapes = _shapes_for(duration)
    guard = 0
    while remaining > 0 and guard < 16:
        guard += 1
        piece = next((v for v in shapes if v <= remaining), None)
        if piece is None:
            # Nothing writable fits. Drop the residue rather than grow the
            # note to swallow it - see the docstring.
            if not out:
                out.append((cursor, remaining))   # nothing emitted yet: keep it
            break
        out.append((cursor, piece))
        cursor += piece
        remaining -= piece
    return out or [(start, duration)]


def split_for_hierarchy(start: Fraction, duration: Fraction,
                        numerator: int = 4, denominator: int = 4,
                        is_tuplet: bool = False) -> List[Tuple[Fraction, Fraction]]:
    """Split one sounding note into the tied chain the page should show.

    Returns [(start, duration), ...] in beat units, contiguous and summing to
    AT MOST the original duration - never more. A tail shorter than the
    shortest writable note value is dropped rather than absorbed by growing its
    neighbour; see _notatable_chain. A single-element result means the note
    already respects the hierarchy and is written as-is.

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

    # Every link in the chain must be writable. The hierarchy decides WHERE to
    # cut; this decides whether what it produced can be drawn.
    chain: List[Tuple[Fraction, Fraction]] = []
    for seg_start, seg_dur in segments:
        chain.extend(_notatable_chain(seg_start, seg_dur))
    return chain


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
