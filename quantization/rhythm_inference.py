# =================================================================
# MODULE: quantization/rhythm_inference.py
# Beat-level Bayesian rhythm inference - the sequence-level upgrade to
# notation_quantizer.py's per-note greedy snapping. For each beat, it
# asks "which symbolic rhythm most likely produced these onsets AND is
# something a musician would want to read", maximizing a posterior
#
#     P(S | X)  proportional to  P(X | S) * P(S)
#
# over a small vocabulary of candidate beat-fillings S, where:
#   - P(X | S), the LIKELIHOOD, is a Gaussian fit of the beat's observed
#     onset positions to the candidate's ideal positions (a spread wide
#     enough to absorb human microtiming). This is the "how well does
#     this reading explain the performance" term.
#   - P(S), the READABILITY PRIOR, is exp(-cost), where cost rises for
#     harder-to-read fillings (a triplet costs more than two eighths;
#     finer subdivisions cost more than coarser). This is the "what
#     would a copyist actually write" term - the essay's penalty scoring
#     function, kept minimal and with defensible RELATIVE orderings
#     rather than a sprawl of tuned magic numbers.
#
# WHY THIS BEATS notation_quantizer.py's per-note snap: rhythm is
# contextual. Three onsets at beat-fractions 0.0/0.34/0.66 are
# obviously ONE triplet as a group, but a per-note snapper rounds each
# to its nearest 16th independently and produces garbage. Deciding the
# whole beat at once is the only way to recover the triplet. Per-note
# snapping remains the fallback (notation_quantizer.notation_quantize_
# note) for beats this can't confidently parse.
#
# STYLE PRIOR FROM REAL EVIDENCE (the essay's Rule #6, grounded not
# guessed): swing is WRITTEN as straight eighths. A 0.0/0.68 onset pair
# is two swung eighths, notated 0.0/0.5 - not a dotted-eighth+sixteenth
# (0.0/0.75) and not a triplet. We already measure swing_ratio per
# track, so the two-eighths candidate's LIKELIHOOD is evaluated at the
# swung position while its OUTPUT notation stays straight. No swing
# magic number - the detected ratio drives it.
#
# SCOPE (honest): this decides symbolic ONSET + DURATION per beat, which
# is what MIDI can carry and what makes a clean notation import. It does
# NOT decide ties/beaming/voices (the host importer's job; MusicXML
# frontier), does NOT learn its prior from a score corpus (a separate
# research program - this prior is hand-authored), and does nothing for
# note OVER-DETECTION (it assumes the notes are real and only their
# rhythmic spelling is in question).
# =================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from core import Note
from core.musical_time import MusicalTime
from quantization.duration_witness import notatable_at_most

# Longest a single notehead may be held, in beats. The notatable table tops out
# at 8 (a double whole), and past a couple of bars a "sustain" is far more
# likely to be a detector that never closed the note than a real pedal point -
# so this is a safety rail on a stuck note, not a musical opinion.
MAX_SUSTAIN_BEATS = 8.0

# Microtiming spread (in beat-fraction units) the likelihood tolerates
# before a candidate stops "explaining" the onsets - matches
# duration_witness's 8%-of-the-beat tolerance philosophy.
ONSET_SIGMA_BEAT_FRACTION = 0.08

# Balances the readability prior against the timing likelihood. Modest
# by design: readability should break genuine near-ties and nudge
# ambiguous cases toward simplicity, NOT override a clearly-better
# timing fit. First-pass; the relative filling costs matter more than
# this global scale.
READABILITY_LAMBDA = 0.5

# A beat whose best candidate still fits this poorly (mean onset error,
# beat-fraction) is not confidently any known filling - caller falls
# back to per-note snapping rather than forcing a bad parse.
MAX_MEAN_ONSET_ERROR = 0.10

# How close to the next beat an onset may sit and still be counted as that
# beat's downbeat rather than the current beat's last subdivision. Half of the
# finest grid modelled (1/12), rounded to a sixteenth of a beat.
EDGE_SNAP_BEATS = 1.0 / 16.0


@dataclass(frozen=True)
class BeatFilling:
    """One candidate symbolic rhythm for a single beat. `onset_fractions`
    are the NOTATION positions (what gets written); `readability_cost`
    is the copyist's reading-load for this pattern (lower = simpler)."""
    name: str
    onset_fractions: Tuple[float, ...]
    is_tuplet: bool
    readability_cost: float
    # How many EQUAL parts this filling divides the beat into, when it does.
    # A bare is_tuplet flag says "not binary" but not "into how many", and the
    # page needs the number: a sextuplet's unit is 1/6 of a beat and a
    # quintuplet's is 1/5, which round to completely different noteheads. None
    # for the irregular hand-authored patterns, which are binary by
    # construction and need no divisor.
    divisor: Optional[int] = None

    @property
    def onset_count(self) -> int:
        return len(self.onset_fractions)


# The vocabulary, for a plain (quarter-note) beat. Deliberately the
# common cases a copyist actually uses - not an exhaustive rhythmic
# lattice. Costs: 1 per onset (fragmentation), +1 for a dotted/
# asymmetric split, +3 for a tuplet (the one big, universally-agreed
# reading cost). Relative orderings are engraving consensus; absolute
# values are first-pass.
_BEAT_VOCABULARY: Tuple[BeatFilling, ...] = (
    # NB: unreachable from infer_voice_rhythm, which only ever builds beats
    # that HAVE onsets - an empty beat simply has no entry. Kept because
    # infer_beat is a public function and a caller may legitimately ask it what
    # zero onsets mean; rests on the page come from the exporter's makeRests.
    BeatFilling("rest", (), False, 0.0),
    BeatFilling("quarter", (0.0,), False, 1.0),
    BeatFilling("two_eighths", (0.0, 0.5), False, 2.0),
    BeatFilling("dotted_eighth_sixteenth", (0.0, 0.75), False, 3.0),
    BeatFilling("sixteenth_dotted_eighth", (0.0, 0.25), False, 3.0),
    BeatFilling("eighth_triplet", (0.0, 1.0 / 3, 2.0 / 3), True, 5.0),
    BeatFilling("eighth_two_sixteenths", (0.0, 0.5, 0.75), False, 4.5),
    BeatFilling("two_sixteenths_eighth", (0.0, 0.25, 0.5), False, 4.5),
    BeatFilling("four_sixteenths", (0.0, 0.25, 0.5, 0.75), False, 4.0),
)

# THE BEAT AS A SUBDIVISION GRID WITH A STRUCK SUBSET.
#
# WHY THE FIXED VOCABULARY HAD TO GO (measured on Rubinstein Op.62/1, 2026-08-17).
# Every hand-authored filling above begins at 0.0, so the model could only
# describe a beat that STARTS WITH A NOTE. Measured on the beats carrying a
# single onset: 75% of them sit at or past 0.125 of the way through the beat -
# on the "and", on the second sixteenth, after a rest or under a note held over
# from the previous beat - and the median lone onset landed at 0.452. None of
# those had a candidate that could fit them, so a beat whose first event is not
# on the beat was unreadable by construction. That, not the tuplet ceiling, is
# what held beat-level coverage at 18.8%.
#
# THE MODEL. A beat is divided into `d` equal parts and ANY SUBSET of those d
# positions is struck. This is the decision a copyist actually makes - first
# the subdivision, then which slots carry notes and which carry rests - and it
# subsumes the whole old vocabulary by construction: dotted_eighth_sixteenth is
# {0, 3/4} on d=4, two_sixteenths_eighth is {0, 1/4, 1/2} on d=4,
# eighth_triplet is the full d=3 grid, four_sixteenths the full d=4. Nothing
# that used to be expressible stopped being expressible; rests inside the beat,
# offbeat entries and dense tuplet runs became expressible for the first time.
#
# Onsets are assigned to slots by an exact monotone injective alignment (the DP
# in _best_alignment), never by independent rounding - two onsets must not
# collapse onto one slot and their order must be preserved, or the reading is
# not a rhythm.
DIVISORS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
_BINARY_DIVISIONS = frozenset({1, 2, 4, 8})

# Reading cost, in the same currency as the old hand-authored table (1 per
# onset), plus the two terms the subset model makes explicit. Absolute values
# stay first-pass; the RELATIVE ordering is what decides anything, and it
# reproduces the old table's ordering to within ~0.5.
_TUPLET_PENALTY = 2.0     # a tuplet is the one big, universally-agreed cost

# SWING MAKES TERNARY THE NATIVE SUBDIVISION, NOT AN EXPENSIVE ONE.
#
# On a shuffle the beat IS divided in three, so charging a tuplet the full
# reading cost describes the wrong genre. MEASURED on Hopeful (a shuffle), the
# textbook shuffle beat - onsets at 0.0 and 0.667 - came out as a
# dotted-eighth-plus-sixteenth:
#     d=3  2_of_3[0,2]  sse 0.0000 (perfect)  cost 4.90  posterior -2.448
#     d=4  2_of_4[0,3]  sse 0.0069 (worse)    cost 3.50  posterior -2.288  WON
# A perfectly-fitting triplet lost to a worse-fitting binary reading on the
# prior alone. That is the prior overriding the evidence, which READABILITY_
# LAMBDA's own docstring says it must never do.
#
# swing_ratio is already MEASURED per track by rhythm_engine.estimate_groove,
# so the correction needs no new constant: the penalty on divisors of three
# fades to nothing as the measured swing approaches a true triplet feel. A
# straight track (swing 0.5) is completely unaffected.
SWING_STRAIGHT = 0.5              # even eighths
SWING_FULL_TRIPLET = 2.0 / 3.0    # a fully swung pair sits on the triplet grid


def _tuplet_penalty_for(divisor: int, swing_ratio: float) -> float:
    """The reading cost of a tuplet, discounted by how swung the track is."""
    if divisor % 3 != 0:
        return _TUPLET_PENALTY
    span = SWING_FULL_TRIPLET - SWING_STRAIGHT
    swung = (float(swing_ratio) - SWING_STRAIGHT) / span if span > 0 else 0.0
    return _TUPLET_PENALTY * max(0.0, min(1.0, 1.0 - swung))


# WRITE SWING AS TRIPLETS, NOT AS STRAIGHT EIGHTHS (user directive 2026-08-18:
# "a lot of rhythms in the song are triplets, as it is a shuffle - written and
# transcribed as 16th notes but they should be even 8th note triplets").
#
# Both are real conventions. Jazz lead sheets write a swung pair as two straight
# eighths and put "swing" at the top; the alternative spells the triplet out.
# They are incompatible - only one can be on the page - and the user has chosen
# the explicit triplet, which is also the more honest reading of what was
# played. Set False to restore the lead-sheet convention.
SWING_AS_TRIPLETS = True
# PER EMPTY SLOT, not a flat charge. A flat penalty made "3 notes scattered
# across 8 slots" cost the same as "3 notes filling 4", so a plain triplet came
# out as 3_of_8[0,3,5] - durations of 3/8, 2/8, 3/8, which no copyist writes -
# because the binary grid dodged the tuplet penalty by a hair. Every empty slot
# is another rest or tie the reader has to parse, so the charge scales with how
# many there are, and the triplet wins its own beat back.
_SPARSE_PENALTY = 0.5     # per unfilled slot
_FINENESS_WEIGHT = 0.25   # ...and a finer grid reads harder than a coarser one


# WHAT MAKES A TUPLET A TUPLET (user directive, 2026-08-18).
#
# "A triplet is 3 notes in the space of two... tuplets, if they are to exist,
# must be on some kind of beat to make sense within the context of a measure.
# Certain tuplets make sense only on the beat they are replacing, not displaced
# by some weird rest or jitter."
#
# That is a hard musical constraint and the subset model above violates it
# freely: nothing stopped it reading 3_of_6[1,3,5] - three notes scattered over
# a sextuplet grid with NOTHING ON THE BEAT - which is not a tuplet, it is a
# tuplet-shaped coincidence in the jitter. A tuplet is a REPLACEMENT for a
# beat's normal division, so it has to start where that beat starts and it has
# to actually fill it.
#
# Two conditions, both required, and only for non-binary subdivisions:
#   ANCHORED  slot 0 is struck - the tuplet begins on the beat it replaces.
#   FILLED    at least two thirds of its slots are struck, so it is the beat's
#             division and not a few notes that happen to land near thirds.
# A binary subdivision needs neither: 1_of_4[3] is just a sixteenth on the last
# sixteenth of the beat, an ordinary thing to write.
#
# Rejected candidates do not vanish - the beat is simply read on the best
# BINARY subdivision instead, which is the conservative direction and matches
# this project's standing bias: rather miss than hallucinate.
TUPLET_MIN_FILL = 2.0 / 3.0


def _tuplet_is_anchored(d: int, slots) -> bool:
    """May a `d`-part reading with these struck slots be written as a tuplet?

    WHAT IS ANCHORED IS THE FRAME, NOT THE FIRST NOTE (corrected 2026-08-18).
    The first version of this rule demanded a sounding note in slot 0, which
    made a triplet beginning on "ple" or "let" impossible to write - and that
    is ordinary notation: the container stays on the beat and the silent slots
    become TUPLET RESTS. Requiring a note there forced every such figure back
    onto the binary grid, which is the padding option, not the cleaner one.

    The frame is anchored by construction - the grid spans exactly one beat -
    so nothing needs checking for that. What does need checking is that this is
    genuinely the beat's DIVISION and not a few onsets that happen to land near
    thirds: hence the fill ratio, which is the only guard left and now carries
    the whole weight of distinguishing a real tuplet from a coincidence."""
    if d in _BINARY_DIVISIONS:
        return True
    return len(slots) >= math.ceil(TUPLET_MIN_FILL * d)


def _grid_positions(d):
    return tuple(i / d for i in range(d))


def _best_alignment(observed, d):
    """Assign each observed onset to a DISTINCT slot of a d-part beat, keeping
    order, minimising squared error. Returns (slot_indices, sse) or None when
    there are more onsets than slots.

    Independent nearest-slot rounding is not good enough: it happily maps two
    onsets onto one slot, which is not a rhythm, and it can invert their order.
    This is the standard monotone-alignment DP - O(n*d), nothing at these sizes."""
    n = len(observed)
    if n == 0 or n > d:
        return None
    pos = _grid_positions(d)
    INF = float("inf")
    cost = [[INF] * d for _ in range(n)]
    back = [[-1] * d for _ in range(n)]
    for k in range(d):
        cost[0][k] = (observed[0] - pos[k]) ** 2
    for j in range(1, n):
        best_prev, best_k = INF, -1
        for k in range(d):
            if k > 0 and cost[j - 1][k - 1] < best_prev:
                best_prev, best_k = cost[j - 1][k - 1], k - 1
            if best_prev < INF:
                cost[j][k] = best_prev + (observed[j] - pos[k]) ** 2
                back[j][k] = best_k
    end = min(range(d), key=lambda k: cost[n - 1][k])
    if cost[n - 1][end] == INF:
        return None
    slots = [0] * n
    slots[n - 1] = end
    for j in range(n - 1, 0, -1):
        slots[j - 1] = back[j][slots[j]]
    return tuple(slots), float(cost[n - 1][end])


_FULL_GRID_NAMES = {1: "quarter", 2: "two_eighths", 3: "eighth_triplet",
                    4: "four_sixteenths", 5: "quintuplet", 6: "sextuplet",
                    7: "septuplet", 8: "eight_thirty_seconds", 9: "nonuplet",
                    10: "decuplet", 12: "twelve_tuplet"}


def _filling_for(d, slots, swing_ratio: float = 0.5):
    """The BeatFilling naming one struck subset of a d-part beat."""
    n = len(slots)
    is_tuplet = d not in _BINARY_DIVISIONS
    cost = (float(n)
            + (_tuplet_penalty_for(d, swing_ratio) if is_tuplet else 0.0)
            + _SPARSE_PENALTY * (d - n)
            + _FINENESS_WEIGHT * math.log2(d))
    if n == d:
        name = _FULL_GRID_NAMES.get(d, "%d_even" % d)
    else:
        name = "%d_of_%d[%s]" % (n, d, ",".join(str(k) for k in slots))
    return BeatFilling(
        name=name,
        onset_fractions=tuple(k / d for k in slots),
        is_tuplet=is_tuplet,
        readability_cost=cost,
        divisor=d if is_tuplet else None,
    )


@dataclass(frozen=True)
class BeatRhythm:
    """The inferred rhythm for one beat: the winning filling plus its
    posterior score and mean fit, for the Music_Box trail."""
    filling: BeatFilling
    onset_fractions: Tuple[float, ...]   # notation positions actually used
    mean_onset_error: float
    log_posterior: float

    @property
    def divisor(self) -> Optional[int]:
        return self.filling.divisor


def _expected_fractions_for_likelihood(filling: BeatFilling, swing_ratio: float) -> Tuple[float, ...]:
    """Where this filling's onsets are EXPECTED to fall in the
    performance (for the likelihood), vs. where they're WRITTEN. Only
    the two-eighths filling diverges: on swung material its second onset
    is expected late (at swing_ratio of the beat) even though it's
    notated straight at 0.5. Everything else is written where it's
    played."""
    if filling.name == "two_eighths" and swing_ratio > 0.55:
        return (0.0, float(swing_ratio))
    return filling.onset_fractions


def infer_beat(observed_fractions: Sequence[float], swing_ratio: float = 0.5) -> Optional[BeatRhythm]:
    """Max-posterior reading of one beat, over every subdivision and every
    struck subset of it. Each observed value is a fraction in [0,1) of the
    beat. Returns None when nothing fits well enough to trust - the caller then
    falls back to per-note snapping."""
    observed = tuple(sorted(float(f) for f in observed_fractions))
    n = len(observed)
    if n == 0:
        return BeatRhythm(_BEAT_VOCABULARY[0], (), 0.0, 0.0)

    best = None
    for d in DIVISORS:
        aligned = _best_alignment(observed, d)
        if aligned is None:
            continue
        slots, sse = aligned
        if not _tuplet_is_anchored(d, slots):
            continue          # a displaced or half-empty tuplet is not a tuplet
        filling = _filling_for(d, slots, swing_ratio)

        # SWING: on swung material two eighths are PLAYED long-short and
        # WRITTEN even, so the likelihood is scored where they are played while
        # the emitted notation stays straight. The one place this model
        # deliberately scores against a position it will not write.
        if (not SWING_AS_TRIPLETS
                and d == 2 and n == 2 and slots == (0, 1) and swing_ratio > 0.55):
            swung = (0.0, float(swing_ratio))
            sse = sum((o - e) ** 2 for o, e in zip(observed, swung))

        log_likelihood = -sse / (2.0 * ONSET_SIGMA_BEAT_FRACTION ** 2)
        log_prior = -READABILITY_LAMBDA * filling.readability_cost
        log_posterior = log_likelihood + log_prior
        # RMS, not a mean - the name is kept for the field it feeds, so the
        # threshold is slightly stricter than "average error" suggests.
        error = math.sqrt(sse / n)
        if best is None or log_posterior > best.log_posterior:
            best = BeatRhythm(filling, filling.onset_fractions, error, log_posterior)

    if best is None or best.mean_onset_error > MAX_MEAN_ONSET_ERROR:
        return None
    return best


@dataclass(frozen=True)
class InferredNoteTiming:
    """One note's beat-inference result - same shape the notation_timing
    Annotation needs (start/end/symbolic), so the Conductor writes it
    identically to the per-note path. `is_tuplet` carries the beat-level
    tuplet VERDICT forward so the page never has to re-guess whether a note
    is a triplet from its rounded duration (that per-note guessing is the
    ReverseGeoCrypt antipattern: decide the lattice once, not per event)."""
    note_id: str
    notation_start_ms: float
    notation_end_ms: float
    reason: str
    is_tuplet: bool = False
    # Equal parts this beat was divided into (5 = quintuplet, 6 = sextuplet...).
    # None on a binary beat. The page needs the NUMBER, not just the flag - see
    # BeatFilling.divisor.
    tuplet_divisor: Optional[int] = None


def _sustained_end_ms(
        start_ms: float,
        performed_end_ms: float,
        next_onset_ms: float,
        beat_end_ms: float,
        beat_ms: float,
        allow_tuplet: bool,
        max_sustain_beats: float,
) -> Tuple[float, Optional[float]]:
    """Notation end for an event that is LAST in its beat, i.e. one that may
    genuinely be held past the beat line. Returns (end_ms, held_beats).

    Two ceilings, both hard: the note cannot outlast what was actually played
    (`performed_end_ms`), and it cannot reach the next onset in its own
    monophonic voice (`next_onset_ms`) or the page overlaps. Under those, the
    span is rounded DOWN to a notatable value - down, not nearest, because
    rounding up would breach the ceiling that was just enforced.

    The floor is the rest of the beat: a note that dies early inside its own
    beat is still written to the beat line, which is what the beat vocabulary
    already decided and what a copyist writes. So this only ever LENGTHENS
    relative to the old beat-line truncation, never shortens."""
    ceiling_ms = min(performed_end_ms, next_onset_ms)
    to_beat_line = beat_end_ms - start_ms
    available_beats = (ceiling_ms - start_ms) / beat_ms
    if available_beats <= to_beat_line / beat_ms:
        return beat_end_ms, None

    held = notatable_at_most(min(available_beats, max_sustain_beats), allow_tuplet)
    if held * beat_ms <= to_beat_line:
        return beat_end_ms, None
    return start_ms + held * beat_ms, held


def _chord_group_onsets(notes: Sequence[Note], chord_tolerance_ms: float) -> List[List[Note]]:
    """Collapses near-simultaneous notes (a chord) into ONE rhythmic
    event - a chord is one onset to the rhythm, not several. Notes are
    grouped in time order; each group shares one rhythmic position."""
    groups: List[List[Note]] = []
    for note in sorted(notes, key=lambda x: x.start_ms):
        if groups and note.start_ms - groups[-1][0].start_ms <= chord_tolerance_ms:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def infer_voice_rhythm(
        notes: Sequence[Note],
        beat_ms: float,
        grid_origin_ms: float,
        swing_ratio: float = 0.5,
        chord_tolerance_ms: float = 40.0,
        max_sustain_beats: float = MAX_SUSTAIN_BEATS,
        musical_time: Optional[MusicalTime] = None,
) -> Dict[str, InferredNoteTiming]:
    """Beat-level rhythm inference for one voice's notes (intended per
    stem/line - a 6-stem separation makes 'one stem ~ one voice'
    reasonable). Returns {note_id: InferredNoteTiming} for the notes
    whose beat parsed confidently; note ids absent from the result had
    no confident beat parse and should fall back to per-note snapping.

    Chords (near-simultaneous notes) count as ONE rhythmic onset and all
    receive that onset's inferred position/duration.

    HOW LONG A NOTE IS WRITTEN (fixed 2026-08-17 audit). An event that is
    followed by another event INSIDE its own beat runs to that next onset -
    that is what the beat vocabulary decided and it is correct. An event that
    is LAST in its beat used to be truncated at the beat line, which made one
    beat the longest value this function could ever emit: a four-beat whole
    note came out as a quarter followed by three beats of rest, on the path
    the Conductor prefers. Since rests are then filled or drawn downstream,
    that manufactured a large share of the page's rests all by itself.

    A last-in-beat event now sustains to the SMALLER of (a) the next rhythmic
    onset anywhere in this voice and (b) its own performed end, and that span
    is snapped to a notatable value (quantization.duration_witness's one
    table). The result is still a clean symbolic duration - it just isn't
    capped at a quarter any more. It never overlaps the next onset, so the
    voice still tiles its bars.

    `musical_time` is the map between clock time and musical position. Given
    one, beats are the beats that were actually TRACKED, so a beat inside a
    ritardando is as long as it really was; without one this falls back to an
    isochronous grid of `beat_ms` from `grid_origin_ms`, which is the old
    behaviour exactly."""
    if beat_ms <= 0 or not notes:
        return {}

    use_map = musical_time is not None and musical_time.usable

    def beat_position(t_ms: float) -> float:
        if use_map:
            return musical_time.to_beats(t_ms)
        return (t_ms - grid_origin_ms) / beat_ms

    def beat_index_of(t_ms: float) -> int:
        """Which beat this onset belongs to.

        A PLAIN FLOOR IS WRONG AT THE EDGE. An onset a few milliseconds early -
        which is most of them, since players and onset detectors both lead the
        beat - sits at ~0.97 of the PREVIOUS beat, and the reader downstream
        then writes it as that beat's last thirty-second instead of as the next
        beat's downbeat. MEASURED on Op.62/1: 13.9% of all onset groups landed
        in the final sixteenth of a beat, a spike of 163 mirroring the 185 in
        the first sixteenth - the same musical event, split by the boundary,
        and the source of the spurious `slot 7` readings that dominated the
        first pass at this model.

        So an onset within EDGE_SNAP_BEATS of the next beat belongs to it. The
        tolerance is half of the finest subdivision this module models, which
        is the distance at which the onset is nearer the next downbeat than any
        slot of the beat it is nominally in."""
        pos = beat_position(t_ms)
        idx = int(math.floor(pos))
        return idx + 1 if (pos - idx) > 1.0 - EDGE_SNAP_BEATS else idx

    def beat_bounds(beat_idx: int) -> Tuple[float, float]:
        if use_map:
            return (float(musical_time.to_ms(beat_idx)),
                    float(musical_time.to_ms(beat_idx + 1)))
        start = grid_origin_ms + beat_idx * beat_ms
        return start, start + beat_ms

    # Bucket rhythmic onset-groups by beat index, keeping each group's position
    # in the VOICE-WIDE order - a sustained note's ceiling is the next onset
    # anywhere in this voice, which is usually in a later beat.
    groups = _chord_group_onsets(notes, chord_tolerance_ms)
    next_onset_ms: List[float] = [
        groups[i + 1][0].start_ms for i in range(len(groups) - 1)
    ] + [float("inf")]

    by_beat: Dict[int, List[Tuple[int, List[Note]]]] = {}
    for gi, group in enumerate(groups):
        by_beat.setdefault(beat_index_of(group[0].start_ms), []).append((gi, group))

    result: Dict[str, InferredNoteTiming] = {}
    for beat_idx, beat_groups in by_beat.items():
        beat_start, beat_end = beat_bounds(beat_idx)
        this_beat_ms = max(1.0, beat_end - beat_start)
        # An onset snapped forward across the boundary reads as a small
        # NEGATIVE fraction of its new beat; clamp it to the downbeat, which is
        # what it is.
        observed_fracs = [
            min(0.999, max(0.0, (g[0].start_ms - beat_start) / this_beat_ms))
            for _gi, g in beat_groups
        ]
        rhythm = infer_beat(observed_fracs, swing_ratio)
        if rhythm is None:
            continue  # caller falls back to per-note snapping for this beat's notes

        onset_ms = [beat_start + f * this_beat_ms for f in rhythm.onset_fractions]
        for i, (gi, group) in enumerate(beat_groups):
            start = onset_ms[i]
            if i + 1 < len(onset_ms):
                # Another event inside this beat - the vocabulary already said
                # where it lands, and that is this note's end.
                end = onset_ms[i + 1]
                sustained_beats = None
            elif rhythm.filling.divisor:
                # A TUPLET NEVER CROSSES A BEAT, so it can never cross a
                # barline either. Sustaining the last note of a tuplet past the
                # beat would put part of the group in the next bar, which is
                # not something a tuplet bracket can express - music21 splits
                # it and emits the nonsense ratios we spent this session
                # removing. The note stops at the beat line; if it really rang
                # on, a tie in the next bar is the correct way to say so.
                end, sustained_beats = beat_end, None
            else:
                end, sustained_beats = _sustained_end_ms(
                    start_ms=start,
                    performed_end_ms=max(n.end_ms for n in group),
                    next_onset_ms=next_onset_ms[gi],
                    beat_end_ms=beat_end,
                    beat_ms=this_beat_ms,
                    allow_tuplet=rhythm.filling.is_tuplet,
                    max_sustain_beats=max_sustain_beats,
                )
            reason = (f"beat rhythm '{rhythm.filling.name}' "
                      f"(fit {rhythm.mean_onset_error:.3f}, {len(beat_groups)} onset(s) in beat)")
            if sustained_beats is not None and sustained_beats > 1.0:
                reason += f"; held {sustained_beats:g} beats to its own end"
            for note in group:
                result[note.id] = InferredNoteTiming(
                    note_id=note.id,
                    notation_start_ms=start,
                    notation_end_ms=end,
                    reason=reason,
                    is_tuplet=rhythm.filling.is_tuplet,
                    tuplet_divisor=rhythm.filling.divisor,
                )

    return result


__all__ = [
    "BeatFilling",
    "BeatRhythm",
    "InferredNoteTiming",
    "infer_beat",
    "infer_voice_rhythm",
    "DIVISORS",
    "TUPLET_MIN_FILL",
    "SWING_AS_TRIPLETS",
    "ONSET_SIGMA_BEAT_FRACTION",
    "READABILITY_LAMBDA",
]
