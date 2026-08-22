# =================================================================
# MODULE: output/playability.py
# The playability checker (GRIMLOCK_6.0_PERFORMANCE_ENGRAVING.md §12).
#
# WHY THIS EXISTS: §VI of the open-problems doc laments that engraving has
# no loss function - "the ear is the only oracle." Playability is a partial
# escape on the one axis that needs no ear: a page a human physically
# cannot play is WRONG, computably. Fifteen notes across four octaves
# struck at once is not a hard chord - it is impossible, and its
# impossibility is a number.
#
# ---------------------------------------------------------------------
# WHAT THE FIRST VERSION GOT WRONG, and how we know
# ---------------------------------------------------------------------
# It swept over SOUNDING sets: every instant asked "can two hands hold
# everything currently ringing?" On a pedalled instrument that is the wrong
# question, and the error is large. Measured against the published edition
# of Chopin's Nocturne Op.62 No.1 and against our own detections:
#
#     % of instants called impossible     struck together   sounding together
#     ANSWER KEY (published edition)             2.4%              2.0%
#     KLANGIO                                    5.2%              3.8%
#     OURS                                       4.4%             13.8%
#
# Two thirds of our apparent unplayability was the sweep, not the music. A
# pianist strikes C2, lifts, strikes C6 half a second later, and the pedal
# holds both - a four-octave "simultaneity" that is entirely ordinary.
#
# The old header also carried a false guarantee: it claimed that because
# the pedal can only make things MORE playable, "every unplayable here is a
# true lower bound." That is backwards for a sounding-set model. Not
# modelling the pedal made the check STRICTER than physics, so the count
# was an upper bound wearing a lower bound's label.
#
# ---------------------------------------------------------------------
# THE MODEL NOW
# ---------------------------------------------------------------------
# 1. INSTANTS ARE STRIKES, not sounding intervals. Fingers are needed to
#    STRIKE; the pedal does the holding. This buys pedal masking and rolled
#    chords together, and it needs no pedal timeline - which is just as
#    well, because nothing upstream produces CC64 and inventing one would
#    be a fiction with a config knob on it. `build_timeline` (the sounding
#    sweep) is kept and still correct for its own question, but it is
#    documented as the UPPER bound it always was.
#
# 2. A HAND IS NOT A BLOCK. Total span is a poor constraint on its own: it
#    passes a five-note cluster and it fails a real tenth. Real hands have
#    ONE big gap available - thumb to the rest - while the remaining
#    adjacent fingers stay close. So a hand is feasible when its total span
#    fits, its single largest adjacent gap may be anything up to that span,
#    and every OTHER adjacent gap is small. This is what lets the outer
#    span be loosened for virtuoso repertoire without the model degenerating
#    into "any ten notes inside three octaves are fine."
#
# 3. CROSSING IS BOUNDED, NOT FREE. Hands cross - left hand takes a low
#    bass and reaches over the right for a high melody note. So the split is
#    allowed ONE crossing: each hand is at most two contiguous runs of the
#    sorted pitch set. Fully free subset assignment was considered and
#    rejected: within a single STRUCK chord fingers physically occupy keys
#    and cannot interleave arbitrarily, and allowing it makes almost any ten
#    notes inside ~28 semitones pass, which costs the model the only thing
#    it has - the ability to be wrong. Interlocking in Ravel and Liszt is a
#    SEQUENTIAL gesture between strikes, and the strike-instant model above
#    already covers it.
#
#    Building it turned up the argument's own punchline: crossing barely
#    matters for a SIMULTANEOUS chord at all. A crossed hand still has to
#    span its own two extremes, so the crossed assignment can only ever help
#    when the whole chord already fits inside one hand's span - a narrow
#    window, and one where a contiguous split usually works anyway. The
#    permission is kept because it costs nothing, but anyone hoping free
#    interleaving would unlock virtuoso repertoire should know that the
#    mechanism they are reaching for does not do that; the strike grouping
#    is what does it.
#
# 4. IMPOSSIBLE AND HARD ARE DIFFERENT VERDICTS. Stride leaps, Mephisto-
#    Waltz jumps and repeated-note tremolo are not impossibility, they are
#    difficulty, and they live between instants rather than inside one.
#    They are reported separately and never fold into the pass rate.
#    Only impossibility can ever be used as evidence about DETECTION;
#    difficulty is a property of the music and says nothing about whether
#    we transcribed it correctly.
#
# ---------------------------------------------------------------------
# WHAT THIS MODULE DELIBERATELY REFUSES TO KNOW
# ---------------------------------------------------------------------
# Jazz-pedagogy constraints - rootless voicing register zones (C3-F4),
# left-hand guide-tone completeness, low-root clash with an upright bass -
# were proposed and are not here. They are IDIOMATIC judgments, not
# physical ones. A voicing that violates Levine is not wrong, it is merely
# not how one player would have voiced it, and folding that into the same
# verdict as "no human hand can do this" would make the output
# uninterpretable: a caller could no longer tell whether a flagged bar is
# a detection error or a stylistic opinion. The stride LEAP is physical and
# is modelled; the register etiquette around it is not this module's
# business. If that analysis is wanted it belongs in its own module with
# its own name.
#
# SCOPE, which the caller owns: this asks whether ONE PLAYER's two hands
# can do something. Feeding it the union of every stem in an ensemble asks
# whether one pianist can play the whole band, which is a meaningless
# question with a confident-looking answer. Pass one instrument's notes.
#
# THAT WARNING WAS EARNED THE SECOND TIME TOO (2026-08-21). A diagnostic
# written to evaluate the register-split shape term ran assign_hands over
# EVERY part of a five-part ensemble score, bass and vocal lines included -
# parts no pianist plays and which notation_score never routes to a grand
# staff. It reported 14 unplayable chords on each of Hopeful and HRV. Scoped
# to the parts that actually get a grand staff (other/guitar/piano) the true
# figures are 6 and 3. More than half of that "unplayability" was the
# measurement asking the meaningless question, in the exact shape this
# paragraph describes.
#
# STILL A DIAGNOSTIC. Pure - tuples in, a report out, no output changed. It
# is built to become the objective function the register-split reducer
# optimizes against (§11.3.3) without dragging music21 or audio into core.
# =================================================================

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Verdicts. Only IMPOSSIBLE is a claim about physics; ROLLED is a claim
# that the notes were not actually struck together.
PLAYABLE = "playable"
ROLLED = "rolled"
IMPOSSIBLE = "impossible"


@dataclass(frozen=True)
class PlayabilityConfig:
    """Physical limits of one pianist's two hands.

    Defaults are the 'comfortable' bracket. Callers are expected to run a
    RANGE (see PROFILES) rather than trust one threshold - a single number
    here would be pretending to a precision nobody has.
    """
    # --- one hand, one strike -------------------------------------------
    # The DEFAULT is the widest bracket, not the average hand, and that is
    # deliberate. Calibration (tools/playability_calibrate.py) puts the
    # published edition at 1.4% "impossible" under a 14st span and 0.1%
    # under 17st. Everything downstream reads a positive from this model as
    # a hard claim about physics, so the default has to be the setting
    # where a positive means something. Use PROFILES["comfortable"] when
    # the question is how COMFORTABLE a page is - a different question.
    max_span_semitones: int = 17          # an eleventh: the virtuoso bracket
    max_fingers_per_hand: int = 5
    max_total_notes: int = 10             # ten fingers, hard ceiling

    # The inner-span constraint. Between thumb and the rest of the hand
    # lives one large gap; fingers 2-3-4-5 sit close together. Allowing one
    # unbounded gap and bounding the others is what distinguishes a real
    # tenth (C-E-G-C, one wide thumb reach) from an impossible cluster of
    # the same total span. 6 semitones between adjacent fingers is
    # already a stretch; it is the loose end of the bracket for the same
    # reason the span is.
    max_adjacent_gap_semitones: int = 6

    # 0 = each hand is one contiguous register. 1 = one hand may cross the
    # other (two contiguous runs). Above 1 is not offered; see header.
    max_crossings: int = 1

    # --- what counts as struck together ---------------------------------
    # Onsets within this are one gesture. Onsets further apart than this
    # are separate strikes and the pedal holds the earlier ones.
    strike_window_beats: float = 0.125    # a 32nd at the notated tempo
    # Inside one gesture, notes offset by more than this read as a ROLL
    # (arpeggiando) rather than a block chord.
    roll_tolerance_beats: float = 0.02

    # --- difficulty, NOT impossibility ----------------------------------
    # A stride left hand travels two octaves in an eighth note at 200bpm.
    # That is hard, not impossible, and it must never fail an instant.
    leap_semitones: int = 20              # jump size that starts to count
    leap_within_beats: float = 0.5        # ...if it happens inside this
    # Single-finger repetition ceiling. Faster than this needs alternating
    # fingers, which needs a spare finger to alternate WITH.
    max_repetition_hz: float = 12.0
    beats_per_second: float = 2.0         # 120bpm; only used for the above


# A small, deliberately short list. One profile per composer is how a model
# stops being falsifiable - these are two brackets, not a taxonomy.
PROFILES: Dict[str, PlayabilityConfig] = {
    # What an average adult hand does without strain.
    "comfortable": PlayabilityConfig(max_span_semitones=12),
    # The bracket the standard repertoire actually sits in.
    "standard": PlayabilityConfig(max_span_semitones=14),
    # Chopin/Liszt/Rachmaninoff: tenths and elevenths are written as block
    # chords and expected to be taken. The inner-gap rule is what keeps
    # this from waving everything through.
    "virtuoso": PlayabilityConfig(max_span_semitones=17,
                                  max_adjacent_gap_semitones=6),
}

# Measured floors, tools/playability_calibrate.py on Chopin Op.62 No.1,
# struck-instant mode. The published edition is by definition playable, so
# this column IS the model's false-positive rate:
#
#     profile        ANSWER KEY   KLANGIO   OURS
#     comfortable          1.7%      2.9%   0.6%
#     standard             1.4%      2.5%   0.6%
#     virtuoso             0.1%      0.7%   0.5%
#
# Two things follow. First, only "virtuoso" is calibrated for this
# repertoire - the tighter brackets measure comfort, not possibility.
# Second, and this is the finding that matters: at the calibrated bracket
# our own impossibility (0.5%) sits barely above the edition's, so physical
# impossibility is NOT a useful over-detection signal on this material. It
# was worth building to find that out, and the number is the point.


@dataclass(frozen=True)
class Instant:
    """One evaluated moment, plus its verdict.

    In STRUCK mode `pitches` is what was struck together and [start, end)
    runs to the next strike. In SOUNDING mode it is what was ringing.
    """
    start: float                      # beats (quarterLengths), absolute
    end: float
    pitches: Tuple[int, ...]          # unique, sorted
    playable: bool                    # PLAYABLE or ROLLED both count as True
    split: Optional[int]              # notes taken by the low hand, if playable
    reason: str
    verdict: str = PLAYABLE
    hands: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def n(self) -> int:
        return len(self.pitches)

    @property
    def span(self) -> int:
        return (self.pitches[-1] - self.pitches[0]) if self.pitches else 0


# ---------------------------------------------------------------------
# one hand
# ---------------------------------------------------------------------

def hand_feasible(pitches: Sequence[int],
                  cfg: PlayabilityConfig) -> Tuple[bool, str]:
    """Can ONE hand strike these pitches at once?

    Three constraints, all necessary. Finger count and total span are the
    obvious ones. The third - at most one large gap, the thumb's - is what
    makes the check mean anything once the total span is loosened for
    virtuoso repertoire: without it, `max_span_semitones=17` waves through
    five-note clusters that no hand can shape.
    """
    if not pitches:
        return True, "empty"
    if len(pitches) > cfg.max_fingers_per_hand:
        return False, f"{len(pitches)} notes > {cfg.max_fingers_per_hand} fingers"

    lo, hi = pitches[0], pitches[-1]
    if hi - lo > cfg.max_span_semitones:
        return False, f"span {hi - lo}st > {cfg.max_span_semitones}st"

    gaps = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    if len(gaps) > 1:
        # The single widest gap is the thumb's and is already bounded by
        # the total span; every other pair of adjacent fingers must be
        # close. Dropping the max (rather than assuming it is the first
        # gap) is what handles a hand whose stretch is at the top.
        rest = sorted(gaps)[:-1]
        if rest and max(rest) > cfg.max_adjacent_gap_semitones:
            return (False,
                    f"inner gap {max(rest)}st between adjacent fingers > "
                    f"{cfg.max_adjacent_gap_semitones}st (total span "
                    f"{hi - lo}st is reachable, the shape is not)")
    return True, "ok"


# ---------------------------------------------------------------------
# two hands
# ---------------------------------------------------------------------

def _splits(n: int, max_crossings: int):
    """Yield (low_indices, high_indices) hand assignments.

    With `max_crossings=0` each hand is one contiguous run: the classic
    single split point. With 1, one hand may be TWO runs - `uniq[:i]`
    plus `uniq[j:]` - which is a hand reaching over the other. That is
    the whole relaxation; see the header for why it stops there.
    """
    for i in range(n + 1):
        if max_crossings <= 0:
            yield tuple(range(i)), tuple(range(i, n))
            continue
        for j in range(i, n + 1):
            outer = tuple(range(i)) + tuple(range(j, n))
            inner = tuple(range(i, j))
            yield outer, inner


def two_hand_feasible(
        pitches: Iterable[int], cfg: PlayabilityConfig,
) -> Tuple[bool, Optional[int], str]:
    """Can two hands STRIKE this pitch set at once?

    Returns (playable, low_hand_note_count, reason). The count is kept for
    backward compatibility with callers that read it as a split index; use
    `assign_hands` when the actual grouping is wanted.
    """
    ok, hands, reason = assign_hands(pitches, cfg)
    return ok, (len(hands[0]) if hands else None), reason


def assign_hands(
        pitches: Iterable[int], cfg: PlayabilityConfig,
) -> Tuple[bool, Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]], str]:
    """As `two_hand_feasible`, but returns the two hands it found."""
    uniq = sorted(set(int(p) for p in pitches))
    n = len(uniq)
    if n == 0:
        return True, ((), ()), "silence"
    if n > cfg.max_total_notes:
        return False, None, f"{n} simultaneous notes > {cfg.max_total_notes} fingers"

    best: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None
    for a_idx, b_idx in _splits(n, cfg.max_crossings):
        a = tuple(uniq[i] for i in a_idx)
        b = tuple(uniq[i] for i in b_idx)
        if not hand_feasible(a, cfg)[0] or not hand_feasible(b, cfg)[0]:
            continue
        # Prefer the assignment with no crossing, so the reported split is
        # the natural reading of the chord when a natural reading exists.
        crossed = bool(a) and bool(b) and not (max(a) < min(b) or max(b) < min(a))
        low, high = (a, b) if (not a or not b or a[0] <= b[0]) else (b, a)
        if not crossed:
            return True, (low, high), "ok"
        if best is None:
            best = (low, high)
    if best is not None:
        return True, best, "ok (hands crossed)"

    total_span = uniq[-1] - uniq[0]
    return (False, None,
            f"{n} notes spanning {total_span}st: no two-hand split within "
            f"{cfg.max_span_semitones}st / {cfg.max_fingers_per_hand}-finger limits")


def _excess_for_finger_count(n: int, cfg: PlayabilityConfig) -> int:
    """A hard floor on how many notes MUST be dropped to become playable:
    anything over ten fingers, guaranteed. (Span failures need dropping
    too, but this is the undeniable minimum - useful as a reduction-
    workload floor without over-claiming.)"""
    return max(0, n - cfg.max_total_notes)


# ---------------------------------------------------------------------
# timelines
# ---------------------------------------------------------------------

def build_strike_timeline(
        notes: Iterable[Tuple[float, float, int]], cfg: PlayabilityConfig,
) -> List[Instant]:
    """Group notes by ONSET and evaluate each struck gesture. The default,
    and the only one of the two that answers a question about fingers.

    A gesture that fails as a block chord is re-tested as a ROLL: if the
    onsets inside it are not all coincident, and every sub-chord that
    IS coincident is feasible on its own, the chord is arpeggiated and the
    pedal holds it. That is how a Rachmaninoff tenth or a Liszt spread
    gets played, and it costs no pedal timeline to say so.
    """
    events = sorted((float(s), int(p)) for s, e, p in notes if e > s)
    if not events:
        return []

    groups: List[List[Tuple[float, int]]] = []
    current: List[Tuple[float, int]] = []
    anchor = None
    for t, p in events:
        # Anchored on the group's FIRST onset, never the previous note, so
        # a dense run cannot chain into one arbitrarily long "instant".
        if anchor is None or t - anchor <= cfg.strike_window_beats:
            if anchor is None:
                anchor = t
            current.append((t, p))
        else:
            groups.append(current)
            current, anchor = [(t, p)], t
    if current:
        groups.append(current)

    timeline: List[Instant] = []
    for gi, group in enumerate(groups):
        start = group[0][0]
        end = groups[gi + 1][0][0] if gi + 1 < len(groups) else max(
            float(e) for _s, e, _p in notes)
        pitches = tuple(sorted({p for _t, p in group}))

        ok, hands, reason = assign_hands(pitches, cfg)
        if ok:
            timeline.append(Instant(start, end, pitches, True,
                                    len(hands[0]) if hands else None,
                                    reason, PLAYABLE, hands))
            continue

        # Rolled? Split the gesture into its coincident sub-chords.
        sub: Dict[float, List[int]] = defaultdict(list)
        for t, p in group:
            key = round((t - start) / max(cfg.roll_tolerance_beats, 1e-9))
            sub[key].append(p)
        if len(sub) > 1 and all(assign_hands(tuple(sorted(set(v))), cfg)[0]
                                for v in sub.values()):
            timeline.append(Instant(
                start, end, pitches, True, None,
                f"rolled: {len(pitches)} notes spanning "
                f"{pitches[-1] - pitches[0]}st struck as {len(sub)} sub-chords, "
                f"held by the pedal", ROLLED, None))
            continue

        timeline.append(Instant(start, end, pitches, False, None,
                                reason, IMPOSSIBLE, None))
    return timeline


def build_timeline(
        notes: Iterable[Tuple[float, float, int]], cfg: PlayabilityConfig,
) -> List[Instant]:
    """Sweep-line over SOUNDING sets. Kept because it answers a real
    question - "how dense is the page at this moment" - but it is NOT a
    finger-count model: everything ringing under the pedal is counted as
    though a finger were holding it. On the Chopin edition this calls 2.0%
    of instants impossible where the strike model calls 2.4%, and on our
    own detections 13.8% where the strike model says 4.4%. Treat any
    unplayable count from here as an upper bound.
    """
    starts: Dict[float, List[int]] = defaultdict(list)
    ends: Dict[float, List[int]] = defaultdict(list)
    times = set()
    for s, e, p in notes:
        if e <= s:
            continue                                  # skip zero/negative (grace notes etc.)
        starts[s].append(int(p))
        ends[e].append(int(p))
        times.add(s)
        times.add(e)

    ordered = sorted(times)
    active: Counter = Counter()
    timeline: List[Instant] = []
    for i, t in enumerate(ordered):
        for p in ends.get(t, []):                     # releases first, then presses
            active[p] -= 1
            if active[p] <= 0:
                del active[p]
        for p in starts.get(t, []):
            active[p] += 1
        if i + 1 >= len(ordered):
            break
        nxt = ordered[i + 1]
        pitches = tuple(sorted(active.keys()))
        if not pitches:
            continue                                  # silence: skip, don't inflate denominators
        ok, split, reason = two_hand_feasible(pitches, cfg)
        timeline.append(Instant(t, nxt, pitches, ok, split, reason,
                                PLAYABLE if ok else IMPOSSIBLE, None))
    return timeline


# ---------------------------------------------------------------------
# difficulty - between instants, never folded into the pass rate
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Strain:
    """One hard-but-possible moment. Never fails an instant."""
    start: float
    kind: str                         # "leap" | "repetition"
    magnitude: float                  # semitones, or Hz
    detail: str


def find_strain(timeline: List[Instant], cfg: PlayabilityConfig) -> List[Strain]:
    """Difficulty between consecutive strikes.

    LEAPS: the left-hand centroid travelling far, fast. This is stride -
    a low root on 1 and 3, a mid-register chord on 2 and 4, two octaves of
    travel in an eighth note at tempo. Art Tatum is hard, not impossible,
    and a model that failed him would be measuring the wrong thing.

    REPETITION: one pitch re-struck faster than a single finger can go.
    An earlier version gated this on having "no spare finger to alternate
    with", computed as ten minus the notes in the chord. That gate was
    unsatisfiable - it needed more repeated pitches than free fingers,
    which never happens - so the rule silently never fired on any source.
    A rule that cannot fire is not a conservative rule, it is dead code
    wearing a caveat. Report the rate and let the caller judge; alternating
    fingers roughly doubles the ceiling, which is why this is strain and
    never impossibility.
    """
    out: List[Strain] = []
    prev: Optional[Instant] = None
    for inst in timeline:
        if prev is not None and prev.hands and inst.hands:
            dt = inst.start - prev.start
            for which, idx in (("left", 0), ("right", 1)):
                a, b = prev.hands[idx], inst.hands[idx]
                if not a or not b:
                    continue
                jump = abs(sum(b) / len(b) - sum(a) / len(a))
                if jump >= cfg.leap_semitones and 0 < dt <= cfg.leap_within_beats:
                    out.append(Strain(
                        inst.start, "leap", jump,
                        f"{which} hand travels {jump:.0f}st in {dt:.3f} beats "
                        f"- playable, but a stride-tier leap"))

        if prev is not None:
            dt = inst.start - prev.start
            repeated = set(prev.pitches) & set(inst.pitches)
            if dt > 0 and repeated:
                hz = cfg.beats_per_second / dt
                if hz > cfg.max_repetition_hz:
                    out.append(Strain(
                        inst.start, "repetition", hz,
                        f"pitch(es) {sorted(repeated)} re-struck at {hz:.1f}Hz "
                        f"- above the single-finger ceiling of "
                        f"{cfg.max_repetition_hz:.0f}Hz, so it needs "
                        f"alternating fingers"))
        prev = inst
    return out


# ---------------------------------------------------------------------
# report
# ---------------------------------------------------------------------

@dataclass
class PlayabilityReport:
    config: PlayabilityConfig
    mode: str = "struck"
    sounding_beats: float = 0.0
    playable_beats: float = 0.0
    sounding_instants: int = 0
    playable_instants: int = 0
    rolled_instants: int = 0
    max_simultaneity: int = 0
    max_span: int = 0
    # beats spent unplayable, split by cause
    unplayable_beats_finger_count: float = 0.0        # > max_total_notes
    unplayable_beats_span_or_split: float = 0.0       # <= 10 notes but no legal split
    # reduction-workload floor: (notes-over-ten) weighted by beats
    excess_notehead_beats: float = 0.0
    # time-weighted histogram of simultaneity, by bucket label
    simultaneity_beats: Dict[str, float] = field(default_factory=dict)
    worst: List[Instant] = field(default_factory=list)
    # HARD, not impossible - reported alongside, never inside, the rates
    strain: List[Strain] = field(default_factory=list)

    @property
    def time_pass_rate(self) -> float:
        return self.playable_beats / self.sounding_beats if self.sounding_beats else 1.0

    @property
    def instant_pass_rate(self) -> float:
        return self.playable_instants / self.sounding_instants if self.sounding_instants else 1.0

    @property
    def mean_simultaneity(self) -> float:
        if not self.sounding_beats:
            return 0.0
        total = sum(_BUCKET_MID.get(k, 0.0) * v for k, v in self.simultaneity_beats.items())
        return total / self.sounding_beats

    @property
    def leaps(self) -> List[Strain]:
        return [s for s in self.strain if s.kind == "leap"]

    @property
    def repetitions(self) -> List[Strain]:
        return [s for s in self.strain if s.kind == "repetition"]


# Bucket edges for the simultaneity histogram, and a rough midpoint per
# bucket for a time-weighted mean estimate.
_BUCKETS = [(1, 5, "1-5"), (6, 8, "6-8"), (9, 10, "9-10"),
            (11, 15, "11-15"), (16, 10 ** 6, "16+")]
_BUCKET_MID = {"1-5": 3.0, "6-8": 7.0, "9-10": 9.5, "11-15": 13.0, "16+": 18.0}


def _bucket(n: int) -> str:
    for lo, hi, label in _BUCKETS:
        if lo <= n <= hi:
            return label
    return "1-5"


def summarize(timeline: List[Instant], cfg: PlayabilityConfig,
              worst_k: int = 8, mode: str = "struck") -> PlayabilityReport:
    rep = PlayabilityReport(config=cfg, mode=mode)
    for inst in timeline:
        d = inst.duration
        rep.sounding_beats += d
        rep.sounding_instants += 1
        rep.max_simultaneity = max(rep.max_simultaneity, inst.n)
        rep.max_span = max(rep.max_span, inst.span)
        rep.simultaneity_beats[_bucket(inst.n)] = (
            rep.simultaneity_beats.get(_bucket(inst.n), 0.0) + d)
        rep.excess_notehead_beats += _excess_for_finger_count(inst.n, cfg) * d
        if inst.verdict == ROLLED:
            rep.rolled_instants += 1
        if inst.playable:
            rep.playable_beats += d
            rep.playable_instants += 1
        elif inst.n > cfg.max_total_notes:
            rep.unplayable_beats_finger_count += d
        else:
            rep.unplayable_beats_span_or_split += d

    rep.worst = sorted(
        (i for i in timeline if not i.playable),
        key=lambda i: (i.n, i.span, i.duration), reverse=True,
    )[:worst_k]
    rep.strain = find_strain(timeline, cfg)
    return rep


def assess(notes: Iterable[Tuple[float, float, int]],
           cfg: Optional[PlayabilityConfig] = None,
           worst_k: int = 8,
           mode: str = "struck") -> PlayabilityReport:
    """End-to-end: (start, end, pitch) triples in beats -> a report.

    `mode="struck"` (default) groups by onset and asks what fingers must
    do. `mode="sounding"` restores the old sweep over ringing notes; it
    reports a strictly harsher number and is not a finger model.

    Pass ONE instrument's notes. See the header on scope.
    """
    cfg = cfg or PlayabilityConfig()
    notes = list(notes)
    if mode == "sounding":
        return summarize(build_timeline(notes, cfg), cfg, worst_k, "sounding")
    return summarize(build_strike_timeline(notes, cfg), cfg, worst_k, "struck")


__all__ = [
    "PlayabilityConfig", "PROFILES", "Instant", "PlayabilityReport", "Strain",
    "PLAYABLE", "ROLLED", "IMPOSSIBLE",
    "hand_feasible", "two_hand_feasible", "assign_hands",
    "build_timeline", "build_strike_timeline", "find_strain",
    "summarize", "assess",
]
