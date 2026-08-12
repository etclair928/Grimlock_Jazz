# =================================================================
# MODULE: output/playability.py
# The playability checker (GRIMLOCK_6.0_PERFORMANCE_ENGRAVING.md §12).
#
# WHY THIS EXISTS: §VI of the open-problems doc laments that engraving
# has no loss function - "the ear is the only oracle." Playability is a
# partial escape on the one axis that matters for a piano reduction: a
# page a human physically cannot play is WRONG, computably, with no ear
# and no dataset. Fifteen notes across four octaves struck at once is not
# a hard chord - it is impossible, and its impossibility is a number.
#
# This is Slice 0: a DIAGNOSTIC. It scores a set of timed notes for
# per-instant two-hand playability and changes no output. It is pure
# (tuples in, a report out) so it can later become the objective function
# the register-split reducer optimizes against (§11.3.3), without
# dragging any of the music21 / audio stack into the core.
#
# MODEL (§12.3), deliberately simple and honest for a first number:
#   - A hand plays a contiguous register, so a simultaneity is playable
#     by two hands iff the sorted set of sounding pitches splits at ONE
#     point into a low group and a high group, each with <= max_fingers
#     notes and a span <= max_span semitones. (Crossed hands ignored -
#     rare, and not what shatters a page.)
#   - Same pitch sounding from two sources = one key = one finger, so
#     pitches are de-duplicated before counting.
#   - The sustain pedal (a real finger-count relaxer, §12.3) is NOT yet
#     modelled: this checker asks "could fingers hold this AS WRITTEN,"
#     the strict floor. Pedal relaxation can only make things MORE
#     playable, so every "unplayable" here is a true lower bound.
# =================================================================

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PlayabilityConfig:
    """Physical limits of one pianist's two hands. Defaults are the
    'comfortable' bracket; the tool also runs a 'stretch' bracket
    (max_span 16 = a tenth) so the reported number is a range, not a
    single threshold pretending to precision we don't have."""
    max_span_semitones: int = 14      # a ninth - comfortable-to-moderate stretch per hand
    max_fingers_per_hand: int = 5
    max_total_notes: int = 10         # ten fingers, hard ceiling


@dataclass(frozen=True)
class Instant:
    """One maximal time interval [start, end) over which the set of
    sounding pitches is constant, plus the playability verdict for it."""
    start: float                      # beats (quarterLengths), absolute
    end: float
    pitches: Tuple[int, ...]          # unique, sorted, currently sounding
    playable: bool
    split: Optional[int]              # #notes in the low (left-hand) group, if playable
    reason: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def n(self) -> int:
        return len(self.pitches)

    @property
    def span(self) -> int:
        return (self.pitches[-1] - self.pitches[0]) if self.pitches else 0


def two_hand_feasible(
        pitches: Iterable[int], cfg: PlayabilityConfig,
) -> Tuple[bool, Optional[int], str]:
    """Can two hands hold this pitch set at once? Returns
    (playable, split_index, reason). split_index is the count of pitches
    taken by the low hand in the cheapest working split (None if none)."""
    uniq = sorted(set(int(p) for p in pitches))
    n = len(uniq)
    if n == 0:
        return True, 0, "silence"
    if n > cfg.max_total_notes:
        return False, None, f"{n} simultaneous notes > {cfg.max_total_notes} fingers"

    # A hand is a contiguous register: low = uniq[:k], high = uniq[k:].
    for k in range(0, n + 1):
        low = uniq[:k]
        high = uniq[k:]
        if len(low) > cfg.max_fingers_per_hand or len(high) > cfg.max_fingers_per_hand:
            continue
        if low and (low[-1] - low[0]) > cfg.max_span_semitones:
            continue
        if high and (high[-1] - high[0]) > cfg.max_span_semitones:
            continue
        return True, k, "ok"

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


def build_timeline(
        notes: Iterable[Tuple[float, float, int]], cfg: PlayabilityConfig,
) -> List[Instant]:
    """Sweep-line over note (start, end, pitch) triples in beats. Between
    consecutive event boundaries the sounding set is constant; each such
    interval becomes one Instant. Half-open intervals [t, next): a note
    ending exactly at t no longer sounds in that interval."""
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
        timeline.append(Instant(t, nxt, pitches, ok, split, reason))
    return timeline


@dataclass
class PlayabilityReport:
    config: PlayabilityConfig
    sounding_beats: float = 0.0
    playable_beats: float = 0.0
    sounding_instants: int = 0
    playable_instants: int = 0
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
              worst_k: int = 8) -> PlayabilityReport:
    rep = PlayabilityReport(config=cfg)
    for inst in timeline:
        d = inst.duration
        rep.sounding_beats += d
        rep.sounding_instants += 1
        rep.max_simultaneity = max(rep.max_simultaneity, inst.n)
        rep.max_span = max(rep.max_span, inst.span)
        rep.simultaneity_beats[_bucket(inst.n)] = (
            rep.simultaneity_beats.get(_bucket(inst.n), 0.0) + d)
        rep.excess_notehead_beats += _excess_for_finger_count(inst.n, cfg) * d
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
    return rep


def assess(notes: Iterable[Tuple[float, float, int]],
           cfg: Optional[PlayabilityConfig] = None,
           worst_k: int = 8) -> PlayabilityReport:
    """End-to-end: (start, end, pitch) triples in beats -> a report."""
    cfg = cfg or PlayabilityConfig()
    return summarize(build_timeline(notes, cfg), cfg, worst_k=worst_k)


__all__ = [
    "PlayabilityConfig", "Instant", "PlayabilityReport",
    "two_hand_feasible", "build_timeline", "summarize", "assess",
]
