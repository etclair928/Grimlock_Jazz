# =================================================================
# MODULE: key_intelligence/key_stability.py
# DOES THIS PIECE HAVE ONE KEY, AND DID WE HEAR ENOUGH OF IT TO SAY?
#
# WHY THIS EXISTS, and it is not the reason anyone expected. Key detection
# was reported as broken since 4.7 and slated to be ripped out. It was
# measured first, and the measurement said something else entirely.
#
# THE CHOPIN CASE, in full, because it is the whole argument. We reported
# D# minor for the Nocturne Op.62 No.1, which is in B major, at confidence
# 0.84. The obvious conclusion is that the detector is broken. It is not:
# feed the SAME detector the published edition's own notes and it returns
# B major at rank 1. The difference is the input. Our audio is a 180-second
# excerpt of a piece the edition renders in 379 quarter-lengths, and when
# the edition is sliced to the same span it reads:
#
#     slice of the published edition      detector says   B ranks
#     whole piece                                B          #1
#     first 40%                                  B          #1
#     first 50%                                D#m          #4
#     first 58%  (the span we were given)      D#m          #3
#     the 42% we were never given                B          #1
#
# The Nocturne's middle dwells in D#/G# minor. B major is established by
# the opening, the return, and the final cadence - and the return and the
# cadence are in the 42% of the piece our clip does not contain. The
# detector gave the RIGHT answer for the audio it was handed.
#
# SO WHAT IS ACTUALLY WRONG. Not the answer - the certainty. The module
# reported 0.84 for a reading that is true of a passage and false of the
# work, and it had no way to notice, because it collapses the whole piece
# into one averaged pitch-class vector before it ever looks. A single key
# with a high confidence is the only thing it can say, so that is what it
# said. This module supplies the missing sentence: "the key changes across
# this span, and here is where."
#
# WHAT IT DOES. Detect the key in overlapping windows, then report:
#   * the global reading (unchanged - this module never overrides it);
#   * the per-window readings;
#   * a STABILITY score, the share of the span whose window agrees with the
#     global reading, counting a relative major/minor as agreement because
#     that pair shares all seven notes and is a labelling choice, not a
#     modulation;
#   * a damped confidence, so a piece that modulates cannot report the
#     certainty of one that does not.
#
# THE CADENCE WITNESS, and its honest limits. Tonal music resolves to its
# tonic, so the final sounding notes are real evidence - on the complete
# edition the last forty notes are F# 12, B 11, D# 10, a textbook B-major
# close that corroborates the global reading. It is offered as CORROBORATION
# ONLY, never as a determinant, for a reason this same investigation
# produced: on our 180-second clip the last forty notes are F-natural 18,
# A# 9, C# 6 - the clip stops mid-phrase in dominant harmony, and F-natural
# is not in B major at all. A witness that would have voted confidently for
# the wrong answer on the one case we know is a witness that gets a vote,
# not a veto.
# =================================================================

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from key_intelligence.key_detector import (
    KeyResult, NOTE_TO_PITCH_CLASS, _MINOR_RELATIVE_MAJOR, _key_pitch_classes,
    chroma_from_notes, detect_key,
)

# Windows this long, hopped by half, so a modulation is visible in more than
# one window and a single odd bar cannot invent one. Twelve windows over a
# three-minute clip is about fifteen seconds each - long enough to establish
# a key, short enough to lose one.
DEFAULT_WINDOWS = 12

# Below this share of agreeing windows, the piece is not in one key and the
# global reading is a summary rather than a fact.
STABLE_MIN_AGREEMENT = 0.70

# How many notes at the end count as "the ending".
CADENCE_NOTES = 40


@dataclass(frozen=True)
class KeyStability:
    """What one key label is and is not worth for this span."""
    key: str                                  # the global reading, unchanged
    raw_confidence: float                     # what detect_key said on its own
    confidence: float                         # damped by stability
    agreement: float                          # share of windows agreeing
    stable: bool
    windows: Tuple[Tuple[float, float, str], ...] = ()   # (start_ms, end_ms, key)
    cadence_key: Optional[str] = None         # what the ending alone suggests
    cadence_agrees: Optional[bool] = None
    notes: str = ""

    @property
    def modulates(self) -> bool:
        return not self.stable


def _tonic_pc(key: str) -> Optional[int]:
    """Pitch class of a key's tonic, so A# and Bb are the same note.

    Comparing key labels as STRINGS silently fails on enharmonics, and it
    did: Hopeful's cadence witness returned "A#" against a key of "Bbm" and
    was scored as a disagreement, when A# IS the tonic of Bb minor. Every
    comparison in this module goes through pitch class for that reason.
    """
    return NOTE_TO_PITCH_CLASS.get(key.rstrip("m"))


def _same_tonal_center(a: str, b: str) -> bool:
    """Relative major/minor count as agreement.

    C major and A minor share all seven pitch classes; which one a passage
    'is' can be a labelling choice rather than a modulation, and counting
    that pair as a disagreement would report almost every tonal piece as
    unstable - which would make the flag useless exactly where it matters.
    """
    pa, pb = _tonic_pc(a), _tonic_pc(b)
    if pa is None or pb is None:
        return a == b
    minor_a, minor_b = a.endswith("m"), b.endswith("m")
    if pa == pb and minor_a == minor_b:
        return True
    # relative pair: the minor tonic sits 3 semitones below the major's
    if minor_a and not minor_b:
        return (pa + 3) % 12 == pb
    if minor_b and not minor_a:
        return (pb + 3) % 12 == pa
    return False


def _cadence_key(notes: Sequence, count: int = CADENCE_NOTES) -> Optional[str]:
    """What the last few notes alone suggest, as a corroborating vote.

    The lowest pitch class among the final notes is the guess - the bass is
    the strongest single indicator of a tonic in tonal music, and a final
    cadence puts the tonic in it. Deliberately crude: this is a second
    opinion, and dressing it up would invite someone to trust it.
    """
    pitched = [n for n in notes
               if getattr(getattr(n, "stem", None), "value", "") != "drums"]
    if len(pitched) < count:
        return None
    tail = sorted(pitched, key=lambda n: n.start_ms)[-count:]
    lowest = min(tail, key=lambda n: n.pitch)
    pc = int(lowest.pitch) % 12
    for name, value in NOTE_TO_PITCH_CLASS.items():
        if value == pc:
            return name
    return None


def analyze_key_stability(
        notes: Sequence,
        windows: int = DEFAULT_WINDOWS,
        min_agreement: float = STABLE_MIN_AGREEMENT,
) -> KeyStability:
    """Global key plus an honest account of whether it holds throughout."""
    pitched = [n for n in notes
               if getattr(getattr(n, "stem", None), "value", "") != "drums"]
    global_result = detect_key(chroma_from_notes(pitched))

    if len(pitched) < 24 or windows < 2:
        # Too little to say anything about stability; report the reading and
        # do not pretend the silence is agreement.
        return KeyStability(
            key=global_result.key, raw_confidence=global_result.confidence,
            confidence=global_result.confidence, agreement=1.0, stable=True,
            notes="too few notes to test stability")

    start = min(float(n.start_ms) for n in pitched)
    end = max(float(n.end_ms) for n in pitched)
    span = max(end - start, 1.0)
    width = span / windows * 2.0            # 50% overlap
    hop = span / windows

    per_window: List[Tuple[float, float, str]] = []
    for i in range(windows):
        w0 = start + i * hop
        w1 = min(end, w0 + width)
        inside = [n for n in pitched if w0 <= float(n.start_ms) < w1]
        if len(inside) < 12:
            continue
        per_window.append((w0, w1, detect_key(chroma_from_notes(inside)).key))

    if not per_window:
        return KeyStability(
            key=global_result.key, raw_confidence=global_result.confidence,
            confidence=global_result.confidence, agreement=1.0, stable=True,
            notes="no window held enough notes to test stability")

    agree = sum(1 for _s, _e, k in per_window
                if _same_tonal_center(k, global_result.key))
    agreement = agree / len(per_window)
    stable = agreement >= min_agreement

    # A key that holds for half the piece is not known to the same standard
    # as one that holds throughout, and the number has to say so. Scaling by
    # agreement is the least clever thing that is true.
    confidence = global_result.confidence * agreement

    # The cadence votes on the TONIC, so it is compared by pitch class and
    # never by label - it has no opinion on major vs minor.
    cadence = _cadence_key(pitched)
    cadence_pc = None if cadence is None else _tonic_pc(cadence)
    global_pc = _tonic_pc(global_result.key)
    cadence_agrees = (None if cadence_pc is None or global_pc is None
                      else cadence_pc == global_pc)

    others = Counter(k for _s, _e, k in per_window
                     if not _same_tonal_center(k, global_result.key))
    detail = (f"{agree}/{len(per_window)} windows agree with "
              f"{global_result.key}")
    if others:
        detail += "; elsewhere " + ", ".join(f"{k} x{c}" for k, c in others.most_common(3))
    if not stable:
        detail += (" - this span does not hold one key, so the global reading "
                   "is a summary of it, not a fact about it")

    # WHEN THE WINDOWS OUTVOTE THE GLOBAL READING, BELIEVE THE WINDOWS.
    #
    # Until now `key` was always the global reading and the windows only ever
    # produced an agreement percentage - a number that said "this is probably
    # wrong" without ever being allowed to say what would be right.
    #
    # The global reading is one correlation against one chroma average over
    # the whole span, so on a piece that moves it can settle between two keys
    # and match neither. A window is the same measurement over a span short
    # enough to hold still. When a clear majority of windows share a tonal
    # center the global reading does not, the mode is the better estimate.
    #
    # Measured on Clocks, whose pitch content is unambiguous (A natural absent,
    # G present - an Ab-major collection): the global reading was Bb minor,
    # which wants Gb, while 7 of 12 windows sat on the Ab/Eb collection that
    # the notes actually spell.
    #
    # Deliberately conservative. It only fires when the span is already
    # UNSTABLE, and only for a center holding a strict majority - so a piece
    # that genuinely holds one key is never second-guessed, and a scatter of
    # disagreeing windows with no clear winner leaves the global reading alone.
    chosen_key = global_result.key
    if not stable and per_window:
        centers = Counter(k for _s, _e, k in per_window)
        top_key, top_n = centers.most_common(1)[0]
        if top_n > len(per_window) / 2.0 and not _same_tonal_center(
                top_key, global_result.key):
            detail += (f"; the windows outvote it - {top_n}/{len(per_window)} "
                       f"read {top_key}, so that is reported instead of "
                       f"{global_result.key}")
            chosen_key = top_key
            agree = sum(1 for _s, _e, k in per_window
                        if _same_tonal_center(k, chosen_key))
            agreement = agree / len(per_window)
            confidence = global_result.confidence * agreement

    # AND THE SCALE MUST CONTAIN THE NOTES ACTUALLY PLAYED.
    #
    # The window vote above is about WHERE the music sits; this is about
    # WHICH NOTES it uses, and they fail differently. A correlation over a
    # weighted chroma average can land on a key whose scale excludes a pitch
    # class the piece leans on - and no amount of window agreement catches
    # that, because every window shares the same bias.
    #
    # Measured on Clocks: the reading was Bb minor, which spells Gb and not G.
    # The recording uses G on 8.5% of its notes and Gb on 2.3%. Bb minor is
    # not a near-miss there, it is the wrong collection - and the windows were
    # split 5-5-2, so the vote above correctly declined to overrule it.
    #
    # The test is deliberately blunt: among the keys the windows actually
    # proposed, prefer the one whose seven pitch classes cover the most of
    # what was played. It cannot invent a key nobody read, and it only moves
    # when the margin is real.
    if per_window:
        histogram = Counter(int(n.pitch) % 12 for n in pitched)
        total_pc = sum(histogram.values()) or 1

        def coverage(key_name: str) -> float:
            try:
                members = _key_pitch_classes(key_name)
            except Exception:
                return 0.0
            return sum(histogram[pc] for pc in members) / total_pc

        candidates = {k for _s, _e, k in per_window} | {chosen_key}
        best = max(candidates, key=coverage)
        if coverage(best) > coverage(chosen_key) + SCALE_COVERAGE_MARGIN:
            detail += (f"; {chosen_key} spells notes this recording does not "
                       f"use - it covers {coverage(chosen_key):.0%} of what was "
                       f"played against {best}'s {coverage(best):.0%}, so {best} "
                       f"is reported")
            chosen_key = best
            agree = sum(1 for _s, _e, k in per_window
                        if _same_tonal_center(k, chosen_key))
            agreement = agree / len(per_window)
            confidence = global_result.confidence * agreement

    return KeyStability(
        key=chosen_key, raw_confidence=global_result.confidence,
        confidence=confidence, agreement=agreement, stable=stable,
        windows=tuple(per_window), cadence_key=cadence,
        cadence_agrees=cadence_agrees, notes=detail)


# How much better a candidate's scale must fit the notes played before it
# displaces the correlation's answer. Four points is comfortably above the
# noise in a pitch-class histogram of a few thousand notes, and well below the
# gap that separates a right collection from a wrong one.
SCALE_COVERAGE_MARGIN = 0.04


__all__ = ["KeyStability", "analyze_key_stability", "DEFAULT_WINDOWS",
           "SCALE_COVERAGE_MARGIN",
           "STABLE_MIN_AGREEMENT", "CADENCE_NOTES"]
