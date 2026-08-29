# =================================================================
# MODULE: epistemic/referee.py
# Epistemic layer (GRIMLOCK_6.0_DESIGN_DECISIONS.md §7), scoped DOWN: a
# small scalar-contradiction referee, NOT a gate every musical statement
# passes through. It arbitrates genuine scalar disputes ONLY - tempo,
# meter - via weighted voting and ratio-folding. Notes never pass
# through here; Basic Pitch already decided a note exists, there is no
# "dispute" on a note for this module to arbitrate.
#
# Ratio-folding handles the classic tempo-tracker ambiguity: two honest
# trackers reporting 63bpm and 126bpm aren't in real disagreement, one
# just locked onto the half/double-time reading of the same pulse. This
# folds every witness onto the highest-confidence witness's octave
# before voting, and records real contention only when disagreement
# SURVIVES folding (or when folding itself had to correct a witness -
# that correction is exactly the disagreement being resolved).
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import Provenance, TempoMeter
from rhythm_engine.tempo_witness import TempoWitness

TEMPO_AGREEMENT_TOLERANCE_BPM = 3.0
FOLD_RATIOS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
ROUND_BPM_SNAP_TOLERANCE = 1.5
# A witness whose BEST fold still lands beyond this fraction of the anchor did
# not genuinely agree (it is not a clean octave/triple relative of the anchor).
# It is excluded from the tempo average but still counts against confidence.
# Chosen from real evidence: on Hopeful the anchor (madmom) read 146.3 while a
# note-onset witness read 173.6 (~19% high) and a lattice half-time read folded
# to 174 - both should be excluded, both sit >8% from the anchor; the one
# legitimate disagreeing witness (librosa 152, ~4%) is kept.
FOLD_ALIGNMENT_TOLERANCE_FRACTION = 0.08


def _snap_to_round_bpm(bpm: float, tolerance: float = ROUND_BPM_SNAP_TOLERANCE) -> float:
    """Real production tempos are overwhelmingly whole numbers - a DAW's
    tempo field gets set to 145, not 146.34. Any measurement taken off
    real audio carries micro-timing jitter that a weighted average never
    cancels out to an exact integer, even when every witness is
    individually close to one. Snaps to the nearest whole BPM only when
    it's well within ordinary measurement noise - a general affinity
    toward how tempos actually get set, not a per-track special case."""
    nearest = round(bpm)
    if abs(bpm - nearest) <= tolerance:
        return float(nearest)
    return bpm


@dataclass(frozen=True)
class TempoResolution:
    tempo_meter: TempoMeter
    contention: Optional[Dict[str, Any]]  # None means witnesses genuinely agreed


@dataclass(frozen=True)
class MeterResolution:
    numerator: int
    denominator: int
    confidence: float
    contention: Optional[Dict[str, Any]]


def _fold_to_reference(tempo_bpm: float, reference_bpm: float) -> Tuple[float, float]:
    """Returns (folded_bpm, ratio_applied) - whichever of tempo_bpm's
    octave/triple-time relatives sits closest to reference_bpm."""
    if reference_bpm <= 0:
        return tempo_bpm, 1.0
    best_ratio = min(FOLD_RATIOS, key=lambda r: abs(tempo_bpm * r - reference_bpm))
    return tempo_bpm * best_ratio, best_ratio


def resolve_tempo(
        witnesses: Sequence[TempoWitness],
        time_signature: Tuple[int, int] = (4, 4),
) -> TempoResolution:
    """Anchor-and-align tempo resolution across independent witnesses -
    ports Symphony's anchor_and_align_tempo_witnesses (tempo_intelligence.py):
    anchor on the witness with the highest confidence*weight, ratio-fold
    every other witness onto ITS octave (not just onto whichever is
    listed first), then vote weighted by `weight` alone (a witness that
    doesn't align to any ratio near the anchor is excluded from the
    numerator but still counts against the denominator - a real
    disagreement occurred, and confidence should reflect that even
    though its (mis-scaled) number can't sensibly enter the average).

    Witnesses with tempo_bpm <= 0 (a witness that found no evidence,
    e.g. note_onset_tempo_witness on too few notes) are dropped first -
    absence of a reading isn't a vote, disagreeing or otherwise."""
    usable = [w for w in witnesses if w.tempo_bpm > 0 and w.confidence > 0]
    if not usable:
        raise ValueError("resolve_tempo requires at least one witness with a real reading")

    if len(usable) == 1:
        w = usable[0]
        tempo_meter = TempoMeter(
            tempo_bpm=_snap_to_round_bpm(w.tempo_bpm), confidence=w.confidence,
            time_signature_numerator=time_signature[0], time_signature_denominator=time_signature[1],
            beat_times_ms=w.beat_times_ms, source=w.source,
        )
        return TempoResolution(tempo_meter, None)

    reference = max(usable, key=lambda w: w.confidence * w.weight)
    anchor_bpm = reference.tempo_bpm
    tolerance_bpm = anchor_bpm * FOLD_ALIGNMENT_TOLERANCE_FRACTION

    # Fold every non-reference witness onto the anchor's octave, then split
    # into INCLUDED (folds within tolerance of the anchor - a genuine
    # octave/triple relative) and EXCLUDED (best fold still off - not really
    # the same pulse). Excluded witnesses do NOT enter the tempo average, but
    # their weight still counts against confidence: a real disagreement lowers
    # confidence without letting a mis-scaled number corrupt the resolved
    # tempo. (Before this, two ~20%-high witnesses dragged a correct 146bpm
    # anchor up to 157 on Hopeful - the exact failure this fixes.)
    included: List[Tuple[float, float, TempoWitness, float]] = [
        (anchor_bpm, reference.confidence * reference.weight, reference, 1.0)
    ]
    excluded: List[Tuple[float, float, TempoWitness, float]] = []
    any_correction = False
    for w in usable:
        if w is reference:
            continue
        folded_bpm, ratio_applied = _fold_to_reference(w.tempo_bpm, anchor_bpm)
        if ratio_applied != 1.0:
            any_correction = True
        entry = (folded_bpm, w.confidence * w.weight, w, ratio_applied)
        if abs(folded_bpm - anchor_bpm) <= tolerance_bpm:
            included.append(entry)
        else:
            excluded.append(entry)

    total_raw_weight = sum(w.weight for w in usable)
    excluded_weight = sum(w.weight for _, _, w, _ in excluded)

    # Anchor-lean on contention: the more witness-weight had to be excluded
    # (the more the witnesses disagreed), the less a mean of the survivors is
    # trustworthy and the more the single highest-confidence anchor should
    # dominate. Boost the anchor's weight in the average by the excluded
    # fraction - a bounded, self-scaling lean (0 when everyone agrees), not a
    # fixed magic constant.
    excluded_fraction = excluded_weight / total_raw_weight if total_raw_weight > 0 else 0.0
    anchor_boost = 1.0 + excluded_fraction

    num = den = 0.0
    for bpm, cw, w, _ in included:
        eff = cw * (anchor_boost if w is reference else 1.0)
        num += bpm * eff
        den += eff
    weighted_bpm = num / den if den > 0 else anchor_bpm

    included_cw = sum(cw for _, cw, _, _ in included)
    resolved_confidence = included_cw / total_raw_weight if total_raw_weight > 0 else reference.confidence

    max_deviation = max((abs(b - weighted_bpm) for b, _, _, _ in included), default=0.0)
    disagreed = max_deviation > TEMPO_AGREEMENT_TOLERANCE_BPM or bool(excluded)

    contention = None
    if disagreed or any_correction:
        contention = {
            "anchor_source": reference.source.value,
            "anchor_tempo_bpm": anchor_bpm,
            "witnesses": [
                {"source": w.source.value, "raw_tempo_bpm": w.tempo_bpm, "confidence": w.confidence,
                 "weight": w.weight, "folded_tempo_bpm": b, "fold_ratio": r, "included": incl}
                for entries, incl in ((included, True), (excluded, False))
                for b, _, w, r in entries
            ],
            "excluded_sources": [w.source.value for _, _, w, _ in excluded],
            "resolved_tempo_bpm": weighted_bpm,
            "max_deviation_bpm": max_deviation,
        }

    resolved_confidence = float(np.clip(resolved_confidence, 0.3, 0.95))
    tempo_meter = TempoMeter(
        tempo_bpm=_snap_to_round_bpm(weighted_bpm), confidence=resolved_confidence,
        time_signature_numerator=time_signature[0], time_signature_denominator=time_signature[1],
        beat_times_ms=reference.beat_times_ms, source=Provenance.RHYTHM_ENGINE,
    )
    return TempoResolution(tempo_meter, contention)


# --- #2 tempo-octave arbiter -------------------------------------------------
# A confident, trained downbeat witness is the one thing that can break the
# tempo OCTAVE cleanly. The per-witness octave correction (octave_correction.py)
# uses a log-normal tempo prior + a compound-tactus test; that test correctly
# halves a genuinely-slow triplet-feel song (Gospel 136->67) but OVER-fires on a
# song that only has a triplet-shuffle FEEL over a normal pulse (No Pasaran:
# user's ear says quarter=132, the pipeline halved it to 66). Measured: madmom's
# downbeat tracker reads No Pasaran at 130 with confidence 0.92, but Gospel at
# 130 with confidence only 0.30 - so a confidence floor lets the witness rescue
# the false halving without touching the true one.
OCTAVE_ARBITER_MIN_CONFIDENCE = 0.70
OCTAVE_ARBITER_RATIO_TOLERANCE = 0.15   # how close witness/resolved must be to exactly 2:1
OCTAVE_ARBITER_MAX_BPM = 180.0          # never adopt an implausibly fast "tactus"


def arbitrate_tempo_octave(
        resolution: TempoResolution,
        witness_bpm: float,
        witness_beat_times_ms: Sequence[float],
        witness_confidence: float,
        witness_downbeat_times_ms: Sequence[float] = (),
) -> TempoResolution:
    """#2: when a CONFIDENT downbeat witness reads a clean ~2x the resolved
    tempo, the resolved tempo was octave-HALVED - adopt the witness's octave
    and its tracked grid. Deliberately narrow so it fixes only the real bug:
      * confidence floor - a low-confidence witness (Gospel's 130 @ 0.30)
        cannot override a genuinely slow tempo (Gospel's real 67);
      * exactly 2:1 only - a 3:2 compound relation (Copper 140 vs 94,
        Wallet 118 vs 78) is NOT the halving bug and is left alone;
      * sane tactus ceiling. Everything else returns unchanged."""
    resolved = resolution.tempo_meter
    if (witness_bpm <= 0 or resolved.tempo_bpm <= 0
            or witness_confidence < OCTAVE_ARBITER_MIN_CONFIDENCE
            or witness_bpm > OCTAVE_ARBITER_MAX_BPM):
        return resolution
    if abs(witness_bpm / resolved.tempo_bpm - 2.0) > OCTAVE_ARBITER_RATIO_TOLERANCE:
        return resolution

    corrected = TempoMeter(
        tempo_bpm=_snap_to_round_bpm(witness_bpm),
        confidence=max(resolved.confidence, witness_confidence),
        time_signature_numerator=resolved.time_signature_numerator,
        time_signature_denominator=resolved.time_signature_denominator,
        beat_times_ms=tuple(witness_beat_times_ms) if witness_beat_times_ms else resolved.beat_times_ms,
        downbeat_times_ms=tuple(witness_downbeat_times_ms),
        source=resolved.source,
    )
    contention = dict(resolution.contention or {})
    contention["octave_arbiter"] = {
        "corrected_from_bpm": resolved.tempo_bpm,
        "corrected_to_bpm": corrected.tempo_bpm,
        "witness_confidence": witness_confidence,
    }
    return TempoResolution(corrected, contention)


def resolve_meter(candidates: Sequence[Tuple[int, int, float]]) -> MeterResolution:
    """Weighted vote among (numerator, denominator, confidence)
    candidates from one or more independent meter estimates."""
    if not candidates:
        raise ValueError("resolve_meter requires at least one candidate")

    if len(candidates) == 1:
        n, d, c = candidates[0]
        return MeterResolution(n, d, c, None)

    votes: Dict[Tuple[int, int], float] = {}
    for n, d, c in candidates:
        votes[(n, d)] = votes.get((n, d), 0.0) + c

    # GROUPING IS DECIDED BY HOW MANY WITNESSES SAW IT, NOT BY THE LOUDEST ONE.
    #
    # Duple-or-triple is a different KIND of question from how many beats go in
    # a bar. Writing 6/4 where 3/4 belongs is a barring choice and either can be
    # read; writing 6/4 where 4/4 belongs makes every bar after it wrong.
    #
    # Summed confidence cannot tell those apart. On HRV two independent
    # witnesses said 4/4 (0.333, 0.158) and one said 6/4 (0.877), so 0.877 beat
    # 0.491 and the page has been in 6/4 since 2026-08-06. Measured against the
    # kit and the bass - the two parts that carry meter - HRV's onsets accent a
    # duple grouping (drums 1.23 at g=2, bass 1.27 at g=4) and not a triple one
    # (1.02 at g=3). The majority was right and was outvoted by one number.
    #
    # So: settle the CATEGORY by counting witnesses, then pick the bar length
    # within it by confidence as before. A tie in count falls back to
    # confidence, which is the old behaviour and the right default when the
    # witnesses genuinely split.
    def _is_triple(numerator: int) -> bool:
        return numerator % 3 == 0

    triple_n = sum(1 for n, _d, _c in candidates if _is_triple(n))
    duple_n = len(candidates) - triple_n
    grouping_note = None
    if triple_n and duple_n:                      # they disagree on the CATEGORY
        if triple_n != duple_n:
            want_triple = triple_n > duple_n
            eligible = {k: v for k, v in votes.items()
                        if _is_triple(k[0]) == want_triple}
            if eligible:
                grouping_note = {
                    "duple_witnesses": duple_n,
                    "triple_witnesses": triple_n,
                    "chose": "triple" if want_triple else "duple",
                    "why": ("grouping settled by witness count, not summed "
                            "confidence - duple-or-triple is a category, and a "
                            "majority of independent witnesses on it outranks "
                            "one confident outlier"),
                }
                votes = eligible

    winner = max(votes, key=lambda k: votes[k])
    total_weight = sum(votes.values())
    winner_share = votes[winner] / total_weight if total_weight > 0 else 1.0

    contention = None
    if len(votes) > 1 or grouping_note is not None:
        contention = {
            "candidates": [{"numerator": n, "denominator": d, "confidence": c} for n, d, c in candidates],
            "resolved": {"numerator": winner[0], "denominator": winner[1]},
            "winner_share": winner_share,
        }
        if grouping_note is not None:
            contention["grouping"] = grouping_note

    confidence = float(np.clip(winner_share, 0.3, 0.95))
    return MeterResolution(winner[0], winner[1], confidence, contention)


__all__ = ["TempoResolution", "MeterResolution", "resolve_tempo", "resolve_meter", "arbitrate_tempo_octave"]
