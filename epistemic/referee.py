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

    winner = max(votes, key=lambda k: votes[k])
    total_weight = sum(votes.values())
    winner_share = votes[winner] / total_weight if total_weight > 0 else 1.0

    contention = None
    if len(votes) > 1:
        contention = {
            "candidates": [{"numerator": n, "denominator": d, "confidence": c} for n, d, c in candidates],
            "resolved": {"numerator": winner[0], "denominator": winner[1]},
            "winner_share": winner_share,
        }

    confidence = float(np.clip(winner_share, 0.3, 0.95))
    return MeterResolution(winner[0], winner[1], confidence, contention)


__all__ = ["TempoResolution", "MeterResolution", "resolve_tempo", "resolve_meter"]
