# =================================================================
# MODULE: quantization/duration_witness.py
# Ports the CORE of Symphony's QuaverIntelligence (epistemic/
# quaver_intelligence.py) - a symbolic-duration witness that proposes
# competing hypotheses for what a note's duration "really is"
# (quarter/eighth/dotted/triplet/etc.), scored by how closely the
# note's raw duration matches each candidate against the resolved
# tempo. "Does NOT quantize. Does NOT decide. Only TESTIFIES" - per
# its own module docstring, already annotation-shaped by design; this
# port keeps that law literally (a function returning a value, never
# touching a Note).
#
# Deliberately reduced from the original's full "adversarial evidence"
# gathering (which cross-examines 8 other subsystems - AnechoicMa
# resonance, per-note drum-anchor metrical evidence, voice-continuity
# phrase grouping, groove-field per-note deviation - none of which
# exist in Jazz yet in the shape QuaverIntelligence expects). This
# keeps the always-available core signal (raw duration vs. the beat
# grid) rather than faking evidence from subsystems that aren't built.
# Extend this once those subsystems exist and have something real to
# contribute, per this project's "don't build ahead of need" discipline.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Tuple

from core import Note

SYMBOLIC_DURATIONS = {
    "whole": 4.000,
    "dotted_half": 3.000,
    "half": 2.000,
    "dotted_quarter": 1.500,
    "quarter": 1.000,
    "dotted_eighth": 0.750,
    "eighth": 0.500,
    "dotted_sixteenth": 0.375,
    "sixteenth": 0.250,
    "thirty_second": 0.125,
}
TUPLET_RATIOS = {
    "triplet": 2.0 / 3.0,
    "quintuplet": 4.0 / 5.0,
    "septuplet": 4.0 / 7.0,
    "compound_triplet": 3.0 / 4.0,
}
DEFAULT_TOLERANCE_FRACTION = 0.08  # 8% of the beat

# THE ONE TABLE OF DURATIONS A NOTEHEAD CAN ACTUALLY BE WRITTEN AS, in beats.
#
# Snapping to a GRID is not enough: a k/3 triplet grid happily produces 5/3 and
# 7/3, which no note value represents, so music21 invents ratios like 24:13 and
# 12:7 to make the arithmetic work and MuseScore boxes the measure in red
# because it cannot reconcile them either. Snapping to this explicit SET makes
# every emitted duration notatable by construction.
#
# This lives HERE, not in the exporter, because two layers need it and the
# exporter is downstream of quantization - it had its own private copy of the
# same numbers, which is exactly the duplicated-fact disease §9 exists to
# remove. output/musicxml_exporter.py imports these.
# THE FLOOR IS A SIXTEENTH (user directive, 2026-08-22): "I NEVER NEVER NEVER
# WANT TO SEE 32nd or 64th notes again", and the reason it is a rule rather
# than a preference: "If you are writing divisions of rhythms like that CHECK
# the Tempo and Meter again because something is off."
#
# That is the correct causal reading. A 32nd note is almost never what a
# player reads; it is what an engraver produces when the beat is in the wrong
# place or the tempo is out by a factor of two, at which point every eighth
# becomes a sixteenth and every sixteenth a thirty-second. Leaving the finer
# values available lets a GRID error hide as a notation choice. Removing them
# forces it to surface as a bad fit, which is information.
#
# In beats, so 1.0 is one beat: 0.25 is a sixteenth, and there is nothing
# below it. Ornaments and grace notes are a separate mechanism and are not
# durations in this vocabulary.
NOTATABLE_BEAT_VALUES: Tuple[float, ...] = (
    0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
)
# THE SHORT VALUES WERE MISSING, and that omission is where k/24 came from.
# This series used to start at 1/3 - a triplet EIGHTH - so a triplet sixteenth
# (1/6 of a beat) and a triplet thirty-second (1/12) were simply not writable,
# even though both are ordinary notation. The consequence was measured on the
# Chopin page: when a note's next onset sat 1/12 of a beat away,
# `notatable_at_most` had nothing in the table that fit, fell through to the
# smallest BINARY value it had, and wrote the note 1/8 long - overrunning by
# exactly 1/8 - 1/12 = 1/24, which then displaced every following onset in
# that voice onto a k/24 position and pushed durations across barlines.
#
# The fix is to finish the series rather than to special-case the symptom.
# This series starts at a triplet EIGHTH, so a triplet sixteenth (1/6 of a
# beat) and a triplet thirty-second (1/12) are not writable from this table
# alone - they are reachable only when a per-beat divisor verdict supplies
# them via tuplet_unit_values(). That looks like an omission and it was
# investigated as the cause of k/24; it is not. Completing the series was
# implemented, measured, and REVERTED: off-grid onsets and barline crossings
# went to zero either way (0 vs 0), note counts differed by six out of 2600,
# and junk ratios by one out of 28. A change to the table every duration in
# the system passes through has to earn more than that.
NOTATABLE_TUPLET_BEAT_VALUES: Tuple[float, ...] = (
    1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0,
)


def tuplet_unit_values(divisor: int, max_units: Optional[int] = None) -> Tuple[Fraction, ...]:
    """Every duration a note inside a `divisor`-tuplet can take, as EXACT
    fractions of a beat: 1/d, 2/d, ... up to a beat and a bit beyond for a
    tuplet note tied over.

    EXACT, not float. A sextuplet unit is 1/6 = 0.1666666666666..., and once a
    handful of those are added up the result is not any rational music21
    recognises - which is where ratios like 24:13 and 48:43 come from. They are
    not tuplets anyone detected; they are music21 reconciling accumulated
    binary-float error. Fractions remove the error at the source."""
    if divisor < 1:
        return ()
    top = max_units if max_units is not None else divisor * 2
    return tuple(Fraction(k, divisor) for k in range(1, top + 1))


def notatable_values(allow_tuplet: bool = False,
                     divisor: Optional[int] = None) -> Tuple[float, ...]:
    """The permitted duration vocabulary, in beats.

    `divisor` is the subdivision the BEAT was read as (rhythm_inference's
    per-beat verdict): pass 6 and a sextuplet's 1/6 becomes writable. Without
    it the old behaviour holds - binary, plus the four triplet values when
    `allow_tuplet` is set - which is what silently flattened every non-triplet
    tuplet to a binary value: 1/6 to 1/8, 1/5 to 1/4, 1/12 to 1/8. The model
    could find a sextuplet; the page had no way to write one down.

    A tuplet WIDENS the set, it never replaces the binary one: a beat read as a
    sextuplet can still contain a note lasting half the beat."""
    if divisor:
        return NOTATABLE_BEAT_VALUES + tuple(float(f) for f in tuplet_unit_values(divisor))
    return NOTATABLE_BEAT_VALUES + (NOTATABLE_TUPLET_BEAT_VALUES if allow_tuplet else ())


def nearest_notatable(beats: float, allow_tuplet: bool = False,
                      divisor: Optional[int] = None) -> float:
    """Nearest duration a notehead can be written as."""
    return min(notatable_values(allow_tuplet, divisor), key=lambda v: abs(v - beats))


def notatable_at_most(beats: float, allow_tuplet: bool = False,
                      divisor: Optional[int] = None) -> float:
    """Largest notatable duration that does NOT exceed `beats`. For the callers
    that are enforcing a ceiling (a note may not run into the next onset in its
    own monophonic voice), where rounding to the NEAREST value could round up
    and reintroduce the overlap the ceiling exists to prevent. Returns the
    0.0 when nothing fits, so the ceiling is never violated.

    IT USED TO RETURN min(allowed) INSTEAD, which broke the single guarantee
    the function exists to provide. The docstring said callers were "expected
    to drop durations they consider too short" - but the value handed back was
    too LONG, not too short, and no caller was checking for that. Measured on
    Chopin: fifteen notes were asked for at most 1/12 of a beat and given 1/8,
    overrunning the next onset by 1/24 each and displacing everything after
    them.

    Returning 0.0 is safe at every call site because each already tests it:
    the exporter's gap clamp drops the note, `_apply_tuplet` falls back to the
    untupleted length, and rhythm_inference falls back to the beat line. A
    ceiling that can be exceeded is not a ceiling.
    """
    allowed = notatable_values(allow_tuplet, divisor)
    fitting = [v for v in allowed if v <= beats + 1e-9]
    return max(fitting) if fitting else 0.0


@dataclass(frozen=True)
class DurationHypothesis:
    symbolic_value: str
    duration_ms: float
    probability: float


@dataclass(frozen=True)
class DurationTestimony:
    note_id: str
    raw_duration_ms: float
    hypotheses: Tuple[DurationHypothesis, ...]

    @property
    def primary(self) -> DurationHypothesis:
        return max(self.hypotheses, key=lambda h: h.probability)

    @property
    def has_contention(self) -> bool:
        """True if the top two hypotheses are within 0.15 of each
        other in probability - the note's duration is genuinely
        ambiguous between two symbolic readings."""
        if len(self.hypotheses) < 2:
            return False
        ranked = sorted(self.hypotheses, key=lambda h: -h.probability)
        return (ranked[0].probability - ranked[1].probability) < 0.15


def infer_duration(note: Note, beat_ms: float, tolerance_fraction: float = DEFAULT_TOLERANCE_FRACTION) -> DurationTestimony:
    """Generates competing symbolic-duration hypotheses for `note`
    against the resolved beat grid. Falls back to a single low-
    confidence "unknown" hypothesis when nothing scores above zero
    (a very short or noisy note)."""
    raw_dur = note.duration_ms

    candidates: List[Tuple[str, float]] = [(name, mult * beat_ms) for name, mult in SYMBOLIC_DURATIONS.items()]
    candidates += [(name, ratio * beat_ms) for name, ratio in TUPLET_RATIOS.items()]

    scored = []
    for name, candidate_ms in candidates:
        if candidate_ms <= 0:
            continue
        error = abs(raw_dur - candidate_ms) / candidate_ms
        if error > tolerance_fraction * 3:
            continue  # clearly not this candidate, don't dilute the vote
        score = max(0.0, 1.0 - error / (tolerance_fraction * 3))
        scored.append((name, candidate_ms, score))

    if not scored:
        return DurationTestimony(
            note_id=note.id, raw_duration_ms=raw_dur,
            hypotheses=(DurationHypothesis("unknown", raw_dur, 0.3),),
        )

    total_score = sum(s for _, _, s in scored)
    hypotheses = tuple(
        DurationHypothesis(name, candidate_ms, s / total_score if total_score > 0 else 0.0)
        for name, candidate_ms, s in scored
    )
    return DurationTestimony(note_id=note.id, raw_duration_ms=raw_dur, hypotheses=hypotheses)


__all__ = ["tuplet_unit_values", "DurationHypothesis", "DurationTestimony", "infer_duration", "SYMBOLIC_DURATIONS", "TUPLET_RATIOS"]
