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


@dataclass(frozen=True)
class BeatFilling:
    """One candidate symbolic rhythm for a single beat. `onset_fractions`
    are the NOTATION positions (what gets written); `readability_cost`
    is the copyist's reading-load for this pattern (lower = simpler)."""
    name: str
    onset_fractions: Tuple[float, ...]
    is_tuplet: bool
    readability_cost: float

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


@dataclass(frozen=True)
class BeatRhythm:
    """The inferred rhythm for one beat: the winning filling plus its
    posterior score and mean fit, for the Music_Box trail."""
    filling: BeatFilling
    onset_fractions: Tuple[float, ...]   # notation positions actually used
    mean_onset_error: float
    log_posterior: float


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
    """Max-posterior beat-filling for the onsets observed within one
    beat (each a fraction in [0,1) of the beat). Returns None when no
    vocabulary filling matches the onset count, or the best fit is still
    too poor to trust - the caller then falls back to per-note snapping."""
    observed = tuple(sorted(float(f) for f in observed_fractions))
    n = len(observed)

    best: Optional[BeatRhythm] = None
    for filling in _BEAT_VOCABULARY:
        if filling.onset_count != n:
            continue
        if n == 0:
            # empty beat = rest; trivially the answer, no onsets to fit
            return BeatRhythm(filling, filling.onset_fractions, 0.0, 0.0)

        expected = _expected_fractions_for_likelihood(filling, swing_ratio)
        sse = sum((o - e) ** 2 for o, e in zip(observed, expected))
        log_likelihood = -sse / (2.0 * ONSET_SIGMA_BEAT_FRACTION ** 2)
        log_prior = -READABILITY_LAMBDA * filling.readability_cost
        log_posterior = log_likelihood + log_prior

        mean_error = math.sqrt(sse / n)
        if best is None or log_posterior > best.log_posterior:
            best = BeatRhythm(filling, filling.onset_fractions, mean_error, log_posterior)

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
) -> Dict[str, InferredNoteTiming]:
    """Beat-level rhythm inference for one voice's notes (intended per
    stem/line - a 6-stem separation makes 'one stem ~ one voice'
    reasonable). Returns {note_id: InferredNoteTiming} for the notes
    whose beat parsed confidently; note ids absent from the result had
    no confident beat parse and should fall back to per-note snapping.

    Chords (near-simultaneous notes) count as ONE rhythmic onset and all
    receive that onset's inferred position/duration. A note's notation
    duration runs to the next rhythmic onset in the beat, or the beat
    end for the last - clean symbolic values by construction."""
    if beat_ms <= 0 or not notes:
        return {}

    # Bucket rhythmic onset-groups by beat index.
    groups = _chord_group_onsets(notes, chord_tolerance_ms)
    by_beat: Dict[int, List[List[Note]]] = {}
    for group in groups:
        beat_idx = int(math.floor((group[0].start_ms - grid_origin_ms) / beat_ms))
        by_beat.setdefault(beat_idx, []).append(group)

    result: Dict[str, InferredNoteTiming] = {}
    for beat_idx, beat_groups in by_beat.items():
        beat_start = grid_origin_ms + beat_idx * beat_ms
        observed_fracs = [
            min(0.999, max(0.0, (g[0].start_ms - beat_start) / beat_ms)) for g in beat_groups
        ]
        rhythm = infer_beat(observed_fracs, swing_ratio)
        if rhythm is None:
            continue  # caller falls back to per-note snapping for this beat's notes

        # Notation onset positions in ms, plus each event's end (next
        # event's onset, or beat end for the last).
        onset_ms = [beat_start + f * beat_ms for f in rhythm.onset_fractions]
        for i, group in enumerate(beat_groups):
            start = onset_ms[i]
            end = onset_ms[i + 1] if i + 1 < len(onset_ms) else beat_start + beat_ms
            for note in group:
                result[note.id] = InferredNoteTiming(
                    note_id=note.id,
                    notation_start_ms=start,
                    notation_end_ms=end,
                    reason=f"beat rhythm '{rhythm.filling.name}' "
                           f"(fit {rhythm.mean_onset_error:.3f}, {len(beat_groups)} onset(s) in beat)",
                    is_tuplet=rhythm.filling.is_tuplet,
                )

    return result


__all__ = [
    "BeatFilling",
    "BeatRhythm",
    "InferredNoteTiming",
    "infer_beat",
    "infer_voice_rhythm",
    "ONSET_SIGMA_BEAT_FRACTION",
    "READABILITY_LAMBDA",
]
