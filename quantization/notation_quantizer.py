# =================================================================
# MODULE: quantization/notation_quantizer.py
# A THIRD timing timeline, distinct from raw (playback, default) and
# groove-quantized (lattice_judge, playback-with-feel): notation
# timing, whose only goal is that a human reads the exported score with
# the least cognitive effort. This is the buildable, MIDI-expressible
# distillation of a large "AI copyist" design discussion - see the
# module's companion note below for what was deliberately NOT built
# from that discussion and why.
#
# THE KEY DISTINCTION (why this isn't just lattice_judge again):
# lattice_judge deliberately keeps a tolerance POCKET that PRESERVES a
# note's human microtiming - a note 30ms off the beat stays 30ms off,
# which is right for playback feel and wrong for notation (the host's
# importer then can't cleanly notate it, producing the endless-tied-
# note mess). Notation timing does the opposite: HARD-snap the onset to
# the metrical grid and give the note a CLEAN symbolic duration, so the
# notation software's import quantizer has almost nothing left to guess.
# The two timelines coexist - Scribe Engraver picks which to export.
#
# THE SCORING IDEA (kept from the discussion, scoped honestly): rather
# than snap to the nearest waveform position, choose the symbolic
# reading that balances fidelity-to-performance against reading load.
# The fidelity term is NOT re-invented here - duration_witness already
# computes probability-weighted symbolic-duration hypotheses for a note
# against the beat grid, so this reuses that directly. The only thing
# added is the discussion's single best rule ("when two readings sound
# ~identical, choose the simpler one") as a principled tie-break among
# near-equal-probability hypotheses - NOT a sprawling weighted blend of
# a dozen unvalidated magic-number penalties (that part of the
# discussion was the same over-fitting trap declined elsewhere this
# project).
#
# DELIBERATELY NOT BUILT (all either the notation RENDERER's job, or
# not expressible in MIDI at all): page-turn optimization, clef
# switching, spacing/collision solvers, house-style matching, cue
# notation, dynamics placement, and the entire academic/Urtext critical-
# apparatus layer. MuseScore/Finale/Dorico own the visual engraving;
# Jazz produces the symbolic content they lay out. The genuinely bigger
# frontier the discussion points at - ties/voices/beaming decided by
# the scoring function - is NOT expressible in MIDI (those are the
# importer's decisions); it needs a MusicXML/MEI export path, a separate
# future module that would build ON these symbolic onset/duration
# decisions, not replace them.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core import Note, TempoMeter
from core.musical_time import MusicalTime
from quantization.duration_witness import (
    DurationTestimony, infer_duration, SYMBOLIC_DURATIONS, TUPLET_RATIOS,
)

NOTATION_TIMING_ANNOTATION_KIND = "notation_timing"

# When the top two duration hypotheses are within this probability of
# each other, they "sound ~identical" and the simpler one wins (the
# discussion's Rule #4/#10, as a tie-break not a weighted term).
NOTATION_SIMPLICITY_TIE_MARGIN = 0.15

# Reading-cost rank per symbolic value: finer subdivisions cost more to
# read, a dot costs a little, a tuplet costs a lot (the one big penalty
# every copyist agrees on). Lower = simpler = preferred on a tie.
_SIMPLICITY_COST = {
    "whole": 0.0, "dotted_half": 1.5, "half": 1.0, "dotted_quarter": 2.5,
    "quarter": 2.0, "dotted_eighth": 3.5, "eighth": 3.0, "dotted_sixteenth": 4.5,
    "sixteenth": 4.0, "thirty_second": 5.0,
    "triplet": 5.0, "compound_triplet": 5.5, "quintuplet": 7.0, "septuplet": 8.0,
}
_DEFAULT_SIMPLICITY_COST = 5.0


@dataclass(frozen=True)
class NotationTiming:
    """Hard-snapped, clean-symbolic-duration timing for one note - the
    reading-optimized timeline. Never applied to the Note; written as a
    notation_timing Annotation the Scribe Engraver may export instead of
    raw/groove timing."""
    note_id: str
    notation_start_ms: float
    notation_end_ms: float
    symbolic_duration: str
    reason: str


def _subdivisions_per_beat(tempo_meter: TempoMeter) -> int:
    """Compound meters (6/8, 9/8, 12/8) subdivide the beat into 3; every
    other meter into 4 (the sixteenth-note grid). This is the same
    simple/compound test build_lattice already uses - one place decides
    it, this mirrors it rather than inventing a second rule."""
    if tempo_meter.time_signature_denominator == 8 and tempo_meter.time_signature_numerator in (6, 9, 12):
        return 3
    return 4


def _simplicity_cost(symbolic_value: str) -> float:
    return _SIMPLICITY_COST.get(symbolic_value, _DEFAULT_SIMPLICITY_COST)


def _pick_symbolic_duration(testimony: DurationTestimony) -> tuple:
    """Best symbolic duration for a note: the highest-probability
    hypothesis, EXCEPT that any hypothesis within
    NOTATION_SIMPLICITY_TIE_MARGIN of the top probability competes on
    reading cost and the simplest wins. Returns (symbolic_value,
    duration_ms, was_tie_broken)."""
    ranked = sorted(testimony.hypotheses, key=lambda h: -h.probability)
    top = ranked[0]
    contenders = [h for h in ranked if top.probability - h.probability <= NOTATION_SIMPLICITY_TIE_MARGIN]
    best = min(contenders, key=lambda h: _simplicity_cost(h.symbolic_value))
    return best.symbolic_value, best.duration_ms, (best.symbolic_value != top.symbolic_value)


# --- TEMPO MAP -------------------------------------------------------------
# Musical time is a RELATIONSHIP, not a measurement. A note is not "at 82.3s",
# it is "the second sixteenth of beat 3 of bar 41" - the seconds are an accident
# of how fast the performer was moving right then.
#
# The old grid took `beat_times_ms[0]` - ONE instant - and extrapolated a
# perfectly isochronous lattice across the whole piece, discarding the rest of
# an array that already records every tracked beat. MEASURED on Rubinstein's
# Chopin Op.62/1: 87% of the performance sits >0.35s away from where a constant
# tempo would put it, and the worst drift is 6.02s. The recall-vs-tolerance
# curve against the published edition is the same distribution seen from the
# other side (0.461 at +/-0.35s, 0.777 at +/-1.0s, 0.938 at +/-5s), so ~53
# points of recall sit behind this one assumption.
#
# So: convert to BEAT SPACE through the tracked beats, snap there, convert back.
# Rubato stops being noise the quantizer fights and becomes the coordinate
# system it works in. Falls back to the isochronous grid when no beat array
# exists, so behaviour is unchanged for callers that never had one.
#
# THAT CONVERSION LIVES IN core/musical_time.py AND NOWHERE ELSE (2026-08-17
# audit). This module used to carry its own `_beat_position`/`_beat_to_ms` -
# a second implementation of MusicalTime.to_beats/to_ms with the same
# interpolation, none of its glitch-confidence logic, and no way to keep the
# two in step. Two implementations of one map is the duplicated-fact disease
# §9 exists to remove, and it is exactly how the +9.6% Chopin fix got reverted
# one layer down the first time. `musical_time` is threaded in from the
# Conductor; absent one, it is built from the TempoMeter's own beats here.


def notation_quantize_note(
        note: Note,
        tempo_meter: TempoMeter,
        duration_testimony: Optional[DurationTestimony] = None,
        grid_origin_ms: Optional[float] = None,
        musical_time: Optional[MusicalTime] = None,
) -> Optional[NotationTiming]:
    """Hard-snaps `note`'s onset to the metrical grid and gives it a
    clean symbolic duration (simplicity-tie-broken). Returns None when
    there's no usable tempo grid to snap against (nothing to notate
    cleanly, so leave it to raw). `grid_origin_ms` anchors the grid -
    the Conductor passes the resolved downbeat/first beat so the snap
    lands on real beats, not on an arbitrary t=0."""
    beat_ms = tempo_meter.beat_duration_ms
    if beat_ms <= 0:
        return None

    subdiv = _subdivisions_per_beat(tempo_meter)
    grid_step = beat_ms / subdiv

    origin = grid_origin_ms
    if origin is None:
        origin = tempo_meter.beat_times_ms[0] if tempo_meter.beat_times_ms else 0.0

    # Hard snap onset to the nearest grid subdivision - no pocket. This
    # is the whole difference from groove quantization: reading clarity
    # over microtiming preservation.
    if musical_time is None and tempo_meter.beat_times_ms:
        musical_time = MusicalTime.from_beats(tempo_meter.beat_times_ms,
                                              fallback_beat_ms=beat_ms)
    testimony = duration_testimony or infer_duration(note, beat_ms)
    symbolic_value, notation_dur, tie_broken = _pick_symbolic_duration(testimony)

    if musical_time is not None and musical_time.usable and grid_origin_ms is None:
        # BEAT-SPACE SNAP. Position the onset against the beat it actually
        # belongs to, snap there, and map back - so a note played inside a
        # ritardando lands on the correct subdivision of the correct beat
        # rather than wherever a metronome would have been.
        notation_start, _residual = musical_time.snap(note.start_ms, subdiv)
        # The duration is STRUCTURAL, so carry it in beats: both ends move
        # together under rubato and a stretched quarter still reads as a
        # quarter. (Measured: duration distortion vs the consolidated source
        # was 139% under the isochronous grid.)
        dur_beats = notation_dur / beat_ms if beat_ms > 0 else 0.0
        notation_end = musical_time.to_ms(
            musical_time.to_beats(notation_start) + dur_beats)
        grid_desc = (f"1/{subdiv}-beat grid (tempo map, "
                     f"{len(musical_time.beats_ms)} tracked beats)")
    else:
        steps_from_origin = round((note.start_ms - origin) / grid_step)
        notation_start = origin + steps_from_origin * grid_step
        # Offset lands on the grid by construction (start on a subdivision +
        # a whole symbolic multiple of the beat), which is exactly what makes
        # the host draw clean beat boundaries instead of tied fragments.
        notation_end = notation_start + notation_dur
        grid_desc = f"1/{subdiv}-beat grid (isochronous)"

    onset_shift = notation_start - note.start_ms
    reason = (f"onset hard-snapped {onset_shift:+.0f}ms to {grid_desc}; "
              f"duration -> {symbolic_value}"
              + (" (simpler tie-break over the top-probability reading)" if tie_broken else ""))

    return NotationTiming(
        note_id=note.id,
        notation_start_ms=notation_start,
        notation_end_ms=notation_end,
        symbolic_duration=symbolic_value,
        reason=reason,
    )


__all__ = [
    "NotationTiming",
    "notation_quantize_note",
    "NOTATION_TIMING_ANNOTATION_KIND",
]
