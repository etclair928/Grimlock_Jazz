# =================================================================
# MODULE: quantization/lattice_judge.py
# Ports Symphony's TemporalLattice.resolve() (agents/quantization/
# temporal_lattice.py) - the "Judge" that translates a resolved tempo/
# meter/groove reading into a proposed snapped position for one note.
#
# The critical adaptation for 6.0: Symphony's version stored the result
# as a `snapped_start_ms` FIELD alongside the note's own `start_ms` -
# already non-destructive by construction, just not annotation-shaped.
# Here it's a pure function returning (quantized_start_ms,
# quantized_end_ms, reason); nothing in this module ever touches a
# Note. The Conductor is the one place that decides whether to write
# the result as an Annotation, and Scribe Engraver the one place that
# decides whether to READ it (opt-in, off by default - see §2's
# frozen-detection-floor law).
#
# The algorithm itself, unchanged: ghost notes (very quiet) pass
# through untouched; a tolerance POCKET around the grid line preserves
# raw timing (human nuance, not error); outside the pocket, a
# confidence-weighted LERP nudges toward the grid, never further than
# hard safety bounds, never producing a negative timestamp, never
# pulling a note earlier by more than a small cap.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from core import Note, TempoMeter

QUANTIZATION_ANNOTATION_KIND = "quantization"
GHOST_NOTE_VELOCITY_CEILING = 30
MAX_QUANTIZATION_SHIFT_MS = 80.0
MAX_EARLY_SHIFT_MS = 40.0  # never pull a note earlier than this, even under the strongest nudge


class QuantizationMode(str, Enum):
    PRESERVE_HUMAN = "preserve_human"  # keep original unless egregious
    GROOVE_AWARE = "groove_aware"      # apply groove but respect bounds
    RIGID = "rigid"                    # full quantization to grid


@dataclass
class TemporalLattice:
    """The resolved grid + feel a note's timing gets judged against."""
    period_ms: float
    subdivisions_per_beat: int
    swing_ratio: float
    phase_delta_ms: float = 0.0
    microtiming_window_ms: float = 15.0
    quantize_strength: float = 0.35
    max_quantization_shift_ms: float = MAX_QUANTIZATION_SHIFT_MS
    preserve_ghost_notes: bool = True
    mode: QuantizationMode = QuantizationMode.GROOVE_AWARE

    normalized_phase_ms: float = field(init=False, default=0.0)
    _beat_cache: Dict[int, float] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.subdivisions_per_beat not in (2, 3, 4, 6):
            self.subdivisions_per_beat = 4
        self.period_ms = max(20.0, min(500.0, self.period_ms))
        self.swing_ratio = max(0.5, min(0.85, self.swing_ratio))

        beat_duration_ms = self.period_ms * self.subdivisions_per_beat
        if beat_duration_ms > 0:
            half_beat = beat_duration_ms / 2.0
            self.normalized_phase_ms = ((self.phase_delta_ms + half_beat) % beat_duration_ms) - half_beat
        else:
            self.normalized_phase_ms = 0.0

        for i in range(-256, 257):
            self._beat_cache[i] = i * self.period_ms

    def get_beat_time(self, beat_index: int) -> float:
        return self._beat_cache.get(beat_index, beat_index * self.period_ms)

    def get_nearest_beat_index(self, time_ms: float) -> int:
        if self.period_ms <= 0:
            return 0
        return int(round(time_ms / self.period_ms))

    def _is_offbeat_eighth(self, time_ms: float) -> bool:
        """Is this note landing on the off-beat eighth (the "and" of
        the beat) - the position swing feel actually applies to? In
        compound/ternary meter the grid itself already lands on the
        true subdivisions, so there's no separate off-beat concept to
        additionally swing-push."""
        if self.subdivisions_per_beat != 4:
            return False
        beat_duration_ms = self.period_ms * self.subdivisions_per_beat
        if beat_duration_ms <= 0:
            return False
        beat_position = time_ms % beat_duration_ms
        half_beat = beat_duration_ms / 2.0
        return abs(beat_position - half_beat) < self.period_ms / 2.0

    def _pocket_width_ms(self) -> float:
        """Normalized phase widens the pocket (musical expression) but
        never shifts the grid (preserves the timeline)."""
        return self.microtiming_window_ms + abs(self.normalized_phase_ms) * 0.15

    def resolve(self, raw_ms: float, velocity: int = 80, confidence: float = 0.5) -> Tuple[float, str]:
        """Returns (proposed_ms, reason). This is the pure per-onset
        judge; quantize_note() below is what actually applies it to a
        Note's start/end."""
        if self.preserve_ghost_notes and velocity < GHOST_NOTE_VELOCITY_CEILING:
            return raw_ms, "ghost_note_preserved"

        if self.mode == QuantizationMode.PRESERVE_HUMAN:
            nearest_idx = self.get_nearest_beat_index(raw_ms)
            nearest_time = self.get_beat_time(nearest_idx)
            if abs(raw_ms - nearest_time) > MAX_QUANTIZATION_SHIFT_MS:
                direction = 1 if nearest_time > raw_ms else -1
                return max(0.0, raw_ms + direction * (MAX_QUANTIZATION_SHIFT_MS * 0.2)), "preserve_human_capped"
            return raw_ms, "preserve_human_within_bounds"

        is_offbeat = self._is_offbeat_eighth(raw_ms)
        lattice_count = self.get_nearest_beat_index(raw_ms)
        target_ms = self.get_beat_time(lattice_count)

        if is_offbeat and self.swing_ratio > 0.5:
            target_ms += (self.swing_ratio - 0.5) * self.period_ms

        pocket = self._pocket_width_ms()
        if abs(raw_ms - target_ms) <= pocket:
            return raw_ms, "within_pocket"

        effective_strength = float(np.clip(self.quantize_strength * (1.0 - confidence), 0.0, 1.0))
        new_ms = (1 - effective_strength) * raw_ms + effective_strength * target_ms

        delta = new_ms - raw_ms
        if abs(delta) > self.max_quantization_shift_ms:
            delta = np.sign(delta) * self.max_quantization_shift_ms
            new_ms = raw_ms + delta
        if delta < 0 and abs(delta) > MAX_EARLY_SHIFT_MS:
            new_ms = raw_ms - MAX_EARLY_SHIFT_MS

        new_ms = max(0.0, new_ms)
        reason = f"snapped: period={self.period_ms:.1f}ms, swing={self.swing_ratio:.2f}, strength={effective_strength:.2f}"
        return float(new_ms), reason

    def quantize_note(self, note: Note) -> Tuple[float, float, str]:
        """Returns (quantized_start_ms, quantized_end_ms, reason). The
        same offset applied to onset is applied to the note's end, so
        duration is preserved."""
        snapped_start, reason = self.resolve(note.start_ms, note.velocity, note.confidence)
        delta = snapped_start - note.start_ms
        snapped_end = max(snapped_start + 10.0, note.end_ms + delta)
        return snapped_start, snapped_end, reason


def _blended_confidence(tempo_confidence: float, groove_confidence: float, lattice_confidence: float) -> float:
    return lattice_confidence * 0.3 + groove_confidence * 0.3 + tempo_confidence * 0.4


def build_lattice(
        tempo_meter: TempoMeter,
        swing_ratio: float = 0.5,
        groove_confidence: float = 0.0,
        lattice_confidence: float = 0.0,
        phase_delta_ms: float = 0.0,
) -> TemporalLattice:
    """Factory deriving a TemporalLattice from a resolved TempoMeter +
    groove/lattice readings. `quantize_strength`/`microtiming_window_ms`
    are derived from the blended confidence (lower confidence -> gentler
    snapping, wider tolerance) rather than hardcoded - "when uncertain,
    disturb the audio less"."""
    subdivisions_per_beat = 3 if (tempo_meter.time_signature_denominator == 8
                                   and tempo_meter.time_signature_numerator in (6, 9, 12)) else 4
    period_ms = tempo_meter.beat_duration_ms / subdivisions_per_beat if tempo_meter.tempo_bpm > 0 else 125.0

    blended = _blended_confidence(tempo_meter.confidence, groove_confidence, lattice_confidence)
    quantize_strength = float(np.clip(0.15 + 0.35 * blended, 0.15, 0.5))
    microtiming_window_ms = float(np.clip(25.0 - 15.0 * blended, 10.0, 25.0))

    return TemporalLattice(
        period_ms=period_ms,
        subdivisions_per_beat=subdivisions_per_beat,
        swing_ratio=swing_ratio,
        phase_delta_ms=phase_delta_ms,
        microtiming_window_ms=microtiming_window_ms,
        quantize_strength=quantize_strength,
    )


__all__ = ["TemporalLattice", "QuantizationMode", "build_lattice", "QUANTIZATION_ANNOTATION_KIND"]
