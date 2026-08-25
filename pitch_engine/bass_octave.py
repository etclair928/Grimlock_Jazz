# =================================================================
# MODULE: pitch_engine/bass_octave.py
# CREPE CORRECTS BASIC PITCH'S OCTAVE ON THE BASS, instead of being logged
# and thrown away.
#
# WHAT WAS HAPPENING. orchestration/conductor.py already ran CREPE on the bass
# stem, counted its notes, wrote the count to the MusicBox trail, and DISCARDED
# the notes - "CREPE's read is logged, not merged." Avoiding a merge is right;
# a monophonic tracker and a polyphonic detector both emitting notes would
# double-count the bass. But the alternative to merging is not discarding, it
# is ARBITRATING, and that distinction is the whole of this module.
#
# THE ASYMMETRY THAT MAKES IT WORK. Basic Pitch GENERATES notes: where they
# start, how long they last, how many there are. CREPE DESCRIBES pitch: one
# continuous, unquantized f0 per frame, with no opinion on how many notes
# exist. So let each do what it is good at - Basic Pitch decides that a note
# exists, CREPE decides which octave it is in.
#
# THE PREMISE THIS WAS BUILT ON WAS WRONG, AND THE CORRECTION IS THE POINT.
# It was built to fix a bass reading twelve semitones too low, measured as
# "human bass 41-60, ours 29-75." That comparison was invalid. The published
# transcription's Bass Guitar part carries <octave-change>-1</octave-change> -
# bass guitar is a TRANSPOSING instrument, written an octave above where it
# sounds. The human bass sounds 29-48, not 41-60. Re-measured against sounding
# pitch:
#
#   our bass floor          29        human sounding floor    29   exact match
#   our notes below floor    0 (0%)
#   our notes above ceiling 52 (6%)
#   our median              42        human median            39
#
# There is no low-octave problem. There never was. What there is instead is a
# modest upward skew - three semitones of median and a 6% tail above the
# ceiling, which looks like guitar and piano bleeding into the bass stem rather
# than an octave error at all.
#
# SO WHY IS THIS STILL HERE, AND WHY IS IT OFF. Measured on three songs with
# bass stems on disk, against the one criterion available without per-note
# ground truth - a bass cannot sound below a five-string's low B (MIDI 23):
#
#   song     bass notes   corrected   impossible notes rescued   high notes pulled down
#   HRV            312    51 (16%)                          0    1 of 12
#   Grey           715   140 (20%)                          0    1 of  3
#   Copper          70     1 ( 1%)                          0    0 of  2
#
# It never creates a physically impossible note, so it is safe. It also never
# fixes one, so it is not yet useful. Moving a fifth of the bass to fix one
# note is not a trade worth making by default, and that is why
# `arbitrate_bass_octaves` is opt-in: the mechanism is built, tested and wired,
# and it stays dark until evidence says it helps.
#
# WHAT WOULD SETTLE IT. Clocks is the only song with a per-note human bass, and
# its stems are not on disk - the arbiter has never been run against ground
# truth. Separate Clocks once, keep the stems, and score the corrections note
# by note against the sounding-pitch bass line. That measurement decides
# whether this ships on, and nothing else should.
#
# WHY ONLY OCTAVES. A disagreement of twelve semitones between a monophonic
# tracker and a polyphonic detector is an octave error, which is the specific
# failure a harmonic-series model makes when it locks onto the wrong partial.
# A disagreement of three semitones is a different note, and this module has
# no opinion about it - correcting those would be re-detecting the bass, which
# is not what the evidence supports. Refusing the general case is what keeps
# this a correction rather than a second detector.
#
# ANNOTATION, NOT MUTATION (DESIGN_DECISIONS §2.2). Nothing here rewrites a
# frozen Note. It returns corrections the Conductor writes as annotations and
# the notation builder honours, exactly as consolidation and notation timing
# already do. Delete the annotations and the original pitches come back.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

BASS_OCTAVE_ANNOTATION_KIND = "bass_octave"

# How close to a whole octave the disagreement must be before it counts as one.
# Wide enough to survive CREPE's frame-to-frame wobble and a slightly flat
# bass, narrow enough that a seventh or a ninth is never mistaken for an
# octave.
OCTAVE_TOLERANCE_SEMITONES = 1.5

# Octave distances considered. Two octaves happens on a low bass where the
# fundamental is weakest; three does not, and allowing it would start catching
# real intervals.
OCTAVE_STEPS = (12.0, 24.0)

# A verdict needs evidence. Below this many voiced CREPE frames inside the
# note's own span, there is no reading and the note is left alone.
MIN_VOICED_FRAMES = 3

# The reading must also be steady. If CREPE's own f0 wanders more than this
# within one note, it is not confidently in any octave and must not be used to
# move one.
MAX_SPREAD_SEMITONES = 2.0


@dataclass(frozen=True)
class OctaveCorrection:
    """One note CREPE places in a different octave than Basic Pitch did."""
    note_id: str
    original_pitch: int
    corrected_pitch: int
    crepe_midi: float
    voiced_frames: int
    reason: str

    @property
    def semitones(self) -> int:
        return self.corrected_pitch - self.original_pitch


def _hz_to_midi(hz: np.ndarray) -> np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-9) / 440.0)


def arbitrate_bass_octaves(
        notes: Sequence,
        sample_continuous_pitch_hz: Optional[Callable[[float, float], np.ndarray]],
        octave_tolerance: float = OCTAVE_TOLERANCE_SEMITONES,
        min_voiced: int = MIN_VOICED_FRAMES,
        max_spread: float = MAX_SPREAD_SEMITONES,
) -> List[OctaveCorrection]:
    """Which bass notes did Basic Pitch put in the wrong octave?

    `sample_continuous_pitch_hz` is `make_f0_sampler`'s callable over CREPE's
    raw f0. Without it there is no evidence and nothing is corrected - a
    structural guess about octaves is not worth having.
    """
    if sample_continuous_pitch_hz is None or not notes:
        return []

    corrections: List[OctaveCorrection] = []
    for note in notes:
        hz = sample_continuous_pitch_hz(float(note.start_ms), float(note.end_ms))
        if hz is None or len(hz) < min_voiced:
            continue

        midi = _hz_to_midi(np.asarray(hz, dtype=np.float64))
        # The median, not the mean: one frame locking onto a partial should not
        # drag the reading, and the bass's own attack transient is noisy.
        reading = float(np.median(midi))
        spread = float(np.percentile(midi, 90) - np.percentile(midi, 10))
        if spread > max_spread:
            continue                       # CREPE is not steady here either

        delta = reading - float(note.pitch)
        if abs(delta) <= octave_tolerance:
            continue                       # they already agree

        for step in OCTAVE_STEPS:
            for signed in (step, -step):
                if abs(delta - signed) <= octave_tolerance:
                    corrected = int(round(note.pitch + signed))
                    if not (0 <= corrected <= 127):
                        continue
                    direction = "up" if signed > 0 else "down"
                    corrections.append(OctaveCorrection(
                        note_id=note.id,
                        original_pitch=int(note.pitch),
                        corrected_pitch=corrected,
                        crepe_midi=reading,
                        voiced_frames=int(len(hz)),
                        reason=(
                            f"Basic Pitch read {int(note.pitch)}; CREPE's f0 over "
                            f"the same span reads {reading:.1f} across "
                            f"{len(hz)} voiced frames - {abs(signed):.0f} "
                            f"semitones {direction}, which is an octave, so the "
                            f"note is moved rather than the disagreement "
                            f"ignored"),
                    ))
                    break
            else:
                continue
            break
    return corrections


__all__ = ["OctaveCorrection", "arbitrate_bass_octaves",
           "BASS_OCTAVE_ANNOTATION_KIND", "OCTAVE_TOLERANCE_SEMITONES",
           "OCTAVE_STEPS", "MIN_VOICED_FRAMES", "MAX_SPREAD_SEMITONES"]
