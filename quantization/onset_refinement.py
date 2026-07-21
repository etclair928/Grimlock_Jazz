# =================================================================
# MODULE: quantization/onset_refinement.py
# Corrects a DETECTION artifact, not an expressive one.
#
# Basic Pitch reports a note's onset when its onset probability crosses
# a threshold. How late that lands depends entirely on ATTACK TIME. A
# kick crosses in ~2ms; a sung note or a strummed chord takes 30-80ms to
# speak, so the crossing lands that far AFTER the actual attack. The
# error is one-directional (never early), and it scales with how slowly
# the instrument speaks.
#
# That is why the drift is stem-selective, and the pipeline's own code
# paths predict exactly which stems suffer:
#
#   drums   - librosa.onset.onset_detect on the onset envelope: their
#             start_ms IS a transient detection. Lands right.
#   bass    - Basic Pitch + CREPE (two witnesses) + a fast plucked
#             attack. Lands right.
#   vocals/guitar/piano/other - Basic Pitch alone, slow attacks, and
#             until this module, no refinement at all. Drifts late.
#
# Because attack character is consistent within an instrument, a whole
# phrase lags TOGETHER - the line reads as displaced rather than
# jittery, which is how it presents to the ear.
#
# The correction reuses rhythm_engine's dual-witness onset machinery,
# which already existed and was wired to nothing: librosa's backtracked
# spectral-flux candidates corroborated by madmom's RNN. Two independent
# witnesses must agree, the shift is bounded, and disagreement is
# recorded rather than averaged away.
#
# ANNOTATION-ONLY (§2.2). Note.start_ms is never mutated - the frozen
# detection floor stands. This writes a proposal; the Scribe Engraver
# decides whether to read it.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from audio_engine import AudioEngine, AudioTrack
from core import Note
from rhythm_engine.onsets import detect_onset_candidates, refine_onset_to_nearest

ONSET_REFINEMENT_ANNOTATION_KIND = "onset_refinement"

# A note's true attack cannot be far from where the pitch detector put
# it - this is refinement, not re-detection. 60ms comfortably covers the
# slowest sung/bowed attack while staying under a sixteenth at any
# plausible tempo (101ms at 148 BPM), so a correction can never be large
# enough to relocate a note to a different beat.
MAX_ONSET_SHIFT_MS = 60.0

# Below this the "correction" is inside the detectors' own frame
# resolution (Basic Pitch's hop is ~11.6ms) and would be noise, not
# evidence. Recording it would add annotations that mean nothing.
MIN_ONSET_SHIFT_MS = 5.0

# refine_onset_to_nearest returns disagreement in [0,1]. When the two
# witnesses point to meaningfully different places, that is a contested
# onset, and the burden of proof is on the correction: decline and leave
# Basic Pitch's reading alone. Absence of one witness is NOT
# disagreement (the function already reports 0.0 for that case).
MAX_WITNESS_DISAGREEMENT = 0.5


@dataclass(frozen=True)
class OnsetRefinement:
    note_id: str
    original_start_ms: float
    refined_start_ms: float
    shift_ms: float           # negative = earlier (the expected direction)
    disagreement: float
    confidence: float


def refine_note_onsets(
        engine: AudioEngine,
        track: AudioTrack,
        notes: List[Note],
        max_shift_ms: float = MAX_ONSET_SHIFT_MS,
) -> List[OnsetRefinement]:
    """Proposes a corrected onset for each note in `notes`, resolved
    against onset candidates detected on `track` - which must be that
    stem's OWN audio, not the master mix.

    Returns only the notes it actually has evidence for. A note with no
    nearby transient (an inner voice moving under a held chord, say)
    gets no proposal at all rather than a guessed one - silence here
    means "no evidence," and the engraver falls back to Basic Pitch."""
    if not notes:
        return []

    candidates = detect_onset_candidates(engine, track)

    refinements: List[OnsetRefinement] = []
    for note in notes:
        resolved, disagreement = refine_onset_to_nearest(candidates, note.start_ms, max_shift_ms)
        if resolved is None:
            continue
        if disagreement > MAX_WITNESS_DISAGREEMENT:
            continue

        shift_ms = resolved - note.start_ms
        if abs(shift_ms) < MIN_ONSET_SHIFT_MS:
            continue

        # A correction may never invert or collapse the note. Basic Pitch's
        # END is not in question here - only where the note began - so a
        # refined onset that would run past its own end is not a better
        # reading of the attack, it is a bad match to a neighbouring note's
        # transient.
        if resolved >= note.end_ms:
            continue

        # Two agreeing witnesses is the strong case; the further apart they
        # are, the less the proposal is worth, and this rides on the note's
        # own detection confidence rather than inventing a fresh one.
        confidence = note.confidence * (1.0 - disagreement)

        refinements.append(OnsetRefinement(
            note_id=note.id,
            original_start_ms=note.start_ms,
            refined_start_ms=resolved,
            shift_ms=shift_ms,
            disagreement=disagreement,
            confidence=confidence,
        ))

    return refinements


__all__ = [
    "OnsetRefinement",
    "refine_note_onsets",
    "ONSET_REFINEMENT_ANNOTATION_KIND",
    "MAX_ONSET_SHIFT_MS",
]
