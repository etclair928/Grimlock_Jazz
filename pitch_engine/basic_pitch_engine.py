# =================================================================
# MODULE: pitch_engine/basic_pitch_engine.py
# Pitch Engine (GRIMLOCK_6.0_DESIGN_DECISIONS.md §3 / §7): Basic Pitch
# per stem IS the transcription. No council, no voting, no median-blend,
# no mid-stream refinement - whatever Basic Pitch says a note's pitch is,
# survives as a canonical, frozen Note. Downstream layers (Instrument
# Attribution, Rhythm Engine) attach opinions as Annotations; nothing
# gets to construct a different-pitched Note from this one's evidence.
#
# Audio always comes from AudioEngine.view() - this module never touches
# soundfile/librosa resampling directly, per the Audio Engine boundary.
# =================================================================

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from audio_engine import AudioEngine, AudioTrack
from core import Note, StemType, Provenance
from model_registry import get_basic_pitch_model

BASIC_PITCH_SAMPLE_RATE = 22050

# Basic Pitch's "note" posteriorgram is a fixed 88-bin grid (one bin per
# semitone) starting at A0 - basic_pitch.constants.ANNOTATIONS_BASE_FREQUENCY
# (27.5Hz = MIDI 21) / ANNOTATIONS_N_SEMITONES (88): bin = midi_pitch - 21,
# an exact non-interpolated mapping.
BASIC_PITCH_POSTERIORGRAM_BASE_MIDI_PITCH = 21
BASIC_PITCH_POSTERIORGRAM_N_BINS = 88


@dataclass(frozen=True)
class BasicPitchPosteriorgram:
    """The model's own per-pitch, per-frame "is this note sounding"
    evidence - frame_note[i, pitch_midi - 21] is Basic Pitch's belief
    (0-1) that pitch_midi is actively sounding at frame i. This is the
    same evidence the model already used internally to decide where
    notes begin and end; consulting it again (e.g. for sustain recovery)
    reuses the model's own judgment instead of a separate, cruder energy
    heuristic - the "amplify, don't fight the models" law applied to a
    frame-level signal instead of just the note-level output."""
    frame_note: np.ndarray       # (n_frames, 88) float32, values in [0, 1]
    frame_hop_ms: float          # ~11.6ms at Basic Pitch's internal 22050Hz/256-hop

    def level_at(self, pitch_midi: int, time_ms: float) -> float:
        bin_idx = pitch_midi - BASIC_PITCH_POSTERIORGRAM_BASE_MIDI_PITCH
        if not (0 <= bin_idx < self.frame_note.shape[1]):
            return 0.0
        frame_idx = int(time_ms / self.frame_hop_ms)
        if not (0 <= frame_idx < self.frame_note.shape[0]):
            return 0.0
        return float(self.frame_note[frame_idx, bin_idx])

    def duration_ms(self) -> float:
        return self.frame_note.shape[0] * self.frame_hop_ms


def _run_predict(
        engine: AudioEngine,
        track: AudioTrack,
        onset_threshold: float,
        frame_threshold: float,
        min_note_duration_ms: float,
        min_frequency_hz: Optional[float],
        max_frequency_hz: Optional[float],
) -> Tuple[list, dict, np.ndarray]:
    """Shared predict() call. predict()'s real signature loads audio FROM
    A FILE, not a raw numpy array, so this always goes through a temp WAV."""
    from basic_pitch.inference import predict

    view = engine.view(track, BASIC_PITCH_SAMPLE_RATE)
    model = get_basic_pitch_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    try:
        sf.write(temp_path, view.samples, BASIC_PITCH_SAMPLE_RATE)
        predict_kwargs = dict(
            model_or_model_path=model,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_duration_ms,
        )
        if min_frequency_hz is not None:
            predict_kwargs["minimum_frequency"] = min_frequency_hz
        if max_frequency_hz is not None:
            predict_kwargs["maximum_frequency"] = max_frequency_hz

        model_output, _midi_data, note_events = predict(temp_path, **predict_kwargs)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return note_events, model_output, view.samples


def _notes_from_events(note_events, stem: StemType) -> List[Note]:
    notes: List[Note] = []
    for start_s, end_s, pitch_midi, amplitude, _pitch_bend in note_events:
        velocity = int(np.clip(round(amplitude * 127), 1, 127))
        notes.append(Note(
            pitch=int(pitch_midi),
            start_ms=float(start_s) * 1000.0,
            end_ms=float(end_s) * 1000.0,
            velocity=velocity,
            confidence=float(amplitude),
            stem=stem,
            source=Provenance.BASIC_PITCH,
        ))
    return notes


def transcribe(
        engine: AudioEngine,
        track: AudioTrack,
        stem: StemType,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        min_note_duration_ms: float = 58.0,
        min_frequency_hz: Optional[float] = None,
        max_frequency_hz: Optional[float] = None,
) -> List[Note]:
    """Runs Basic Pitch on `track` and returns canonical, frozen Notes
    tagged with `stem` and Provenance.BASIC_PITCH. Thresholds default to
    Basic Pitch's own library defaults, not any ensemble-tuned override -
    there is no ensemble in 6.0 for them to be tuned against."""
    note_events, _model_output, _samples = _run_predict(
        engine, track, onset_threshold, frame_threshold,
        min_note_duration_ms, min_frequency_hz, max_frequency_hz,
    )
    return _notes_from_events(note_events, stem)


def transcribe_with_posteriorgram(
        engine: AudioEngine,
        track: AudioTrack,
        stem: StemType,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        min_note_duration_ms: float = 58.0,
        min_frequency_hz: Optional[float] = None,
        max_frequency_hz: Optional[float] = None,
) -> Tuple[List[Note], BasicPitchPosteriorgram]:
    """Same call as transcribe(), but also returns the model's raw
    frame-level note posteriorgram - discarded as `_model_output` inside
    plain transcribe(). Kept as a separate function so existing callers
    are unaffected; only a caller that actually wants the posteriorgram
    (sustain recovery) uses this one."""
    note_events, model_output, samples = _run_predict(
        engine, track, onset_threshold, frame_threshold,
        min_note_duration_ms, min_frequency_hz, max_frequency_hz,
    )
    notes = _notes_from_events(note_events, stem)
    frame_note = np.asarray(model_output["note"], dtype=np.float32)
    frame_hop_ms = (len(samples) / BASIC_PITCH_SAMPLE_RATE * 1000.0) / max(1, frame_note.shape[0])
    posteriorgram = BasicPitchPosteriorgram(frame_note=frame_note, frame_hop_ms=frame_hop_ms)
    return notes, posteriorgram


__all__ = [
    "transcribe", "transcribe_with_posteriorgram",
    "BasicPitchPosteriorgram", "BASIC_PITCH_SAMPLE_RATE",
]
