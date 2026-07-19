# =================================================================
# MODULE: pitch_engine/crepe_bass.py
# Pitch Engine (GRIMLOCK_6.0_DESIGN_DECISIONS.md §3 / §7): CREPE is a
# bass specialist, run as its OWN independent read - not a mid-stream
# refiner of Basic Pitch's notes, not a voting witness in a council that
# no longer exists. CREPE is a strong monophonic pitch tracker, well
# suited to a bass line's mostly-monophonic register; this module
# segments its continuous f0/confidence stream into its own Notes,
# tagged Provenance.CREPE, standing alongside (never merged into)
# basic_pitch_engine's bass-stem read. Any reconciliation between the
# two is Instrument Attribution / Epistemic's job, not this module's.
# =================================================================

from __future__ import annotations

from typing import List, Optional

import numpy as np
import librosa

from audio_engine import AudioEngine, AudioTrack
from core import Note, StemType, Provenance
from model_registry import get_crepe_model

CREPE_SAMPLE_RATE = 16000
CREPE_STEP_SIZE_MS = 10
CREPE_MODEL_CAPACITY = "small"


def transcribe_bass(
        engine: AudioEngine,
        track: AudioTrack,
        confidence_floor: float = 0.50,
        min_note_duration_ms: float = 30.0,
) -> List[Note]:
    """Runs CREPE on `track`'s bass audio and segments its per-frame f0
    stream into canonical, frozen Notes tagged StemType.BASS and
    Provenance.CREPE. `track` should already be the isolated bass stem -
    this module has no stem-separation logic of its own."""
    import crepe

    view = engine.view(track, CREPE_SAMPLE_RATE)
    audio = view.samples

    # Forces the load through the registry (so it's tracked for unload)
    # before crepe.predict() runs - predict() then hits crepe's own
    # internal cache, which the registry call above already populated.
    get_crepe_model(CREPE_MODEL_CAPACITY)

    time_s, frequency, confidence, _ = crepe.predict(
        audio, CREPE_SAMPLE_RATE, viterbi=True, model_capacity=CREPE_MODEL_CAPACITY,
        step_size=CREPE_STEP_SIZE_MS, verbose=0,
    )

    hop_length = int(CREPE_SAMPLE_RATE * CREPE_STEP_SIZE_MS / 1000)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    rms = rms / (np.max(rms) + 1e-8)

    def _rms_at(frame_idx: int) -> float:
        idx = min(int(frame_idx / len(time_s) * len(rms)), len(rms) - 1)
        return float(rms[idx])

    valid = (confidence >= confidence_floor) & (frequency > 0)
    midi_pitches = np.full(len(frequency), -1, dtype=int)
    midi_pitches[valid] = np.round(librosa.hz_to_midi(frequency[valid])).astype(int)

    min_duration_s = min_note_duration_ms / 1000.0
    notes: List[Note] = []

    def _emit(pitch: int, start_idx: int, end_idx: int) -> None:
        start_t, end_t = time_s[start_idx], time_s[end_idx]
        if end_t - start_t < min_duration_s:
            return
        seg_rms = float(np.mean([_rms_at(j) for j in range(start_idx, max(end_idx, start_idx + 1))]))
        seg_conf = float(np.mean(confidence[start_idx:max(end_idx, start_idx + 1)]))
        velocity = int(np.clip(30 + seg_rms * 97, 1, 127))
        notes.append(Note(
            pitch=int(pitch),
            start_ms=float(start_t) * 1000.0,
            end_ms=float(end_t) * 1000.0,
            velocity=velocity,
            confidence=seg_conf,
            stem=StemType.BASS,
            source=Provenance.CREPE,
            fundamental_freq_hz=float(np.mean(frequency[start_idx:max(end_idx, start_idx + 1)])),
        ))

    cur_pitch: Optional[int] = None
    cur_start_idx: Optional[int] = None
    for i in range(len(midi_pitches)):
        frame_pitch: Optional[int] = int(midi_pitches[i]) if valid[i] else None
        if frame_pitch != cur_pitch:
            if cur_pitch is not None:
                _emit(cur_pitch, cur_start_idx, i)
            cur_pitch, cur_start_idx = frame_pitch, i
    if cur_pitch is not None:
        _emit(cur_pitch, cur_start_idx, len(midi_pitches) - 1)

    return notes


__all__ = ["transcribe_bass", "CREPE_SAMPLE_RATE"]
