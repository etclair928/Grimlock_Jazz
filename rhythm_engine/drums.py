# =================================================================
# MODULE: rhythm_engine/drums.py
# Rhythm Engine: drum onset detection + coarse classification. This is
# a first, honestly-scoped working version - Symphony's 5.x
# DrumIntelligence package (onset/classification/articulation/
# consensus, its own sub-package) went through extensive iteration
# against real audio (brush strokes, flams, rolls, refractory-period
# tuning). Porting that whole package wholesale is future work; this
# module covers the acoustically clearest 4-way split (kick/snare/
# closed-hihat/cymbal) via real band-energy + decay features, not a
# placeholder.
#
# Drum hits are still canonical Notes - GM drum-map pitch numbers
# (kick=36, snare=38, closed_hihat=42, cymbal=49) are a real, standard
# vocabulary for "which drum," not a repurposing of Note.pitch. The
# human-readable label also gets written as a "drum_type" Annotation
# for anything that would rather read a string than a MIDI number.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import librosa

from audio_engine import AudioEngine, AudioTrack
from core import AnnotationStore, Annotation, Note, Provenance, StemType

DRUM_SAMPLE_RATE = 22050
DRUM_HOP_LENGTH = 512
ANALYSIS_WINDOW_MS = 100.0

GM_PITCH = {
    "kick": 36,
    "snare": 38,
    "closed_hihat": 42,
    "cymbal": 49,
}
DEFAULT_DURATION_MS = {
    "kick": 90.0,
    "snare": 110.0,
    "closed_hihat": 60.0,
    "cymbal": 350.0,
}


@dataclass(frozen=True)
class _HitFeatures:
    low_ratio: float
    high_ratio: float
    zcr: float
    decay_ratio: float
    peak_amplitude: float


def _extract_hit_features(samples: np.ndarray, sr: int) -> _HitFeatures:
    n = len(samples)
    if n < 8:
        return _HitFeatures(0.0, 0.0, 0.0, 1.0, 0.0)

    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    total_energy = float(np.sum(spectrum ** 2)) + 1e-8

    low_energy = float(np.sum(spectrum[(freqs >= 20) & (freqs < 150)] ** 2))
    high_energy = float(np.sum(spectrum[freqs >= 3000] ** 2))

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=samples, frame_length=min(2048, n), hop_length=max(1, n // 4))))

    half = n // 2
    first_half_energy = float(np.sum(samples[:half] ** 2)) + 1e-8
    second_half_energy = float(np.sum(samples[half:] ** 2))
    decay_ratio = second_half_energy / first_half_energy

    return _HitFeatures(
        low_ratio=low_energy / total_energy,
        high_ratio=high_energy / total_energy,
        zcr=zcr,
        decay_ratio=decay_ratio,
        peak_amplitude=float(np.max(np.abs(samples))),
    )


def _classify_hit(features: _HitFeatures) -> tuple:
    """Returns (drum_type, confidence). Rule order matters - kick's low-
    frequency dominance is checked first since it's the least ambiguous
    signal; the rest disambiguate noisy/bright hits by decay length."""
    if features.low_ratio > 0.45:
        confidence = float(np.clip(0.5 + features.low_ratio, 0.6, 0.9))
        return "kick", confidence

    if features.high_ratio > 0.35 and features.zcr > 0.25:
        if features.decay_ratio > 0.35:
            confidence = float(np.clip(0.5 + features.decay_ratio * 0.4, 0.6, 0.9))
            return "cymbal", confidence
        confidence = float(np.clip(0.6 + features.high_ratio * 0.3, 0.6, 0.9))
        return "closed_hihat", confidence

    confidence = float(np.clip(0.5 + features.zcr, 0.5, 0.85))
    return "snare", confidence


def detect_drums(
        engine: AudioEngine,
        track: AudioTrack,
        annotations: AnnotationStore,
) -> List[Note]:
    """Detects drum hits on `track` (should already be the isolated
    drums stem) and returns them as canonical Notes tagged
    StemType.DRUMS / Provenance.DRUM_INTELLIGENCE, GM-drum-map pitched.
    Also writes a "drum_type" Annotation with the human-readable label."""
    view = engine.view(track, DRUM_SAMPLE_RATE)
    onset_env = engine.onset_envelope(track, DRUM_SAMPLE_RATE, hop_length=DRUM_HOP_LENGTH)
    onset_times_ms = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=DRUM_SAMPLE_RATE, hop_length=DRUM_HOP_LENGTH, units="time",
    ) * 1000.0

    notes: List[Note] = []
    for onset_ms in onset_times_ms:
        window = view.slice_ms(onset_ms, onset_ms + ANALYSIS_WINDOW_MS)
        if len(window.samples) < 8:
            continue

        features = _extract_hit_features(window.samples, DRUM_SAMPLE_RATE)
        drum_type, confidence = _classify_hit(features)
        velocity = int(np.clip(30 + features.peak_amplitude * 97, 1, 127))
        duration_ms = DEFAULT_DURATION_MS[drum_type]

        note = Note(
            pitch=GM_PITCH[drum_type],
            start_ms=float(onset_ms),
            end_ms=float(onset_ms) + duration_ms,
            velocity=velocity,
            confidence=confidence,
            stem=StemType.DRUMS,
            source=Provenance.DRUM_INTELLIGENCE,
        )
        notes.append(note)
        annotations.add(Annotation(
            note_id=note.id,
            kind="drum_type",
            value=drum_type,
            source=Provenance.DRUM_INTELLIGENCE,
            confidence=confidence,
        ))

    return notes


__all__ = ["detect_drums", "GM_PITCH"]
