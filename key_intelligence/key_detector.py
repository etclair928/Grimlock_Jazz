# =================================================================
# MODULE: key_intelligence/key_detector.py
# Ports the KeyDetector piece of Symphony's agents/detection/
# harmonic_intelligence.py (5.6.1) - a Krumhansl-Schmuckler
# correlation-based key detector, made jazz-aware via two already-
# tuned corroborating fixes for its single most common failure mode:
# a relative major/minor pair (e.g. C major / A minor) shares the
# exact same 7 pitch classes, so a correlation match this close can
# genuinely be a coin flip on the raw profile alone.
#   - Weighted chroma mean, favoring the piece's opening and closing
#     sections over naive full-track averaging - both are strongly
#     correlated with the true tonal center in tonal music.
#   - When the top two candidates are a relative pair with a
#     correlation gap too small to be meaningful, checking the minor
#     key's raised leading tone (harmonic/melodic minor's distinctive
#     7th, real minor-key music emphasizes it for cadential motion)
#     against its natural (Aeolian) 7th - a distinction the shared
#     7-note profile can't make on its own.
#
# NOT ported from the same source file, and why: HarmonicValidator
# (STFT partial-picking, weighted inharmonicity, ZCR/noise verdicting)
# and NoteTransformer.apply_harmonic_evidence are both redundant with
# acoustic_witness/schoenberg_mirror.py, already ported and already
# doing real STFT-based partial-picking + inharmonicity + ZCR/noise
# verdicting directly against raw audio, corroborated by retrograde-
# symmetry and spectral-inversion witnesses Symphony's original never
# had. Porting a second implementation of the same evidence would be
# exactly the redundant-subsystem trap this project exists to avoid.
#
# 6.0 adaptation: Symphony's NoteTransformer.apply_key_context and
# KeyDetector.get_key_weight directly MUTATED a note's confidence.
# Per this session's law (annotation, not mutation - same as every
# other pass tonight), key is a SONG-level fact (MusicalFindingsMap.
# key/key_confidence, same tier as tempo_meter/swing_ratio - it
# describes the whole piece, not one note) and per-note key-fit is a
# `key_fit` Annotation a caller may read; nothing here ever touches a
# Note or its confidence directly.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np
import librosa

from audio_engine import AudioEngine, AudioTrack

KEY_INTELLIGENCE_SAMPLE_RATE = 22050
KEY_FIT_ANNOTATION_KIND = "key_fit"
KEY_CONFIDENCE_THRESHOLD = 0.5  # below this, key evidence is too weak to weight notes by

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

KEYS_MAJOR = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
KEYS_MINOR = ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'Bbm', 'Fm', 'Cm', 'Gm', 'Dm']

NOTE_TO_PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}

_MAJOR_KEY_PITCH_CLASSES = {
    'C': {0, 2, 4, 5, 7, 9, 11}, 'G': {7, 9, 11, 0, 2, 4, 6}, 'D': {2, 4, 6, 7, 9, 11, 1},
    'A': {9, 11, 1, 2, 4, 6, 8}, 'E': {4, 6, 8, 9, 11, 1, 3}, 'B': {11, 1, 3, 4, 6, 8, 10},
    'F#': {6, 8, 10, 11, 1, 3, 5}, 'Db': {1, 3, 5, 6, 8, 10, 0}, 'Ab': {8, 10, 0, 1, 3, 5, 7},
    'Eb': {3, 5, 7, 8, 10, 0, 2}, 'Bb': {10, 0, 2, 3, 5, 7, 9}, 'F': {5, 7, 9, 10, 0, 2, 4},
}
_MINOR_RELATIVE_MAJOR = {
    'Am': 'C', 'Em': 'G', 'Bm': 'D', 'F#m': 'A', 'C#m': 'E', 'G#m': 'B',
    'D#m': 'F#', 'Bbm': 'Db', 'Fm': 'Ab', 'Cm': 'Eb', 'Gm': 'Bb', 'Dm': 'F',
}


@dataclass(frozen=True)
class KeyResult:
    key: str            # e.g. "C", "Am" - trailing 'm' means minor
    confidence: float


def _weighted_chroma_mean(chroma: np.ndarray) -> np.ndarray:
    """Averages chroma frames with extra weight on the piece's opening
    and closing sections - both strongly correlated with the true
    tonal center in tonal music, whereas naive full-track averaging
    can wash that emphasis out entirely."""
    n_frames = chroma.shape[1] if chroma.ndim > 1 else 1
    if n_frames < 10:
        return np.mean(chroma, axis=1) if chroma.ndim > 1 else chroma

    weights = np.ones(n_frames)
    edge = max(1, int(n_frames * 0.15))
    weights[:edge] *= 2.0
    weights[-edge:] *= 2.0

    weighted = chroma * weights[np.newaxis, :]
    return np.sum(weighted, axis=1) / np.sum(weights)


def _correlate_with_key(chroma: np.ndarray, profile: np.ndarray, pitch_class: int) -> float:
    """Pearson correlation between the chroma and the key profile rotated
    so its TONIC sits at `pitch_class`.

    BUG FIX (2026-08-06, §XVII.2): this was `np.roll(profile, -key_idx)`,
    which is wrong twice over. MAJOR_PROFILE/MINOR_PROFILE are written
    tonic-first (index 0 = tonic), and chroma index 0 = C, so the tonic
    must be moved TO `pitch_class` - that is `np.roll(profile, +pc)`. The
    negative sign instead placed it at `12 - pc`, which only coincides
    with the truth for pc 0 and 6. Proven with a synthetic test: a pure
    A-flat-major profile was reported as "E".
    """
    rotated_profile = np.roll(profile, pitch_class)
    chroma_mean = np.mean(chroma)
    profile_mean = np.mean(rotated_profile)
    numerator = np.sum((chroma - chroma_mean) * (rotated_profile - profile_mean))
    denominator = np.sqrt(np.sum((chroma - chroma_mean) ** 2) * np.sum((rotated_profile - profile_mean) ** 2))
    return numerator / denominator if denominator > 0 else 0.0


def _break_relative_tie(avg_chroma: np.ndarray, major_key: str, minor_key: str) -> Tuple[str, bool]:
    """major_key and minor_key are a relative pair (identical 7 natural
    notes). Checks the minor key's raised leading tone against its
    natural (Aeolian) 7th - meaningfully more of the former is real
    evidence of actual minor-key harmonic usage (dominant/cadential
    motion), which nothing in the shared natural-scale profile could
    otherwise reveal."""
    minor_tonic_pc = NOTE_TO_PITCH_CLASS.get(minor_key.rstrip('m'))
    if minor_tonic_pc is None:
        return major_key, False

    raised_leading_tone_pc = (minor_tonic_pc + 11) % 12
    natural_seventh_pc = (minor_tonic_pc + 10) % 12
    raised = avg_chroma[raised_leading_tone_pc]
    natural = avg_chroma[natural_seventh_pc]

    if raised > natural * 1.15:
        return minor_key, True
    return major_key, False


def detect_key(chroma: np.ndarray) -> KeyResult:
    """Detects key from chromagram evidence - the pure, testable core
    of this module (analyze_key below just supplies the chroma)."""
    if chroma is None or chroma.size == 0:
        return KeyResult(key="Cm", confidence=0.0)

    avg_chroma = _weighted_chroma_mean(chroma)
    if np.sum(avg_chroma) > 0:
        avg_chroma = avg_chroma / np.sum(avg_chroma)

    # BUG FIX (2026-08-06, GRIMLOCK_6.0_OPEN_PROBLEMS.md §XVII.2): the
    # profile must be rotated by the key's PITCH CLASS, not by its index
    # in these lists. KEYS_MAJOR/KEYS_MINOR are in circle-of-fifths order
    # (deliberately, so that index i is always a relative major/minor
    # pair - C/Am, G/Em, ...), but _correlate_with_key does
    # np.roll(profile, -key_idx), which is a CHROMATIC shift. Passing the
    # list index therefore tested the wrong profile for any key whose
    # index != its pitch class:
    #     6 of 12 MAJOR keys were mislabeled by a TRITONE (G, A, B, Db,
    #     Eb, F), and ALL 12 MINOR keys were wrong (off by 3 or 9).
    # Measured consequence: prospering reported E (true Ab major),
    # Hopeful reported Bm (true Bb minor) - both confirmed against an
    # independent transcription. key_idx is still carried through for the
    # relative-pair check below, which DOES depend on list position.
    candidates: List[Tuple[float, str, bool, int]] = []
    for key_idx, key in enumerate(KEYS_MAJOR):
        pitch_class = NOTE_TO_PITCH_CLASS.get(key, key_idx)
        candidates.append((_correlate_with_key(avg_chroma, MAJOR_PROFILE, pitch_class), key, False, key_idx))
    for key_idx, key in enumerate(KEYS_MINOR):
        pitch_class = NOTE_TO_PITCH_CLASS.get(key.rstrip('m'), key_idx)
        candidates.append((_correlate_with_key(avg_chroma, MINOR_PROFILE, pitch_class), key, True, key_idx))

    candidates.sort(key=lambda c: -c[0])
    best_correlation, best_key, best_is_minor, best_idx = candidates[0]

    # KEYS_MAJOR[i] and KEYS_MINOR[i] are always a relative pair (same
    # key_idx = same 7 natural notes) - check the runner-up for exactly
    # this ambiguity before trusting the raw winner.
    for correlation, key, is_minor, key_idx in candidates[1:]:
        if key_idx != best_idx or is_minor == best_is_minor:
            continue
        if abs(correlation - best_correlation) < 0.05:
            major_key, minor_key = (best_key, key) if not best_is_minor else (key, best_key)
            resolved_key, resolved_is_minor = _break_relative_tie(avg_chroma, major_key, minor_key)
            if resolved_is_minor != best_is_minor:
                best_key = resolved_key
                best_correlation = correlation
        break

    confidence = max(0.0, min(1.0, (best_correlation + 0.5) / 1.5))
    return KeyResult(key=best_key, confidence=confidence)


def analyze_key(engine: AudioEngine, track: AudioTrack) -> KeyResult:
    """Song-level key detection from the full mix - a chord/harmony
    fact needs every instrument's contribution, not one isolated
    stem's. One entry point, matching every other port this session."""
    view = engine.view(track, KEY_INTELLIGENCE_SAMPLE_RATE)
    chroma = librosa.feature.chroma_cqt(y=view.samples, sr=KEY_INTELLIGENCE_SAMPLE_RATE)
    return detect_key(chroma)


def _key_pitch_classes(key: str) -> Set[int]:
    if key in _MAJOR_KEY_PITCH_CLASSES:
        return _MAJOR_KEY_PITCH_CLASSES[key]
    if key in _MINOR_RELATIVE_MAJOR:
        major_set = _MAJOR_KEY_PITCH_CLASSES.get(_MINOR_RELATIVE_MAJOR[key], {0, 2, 4, 5, 7, 9, 11})
        return {(pc - 3) % 12 for pc in major_set}
    return {0, 2, 4, 5, 7, 9, 11}


def key_fit(pitch: int, key_result: KeyResult) -> Tuple[bool, float]:
    """Returns (is_in_key, weight) for a note's pitch against the
    resolved key. Mirrors Symphony's get_key_weight() exactly (a small
    boost for in-key notes, a confidence-scaled penalty for out-of-key
    ones, no opinion at all when key confidence itself is too weak to
    trust) - but this never applies the weight itself. A caller (the
    Conductor) writes it as a key_fit Annotation; nothing here mutates
    a Note."""
    if key_result.confidence < KEY_CONFIDENCE_THRESHOLD:
        return True, 1.0

    pitch_class = pitch % 12
    is_in_key = pitch_class in _key_pitch_classes(key_result.key)

    if is_in_key:
        return True, 1.05
    if key_result.confidence > 0.8:
        return False, 0.90
    if key_result.confidence > 0.65:
        return False, 0.95
    return False, 0.98


def chroma_from_notes(notes, n_frames: int = 200) -> np.ndarray:
    """Build a 12 x n_frames chromagram from TRANSCRIBED NOTES instead of
    from mix audio.

    WHY (measured 2026-08-06, GRIMLOCK_6.0_OPEN_PROBLEMS.md §XVII.2): the
    audio path (`analyze_key`) reads chroma off the FULL MIX, where drums
    and percussion smear the pitch-class profile - and it was wrong on
    both songs we could check against an independent transcription
    (prospering: reported E, actual Ab major; Hopeful: reported Bm, actual
    Bb minor - a semitone off). Our own transcribed notes are 90-99%
    diatonic, a far cleaner signal we already compute and were throwing
    away.

    Deliberately produces a real TIME SERIES, not a single averaged
    vector, so detect_key's existing opening/closing edge weighting
    (_weighted_chroma_mean) and its relative-major/minor tie-breaker keep
    working unchanged. Drums are excluded: their pitch values are
    percussion slot ids, not tonal content.
    """
    pitched = [n for n in notes
               if getattr(getattr(n, "stem", None), "value", str(getattr(n, "stem", ""))) != "drums"]
    if not pitched:
        return np.zeros((12, 1), dtype=np.float32)

    span_end = max(float(n.end_ms) for n in pitched)
    span_start = min(float(n.start_ms) for n in pitched)
    span = max(1.0, span_end - span_start)
    frames = max(1, int(n_frames))
    chroma = np.zeros((12, frames), dtype=np.float32)

    for note in pitched:
        start = float(note.start_ms); end = max(start + 1.0, float(note.end_ms))
        pc = int(note.pitch) % 12
        f0 = int((start - span_start) / span * (frames - 1)) if frames > 1 else 0
        f1 = int((end - span_start) / span * (frames - 1)) if frames > 1 else 0
        f0 = max(0, min(frames - 1, f0)); f1 = max(f0, min(frames - 1, f1))
        # Weight by real sounding duration, spread across the frames it covers.
        per = (end - start) / 1000.0 / max(1, (f1 - f0 + 1))
        chroma[pc, f0:f1 + 1] += per
    return chroma


def analyze_key_from_notes(notes, n_frames: int = 200) -> KeyResult:
    """Key from our own transcribed notes (see chroma_from_notes). Reuses
    detect_key verbatim - only the evidence changes, not the logic."""
    return detect_key(chroma_from_notes(notes, n_frames=n_frames))


__all__ = [
    "KeyResult",
    "detect_key",
    "analyze_key",
    "analyze_key_from_notes",
    "chroma_from_notes",
    "key_fit",
    "KEY_FIT_ANNOTATION_KIND",
    "KEY_CONFIDENCE_THRESHOLD",
]
