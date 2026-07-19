# =================================================================
# MODULE: quantization/sustain_recovery.py
# Ports Symphony's SustainRecovery (agents/quantization/ritornello.py,
# Pass 2) - the cleanest, most directly Basic-Pitch-aligned piece of
# Ritornello, per user request. Extends a note's END when it is still
# acoustically ringing past its nominal end - never touches pitch,
# never touches start, never invents a note that doesn't already exist.
#
# Two independent witnesses vote on how far to extend, weighted toward
# the one with real model-level evidence:
#   - Basic Pitch's own frame-level note posteriorgram (pitch_engine.
#     BasicPitchPosteriorgram) - the model's own belief that THIS pitch
#     is still sounding, the same evidence it used internally to decide
#     where the note should end in the first place. Weight 0.7.
#   - Energy in narrow bands around the note's own fundamental (+2nd
#     harmonic) - never broadband RMS, which asks "is anything still
#     playing in this stem" and in a real mix is almost always yes.
#     Weight 0.3.
#
# When only one witness has evidence (no posteriorgram, or no stem
# audio), its estimate is used alone at full weight - absence of the
# other witness isn't disagreement. When both agree, extension survives
# with full confidence; when they disagree, the note's confidence is
# docked proportional to the disagreement rather than either witness
# unilaterally winning or the extension being blocked outright - the
# Epistemic layer / downstream confidence floor is the actual arbiter
# of whether a shakily-supported extension survives.
#
# Second guard, independent of either witness: never extend into the
# next detected onset of the same pitch - that onset is a
# re-articulation Basic Pitch explicitly found; overwriting it would
# delete real rhythmic information (the "amplify, don't fight" law).
#
# 6.0 adaptation: this NEVER mutates Note.end_ms. It returns a
# SustainExtension testimony per note that got one; the Conductor
# writes it as a `sustain_extension` Annotation (kind =
# SUSTAIN_RECOVERY_ANNOTATION_KIND), keyed by Note.id. Scribe Engraver
# decides at export time whether to read it (opt-in, off by default -
# same law as quantization/lattice_judge.py's proposed timing).
# =================================================================

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core import Note
from pitch_engine import BasicPitchPosteriorgram

SUSTAIN_RECOVERY_ANNOTATION_KIND = "sustain_extension"

SUSTAIN_RECOVERY_RMS_FLOOR_RATIO = 0.15       # Fraction of note's own onset level still counted as "ringing"
SUSTAIN_RECOVERY_MAX_EXTEND_MS = 400.0        # Never extend a note more than this
SUSTAIN_RECOVERY_FRAME_MS = 20.0              # Extension march hop size
SUSTAIN_RECOVERY_FFT_SIZE = 2048
SUSTAIN_RECOVERY_NEXT_ONSET_MARGIN_MS = 10.0
SUSTAIN_RECOVERY_POSTERIORGRAM_WEIGHT = 0.7
SUSTAIN_RECOVERY_ENERGY_WEIGHT = 1.0 - SUSTAIN_RECOVERY_POSTERIORGRAM_WEIGHT
SUSTAIN_RECOVERY_POSTERIORGRAM_THRESHOLD = 0.5
SUSTAIN_RECOVERY_MAX_DISAGREEMENT_CONFIDENCE_PENALTY = 0.3


@dataclass(frozen=True)
class SustainExtension:
    """Proposed new end_ms for one note, plus the confidence the
    extension itself should carry (already docked for witness
    disagreement) and a human-readable reason for the Music_Box trail."""
    note_id: str
    extended_end_ms: float
    confidence: float
    reason: str


def _pitch_hz(pitch_midi: int) -> float:
    return 440.0 * 2.0 ** ((pitch_midi - 69) / 12.0)


def _band_level(
        frame: np.ndarray, sample_rate: int, f0_hz: float,
        fft_cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> float:
    """RMS-equivalent level of the energy within +/-1 semitone of the
    note's fundamental and 2nd harmonic. Kept in amplitude units (not
    energy) so SUSTAIN_RECOVERY_RMS_FLOOR_RATIO retains its original
    "fraction of the onset level" meaning. fft_cache holds (hanning
    window, rfft bin frequencies) per frame length - recomputing those
    per call dominated runtime on large note counts in Symphony."""
    n = len(frame)
    cached = fft_cache.get(n)
    if cached is None:
        cached = (np.hanning(n), np.fft.rfftfreq(n, 1.0 / sample_rate))
        fft_cache[n] = cached
    window, freqs = cached

    spec = np.abs(np.fft.rfft(frame * window))

    band_energy = 0.0
    for harmonic in (1, 2):
        target = f0_hz * harmonic
        if target >= sample_rate / 2:
            break
        lo = target * 2.0 ** (-1.0 / 12.0)
        hi = target * 2.0 ** (1.0 / 12.0)
        lo_i = int(np.searchsorted(freqs, lo))
        hi_i = int(np.searchsorted(freqs, hi))
        if hi_i <= lo_i:
            # Bass fundamentals sit below one bin's width at this FFT
            # size - fall back to the nearest bin so low notes still
            # get a real (if coarse) narrowband reading.
            lo_i = int(np.argmin(np.abs(freqs - target)))
            hi_i = lo_i + 1
        lo_i = max(0, lo_i - 1)
        hi_i = min(len(spec), hi_i + 1)
        band = spec[lo_i:hi_i]
        band_energy += float(np.dot(band, band))

    return float(np.sqrt(band_energy / max(1, n)))


def _energy_extend_ms(
        audio: np.ndarray, sample_rate: int, f0_hz: float,
        onset_level: float, onset_end_ms: float, onset_end_sample: int,
        ceiling_ms: float, max_extend_ms: float,
        hop_samples: int, fft_samples: int,
        fft_cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> float:
    floor = onset_level * SUSTAIN_RECOVERY_RMS_FLOOR_RATIO
    extend_samples = 0
    max_extend_samples = int(max_extend_ms * sample_rate / 1000)
    pos = onset_end_sample
    while extend_samples < max_extend_samples:
        candidate_end_ms = onset_end_ms + (extend_samples + hop_samples) * 1000.0 / sample_rate
        if candidate_end_ms > ceiling_ms:
            break
        frame = audio[pos:pos + fft_samples]
        if len(frame) < fft_samples // 2:
            break
        if _band_level(frame, sample_rate, f0_hz, fft_cache) < floor:
            break
        pos += hop_samples
        extend_samples += hop_samples
    return extend_samples * 1000.0 / sample_rate


def _posteriorgram_extend_ms(
        posteriorgram: BasicPitchPosteriorgram, pitch: int,
        onset_end_ms: float, ceiling_ms: float, max_extend_ms: float,
) -> float:
    step_ms = posteriorgram.frame_hop_ms
    if step_ms <= 0:
        return 0.0
    threshold = SUSTAIN_RECOVERY_POSTERIORGRAM_THRESHOLD
    extend_ms = 0.0
    t_ms = onset_end_ms
    while extend_ms < max_extend_ms:
        candidate_end_ms = onset_end_ms + extend_ms + step_ms
        if candidate_end_ms > ceiling_ms:
            break
        if posteriorgram.level_at(pitch, t_ms) < threshold:
            break
        t_ms += step_ms
        extend_ms += step_ms
    return extend_ms


def propose_sustain_extensions(
        notes: List[Note],
        stem_samples: Optional[np.ndarray],
        sample_rate: int,
        posteriorgram: Optional[BasicPitchPosteriorgram] = None,
) -> List[SustainExtension]:
    """One stem's worth of notes against that same stem's own audio +
    posteriorgram (both already scoped to one stem by the Pitch Engine
    and Conductor - unlike Symphony's multi-stem-dict version, Jazz
    processes one stem at a time so there's nothing to key by stem tag
    here). Returns only the notes that got a genuine positive
    extension; the Conductor writes each as a `sustain_extension`
    Annotation."""
    if sample_rate <= 0 or not notes:
        return []

    hop_samples = max(64, int(SUSTAIN_RECOVERY_FRAME_MS * sample_rate / 1000))
    fft_samples = max(hop_samples, int(SUSTAIN_RECOVERY_FFT_SIZE))
    max_extend_ms = SUSTAIN_RECOVERY_MAX_EXTEND_MS
    margin_ms = SUSTAIN_RECOVERY_NEXT_ONSET_MARGIN_MS
    posteriorgram_weight = SUSTAIN_RECOVERY_POSTERIORGRAM_WEIGHT
    energy_weight = SUSTAIN_RECOVERY_ENERGY_WEIGHT
    max_penalty = SUSTAIN_RECOVERY_MAX_DISAGREEMENT_CONFIDENCE_PENALTY

    # Next same-pitch onset per note: the hard ceiling no extension may
    # cross (that onset is a re-articulation Basic Pitch found).
    onsets_by_pitch: Dict[int, List[float]] = defaultdict(list)
    for n in notes:
        onsets_by_pitch[n.pitch].append(n.start_ms)
    for onsets in onsets_by_pitch.values():
        onsets.sort()

    # (hanning window, rfft bin freqs) per frame length, shared across
    # every note this call - see _band_level.
    fft_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    extensions: List[SustainExtension] = []
    for note in notes:
        same_pitch_onsets = onsets_by_pitch[note.pitch]
        next_idx = bisect.bisect_right(same_pitch_onsets, note.start_ms)
        next_onset_ms = (same_pitch_onsets[next_idx]
                         if next_idx < len(same_pitch_onsets) else float("inf"))
        extension_ceiling_ms = next_onset_ms - margin_ms
        onset_end_ms = note.end_ms

        energy_ms: Optional[float] = None
        if stem_samples is not None:
            f0_hz = _pitch_hz(note.pitch)
            onset_start_sample = int(note.start_ms * sample_rate / 1000)
            onset_end_sample = int(onset_end_ms * sample_rate / 1000)
            onset_slice = stem_samples[onset_start_sample:onset_start_sample + fft_samples]
            if len(onset_slice) >= 256:
                onset_level = _band_level(onset_slice, sample_rate, f0_hz, fft_cache)
                if onset_level > 1e-8:
                    energy_ms = _energy_extend_ms(
                        stem_samples, sample_rate, f0_hz, onset_level, onset_end_ms,
                        onset_end_sample, extension_ceiling_ms, max_extend_ms,
                        hop_samples, fft_samples, fft_cache,
                    )

        posterior_ms: Optional[float] = None
        if posteriorgram is not None:
            posterior_ms = _posteriorgram_extend_ms(
                posteriorgram, note.pitch, onset_end_ms, extension_ceiling_ms, max_extend_ms,
            )

        if energy_ms is None and posterior_ms is None:
            continue

        if energy_ms is not None and posterior_ms is not None:
            resolved_ms = posteriorgram_weight * posterior_ms + energy_weight * energy_ms
            disagreement = abs(posterior_ms - energy_ms) / max(max_extend_ms, 1.0)
            witness_desc = f"posteriorgram={posterior_ms:.0f}ms, energy={energy_ms:.0f}ms"
        elif posterior_ms is not None:
            resolved_ms = posterior_ms
            disagreement = 0.0
            witness_desc = f"posteriorgram={posterior_ms:.0f}ms (no stem audio for energy check)"
        else:
            resolved_ms = energy_ms
            disagreement = 0.0
            witness_desc = f"energy={energy_ms:.0f}ms (no posteriorgram for this stem)"

        if resolved_ms <= 0:
            continue

        confidence_penalty = disagreement * max_penalty
        new_confidence = max(0.0, note.confidence - confidence_penalty)

        extensions.append(SustainExtension(
            note_id=note.id,
            extended_end_ms=onset_end_ms + resolved_ms,
            confidence=new_confidence,
            reason=witness_desc,
        ))

    return extensions


__all__ = [
    "SustainExtension",
    "propose_sustain_extensions",
    "SUSTAIN_RECOVERY_ANNOTATION_KIND",
]
