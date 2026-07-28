# =================================================================
# MODULE: rhythm_engine/drums.py
# Rhythm Engine: multi-voice drum detection.
#
# WHY THIS IS A REDESIGN, NOT A CLASSIFIER TWEAK
# ----------------------------------------------
# The old detector (and a faithful port of Symphony's SpectralDrumClassifier)
# emitted ONE note per onset - it picked a single label per moment. Measured
# on You Say God Says that collapsed to ~70% "kick" and ZERO snares, and it
# "missed the 16th hats". Those are the SAME bug: a real kit plays kick+hat
# (and snare+hat) SIMULTANEOUSLY, so when they land together the loud kick
# wins the single-label vote and the hat/snare is destroyed. So each voice is
# detected INDEPENDENTLY, per band, and they coexist (multiple Notes may share
# a start_ms - exactly how a GM drum channel represents a layered kit).
#
# WHY PRESENCE-IN-BAND WASN'T ENOUGH (the over-detection study, 2026-07-28)
# ------------------------------------------------------------------------
# Detecting each voice by "did its band have an onset here" over-fired badly:
# on You Say God Says (123 BPM, 136 bars) it found kick 6.9/bar, snare 9.0/bar,
# hat 9.7/bar - a snare on ~9 of the 16 slots per bar is not a backbeat, it's
# noise. The study showed WHY: it is NOT double-triggering (snapping to the
# 16th grid collapsed only 2-3%), it is CROSS-BAND BLEED - the kick's body and
# the hat's tail both put energy in a wide "snare" band, so the snare band
# fired on nearly every musical event. Threshold tuning didn't help (the
# onsets are real energy, just not real snares). Two changes fix it:
#   1. A voice counts only when its band LEADS, not merely has energy. Kick
#      requires the low band to dominate the mid+high bands (a real kick, not
#      a kick-harmonic leaking up); snare uses a narrow 2-5 kHz "crack" band
#      (above the kick's rolloff, below the hat's sizzle) that is genuinely
#      snare-specific instead of the wide 200-2500 catch-all.
#   2. An ENERGY GATE per band drops quiet bleed/ghost hits (below a fraction
#      of that band's own loud-hit level) - this is the "it's not that many
#      notes" fix; a listener doesn't hear the ghosts as separate notes.
# Result: kick 4.0/bar, snare 2.6/bar, hat 4.5/bar - a real kit shape.
#
# Deferred (needs more than blind spectral cues, and guessing them would
# reintroduce noise): open vs closed hat, ride/crash/tom/rimshot. Everything
# bright is written as a closed hi-hat.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

from audio_engine import AudioEngine, AudioTrack
from core import AnnotationStore, Annotation, Note, Provenance, StemType

DRUM_SAMPLE_RATE = 22050
DRUM_HOP_LENGTH = 256

# band edges (Hz)
KICK_MAX_HZ = 150
SNARE_CRACK_HZ = (2000, 5000)    # snare-specific: above kick rolloff, below hat sizzle
HAT_MIN_HZ = 6000

MERGE_MS = 35.0                  # union-of-bands dedup / coincidence window
COINCIDENCE_MS = 45.0            # a moment "has" a voice if that band's kept onset is within this
ENERGY_MS = 30.0                 # window for the per-onset RMS energy read
ENERGY_GATE_FRAC = 0.25         # keep onsets whose band energy >= this * that band's p90
KICK_DOMINANCE = 1.2            # low-band energy must exceed this * the louder of snare/hat bands

GM_PITCH: Dict[str, int] = {"kick": 36, "snare": 38, "closed_hihat": 42}
DEFAULT_DURATION_MS: Dict[str, float] = {"kick": 90.0, "snare": 110.0, "closed_hihat": 60.0}


@dataclass(frozen=True)
class _Voice:
    name: str
    times: np.ndarray            # kept onset times (s), after gate/dominance
    signal: np.ndarray           # this voice's band-limited signal (for velocity)


def _bandpass(y: np.ndarray, lo: float | None, hi: float | None, sr: int) -> np.ndarray:
    if lo and hi:
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    elif lo:
        sos = butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        sos = butter(4, hi, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def _band_onset_times(sig: np.ndarray, sr: int) -> np.ndarray:
    env = librosa.onset.onset_strength(y=sig, sr=sr, hop_length=DRUM_HOP_LENGTH)
    return librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=DRUM_HOP_LENGTH, units="time",
    )


def _rms_at(sig: np.ndarray, t: float, sr: int, span_ms: float = ENERGY_MS) -> float:
    i0 = int(t * sr)
    seg = sig[i0: i0 + int(span_ms / 1000.0 * sr)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def _near(times: np.ndarray, t: float, tol_ms: float = COINCIDENCE_MS) -> bool:
    return len(times) > 0 and float(np.min(np.abs(times - t))) <= tol_ms / 1000.0


def _peak_velocity(sig: np.ndarray, sr: int, t: float, span_ms: float = 40.0) -> int:
    i0 = int(t * sr)
    seg = sig[i0: i0 + int(span_ms / 1000.0 * sr)]
    peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
    return int(np.clip(30 + peak * 97, 1, 127))


def detect_drums(
        engine: AudioEngine,
        track: AudioTrack,
        annotations: AnnotationStore,
) -> List[Note]:
    """Detects drum hits on `track` (the isolated drums stem) as layered
    voices and returns them as canonical Notes (StemType.DRUMS /
    DRUM_INTELLIGENCE), GM-drum-map pitched, each with a "drum_type"
    Annotation. Kick/snare/hat are detected independently - each only where
    its band LEADS and is loud enough - so a single moment can produce
    several coincident Notes without the cross-band bleed that over-fired
    the presence-only version."""
    view = engine.view(track, DRUM_SAMPLE_RATE)
    y = np.ascontiguousarray(view.samples, dtype=np.float64)
    if y.size < 32:
        return []
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

    kick_sig = _bandpass(y, None, KICK_MAX_HZ, DRUM_SAMPLE_RATE)
    snare_sig = _bandpass(y, SNARE_CRACK_HZ[0], SNARE_CRACK_HZ[1], DRUM_SAMPLE_RATE)
    hat_sig = _bandpass(y, HAT_MIN_HZ, None, DRUM_SAMPLE_RATE)

    def gate(sig: np.ndarray, dominators: List[np.ndarray]) -> np.ndarray:
        """Keep onsets loud enough in this band (>= ENERGY_GATE_FRAC * p90) and,
        if `dominators` given, only where this band's energy leads them."""
        times = _band_onset_times(sig, DRUM_SAMPLE_RATE)
        if times.size == 0:
            return times
        energies = np.array([_rms_at(sig, t, DRUM_SAMPLE_RATE) for t in times])
        threshold = ENERGY_GATE_FRAC * float(np.percentile(energies, 90))
        kept = []
        for t, e in zip(times, energies):
            if e < threshold:
                continue
            if dominators:
                rival = max(_rms_at(d, t, DRUM_SAMPLE_RATE) for d in dominators)
                if e < KICK_DOMINANCE * rival:
                    continue
            kept.append(t)
        return np.array(kept)

    voices = [
        _Voice("kick", gate(kick_sig, [snare_sig, hat_sig]), kick_sig),
        _Voice("snare", gate(snare_sig, []), snare_sig),
        _Voice("closed_hihat", gate(hat_sig, []), hat_sig),
    ]

    all_t = np.sort(np.concatenate([v.times for v in voices if v.times.size]))
    if all_t.size == 0:
        return []
    grid: List[float] = [float(all_t[0])]
    for t in all_t[1:]:
        if t - grid[-1] > MERGE_MS / 1000.0:
            grid.append(float(t))

    notes: List[Note] = []
    for t in grid:
        onset_ms = t * 1000.0
        for v in voices:
            if not _near(v.times, t):
                continue
            note = Note(
                pitch=GM_PITCH[v.name],
                start_ms=onset_ms,
                end_ms=onset_ms + DEFAULT_DURATION_MS[v.name],
                velocity=_peak_velocity(v.signal, DRUM_SAMPLE_RATE, t),
                confidence=0.7,
                stem=StemType.DRUMS,
                source=Provenance.DRUM_INTELLIGENCE,
            )
            notes.append(note)
            annotations.add(Annotation(
                note_id=note.id, kind="drum_type", value=v.name,
                source=Provenance.DRUM_INTELLIGENCE, confidence=0.7,
            ))

    return notes


__all__ = ["detect_drums", "GM_PITCH"]
