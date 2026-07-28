# =================================================================
# MODULE: rhythm_engine/drums.py
# Rhythm Engine: multi-voice drum detection.
#
# WHY THIS IS A REDESIGN, NOT A CLASSIFIER TWEAK
# ----------------------------------------------
# The old detector (and a faithful port of Symphony's SpectralDrumClassifier)
# emitted ONE note per onset - it picked a single label per moment. Measured
# on You Say God Says that collapsed to ~70% "kick" and ZERO snares, and it
# "missed the 16th hats". Looking at the audio showed those are the SAME bug:
# a real kit plays kick+hat (and snare+hat) SIMULTANEOUSLY, so when they land
# together the loud kick wins the single-label vote and the hat/snare is
# destroyed. On this stem 899 of the hat onsets had a kick underneath - all
# 899 hats were being eaten.
#
# THE FIX: detect each voice INDEPENDENTLY, per band, and let them coexist.
#   * kick  = a fresh attack in the low band (<150 Hz)
#   * snare = a fresh attack in the body band (200-2500 Hz) that SUSTAINS
#             broadband noise (1.5-7 kHz) past the first 15 ms - this is what
#             separates a real snare/clap burst from a kick's attack CLICK,
#             which dies in <10 ms.
#   * hat   = a fresh attack in the sizzle band (>6 kHz)
# Per-band onset detection is bleed-robust: a ringing kick TAIL creates no new
# low-band ATTACK, so it can't fake a second kick, and a kick under a hat
# can't hide the hat's high-band attack. Multiple Notes may share a start time
# (this is exactly how a GM drum channel represents a layered kit).
#
# Deferred (needs more than blind spectral cues, and guessing them would
# reintroduce the noise the user asked us to remove): open vs closed hat,
# ride vs crash vs hat, toms, rimshot. Everything currently bright is written
# as a closed hi-hat.
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
SNARE_BAND_HZ = (200, 2500)
HAT_MIN_HZ = 6000
SUSTAIN_BAND_HZ = (1500, 7000)   # snare noise-burst band

COINCIDENCE_MS = 45.0            # a moment "has" a voice if that band fired within this
MERGE_MS = 35.0                  # union-of-bands dedup
SNARE_SUSTAIN_RATIO = 0.6        # late/early brightness; >this = real noise burst, not a click
EARLY_MS, LATE_MS = 15.0, 70.0

GM_PITCH: Dict[str, int] = {"kick": 36, "snare": 38, "closed_hihat": 42}
DEFAULT_DURATION_MS: Dict[str, float] = {"kick": 90.0, "snare": 110.0, "closed_hihat": 60.0}


@dataclass(frozen=True)
class _BandOnsets:
    kick: np.ndarray             # onset times (s)
    snare: np.ndarray
    hat: np.ndarray


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


def _near(times: np.ndarray, t: float, tol_ms: float = COINCIDENCE_MS) -> bool:
    return len(times) > 0 and float(np.min(np.abs(times - t))) <= tol_ms / 1000.0


def _snare_sustains(y: np.ndarray, sr: int, t: float) -> bool:
    """A snare/clap sustains broadband noise past its attack click; a kick's
    click in the same band does not. Compare 1.5-7 kHz energy in the 15-70 ms
    window against the first 15 ms."""
    i0 = int(t * sr)
    early = y[i0: i0 + int(EARLY_MS / 1000.0 * sr)]
    late = y[i0 + int(EARLY_MS / 1000.0 * sr): i0 + int(LATE_MS / 1000.0 * sr)]

    def bright_energy(seg: np.ndarray) -> float:
        if len(seg) < 16:
            return 0.0
        spec = np.abs(np.fft.rfft(seg)) ** 2
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        lo, hi = SUSTAIN_BAND_HZ
        return float(np.sum(spec[(freqs >= lo) & (freqs < hi)])) / len(seg)

    late_e = bright_energy(late)
    return late_e > 1e-4 and late_e / (bright_energy(early) + 1e-12) > SNARE_SUSTAIN_RATIO


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
    Annotation. Kick/snare/hat are detected independently, so a single
    moment can produce several coincident Notes."""
    view = engine.view(track, DRUM_SAMPLE_RATE)
    y = np.ascontiguousarray(view.samples, dtype=np.float64)
    if y.size < 32:
        return []
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

    kick_sig = _bandpass(y, None, KICK_MAX_HZ, DRUM_SAMPLE_RATE)
    snare_sig = _bandpass(y, SNARE_BAND_HZ[0], SNARE_BAND_HZ[1], DRUM_SAMPLE_RATE)
    hat_sig = _bandpass(y, HAT_MIN_HZ, None, DRUM_SAMPLE_RATE)
    bands = _BandOnsets(
        kick=_band_onset_times(kick_sig, DRUM_SAMPLE_RATE),
        snare=_band_onset_times(snare_sig, DRUM_SAMPLE_RATE),
        hat=_band_onset_times(hat_sig, DRUM_SAMPLE_RATE),
    )

    all_t = np.sort(np.concatenate([bands.kick, bands.snare, bands.hat]))
    if all_t.size == 0:
        return []
    grid: List[float] = [float(all_t[0])]
    for t in all_t[1:]:
        if t - grid[-1] > MERGE_MS / 1000.0:
            grid.append(float(t))

    notes: List[Note] = []
    for t in grid:
        voices: List[str] = []
        if _near(bands.kick, t):
            voices.append("kick")
        if _near(bands.snare, t) and _snare_sustains(y, DRUM_SAMPLE_RATE, t):
            voices.append("snare")
        if _near(bands.hat, t):
            voices.append("closed_hihat")
        if not voices:
            continue

        onset_ms = t * 1000.0
        sig_for = {"kick": kick_sig, "snare": snare_sig, "closed_hihat": hat_sig}
        for drum_type in voices:
            note = Note(
                pitch=GM_PITCH[drum_type],
                start_ms=onset_ms,
                end_ms=onset_ms + DEFAULT_DURATION_MS[drum_type],
                velocity=_peak_velocity(sig_for[drum_type], DRUM_SAMPLE_RATE, t),
                confidence=0.7,
                stem=StemType.DRUMS,
                source=Provenance.DRUM_INTELLIGENCE,
            )
            notes.append(note)
            annotations.add(Annotation(
                note_id=note.id, kind="drum_type", value=drum_type,
                source=Provenance.DRUM_INTELLIGENCE, confidence=0.7,
            ))

    return notes


__all__ = ["detect_drums", "GM_PITCH"]
