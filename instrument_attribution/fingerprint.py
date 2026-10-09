# =================================================================
# MODULE: instrument_attribution/fingerprint.py
# Step 1 of "fingerprint -> stream -> resolve".
#
# Per-NOTE overtone features, measured by HETERODYNING (not an STFT):
# for a note at f0, multiply by exp(-j*2*pi*k*f0*t) and low-pass at
# ~40 Hz. That gives the amplitude a_k(t) and the instantaneous
# frequency of partial k, even in the bass where STFT bins are too
# wide to tell partials apart.
#
# Only the STEADY STATE is used (from ~40 ms after onset until the
# note clearly decays). Partials that land within 40 cents of a
# partial of another simultaneous note are masked (NaN).
#
# Individual notes are still noisy. The line-level median across a
# voice_continuity line is what identifies an instrument:
# use pool_fingerprints() at the bottom for that.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from audio_engine import AudioEngine, AudioTrack
from core import Note

FINGERPRINT_SAMPLE_RATE = 22050

N_PARTIALS = 12            # partials 1..12 are measured
STEADY_START_MS = 40.0     # skip hammer/pick/tongue transient
MIN_STEADY_MS = 80.0       # shorter steady state = not enough evidence
PRE_ROLL_MS = 20.0         # audio taken before onset (filter settling, attack residual)
ATTACK_MS = 30.0           # window for the attack-noise measure
LOWPASS_HZ = 40.0          # heterodyne low-pass cutoff
ENV_RATE_HZ = 500.0        # rate the partial envelopes are decimated to
DECAY_FRACTION = 0.25      # steady state ends when partial 1 drops below 25% of peak (-12 dB)
COLLISION_CENTS = 40.0     # mask partials this close to another note's partial
MIN_OVERLAP_MS = 20.0      # another note must overlap the steady state by this much to count
MIN_PARTIAL_DB = -40.0     # partials weaker than this (vs partial 1) are below the noise floor
VIBRATO_BAND_HZ = (4.0, 8.0)
MIN_VIBRATO_MS = 300.0     # need ~1+ vibrato cycle to measure it

_NAN = float("nan")


@dataclass(frozen=True)
class TimbreFingerprint:
    note_id: str
    f0_hz: float
    # dB of partial k relative to partial 1, for k = 2..12. NaN = masked/unavailable.
    # Weak partials are floored at MIN_PARTIAL_DB.
    partial_log_ratios: Tuple[float, ...]
    inharmonicity_b: float            # string stiffness B; ~0 for winds/voice/bowed
    even_odd_ratio: float             # RMS even-partial / RMS odd-partial (clarinet is low)
    harmonic_centroid_slope: float    # partials per second (piano falls, organ flat)
    vibrato_depth_cents: float        # peak FM depth of partial 1 in 4-8 Hz (NaN if note too short)
    vibrato_rate_hz: float
    attack_noise_ratio: float         # energy the harmonic model can't explain in first 30 ms
    n_partials_used: int              # unmasked partials among 1..12
    steady_ms: float


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _note_f0_hz(note) -> Optional[float]:
    """Find the pitch on a Note. EDIT HERE if your Note uses another field name."""
    for name in ("frequency_hz", "freq_hz", "f0_hz", "hz"):
        v = getattr(note, name, None)
        if v:
            return float(v)
    for name in ("midi", "pitch", "midi_pitch", "pitch_midi", "midi_note"):
        v = getattr(note, name, None)
        if v is not None:
            return 440.0 * 2.0 ** ((float(v) - 69.0) / 12.0)
    return None


def _lowpass_complex(z: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    sos = butter(4, cutoff_hz, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, z.real) + 1j * sosfiltfilt(sos, z.imag)


def _refine_f0(seg: np.ndarray, sr: int, f0_nom: float) -> float:
    """Basic Pitch pitches are rounded to semitones; find the real f0 within +/-50 cents."""
    n = len(seg)
    if n < 64:
        return f0_nom
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(n), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    lo, hi = f0_nom * 2 ** (-50 / 1200), f0_nom * 2 ** (50 / 1200)
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    if len(idx) < 3:
        return f0_nom
    i = int(idx[np.argmax(spec[idx])])
    if 0 < i < len(spec) - 1:
        a, b, c = (np.log(spec[j] + 1e-12) for j in (i - 1, i, i + 1))
        denom = a - 2 * b + c
        off = 0.0 if denom == 0 else float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))
        return float(freqs[i] + off * (freqs[1] - freqs[0]))
    return float(freqs[i])


def _fingerprint_one(
        note_id: str,
        x: np.ndarray,
        onset_idx: int,
        sr: int,
        f0_nom: float,
        win_start_ms: float,
        overlapping_f0s,
) -> Optional[TimbreFingerprint]:
    n = len(x)
    D = max(1, int(round(sr / ENV_RATE_HZ)))
    env_rate = sr / D

    i_s0 = onset_idx + int(STEADY_START_MS * sr / 1000.0)
    if n - i_s0 < int(MIN_STEADY_MS * sr / 1000.0):
        return None

    f0 = _refine_f0(x[i_s0:i_s0 + int(0.4 * sr)], sr, f0_nom)
    k_max = min(N_PARTIALS, int(0.45 * sr / f0))
    if k_max < 2:
        return None
    cutoff = min(LOWPASS_HZ, 0.45 * f0)  # never wide enough to swallow the neighbouring partial
    t = np.arange(n) / sr

    # --- Partial 1 gives the phase reference (follows vibrato / pitch drift) ---
    z1 = _lowpass_complex(x * np.exp(-2j * np.pi * f0 * t), cutoff, sr)
    p1 = np.unwrap(np.angle(z1))
    phi = 2.0 * np.pi * f0 * t + p1
    f1_inst = f0 + np.gradient(np.unwrap(np.angle(z1[::D])), 1.0 / env_rate) / (2.0 * np.pi)
    f1_inst = np.convolve(f1_inst, np.ones(5) / 5.0, mode="same")

    # --- Heterodyne every partial against k * phi ---
    d_n = len(range(0, n, D))
    amps = np.zeros((k_max + 1, d_n))
    phs = np.zeros((k_max + 1, d_n))
    att = slice(onset_idx, min(n, onset_idx + int(ATTACK_MS * sr / 1000.0)))
    recon = np.zeros(att.stop - att.start)
    for k in range(1, k_max + 1):
        zk = _lowpass_complex(x * np.exp(-1j * k * phi), cutoff, sr)
        zd = zk[::D]
        amps[k] = 2.0 * np.abs(zd)
        phs[k] = np.unwrap(np.angle(zd))
        recon += 2.0 * np.real(zk[att] * np.exp(1j * k * phi[att]))

    # --- Steady-state window: after the transient, until clearly decaying ---
    d0 = int(np.ceil(i_s0 / D))
    d_hard_end = d_n - int(np.ceil(0.02 * env_rate))  # drop filter edge effects
    if d_hard_end - d0 < 3:
        return None
    a1_env = amps[1]
    pk = d0 + int(np.argmax(a1_env[d0:d_hard_end]))
    below = np.where(a1_env[pk:d_hard_end] < DECAY_FRACTION * a1_env[pk])[0]
    d1 = pk + int(below[0]) if len(below) else d_hard_end
    steady_ms = (d1 - d0) / env_rate * 1000.0
    if steady_ms < MIN_STEADY_MS:
        return None

    A = np.array([np.median(amps[k, d0:d1]) for k in range(k_max + 1)])
    a1 = A[1]
    if a1 < 1e-5:
        return None

    # --- Collision mask against other simultaneous notes ---
    s0_ms = win_start_ms + d0 * D / sr * 1000.0
    s1_ms = win_start_ms + d1 * D / sr * 1000.0
    masked = np.zeros(N_PARTIALS + 1, dtype=bool)
    masked[k_max + 1:] = True
    ks_arr = np.arange(1, k_max + 1)[:, None]
    for fo in overlapping_f0s(s0_ms, s1_ms):
        m = np.arange(1, int(np.ceil(k_max * f0 * 1.05 / fo)) + 2)
        cents = np.abs(1200.0 * np.log2((ks_arr * f0) / (m[None, :] * fo)))
        masked[1:k_max + 1] |= cents.min(axis=1) < COLLISION_CENTS
    if masked[1]:
        return None  # no clean reference partial; this note can't be normalised

    ratios_db: List[float] = []
    for k in range(2, N_PARTIALS + 1):
        if masked[k]:
            ratios_db.append(_NAN)
        else:
            ratios_db.append(max(20.0 * np.log10(max(A[k] / a1, 1e-9)), MIN_PARTIAL_DB))

    t_d = np.arange(d0, d1) / env_rate
    F1 = float(np.median(f1_inst[d0:d1]))

    # --- Inharmonicity: (f_k / (k f1))^2 ~ 1 + B k^2 ---
    num = den = 0.0
    used = 0
    for k in range(2, k_max + 1):
        if masked[k] or 20.0 * np.log10(max(A[k] / a1, 1e-9)) <= MIN_PARTIAL_DB:
            continue
        slope_hz = np.polyfit(t_d, phs[k, d0:d1], 1)[0] / (2.0 * np.pi)
        delta = slope_hz / (k * F1)
        y = (1.0 + delta) ** 2 - 1.0
        num += y * k ** 2
        den += float(k) ** 4
        used += 1
    inharm_b = float(num / den) if used >= 2 else _NAN

    # --- Odd / even (partial 1 counts as odd, at ratio 1) ---
    ks = [k for k in range(1, k_max + 1) if not masked[k]]
    r = {k: max(A[k] / a1, 1e-3) for k in ks}
    even = [r[k] for k in ks if k % 2 == 0]
    odd = [r[k] for k in ks if k % 2 == 1]
    even_odd = float(np.sqrt(np.mean(np.square(even))) / np.sqrt(np.mean(np.square(odd)))) if even else _NAN

    # --- Harmonic centroid slope across the note ---
    if len(ks) >= 2:
        w = amps[ks, d0:d1]
        cen = (np.array(ks)[:, None] * w).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-12)
        cen_slope = float(np.polyfit(t_d, cen, 1)[0])
    else:
        cen_slope = _NAN

    # --- Vibrato: FM of partial 1 in 4-8 Hz ---
    vib_depth = vib_rate = _NAN
    seg = f1_inst[d0:d1]
    if steady_ms >= MIN_VIBRATO_MS and np.all(seg > 0):
        cents_track = 1200.0 * np.log2(seg / np.median(seg))
        tt = np.arange(len(cents_track)) / env_rate
        cents_track = cents_track - np.polyval(np.polyfit(tt, cents_track, 1), tt)
        win = np.hanning(len(cents_track))
        nfft = 1 << int(np.ceil(np.log2(len(cents_track) * 8)))
        sp = 2.0 * np.abs(np.fft.rfft(cents_track * win, nfft)) / np.sum(win)
        fr = np.fft.rfftfreq(nfft, 1.0 / env_rate)
        band = (fr >= VIBRATO_BAND_HZ[0]) & (fr <= VIBRATO_BAND_HZ[1])
        if band.any():
            j = int(np.argmax(sp[band]))
            vib_depth = float(sp[band][j])
            vib_rate = float(fr[band][j])

    # --- Attack noise: energy the harmonic model can't explain in the first 30 ms ---
    xs = x[att]
    e = float(np.sum(xs ** 2))
    attack = float(np.sum((xs - recon) ** 2) / e) if e > 1e-12 and len(xs) >= 16 else _NAN

    return TimbreFingerprint(
        note_id=note_id,
        f0_hz=float(f0),
        partial_log_ratios=tuple(ratios_db),
        inharmonicity_b=inharm_b,
        even_odd_ratio=even_odd,
        harmonic_centroid_slope=cen_slope,
        vibrato_depth_cents=vib_depth,
        vibrato_rate_hz=vib_rate,
        attack_noise_ratio=attack,
        n_partials_used=len(ks),
        steady_ms=float(steady_ms),
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def fingerprint_notes(
        engine: AudioEngine,
        track: AudioTrack,
        notes: List[Note],
        sample_rate: int = FINGERPRINT_SAMPLE_RATE,
) -> Dict[str, TimbreFingerprint]:
    """One TimbreFingerprint per usable note, keyed by Note.id.

    Notes are SKIPPED (no entry) when they have no usable steady state
    (under ~120 ms total), or when partial 1 collides with another
    simultaneous note's partial (e.g. an octave doubling). Pass ALL
    notes that sound together, not just one line, so collisions can
    be detected."""
    view = engine.view(track, sample_rate)

    f0_list = [_note_f0_hz(nt) for nt in notes]
    if notes and all(f is None for f in f0_list):
        raise AttributeError(
            "fingerprint.py: could not find a pitch on Note. "
            "Edit _note_f0_hz() to use your Note's pitch field name."
        )
    f0_arr = np.array([f if f else np.nan for f in f0_list], dtype=float)
    starts = np.array([nt.start_ms for nt in notes], dtype=float)
    ends = np.array([nt.end_ms for nt in notes], dtype=float)

    fingerprints: Dict[str, TimbreFingerprint] = {}
    for i, note in enumerate(notes):
        f0_nom = f0_list[i]
        if not f0_nom:
            continue

        def overlapping(s0: float, s1: float, i: int = i) -> np.ndarray:
            ov = np.minimum(ends, s1) - np.maximum(starts, s0)
            sel = (ov > MIN_OVERLAP_MS) & np.isfinite(f0_arr)
            sel[i] = False
            return f0_arr[sel]

        win_start_ms = max(0.0, note.start_ms - PRE_ROLL_MS)
        window = view.slice_ms(win_start_ms, note.end_ms)
        x = np.asarray(window.samples, dtype=np.float64)
        onset_idx = int(round((note.start_ms - win_start_ms) * sample_rate / 1000.0))

        fp = _fingerprint_one(note.id, x, onset_idx, sample_rate, f0_nom, win_start_ms, overlapping)
        if fp is not None:
            fingerprints[note.id] = fp

    return fingerprints


# ---------------------------------------------------------------
# Line-level pooling (one note is noisy; a line of notes is an instrument)
# ---------------------------------------------------------------

@dataclass(frozen=True)
class LineTimbre:
    n_notes: int
    partial_log_ratios: Tuple[float, ...]  # median envelope, k = 2..12, dB vs partial 1
    inharmonicity_b: float                 # fit on the LOWEST third of notes only
    even_odd_ratio: float
    harmonic_centroid_slope: float
    vibrato_depth_cents: float
    attack_noise_ratio: float


def _median(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else _NAN


def pool_fingerprints(fps: List[TimbreFingerprint]) -> Optional[LineTimbre]:
    """Median-pool the fingerprints of one voice line (from voice_continuity)."""
    if not fps:
        return None
    n_k = len(fps[0].partial_log_ratios)
    envelope = tuple(_median(fp.partial_log_ratios[i] for fp in fps) for i in range(n_k))
    # Piano stiffness is unambiguous in the bass strings, so fit B on the low notes.
    lowest = sorted(fps, key=lambda fp: fp.f0_hz)[:max(1, len(fps) // 3)]
    return LineTimbre(
        n_notes=len(fps),
        partial_log_ratios=envelope,
        inharmonicity_b=_median(fp.inharmonicity_b for fp in lowest),
        even_odd_ratio=_median(fp.even_odd_ratio for fp in fps),
        harmonic_centroid_slope=_median(fp.harmonic_centroid_slope for fp in fps),
        vibrato_depth_cents=_median(fp.vibrato_depth_cents for fp in fps),
        attack_noise_ratio=_median(fp.attack_noise_ratio for fp in fps),
    )


__all__ = [
    "TimbreFingerprint",
    "LineTimbre",
    "fingerprint_notes",
    "pool_fingerprints",
    "FINGERPRINT_SAMPLE_RATE",
]
