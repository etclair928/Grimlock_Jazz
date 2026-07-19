# =================================================================
# MODULE: acoustic_witness/anechoic_ma.py
# Ports the core of Symphony's AnechoicMa (agents/analysis/
# anechoic_ma.py) - a frame-level "what's really happening
# acoustically right now" witness. It produces a continuous timeline
# of evidence about whether a moment is genuinely silent (rhythmic
# void), a ringing/decaying tail (resonance - cymbal wash, pedal tone,
# reverb), or real new musical activity - queryable over any time
# window. Like every other pass in this codebase, it is a WITNESS,
# not a judge: it never mutates a Note, it only produces evidence a
# caller (the Conductor) writes as an Annotation.
#
# Directly motivated by tonight's "other"/vocals-stem investigation:
# warm_sustained's absurd density (9.77 notes/sec, 246% cross-pitch
# overlap on a 60s Hopeful.mp3 clip) is plausibly inflated by
# resonance tails Basic Pitch misread as new note onsets rather than
# the continuation of a real note - resonance_probability is built to
# catch exactly that, as an acoustic-grounded complement to the purely
# symbolic evidence (duration, confidence, centroid) used so far.
#
# Cut from the original, and why (same "don't build ahead of need" /
# "no redundant subsystems" discipline as every other port this
# session):
#   - Swing-aware subdivision alignment (SubdivisionAlignment,
#     _subdivision_maps): Jazz's own Rhythm Engine (PulseField,
#     lattice_witness, meter.py's phase-locked grid) already owns
#     rhythmic-placement judgment, far more rigorously. A second,
#     cruder subdivision-alignment system here would be exactly the
#     redundant-subsystem trap this project exists to avoid.
#   - Reverb-profile detection (AnechoicProfile: STUDIO/LIVE_ROOM/
#     CHURCH/OUTDOOR): no downstream consumer needs this metadata.
#   - Agent-framework plumbing (StageResult, StatusReporterProtocol,
#     MemoryManagedProtocol, run()/validate()/get_confidence()): Jazz
#     has no agent-runner architecture; every port this session is a
#     plain function, not an agent class.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

from audio_engine import AudioEngine, AudioTrack
from core import StemType

ACOUSTIC_ACTIVITY_ANNOTATION_KIND = "acoustic_activity"

ANECHOIC_SAMPLE_RATE = 22050
ANECHOIC_HOP_LENGTH = 512
ANECHOIC_FFT_SIZE = 4096
ANECHOIC_ADAPTIVE_WINDOW_SEC = 2.0
ANECHOIC_REGION_MERGE_GAP_SEC = 0.10

EPS = 1e-8


# =====================================================================
# Vectorized utility functions (unchanged from the original - already
# correct, self-contained numpy)
# =====================================================================

def _safe_norm(x: np.ndarray) -> np.ndarray:
    """Normalize to [0,1]. Handles NaN and constant arrays."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if (hi - lo) < EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _rolling_percentile(x: np.ndarray, win: int, pct: float) -> np.ndarray:
    """Rolling percentile via vectorized numpy broadcasting - ~40x
    faster than a Python loop (verified against real, minute-plus-long
    stems in Symphony where the loop version was a real bottleneck)."""
    n = len(x)
    half = win // 2
    padded = np.pad(x, (half, half), mode="edge")
    idx = np.arange(win)[None, :] + np.arange(n)[:, None]
    windows = padded[idx]
    return np.percentile(windows, pct, axis=1)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    return np.convolve(x, np.ones(win) / win, mode="same")


def _band_rms(y: np.ndarray, sr: int, n_frames: int, cutoff_hz: float, highpass: bool) -> np.ndarray:
    """RMS energy in one frequency band (high-pass for cymbals/
    transients, low-pass for bass/pedal tones)."""
    try:
        nyq = sr / 2.0
        sos = butter(4, cutoff_hz / nyq, btype="high" if highpass else "low", output="sos")
        filtered = sosfilt(sos, y)
        rms = librosa.feature.rms(y=filtered, hop_length=ANECHOIC_HOP_LENGTH)[0]
    except Exception:
        rms = np.zeros(n_frames)
    if len(rms) >= n_frames:
        return rms[:n_frames]
    return np.pad(rms, (0, n_frames - len(rms)), mode="edge")


# =====================================================================
# Stem-adaptive weights - unchanged from the original's calibration,
# extended with Jazz's additional StemType members (VOCALS/GUITAR/
# PIANO fall back to the same defaults FULL_MIX uses, since the
# original never distinguished them either - htdemucs_6s wasn't
# splitting those stems out yet when this was tuned).
# =====================================================================

def _margin_multiplier(stem: StemType) -> float:
    return {
        StemType.DRUMS: 0.7,
        StemType.BASS: 0.6,
        StemType.OTHER: 0.8,
        StemType.FULL_MIX: 1.0,
    }.get(stem, 1.0)


def _composite_weights(stem: StemType) -> Dict[str, float]:
    defaults = {
        "void_energy": 0.30, "void_stasis": 0.20, "void_transient": 0.20,
        "void_decay": 0.15, "void_noise": 0.15,
        "res_harmonic": 0.35, "res_stasis": 0.20, "res_cymbal": 0.20,
        "res_pedal": 0.15, "res_decay": 0.10,
        "active_flux": 0.45, "active_energy": 0.25,
        "active_stasis": 0.20, "active_harmonic": 0.10,
    }
    overrides = {
        StemType.DRUMS: {"void_transient": 0.30, "res_cymbal": 0.35, "active_flux": 0.55},
        StemType.BASS: {"void_energy": 0.35, "res_pedal": 0.40, "active_harmonic": 0.25},
        StemType.OTHER: {"res_harmonic": 0.45, "void_decay": 0.20, "active_harmonic": 0.20},
    }
    w = {**defaults, **overrides.get(stem, {})}
    for group in ("void", "res", "active"):
        keys = [k for k in w if k.startswith(group)]
        total = sum(w[k] for k in keys)
        if total > 0:
            for k in keys:
                w[k] /= total
    return w


def _region_threshold(stem: StemType) -> float:
    return {
        StemType.DRUMS: 0.55,
        StemType.BASS: 0.60,
        StemType.OTHER: 0.58,
        StemType.FULL_MIX: 0.65,
    }.get(stem, 0.62)


# =====================================================================
# Result types
# =====================================================================

class RegionType:
    SILENT = "silent"
    RESONANT = "resonant"
    ACTIVE = "active"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class SilenceState:
    """Immutable evidence for one time window - never a verdict on its
    own; a caller (the Conductor) decides what, if anything, to do
    with it (e.g. write it as an acoustic_activity Annotation)."""
    start_ms: float
    end_ms: float
    energy_floor: float
    spectral_stasis: float
    harmonic_persistence: float
    noise_floor_probability: float
    decay_completion: float
    transient_absence: float
    rhythmic_void_probability: float
    resonance_probability: float
    active_material_probability: float
    cymbal_wash_probability: float
    pedal_tone_probability: float
    confidence: float
    region_type: str = RegionType.UNCERTAIN

    def __post_init__(self) -> None:
        if self.rhythmic_void_probability > 0.65:
            object.__setattr__(self, "region_type", RegionType.SILENT)
        elif self.resonance_probability > 0.6:
            object.__setattr__(self, "region_type", RegionType.RESONANT)
        elif self.active_material_probability > 0.7:
            object.__setattr__(self, "region_type", RegionType.ACTIVE)

    def get_confidence_penalty(self) -> float:
        """Recommended confidence penalty for a note falling in this
        window - silent regions penalize hardest, resonant regions
        moderately, active regions not at all. A caller applies this;
        this method never mutates anything itself."""
        if self.region_type == RegionType.SILENT:
            return min(0.8, self.rhythmic_void_probability * 0.9)
        if self.region_type == RegionType.RESONANT:
            return min(0.4, self.resonance_probability * 0.5)
        return 0.0


@dataclass(frozen=True)
class SilenceRegion:
    start_ms: float
    end_ms: float
    state: SilenceState


@dataclass
class AnechoicReport:
    """Full frame-level analysis for one audio stem. `query()` is the
    primary interface callers use - averaged evidence for an arbitrary
    time window, e.g. one Note's [start_ms, end_ms)."""
    frame_times_ms: np.ndarray
    feature_maps: Dict[str, np.ndarray]
    regions: List[SilenceRegion]
    stem: StemType
    duration_seconds: float

    def query(self, start_ms: float, end_ms: float) -> SilenceState:
        idx = self._window_slice(start_ms, end_ms)
        if idx.size == 0:
            return self._empty_state(start_ms, end_ms)

        def avg(name: str) -> float:
            return float(np.mean(self.feature_maps[name][idx]))

        return SilenceState(
            start_ms=start_ms, end_ms=end_ms,
            energy_floor=avg("energy_floor"),
            spectral_stasis=avg("spectral_stasis"),
            harmonic_persistence=avg("harmonic_persistence"),
            noise_floor_probability=avg("noise_floor_probability"),
            decay_completion=avg("decay_completion"),
            transient_absence=avg("transient_absence"),
            rhythmic_void_probability=avg("rhythmic_void_probability"),
            resonance_probability=avg("resonance_probability"),
            active_material_probability=avg("active_material_probability"),
            cymbal_wash_probability=avg("cymbal_wash_probability"),
            pedal_tone_probability=avg("pedal_tone_probability"),
            confidence=avg("confidence"),
        )

    def get_silent_regions(self, threshold: float = 0.65) -> List[SilenceRegion]:
        return [r for r in self.regions if r.state.rhythmic_void_probability >= threshold]

    def get_resonant_regions(self, threshold: float = 0.60) -> List[SilenceRegion]:
        return [r for r in self.regions if r.state.resonance_probability >= threshold]

    def _window_slice(self, start_ms: float, end_ms: float) -> np.ndarray:
        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms
        return np.where((self.frame_times_ms >= start_ms) & (self.frame_times_ms <= end_ms))[0]

    def _empty_state(self, start_ms: float, end_ms: float) -> SilenceState:
        return SilenceState(
            start_ms=start_ms, end_ms=end_ms,
            energy_floor=0.0, spectral_stasis=0.0, harmonic_persistence=0.0,
            noise_floor_probability=0.0, decay_completion=0.0, transient_absence=0.0,
            rhythmic_void_probability=0.0, resonance_probability=0.0,
            active_material_probability=0.0, cymbal_wash_probability=0.0,
            pedal_tone_probability=0.0, confidence=0.0,
        )


def _derive_regions(times_ms: np.ndarray, fmap: Dict[str, np.ndarray], threshold: float) -> List[SilenceRegion]:
    """Contiguous rhythmic-void regions above `threshold`, merging
    fragments less than ANECHOIC_REGION_MERGE_GAP_SEC apart."""
    score = fmap["rhythmic_void_probability"]
    mask = score > threshold

    regions: List[SilenceRegion] = []
    start_idx: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            r = _make_region(times_ms, fmap, start_idx, i - 1)
            if r:
                regions.append(r)
            start_idx = None
    if start_idx is not None:
        r = _make_region(times_ms, fmap, start_idx, len(mask) - 1)
        if r:
            regions.append(r)

    if len(regions) > 1:
        merged: List[SilenceRegion] = []
        cur = regions[0]
        merge_gap_ms = ANECHOIC_REGION_MERGE_GAP_SEC * 1000.0
        for nxt in regions[1:]:
            if nxt.start_ms - cur.end_ms < merge_gap_ms:
                cur = SilenceRegion(cur.start_ms, nxt.end_ms, cur.state)
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        regions = merged

    return regions


def _make_region(times_ms: np.ndarray, fmap: Dict[str, np.ndarray], a: int, b: int) -> Optional[SilenceRegion]:
    if b <= a:
        return None
    idx = np.arange(a, b + 1)

    def avg(name: str) -> float:
        return float(np.mean(fmap[name][idx]))

    state = SilenceState(
        start_ms=float(times_ms[a]), end_ms=float(times_ms[b]),
        energy_floor=avg("energy_floor"),
        spectral_stasis=avg("spectral_stasis"),
        harmonic_persistence=avg("harmonic_persistence"),
        noise_floor_probability=avg("noise_floor_probability"),
        decay_completion=avg("decay_completion"),
        transient_absence=avg("transient_absence"),
        rhythmic_void_probability=avg("rhythmic_void_probability"),
        resonance_probability=avg("resonance_probability"),
        active_material_probability=avg("active_material_probability"),
        cymbal_wash_probability=avg("cymbal_wash_probability"),
        pedal_tone_probability=avg("pedal_tone_probability"),
        confidence=avg("confidence"),
    )
    return SilenceRegion(start_ms=float(times_ms[a]), end_ms=float(times_ms[b]), state=state)


def analyze_stem(engine: AudioEngine, track: AudioTrack, stem: StemType) -> AnechoicReport:
    """Analyzes one stem's audio for silence/resonance/activity
    evidence. This is the one entry point - a plain function, not an
    agent class, matching every other pass in this codebase."""
    view = engine.view(track, ANECHOIC_SAMPLE_RATE)
    y = view.samples
    sr = ANECHOIC_SAMPLE_RATE

    rms = librosa.feature.rms(y=y, frame_length=ANECHOIC_FFT_SIZE, hop_length=ANECHOIC_HOP_LENGTH)[0]
    times_ms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=ANECHOIC_HOP_LENGTH) * 1000.0

    # Reuses AudioEngine's own cached onset envelope - identical call
    # (librosa.onset.onset_strength) to what the original computed
    # fresh every time; no reason to pay for it twice.
    flux = engine.onset_envelope(track, sr, hop_length=ANECHOIC_HOP_LENGTH)

    flatness = librosa.feature.spectral_flatness(y=y, n_fft=ANECHOIC_FFT_SIZE, hop_length=ANECHOIC_HOP_LENGTH)[0]

    # Reuses AudioEngine's cached STFT rather than recomputing it here -
    # identical call (librosa.stft over the same mono view at the same
    # n_fft/hop), and the Audio Engine boundary (§7) is the one place
    # transforms are meant to be computed and cached.
    S = np.abs(engine.stft(track, sr, n_fft=ANECHOIC_FFT_SIZE, hop_length=ANECHOIC_HOP_LENGTH))
    harmonic_proxy = np.mean(S, axis=0)

    min_len = min(len(rms), len(flux), len(flatness), len(harmonic_proxy))
    rms, flux, flatness, harmonic_proxy = rms[:min_len], flux[:min_len], flatness[:min_len], harmonic_proxy[:min_len]
    times_ms = times_ms[:min_len]

    fps = sr / ANECHOIC_HOP_LENGTH
    roll_win = max(5, int(fps * ANECHOIC_ADAPTIVE_WINDOW_SEC))
    floor = _rolling_percentile(rms, roll_win, 10)

    margin = np.percentile(rms, 60) * 0.05 * _margin_multiplier(stem)
    quietness = np.clip((floor + margin - rms) / (floor + margin + EPS), 0, 1)
    energy_floor = _safe_norm(quietness)

    flux_norm = _safe_norm(flux)
    spectral_stasis = 1.0 - flux_norm
    noise_floor_probability = _safe_norm(flatness)
    harmonic_persistence = _safe_norm(harmonic_proxy)

    smooth_rms = _moving_average(rms, 5)
    decay_completion = _safe_norm(np.clip(-np.gradient(smooth_rms), 0, None))
    transient_absence = 1.0 - flux_norm

    high_freq_energy = _band_rms(y, sr, min_len, cutoff_hz=2000.0, highpass=True)
    low_freq_energy = _band_rms(y, sr, min_len, cutoff_hz=250.0, highpass=False)
    cymbal_wash_probability = _safe_norm(high_freq_energy * (1.0 - flux_norm))
    pedal_tone_probability = _safe_norm(low_freq_energy * harmonic_persistence)

    w = _composite_weights(stem)
    rhythmic_void_probability = np.clip(
        w["void_energy"] * energy_floor + w["void_stasis"] * spectral_stasis
        + w["void_transient"] * transient_absence + w["void_decay"] * decay_completion
        + w["void_noise"] * (1.0 - noise_floor_probability),
        0, 1,
    )
    resonance_probability = np.clip(
        w["res_harmonic"] * harmonic_persistence + w["res_stasis"] * spectral_stasis
        + w["res_cymbal"] * cymbal_wash_probability + w["res_pedal"] * pedal_tone_probability
        + w["res_decay"] * (1.0 - decay_completion),
        0, 1,
    )
    active_material_probability = np.clip(
        w["active_flux"] * flux_norm + w["active_energy"] * (1.0 - energy_floor)
        + w["active_stasis"] * (1.0 - spectral_stasis) + w["active_harmonic"] * harmonic_persistence,
        0, 1,
    )
    confidence = np.clip(1.0 - np.abs(rhythmic_void_probability - 0.5) * 1.5, 0.3, 0.95)

    feature_maps = {
        "energy_floor": energy_floor,
        "spectral_stasis": spectral_stasis,
        "harmonic_persistence": harmonic_persistence,
        "noise_floor_probability": noise_floor_probability,
        "decay_completion": decay_completion,
        "transient_absence": transient_absence,
        "rhythmic_void_probability": rhythmic_void_probability,
        "resonance_probability": resonance_probability,
        "active_material_probability": active_material_probability,
        "cymbal_wash_probability": cymbal_wash_probability,
        "pedal_tone_probability": pedal_tone_probability,
        "confidence": confidence,
    }

    regions = _derive_regions(times_ms, feature_maps, _region_threshold(stem))

    return AnechoicReport(
        frame_times_ms=times_ms,
        feature_maps=feature_maps,
        regions=regions,
        stem=stem,
        duration_seconds=len(y) / sr,
    )


__all__ = [
    "AnechoicReport",
    "SilenceState",
    "SilenceRegion",
    "RegionType",
    "analyze_stem",
    "ACOUSTIC_ACTIVITY_ANNOTATION_KIND",
]
