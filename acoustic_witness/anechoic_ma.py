# =================================================================
# MODULE: acoustic_witness/anechoic_ma.py
# Ports the core of Symphony's AnechoicMa - a frame-level "what's
# really happening acoustically right now" witness. It produces a
# continuous timeline of evidence about whether a moment is
# genuinely silent, a ringing/decaying tail (resonance), or real new
# musical activity - queryable over any time window. Like every
# other pass in this codebase, it is a WITNESS, not a judge: it never
# mutates a Note, it only produces evidence a caller (the Conductor)
# writes as an Annotation.
#
# REWRITE NOTES (v2) - why the probability block changed:
#   The original three probabilities were all functions of loudness,
#   so they could not disagree (void vs active correlated -0.93).
#   Fixes:
#     1. No per-stem min-max. Level = dB above a 2 s local floor
#        divided by a FIXED 18 dB range, forced to 0 below -60 dBFS.
#     2. harmonic_persistence = 1 - spectral_flatness (was mean |STFT|,
#        which is just loudness).
#     3. Resonance REQUIRES a fall (negative dB slope), no onset, and
#        energy still above the floor. A held note is not a tail.
#     4. silent / resonant / active are a PARTITION (sum to 1).
#     5. confidence = margin between the top two scores.
#     6. Analysis window 4096 -> 2048 samples (hop unchanged).
#     7. cymbal / pedal are band FRACTIONS, not _safe_norm products.
#     8. Stem weight tables removed (the partition does not use
#        them). Deliberately NOT retuned - see the bug report.
#   Also fixed: get_resonant_regions() used to filter the silent
#   regions list; merged regions kept a stale state.
#
# STATUS: thresholds below are starting values, not yet validated on
# real vibraphone material. Do not treat active_material_probability
# < 0.15 as hard evidence in note_support until re-measured.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import librosa
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfilt

from audio_engine import AudioEngine, AudioTrack
from core import StemType

ACOUSTIC_ACTIVITY_ANNOTATION_KIND = "acoustic_activity"

# Flip to True only after validating on a real take you did not build.
# note_support can read this before trusting active_material_probability.
ANECHOIC_VALIDATED = False

ANECHOIC_SAMPLE_RATE = 22050
ANECHOIC_HOP_LENGTH = 512
ANECHOIC_FFT_SIZE = 2048            # was 4096 (186 ms); now ~93 ms
ANECHOIC_ADAPTIVE_WINDOW_SEC = 2.0
ANECHOIC_REGION_MERGE_GAP_SEC = 0.10

# --- Level (replaces min-max) ---------------------------------------
ANECHOIC_LEVEL_RANGE_DB = 18.0      # fixed dB range mapped to level 0..1
ANECHOIC_LEVEL_FLOOR_PCT = 15       # local floor percentile
ANECHOIC_ABS_GATE_DBFS = -60.0      # below this, level is forced to 0
ANECHOIC_REF_PCT = 95               # robust "loud" reference for floor cap
ANECHOIC_PRESENT_LEVEL = 0.25       # level at which energy counts as present

# --- Fall (resonance needs a falling envelope) -----------------------
ANECHOIC_SLOPE_SMOOTH_FRAMES = 5
ANECHOIC_FALL_DEADZONE_DB = 0.04    # dB/frame below which = steady
ANECHOIC_FALL_FULL_DB = 0.30        # dB/frame at which fall = 1

# --- Onsets ----------------------------------------------------------
ANECHOIC_ONSET_REF_MIN = 1.0        # floor on the onset scale (silent stems)
ANECHOIC_ONSET_REF_PCT = 95

# --- Region labelling ------------------------------------------------
ANECHOIC_REGION_MIN_MARGIN = 0.10   # top class must beat 2nd by this
ANECHOIC_SILENT_REGION_THRESHOLD = 0.50
ANECHOIC_RESONANT_REGION_THRESHOLD = 0.50

EPS = 1e-8


# =====================================================================
# Utility functions
# =====================================================================

def _rolling_percentile(x: np.ndarray, win: int, pct: float) -> np.ndarray:
    """Rolling percentile via vectorized numpy broadcasting."""
    n = len(x)
    half = win // 2
    padded = np.pad(x, (half, half), mode="edge")
    idx = np.arange(win)[None, :] + np.arange(n)[:, None]
    windows = padded[idx]
    return np.percentile(windows, pct, axis=1)


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Moving average with edge padding (no zero-padding droop at the
    ends, unlike np.convolve(..., mode='same'))."""
    if win <= 1 or len(x) == 0:
        return np.asarray(x, dtype=float).copy()
    pad_l = win // 2
    pad_r = win - 1 - pad_l
    padded = np.pad(x, (pad_l, pad_r), mode="edge")
    return np.convolve(padded, np.ones(win) / win, mode="valid")


def _band_rms(y: np.ndarray, sr: int, n_frames: int, cutoff_hz: float, highpass: bool) -> np.ndarray:
    """RMS energy in one frequency band, same framing as the main RMS."""
    try:
        nyq = sr / 2.0
        sos = butter(4, cutoff_hz / nyq, btype="high" if highpass else "low", output="sos")
        filtered = sosfilt(sos, y)
        rms = librosa.feature.rms(
            y=filtered, frame_length=ANECHOIC_FFT_SIZE, hop_length=ANECHOIC_HOP_LENGTH
        )[0]
    except Exception:
        rms = np.zeros(n_frames)
    if len(rms) >= n_frames:
        return rms[:n_frames]
    return np.pad(rms, (0, n_frames - len(rms)), mode="edge")


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
    with it (e.g. write it as an acoustic_activity Annotation).

    rhythmic_void_probability + resonance_probability +
    active_material_probability sum to ~1 (a partition)."""
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
        # Top class of the partition, only if it clearly beats the
        # runner-up. Otherwise the window stays UNCERTAIN.
        scored = sorted(
            [
                (self.rhythmic_void_probability, RegionType.SILENT),
                (self.resonance_probability, RegionType.RESONANT),
                (self.active_material_probability, RegionType.ACTIVE),
            ],
            reverse=True,
        )
        top, runner_up = scored[0], scored[1]
        if top[0] > 0.0 and (top[0] - runner_up[0]) >= ANECHOIC_REGION_MIN_MARGIN:
            object.__setattr__(self, "region_type", top[1])

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


_STATE_FIELDS = (
    "energy_floor", "spectral_stasis", "harmonic_persistence",
    "noise_floor_probability", "decay_completion", "transient_absence",
    "rhythmic_void_probability", "resonance_probability",
    "active_material_probability", "cymbal_wash_probability",
    "pedal_tone_probability", "confidence",
)


def _state_from_map(fmap: Dict[str, np.ndarray], idx: np.ndarray,
                    start_ms: float, end_ms: float) -> SilenceState:
    """Average every feature map over `idx` into one SilenceState."""
    values = {name: float(np.mean(fmap[name][idx])) for name in _STATE_FIELDS}
    return SilenceState(start_ms=start_ms, end_ms=end_ms, **values)


@dataclass
class AnechoicReport:
    """Full frame-level analysis for one audio stem. `query()` is the
    primary interface callers use - averaged evidence for an arbitrary
    time window, e.g. one Note's [start_ms, end_ms)."""
    frame_times_ms: np.ndarray
    feature_maps: Dict[str, np.ndarray]
    regions: List[SilenceRegion]            # silent regions (kept name)
    stem: StemType
    duration_seconds: float
    resonant_regions: List[SilenceRegion] = field(default_factory=list)

    def query(self, start_ms: float, end_ms: float) -> SilenceState:
        idx = self._window_slice(start_ms, end_ms)
        if idx.size == 0:
            return self._empty_state(start_ms, end_ms)
        return _state_from_map(self.feature_maps, idx, start_ms, end_ms)

    def get_silent_regions(self, threshold: float = ANECHOIC_SILENT_REGION_THRESHOLD) -> List[SilenceRegion]:
        return [r for r in self.regions if r.state.rhythmic_void_probability >= threshold]

    def get_resonant_regions(self, threshold: float = ANECHOIC_RESONANT_REGION_THRESHOLD) -> List[SilenceRegion]:
        return [r for r in self.resonant_regions if r.state.resonance_probability >= threshold]

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


def _find_spans(score: np.ndarray, threshold: float, merge_gap_frames: int,
                min_frames: int = 2) -> List[Tuple[int, int]]:
    """Contiguous index spans where score > threshold, merging spans
    separated by fewer than `merge_gap_frames` frames."""
    if len(score) == 0:
        return []
    mask = (score > threshold).astype(int)
    d = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    if len(starts) == 0:
        return []

    merged: List[Tuple[int, int]] = []
    cur_a, cur_b = int(starts[0]), int(ends[0])
    for a, b in zip(starts[1:], ends[1:]):
        if int(a) - cur_b - 1 < merge_gap_frames:
            cur_b = int(b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = int(a), int(b)
    merged.append((cur_a, cur_b))

    return [(a, b) for a, b in merged if (b - a + 1) >= min_frames]


def _derive_regions(times_ms: np.ndarray, fmap: Dict[str, np.ndarray],
                    key: str, threshold: float) -> List[SilenceRegion]:
    """Regions where fmap[key] > threshold. Each region's state is
    recomputed from its own (merged) span."""
    fps = ANECHOIC_SAMPLE_RATE / ANECHOIC_HOP_LENGTH
    gap_frames = max(1, int(round(ANECHOIC_REGION_MERGE_GAP_SEC * fps)))
    regions: List[SilenceRegion] = []
    for a, b in _find_spans(fmap[key], threshold, gap_frames):
        idx = np.arange(a, b + 1)
        start_ms, end_ms = float(times_ms[a]), float(times_ms[b])
        regions.append(SilenceRegion(
            start_ms=start_ms, end_ms=end_ms,
            state=_state_from_map(fmap, idx, start_ms, end_ms),
        ))
    return regions


# =====================================================================
# Main entry point
# =====================================================================

def _empty_report(stem: StemType, duration_seconds: float) -> AnechoicReport:
    return AnechoicReport(
        frame_times_ms=np.zeros(0),
        feature_maps={name: np.zeros(0) for name in _STATE_FIELDS},
        regions=[], stem=stem, duration_seconds=duration_seconds,
        resonant_regions=[],
    )


def analyze_stem(engine: AudioEngine, track: AudioTrack, stem: StemType) -> AnechoicReport:
    """Analyzes one stem's audio for silence/resonance/activity
    evidence. This is the one entry point - a plain function, not an
    agent class, matching every other pass in this codebase."""
    view = engine.view(track, ANECHOIC_SAMPLE_RATE)
    y = view.samples
    sr = ANECHOIC_SAMPLE_RATE
    duration_seconds = len(y) / sr

    rms = librosa.feature.rms(y=y, frame_length=ANECHOIC_FFT_SIZE, hop_length=ANECHOIC_HOP_LENGTH)[0]
    times_ms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=ANECHOIC_HOP_LENGTH) * 1000.0

    # AudioEngine's cached onset envelope (same hop as everything else).
    flux = engine.onset_envelope(track, sr, hop_length=ANECHOIC_HOP_LENGTH)

    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=ANECHOIC_FFT_SIZE, hop_length=ANECHOIC_HOP_LENGTH
    )[0]

    min_len = min(len(rms), len(flux), len(flatness))
    if min_len == 0:
        return _empty_report(stem, duration_seconds)
    rms, flux, flatness = rms[:min_len], flux[:min_len], flatness[:min_len]
    times_ms = times_ms[:min_len]

    fps = sr / ANECHOIC_HOP_LENGTH
    roll_win = max(5, int(fps * ANECHOIC_ADAPTIVE_WINDOW_SEC))

    # ---- Level: dB above a local floor, FIXED range, absolute gate ---
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    local_floor_db = _rolling_percentile(rms_db, roll_win, ANECHOIC_LEVEL_FLOOR_PCT)
    # Cap the floor below a robust loud reference so a long held note
    # (which would become its own floor) still reads as present.
    ref_db = float(np.percentile(rms_db, ANECHOIC_REF_PCT))
    floor_db = np.minimum(local_floor_db, ref_db - ANECHOIC_LEVEL_RANGE_DB)
    level = np.clip((rms_db - floor_db) / ANECHOIC_LEVEL_RANGE_DB, 0.0, 1.0)
    level[rms_db < ANECHOIC_ABS_GATE_DBFS] = 0.0   # silent stem can't invent a peak

    present = np.clip(level / ANECHOIC_PRESENT_LEVEL, 0.0, 1.0)

    # ---- Fall: positive slope of -dB/frame --------------------------
    smooth_db = _smooth(rms_db, ANECHOIC_SLOPE_SMOOTH_FRAMES)
    slope_down = -np.gradient(smooth_db)
    fall = np.clip(
        (slope_down - ANECHOIC_FALL_DEADZONE_DB)
        / (ANECHOIC_FALL_FULL_DB - ANECHOIC_FALL_DEADZONE_DB),
        0.0, 1.0,
    )

    # ---- Attack: onset strength on a fixed-ish scale ----------------
    onset_ref = max(float(np.percentile(flux, ANECHOIC_ONSET_REF_PCT)), ANECHOIC_ONSET_REF_MIN)
    onset_norm = np.clip(np.nan_to_num(flux) / onset_ref, 0.0, 1.0)
    attack = maximum_filter1d(onset_norm, size=3)   # onsets smear over ~3 frames

    # ---- Harmonicity ------------------------------------------------
    flatness = np.clip(np.nan_to_num(flatness, nan=1.0), 0.0, 1.0)
    harmonic_persistence = 1.0 - flatness
    noise_floor_probability = flatness

    # ---- The partition (sums to 1 on every frame) --------------------
    # silent   = energy not present
    # resonant = energy present, no new onset, level falling
    # active   = energy present and not (a pure tail): struck or held
    rhythmic_void_probability = 1.0 - present
    resonance_probability = present * (1.0 - attack) * fall
    active_material_probability = present - resonance_probability

    # ---- Confidence: margin between the top two classes --------------
    stacked = np.vstack([rhythmic_void_probability, resonance_probability, active_material_probability])
    top_two = np.sort(stacked, axis=0)[-2:, :]
    confidence = np.clip(top_two[1] - top_two[0], 0.0, 1.0)

    # ---- Band fractions for cymbal wash / pedal tone ------------------
    total_pow = rms ** 2 + EPS
    high_rms = _band_rms(y, sr, min_len, cutoff_hz=2000.0, highpass=True)
    low_rms = _band_rms(y, sr, min_len, cutoff_hz=250.0, highpass=False)
    high_frac = np.clip(high_rms ** 2 / total_pow, 0.0, 1.0)
    low_frac = np.clip(low_rms ** 2 / total_pow, 0.0, 1.0)

    steady = (1.0 - attack) * (1.0 - fall)
    cymbal_wash_probability = present * high_frac * fall * (1.0 - attack)
    pedal_tone_probability = present * low_frac * steady

    feature_maps = {
        "energy_floor": 1.0 - level,                 # quietness, 0..1
        "spectral_stasis": 1.0 - attack,
        "harmonic_persistence": harmonic_persistence,
        "noise_floor_probability": noise_floor_probability,
        "decay_completion": fall,
        "transient_absence": 1.0 - attack,
        "rhythmic_void_probability": rhythmic_void_probability,
        "resonance_probability": resonance_probability,
        "active_material_probability": active_material_probability,
        "cymbal_wash_probability": cymbal_wash_probability,
        "pedal_tone_probability": pedal_tone_probability,
        "confidence": confidence,
    }

    silent_regions = _derive_regions(
        times_ms, feature_maps, "rhythmic_void_probability", ANECHOIC_SILENT_REGION_THRESHOLD
    )
    resonant_regions = _derive_regions(
        times_ms, feature_maps, "resonance_probability", ANECHOIC_RESONANT_REGION_THRESHOLD
    )

    return AnechoicReport(
        frame_times_ms=times_ms,
        feature_maps=feature_maps,
        regions=silent_regions,
        stem=stem,
        duration_seconds=duration_seconds,
        resonant_regions=resonant_regions,
    )


__all__ = [
    "AnechoicReport",
    "SilenceState",
    "SilenceRegion",
    "RegionType",
    "analyze_stem",
    "ACOUSTIC_ACTIVITY_ANNOTATION_KIND",
    "ANECHOIC_VALIDATED",
]
