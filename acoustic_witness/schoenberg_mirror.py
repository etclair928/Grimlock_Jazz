# =================================================================
# MODULE: acoustic_witness/schoenberg_mirror.py
# Ports Symphony's SchoenbergMirror (agents/validation/
# schoenberg_mirror.py) - a per-note harmonic-legitimacy auditor.
# "It does not transcribe. It audits." A note is judged TONAL
# (legitimate harmonic series), PERCUSSION (exempt - no harmonic
# structure expected), UNCERTAIN (ambiguous or witnesses disagree),
# NOISE (no harmonic structure, not percussive either), or
# HALLUCINATION (low confidence with no support at all).
#
# Four independent, self-contained DSP witnesses, corroborating or
# contradicting each other rather than any one deciding alone:
#   - HarmonicSeriesAnalyzer: direct STFT partial-picking against the
#     note's own claimed fundamental (harmonics 1-8, cent-tolerance
#     bands tightening for higher harmonics, weighted toward the low
#     ones that matter most for pitch perception).
#   - ZeroCrossingAnalyzer: onset-vs-sustained ZCR pattern
#     distinguishes tonal (low, stable) from noise (high throughout)
#     from percussive (high at onset, settles fast).
#   - RetrogradeSymmetryAnalyzer: reverses the note's audio and
#     compares the amplitude envelope forward vs backward - a real
#     percussive attack looks structurally different played backwards
#     (asymmetric decay); a sustained tone looks similar either way.
#     Independent of spectral content entirely, so it corroborates or
#     contradicts the ZCR-based call without restating the same
#     evidence.
#   - SpectralInversionAnalyzer: runs Harmonic Product Spectrum on the
#     normal spectrum, then on the frequency-flipped spectrum. Real
#     harmonic energy stays comparatively strong even mirrored; noise
#     stays uniformly weak either way. A different mathematical test
#     than direct partial-picking, so agreement is a real second
#     opinion, not the same evidence restated.
# When independent witnesses disagree, the verdict downgrades to
# UNCERTAIN rather than picking a side.
#
# 6.0 adaptation: Symphony's version had real teeth (should_veto(),
# adjust_confidence() actively zeroing a note's confidence). Per this
# session's law (annotation, not mutation - same as MicroNotePurge,
# TieReconstruction, TroubleMap), this port never touches a Note. It
# returns a HarmonicAudit the Conductor writes as a harmonic_legitimacy
# Annotation; Scribe Engraver decides at export time whether to act on
# a NOISE/HALLUCINATION verdict (mirroring MicroNotePurge's
# PURGE_CANDIDATE export-time filter).
#
# Cut from the original, and why:
#   - FeatureBundleAnalyzer (a fallback path reading a precomputed
#     `harmonic_network` from Symphony's FeatureBundle): Jazz has no
#     harmonic-intelligence pass producing that evidence (same gap
#     already noted in quantization/micro_note_purge.py) - and unlike
#     Symphony's Scribe validation gate (which sometimes only has bare
#     NoteEvents), this always runs with real stem audio available, so
#     the direct-STFT path is always taken; there's nothing for a
#     fallback to fall back FROM.
#   - AudioContract/AcousticIntelligence.create_contract(): Symphony-
#     specific audio-access abstraction. Replaced with AudioEngine.
#     view().slice_ms() - the same pattern instrument_attribution/
#     fingerprint.py already uses for "get a note's raw audio window".
#   - Agent-framework plumbing, ConfidenceComponents forensic
#     breakdown: no consumer for it in Jazz; the Annotation's own
#     confidence float carries the graded signal instead.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa

from audio_engine import AudioEngine, AudioTrack
from core import Note, StemType

HARMONIC_LEGITIMACY_ANNOTATION_KIND = "harmonic_legitimacy"

SCHOENBERG_SAMPLE_RATE = 22050
SCHOENBERG_FFT_SIZE = 4096
SCHOENBERG_HOP_LENGTH = 512

# Partial detection
MIN_PARTIAL_AMPLITUDE_DB = -80.0
HARMONIC_SERIES_MIN_PARTIALS = 2       # Need at least 2 partials to confirm pitch
HARMONIC_SERIES_MATCH_THRESHOLD = 0.7  # 70% of partials must match
MAX_INHARMONICITY = 0.05               # 5% max deviation from perfect harmonics

# Tolerance bands (cents) - musical, not linear
POWER_PARTIAL_TOLERANCE_CENTS = 30.0   # Harmonics 1-3: strict
MID_PARTIAL_TOLERANCE_CENTS = 50.0     # Harmonics 4-6: moderate
HIGH_PARTIAL_TOLERANCE_CENTS = 70.0    # Harmonics 7+: loose (string stretch)

# Zero-crossing / spectral flatness
ZERO_CROSSING_MAX = 0.35               # Above this = noise/snare, not tonal
SPECTRAL_FLATNESS_TONAL_THRESHOLD = 0.3
SPECTRAL_FLATNESS_NOISE_THRESHOLD = 0.7

# Retrograde symmetry
RETROGRADE_PERCUSSION_THRESHOLD = 0.6  # similarity below this = asymmetric = percussive
RETROGRADE_TONAL_THRESHOLD = 0.85      # similarity above this = strongly symmetric = tonal

# Spectral inversion (HPS on normal + frequency-flipped spectrum)
HPS_NUM_HARMONICS = 4
SPECTRAL_INVERSION_WINDOW_MS = 50.0


class HarmonicVerdict(str, Enum):
    TONAL = "tonal"
    PERCUSSION = "percussion"
    UNCERTAIN = "uncertain"
    NOISE = "noise"
    HALLUCINATION = "hallucination"


@dataclass(frozen=True)
class PartialTrack:
    partial_number: int
    frequency_hz: float
    amplitude_db: float
    confidence: float


@dataclass(frozen=True)
class HarmonicSeries:
    fundamental_freq_hz: float
    fundamental_confidence: float
    partials: Tuple[PartialTrack, ...]
    missing_partials: Tuple[int, ...]
    inharmonicity: float


@dataclass(frozen=True)
class HarmonicAudit:
    """One note's harmonic-legitimacy verdict - never applied to the
    Note itself; the Conductor writes this as a harmonic_legitimacy
    Annotation."""
    note_id: str
    verdict: HarmonicVerdict
    reason: str
    confidence: float
    contested: bool
    zero_crossing_rate: float
    spectral_flatness: float
    partial_count: int = 0
    harmonic_match_confidence: float = 0.0


# =====================================================================
# Harmonic series analysis (direct STFT partial-picking)
# =====================================================================

def _compute_stft_magnitude(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    stft = librosa.stft(audio, n_fft=SCHOENBERG_FFT_SIZE, hop_length=SCHOENBERG_HOP_LENGTH, window="hann")
    magnitude = np.mean(np.abs(stft), axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=SCHOENBERG_FFT_SIZE)
    return magnitude, freqs


def _detect_partials(magnitude: np.ndarray, freqs: np.ndarray, fundamental_hz: float, sample_rate: int) -> List[PartialTrack]:
    """Detects harmonic partials 1-8 against `fundamental_hz`. Searches
    a small window around the nearest bin for the true local peak, then
    refines with parabolic interpolation - a raw nearest-bin frequency
    has up to +/-half-a-bin of quantization error (~35-70 cents at this
    FFT size/sample rate for the low harmonics that matter most), which
    alone would exceed POWER_PARTIAL_TOLERANCE_CENTS for a genuinely
    clean tone."""
    partials: List[PartialTrack] = []
    bin_width = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    for harmonic in range(1, 9):
        target_freq = fundamental_hz * harmonic
        if target_freq > sample_rate / 2:
            continue

        center_idx = int(np.argmin(np.abs(freqs - target_freq)))
        lo = max(0, center_idx - 2)
        hi = min(len(magnitude), center_idx + 3)
        idx = lo + int(np.argmax(magnitude[lo:hi]))
        amplitude = magnitude[idx]

        amplitude_db = 20 * np.log10(amplitude + 1e-8)
        if amplitude_db < MIN_PARTIAL_AMPLITUDE_DB:
            continue

        if 0 < idx < len(magnitude) - 1:
            left, right = magnitude[idx - 1], magnitude[idx + 1]
            denom = left - 2 * amplitude + right
            p = 0.5 * (left - right) / denom if abs(denom) > 1e-12 else 0.0
            p = float(np.clip(p, -1.0, 1.0))
        else:
            p = 0.0
        detected_freq = freqs[idx] + p * bin_width

        cent_deviation = 1200 * np.log2(detected_freq / target_freq) if target_freq > 0 else 0.0

        if harmonic <= 3:
            tolerance_cents, weight = POWER_PARTIAL_TOLERANCE_CENTS, 1.0
        elif harmonic <= 5:
            tolerance_cents, weight = MID_PARTIAL_TOLERANCE_CENTS, 0.7
        else:
            tolerance_cents, weight = HIGH_PARTIAL_TOLERANCE_CENTS, 0.4

        if abs(cent_deviation) <= tolerance_cents:
            deviation_ratio = abs(cent_deviation) / tolerance_cents
            confidence = max(0.0, min(1.0, (1.0 - deviation_ratio) * weight))
            partials.append(PartialTrack(
                partial_number=harmonic, frequency_hz=float(detected_freq),
                amplitude_db=float(amplitude_db), confidence=float(confidence),
            ))

    return partials


def _calculate_inharmonicity(partials: List[PartialTrack], fundamental_hz: float) -> float:
    if not partials:
        return 1.0
    total_deviation, total_weight = 0.0, 0.0
    for p in partials:
        expected = fundamental_hz * p.partial_number
        if expected > 0:
            deviation = abs(p.frequency_hz - expected) / expected
            weight = 1.0 / p.partial_number  # lower harmonics matter more for pitch perception
            total_deviation += deviation * weight
            total_weight += weight
    return total_deviation / total_weight if total_weight > 0 else 1.0


def analyze_harmonic_series(audio_slice: np.ndarray, fundamental_hz: float, sample_rate: int) -> Optional[HarmonicSeries]:
    if audio_slice is None or len(audio_slice) < sample_rate * 0.01:
        return None
    if fundamental_hz <= 0 or fundamental_hz > sample_rate / 2:
        return None

    magnitude, freqs = _compute_stft_magnitude(audio_slice, sample_rate)
    partials = _detect_partials(magnitude, freqs, fundamental_hz, sample_rate)
    if len(partials) < HARMONIC_SERIES_MIN_PARTIALS:
        return None

    inharmonicity = _calculate_inharmonicity(partials, fundamental_hz)
    expected_partials = list(range(1, 9))
    present_numbers = {p.partial_number for p in partials}
    missing = tuple(n for n in expected_partials if n not in present_numbers)
    match_ratio = len(partials) / len(expected_partials)

    confidence = match_ratio * (1.0 - min(inharmonicity, 0.1) / 0.1)
    confidence = max(0.0, min(1.0, confidence))

    return HarmonicSeries(
        fundamental_freq_hz=fundamental_hz, fundamental_confidence=confidence,
        partials=tuple(partials), missing_partials=missing, inharmonicity=inharmonicity,
    )


# =====================================================================
# Zero-crossing analysis
# =====================================================================

def analyze_zero_crossing(audio_slice: np.ndarray, sample_rate: int) -> Dict[str, float]:
    if audio_slice is None or len(audio_slice) < sample_rate * 0.005:
        return {"verdict": "insufficient", "noise_ratio": 0.5, "onset_zcr": 0.0}

    zcr = librosa.feature.zero_crossing_rate(audio_slice, hop_length=SCHOENBERG_HOP_LENGTH)[0]
    if len(zcr) < 3:
        return {"verdict": "insufficient", "noise_ratio": 0.5, "onset_zcr": 0.0}

    onset_zcr = float(zcr[0])
    sustained_zcr = float(np.mean(zcr[1:])) if len(zcr) > 1 else onset_zcr

    if onset_zcr > ZERO_CROSSING_MAX and sustained_zcr < ZERO_CROSSING_MAX * 0.6:
        verdict, noise_ratio = "percussive", 0.1
    elif onset_zcr > ZERO_CROSSING_MAX and sustained_zcr > ZERO_CROSSING_MAX:
        verdict, noise_ratio = "noise", 0.9
    elif onset_zcr < ZERO_CROSSING_MAX and sustained_zcr < ZERO_CROSSING_MAX:
        verdict, noise_ratio = "tonal", 0.0
    else:
        verdict, noise_ratio = "uncertain", 0.3

    return {"onset_zcr": onset_zcr, "sustained_zcr": sustained_zcr, "verdict": verdict, "noise_ratio": noise_ratio}


# =====================================================================
# Retrograde symmetry (literal transformation: reverse and compare)
# =====================================================================

def _envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    envelope = np.abs(audio)
    window_size = max(1, int(0.005 * sample_rate))
    if window_size > 1 and len(envelope) > window_size:
        kernel = np.ones(window_size) / window_size
        envelope = np.convolve(envelope, kernel, mode="same")
    return envelope


def _analyze_envelope(envelope: np.ndarray) -> Optional[Dict[str, float]]:
    if len(envelope) == 0:
        return None
    peak_idx = int(np.argmax(envelope))
    peak_val = float(envelope[peak_idx])
    peak_position = peak_idx / len(envelope)
    attack_slope = peak_val / (peak_idx + 1e-8) if peak_idx > 0 else 0.0
    decay_samples = len(envelope) - peak_idx
    decay_slope = peak_val / (decay_samples + 1e-8) if decay_samples > 0 else 0.0
    total_energy = float(np.sum(envelope))
    early_energy = float(np.sum(envelope[:len(envelope) // 2]))
    skewness = early_energy / (total_energy + 1e-8)
    return {"peak_position": peak_position, "attack_slope": attack_slope, "decay_slope": decay_slope, "skewness": skewness}


def _envelope_similarity(fwd: Dict[str, float], rev: Dict[str, float]) -> float:
    """Higher = more symmetric (tonal-like). Lower = more asymmetric (percussive-like)."""
    position_similarity = 1.0 - min(1.0, abs(fwd["peak_position"] - rev["peak_position"]) / 0.5)
    skew_similarity = 1.0 - min(1.0, abs(fwd["skewness"] - rev["skewness"]))
    ratio_fwd = fwd["attack_slope"] / (fwd["decay_slope"] + 1e-8)
    ratio_rev = rev["attack_slope"] / (rev["decay_slope"] + 1e-8)
    slope_similarity = 1.0 - min(1.0, abs(ratio_fwd - ratio_rev) / 10.0)
    return position_similarity * 0.4 + skew_similarity * 0.3 + slope_similarity * 0.3


def analyze_retrograde_symmetry(audio_slice: np.ndarray, sample_rate: int) -> Dict[str, object]:
    """Reverses the note's audio and compares the amplitude envelope
    forward vs backward - independent of spectral content entirely."""
    if audio_slice is None or len(audio_slice) < sample_rate * 0.02:
        return {"similarity": 0.5, "verdict_hint": "uncertain"}

    fwd_analysis = _analyze_envelope(_envelope(audio_slice, sample_rate))
    rev_analysis = _analyze_envelope(_envelope(audio_slice[::-1], sample_rate))
    if fwd_analysis is None or rev_analysis is None:
        return {"similarity": 0.5, "verdict_hint": "uncertain"}

    similarity = _envelope_similarity(fwd_analysis, rev_analysis)
    if similarity < RETROGRADE_PERCUSSION_THRESHOLD:
        hint = "percussive"
    elif similarity > RETROGRADE_TONAL_THRESHOLD:
        hint = "tonal"
    else:
        hint = "uncertain"
    return {"similarity": similarity, "verdict_hint": hint}


# =====================================================================
# Spectral inversion (literal transformation: flip spectrum, run HPS)
# =====================================================================

def harmonic_product_spectrum(spectrum: np.ndarray, num_harmonics: int = HPS_NUM_HARMONICS) -> Tuple[int, float]:
    """Multiplies downsampled copies of the spectrum together, so a
    true fundamental (whose harmonics all line up) produces a strong
    product peak even when the fundamental itself is weak."""
    if len(spectrum) == 0:
        return 0, 0.0
    hps = spectrum.astype(np.float64).copy()
    for h in range(2, num_harmonics + 1):
        downsampled = spectrum[::h]
        n = min(len(hps), len(downsampled))
        hps[:n] *= downsampled[:n]
    peak_idx = int(np.argmax(hps))
    return peak_idx, float(hps[peak_idx])


def analyze_spectral_inversion(audio_slice: np.ndarray, sample_rate: int) -> float:
    """Returns a harmonic_ratio in [0,1] - higher means more harmonic/
    tonal. Runs HPS on the normal spectrum, then LITERALLY flips the
    frequency axis and runs HPS again; a genuinely harmonic sound still
    produces a comparatively strong HPS peak even mirrored, because its
    energy really is concentrated in a small number of related bins -
    noise stays uniformly weak either way."""
    if audio_slice is None or len(audio_slice) < 256:
        return 0.5

    segment = audio_slice[:int(SPECTRAL_INVERSION_WINDOW_MS / 1000 * sample_rate)]
    if len(segment) < 256:
        segment = audio_slice
    segment = segment * np.hanning(len(segment))

    spectrum = np.abs(np.fft.rfft(segment, n=SCHOENBERG_FFT_SIZE))
    if len(spectrum) < 10 or np.max(spectrum) <= 0:
        return 0.5

    _, normal_strength = harmonic_product_spectrum(spectrum)
    _, inverted_strength = harmonic_product_spectrum(spectrum[::-1])

    max_possible = np.max(spectrum) * HPS_NUM_HARMONICS
    normal_ratio = normal_strength / max_possible
    inverted_ratio = inverted_strength / max_possible
    return float(max(0.0, min(1.0, (normal_ratio + inverted_ratio) / 2)))


# =====================================================================
# Spectral flatness
# =====================================================================

def analyze_spectral_flatness(audio_slice: np.ndarray, sample_rate: int) -> float:
    if audio_slice is None or len(audio_slice) < sample_rate * 0.01:
        return 0.5
    stft = librosa.stft(audio_slice, n_fft=SCHOENBERG_FFT_SIZE, hop_length=SCHOENBERG_HOP_LENGTH)
    flatness = librosa.feature.spectral_flatness(S=np.abs(stft))
    return float(np.mean(flatness))


# =====================================================================
# Verdict logic
# =====================================================================

def _determine_verdict(
        harmonic_series: Optional[HarmonicSeries],
        zcr_profile: Dict[str, object],
        retrograde_profile: Dict[str, object],
        spectral_inversion_ratio: float,
        is_drum_stem: bool,
        note_confidence: float,
) -> Tuple[HarmonicVerdict, str]:
    """Two independent witnesses agreeing raises confidence in a call;
    disagreeing downgrades it to UNCERTAIN rather than picking a side."""
    zcr_says_percussive = zcr_profile.get("verdict") == "percussive"
    retrograde_hint = retrograde_profile.get("verdict_hint", "uncertain")

    # Harmonic series checked BEFORE the ZCR onset/decay heuristic on
    # purpose: instruments with percussive attack transients (piano
    # hammers, plucked strings, mallets) routinely trip the ZCR
    # "percussive" verdict on their onset frame even though the note is
    # fully tonal - a clean, well-matched harmonic series is stronger,
    # less ambiguous evidence than an onset-only ZCR heuristic.
    if harmonic_series is not None:
        match_ok = harmonic_series.fundamental_confidence >= HARMONIC_SERIES_MATCH_THRESHOLD
        inharmonicity_ok = harmonic_series.inharmonicity < MAX_INHARMONICITY

        if match_ok and inharmonicity_ok:
            if spectral_inversion_ratio >= 0.4:
                return HarmonicVerdict.TONAL, (
                    f"Valid harmonic series ({len(harmonic_series.partials)} partials, "
                    f"confidence {harmonic_series.fundamental_confidence:.1%}), corroborated by "
                    f"spectral inversion HPS ratio {spectral_inversion_ratio:.2f}"
                )
            if zcr_says_percussive:
                return HarmonicVerdict.TONAL, (
                    f"Valid harmonic series ({len(harmonic_series.partials)} partials, "
                    f"confidence {harmonic_series.fundamental_confidence:.1%}) with percussive attack "
                    f"transient (ZCR) - tonal note with a hard onset, e.g. piano/plucked/mallet"
                )
            return HarmonicVerdict.UNCERTAIN, (
                f"Harmonic partials matched ({harmonic_series.fundamental_confidence:.1%}) but "
                f"spectral inversion HPS disagrees (ratio={spectral_inversion_ratio:.2f})"
            )
        if zcr_says_percussive:
            if retrograde_hint == "tonal":
                return HarmonicVerdict.UNCERTAIN, (
                    f"Witnesses disagree: ZCR suggests percussive but retrograde envelope is "
                    f"symmetric (similarity={retrograde_profile['similarity']:.2f})"
                )
            return HarmonicVerdict.PERCUSSION, (
                "Percussive attack (ZCR) with no strong harmonic series "
                f"(confidence {harmonic_series.fundamental_confidence:.1%}, "
                f"inharmonicity {harmonic_series.inharmonicity:.3f})"
            )
        if not inharmonicity_ok:
            return HarmonicVerdict.UNCERTAIN, f"High inharmonicity: {harmonic_series.inharmonicity:.3f}"
        return HarmonicVerdict.UNCERTAIN, f"Partial match only: {harmonic_series.fundamental_confidence:.1%}"

    # No harmonic series at all: ZCR onset/decay pattern is primary;
    # retrograde symmetry corroborates or contradicts it.
    if zcr_says_percussive:
        if retrograde_hint == "percussive":
            return HarmonicVerdict.PERCUSSION, (
                "Percussive attack confirmed by two independent witnesses "
                f"(ZCR decay pattern, retrograde asymmetry={retrograde_profile['similarity']:.2f})"
            )
        if retrograde_hint == "tonal":
            return HarmonicVerdict.UNCERTAIN, (
                f"Witnesses disagree: ZCR suggests percussive but retrograde envelope is "
                f"symmetric (similarity={retrograde_profile['similarity']:.2f})"
            )
        return HarmonicVerdict.PERCUSSION, "Percussive attack with tonal decay (ZCR pattern)"

    if is_drum_stem:
        return HarmonicVerdict.PERCUSSION, "Drum stem - harmonic analysis not applicable"

    if retrograde_hint == "percussive":
        return HarmonicVerdict.PERCUSSION, (
            f"No harmonic series, but retrograde envelope is asymmetric "
            f"(similarity={retrograde_profile['similarity']:.2f}) - likely an untagged percussive hit"
        )

    if note_confidence < 0.5:
        return HarmonicVerdict.HALLUCINATION, f"Low confidence ({note_confidence:.1%}) without harmonic support"

    return HarmonicVerdict.NOISE, "No harmonic series detected"


def _verdict_confidence(verdict: HarmonicVerdict, harmonic_series: Optional[HarmonicSeries]) -> float:
    if verdict == HarmonicVerdict.TONAL:
        return harmonic_series.fundamental_confidence if harmonic_series else 0.9
    if verdict == HarmonicVerdict.PERCUSSION:
        return 0.9
    if verdict == HarmonicVerdict.UNCERTAIN:
        return 0.5
    return 0.15  # NOISE, HALLUCINATION


# =====================================================================
# Main entry point
# =====================================================================

def audit_note(
        engine: AudioEngine,
        track: AudioTrack,
        note: Note,
        stem: StemType,
        sample_rate: int = SCHOENBERG_SAMPLE_RATE,
) -> HarmonicAudit:
    """Audits one note's harmonic legitimacy against its own stem's raw
    audio. Never touches the Note - returns a verdict the Conductor
    writes as a harmonic_legitimacy Annotation."""
    view = engine.view(track, sample_rate)
    start_idx = int(note.start_ms * sample_rate / 1000)
    end_idx = int(note.end_ms * sample_rate / 1000)
    audio_slice = view.samples[max(0, start_idx):end_idx]

    zcr_profile = analyze_zero_crossing(audio_slice, sample_rate)

    if zcr_profile.get("verdict") == "noise":
        return HarmonicAudit(
            note_id=note.id, verdict=HarmonicVerdict.NOISE,
            reason=f"Persistent noise: ZCR {zcr_profile.get('onset_zcr', 0):.3f} > {ZERO_CROSSING_MAX}",
            confidence=_verdict_confidence(HarmonicVerdict.NOISE, None),
            contested=False,
            zero_crossing_rate=zcr_profile.get("onset_zcr", 0.0),
            spectral_flatness=SPECTRAL_FLATNESS_NOISE_THRESHOLD,
        )

    retrograde_profile = analyze_retrograde_symmetry(audio_slice, sample_rate)

    fundamental_hz = note.fundamental_freq_hz
    if fundamental_hz is None:
        fundamental_hz = 440.0 * 2.0 ** ((note.pitch - 69) / 12.0)
    harmonic_series = analyze_harmonic_series(audio_slice, fundamental_hz, sample_rate)

    spectral_inversion_ratio = analyze_spectral_inversion(audio_slice, sample_rate)

    verdict, reason = _determine_verdict(
        harmonic_series, zcr_profile, retrograde_profile, spectral_inversion_ratio,
        is_drum_stem=(stem == StemType.DRUMS), note_confidence=note.confidence,
    )

    spectral_flatness = analyze_spectral_flatness(audio_slice, sample_rate)

    return HarmonicAudit(
        note_id=note.id, verdict=verdict, reason=reason,
        confidence=_verdict_confidence(verdict, harmonic_series),
        contested=(verdict == HarmonicVerdict.UNCERTAIN),
        zero_crossing_rate=zcr_profile.get("onset_zcr", 0.0),
        spectral_flatness=spectral_flatness,
        partial_count=len(harmonic_series.partials) if harmonic_series else 0,
        harmonic_match_confidence=harmonic_series.fundamental_confidence if harmonic_series else 0.0,
    )


__all__ = [
    "HarmonicVerdict",
    "PartialTrack",
    "HarmonicSeries",
    "HarmonicAudit",
    "analyze_harmonic_series",
    "analyze_zero_crossing",
    "analyze_retrograde_symmetry",
    "analyze_spectral_inversion",
    "analyze_spectral_flatness",
    "harmonic_product_spectrum",
    "audit_note",
    "HARMONIC_LEGITIMACY_ANNOTATION_KIND",
    "ZERO_CROSSING_MAX",
    "SPECTRAL_FLATNESS_TONAL_THRESHOLD",
    "SPECTRAL_FLATNESS_NOISE_THRESHOLD",
]
