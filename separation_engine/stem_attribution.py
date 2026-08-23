# =================================================================
# MODULE: separation_engine/stem_attribution.py
# WHICH INSTRUMENT PLAYED THIS NOTE - answered without letting separation
# decide that the note exists.
#
# WHY THIS EXISTS. Today separation runs first and each stem is transcribed
# independently, so separation decides WHAT NOTES EXIST and every leak becomes
# notes, multiplied by the stem count. Measured on Burden of Sentiment against
# Klangio:
#
#     candidate                          notes   prec  recall     F1
#     raw Basic Pitch, no separation      1602  0.202   0.236  0.218
#     our 'other' stem alone              1977  0.168   0.243  0.199
#     our full six-stem pipeline          3425  0.147   0.369  0.211
#
# The whole pipeline adds 1823 notes and does not beat feeding the raw mix to
# Basic Pitch. Read honestly, separation genuinely helps RECALL (0.369 vs
# 0.236) - it surfaces quiet notes buried in the mix - and badly hurts
# precision.
#
# THE ASYMMETRY THAT MOTIVATES THE FIX. A stage that GENERATES can invent; a
# stage that DESCRIBES can only mislabel. Separation is fallible either way,
# so it belongs where its errors are cheap. Detect once on the mix, then use
# the stems only to say which instrument each detected note belongs to: a leak
# then costs a wrong NAME, never a phantom NOTE.
#
# HOW. A note at pitch p sounding over [t0, t1) has its energy concentrated at
# f0 = 440 * 2^((p-69)/12) and at that frequency's partials. Measure that
# energy in each stem over that window and the stem holding most of it is the
# one carrying the note. This is a per-note soft vote, which is strictly more
# information than a hard split: a piano note whose energy sits 60% in `other`
# and 30% in `vocals` becomes ONE note labelled piano with a confidence, where
# today it becomes two notes in two stems.
#
# WHAT IT IS NOT. It is not a fix for separation quality, and it cannot
# recover a note no detector found on the mix. It moves separation from
# deciding existence to describing identity, and that is all.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ATTRIBUTION_SAMPLE_RATE = 22050

# Partials measured per note. The fundamental alone is too easily masked - a
# piano's second partial often carries more energy than its first - and going
# far up the series starts sampling other instruments' fundamentals.
PARTIALS = (1, 2, 3, 4)

# Half-width of the band summed around each partial, in semitones. Wide enough
# to survive vibrato and tuning drift, narrow enough not to swallow a
# neighbouring semitone.
BAND_SEMITONES = 0.6

# A note shorter than this has too few STFT frames to measure honestly.
MIN_WINDOW_MS = 40.0

# Below this share of the winning stem's energy, the runner-up is not a real
# rival and the attribution is called clean.
CONTESTED_RATIO = 0.6


@dataclass(frozen=True)
class Attribution:
    """Which stem carries a note, and how clearly."""
    note_id: str
    stem: str
    confidence: float                    # winner's share of total energy
    contested_by: Optional[str]          # runner-up, when it is close
    energy_by_stem: Dict[str, float]

    @property
    def is_contested(self) -> bool:
        return self.contested_by is not None


def _midi_to_hz(pitch: float) -> float:
    return 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))


def _band_energy(spec: np.ndarray, freqs: np.ndarray, centre_hz: float,
                 half_semitones: float) -> float:
    """Energy in a band around `centre_hz`, in semitone-proportional width."""
    if centre_hz <= 0:
        return 0.0
    lo = centre_hz * (2.0 ** (-half_semitones / 12.0))
    hi = centre_hz * (2.0 ** (half_semitones / 12.0))
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    return float(spec[mask, :].sum())


class StemAttributor:
    """Holds one STFT per stem and answers per-note questions against them.

    The spectrogram is computed ONCE per stem and reused for every note - a
    note-at-a-time STFT would be the dominant cost and would recompute the
    same frames thousands of times.
    """

    def __init__(self, stem_audio: Dict[str, np.ndarray],
                 sample_rate: int = ATTRIBUTION_SAMPLE_RATE,
                 n_fft: int = 4096, hop_length: int = 512):
        import librosa
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        self.spec: Dict[str, np.ndarray] = {}
        for name, samples in stem_audio.items():
            y = np.asarray(samples, dtype=np.float32)
            if y.ndim > 1:
                y = y.mean(axis=0)
            self.spec[name] = np.abs(librosa.stft(y, n_fft=n_fft,
                                                  hop_length=hop_length))

    def _frames(self, start_ms: float, end_ms: float) -> Tuple[int, int]:
        per_frame_ms = 1000.0 * self.hop_length / self.sample_rate
        a = max(0, int(start_ms / per_frame_ms))
        b = max(a + 1, int(end_ms / per_frame_ms))
        return a, b

    def attribute(self, note) -> Optional[Attribution]:
        """Which stem holds this note's energy? None when unmeasurable."""
        start_ms = float(note.start_ms)
        end_ms = float(note.end_ms)
        if end_ms - start_ms < MIN_WINDOW_MS:
            end_ms = start_ms + MIN_WINDOW_MS
        a, b = self._frames(start_ms, end_ms)

        f0 = _midi_to_hz(note.pitch)
        energy: Dict[str, float] = {}
        for name, spec in self.spec.items():
            if a >= spec.shape[1]:
                continue
            window = spec[:, a:min(b, spec.shape[1])]
            if window.size == 0:
                continue
            total = 0.0
            for k in PARTIALS:
                total += _band_energy(window, self.freqs, f0 * k, BAND_SEMITONES)
            energy[name] = total

        if not energy:
            return None
        grand = sum(energy.values())
        if grand <= 0:
            return None

        ranked = sorted(energy.items(), key=lambda kv: -kv[1])
        winner, top = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        contested = (runner_up[0] if runner_up and top > 0
                     and runner_up[1] / top >= CONTESTED_RATIO else None)
        return Attribution(
            note_id=getattr(note, "id", ""), stem=winner,
            confidence=top / grand, contested_by=contested,
            energy_by_stem=energy,
        )


def attribute_notes(notes: Sequence, stem_audio: Dict[str, np.ndarray],
                    sample_rate: int = ATTRIBUTION_SAMPLE_RATE
                    ) -> List[Attribution]:
    """Label every note with the stem carrying most of its energy."""
    if not notes or not stem_audio:
        return []
    attributor = StemAttributor(stem_audio, sample_rate)
    out: List[Attribution] = []
    for note in notes:
        result = attributor.attribute(note)
        if result is not None:
            out.append(result)
    return out


__all__ = ["Attribution", "StemAttributor", "attribute_notes",
           "ATTRIBUTION_SAMPLE_RATE", "PARTIALS", "BAND_SEMITONES",
           "CONTESTED_RATIO"]
