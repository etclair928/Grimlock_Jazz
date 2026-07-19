# =================================================================
# MODULE: acoustic_witness/note_support.py
# Over-detection / bleed / hallucination filter, built ENTIRELY from a
# 3-song, 3468-note diagnostic (not from intuition). The diagnostic
# measured which per-note signals actually separate spurious notes
# (~16.5% of all notes have near-silent energy at their own fundamental
# in their own stem) from real ones, and the result was blunt:
#   - harmonic legitimacy (SchoenbergMirror) does NOT discriminate -
#     bleed/doubled notes are genuinely tonal (0.87 vs 0.91 match). This
#     is exactly why the existing harmonic-verdict drop filter never
#     touched them.
#   - Basic Pitch's own confidence barely discriminates (0.45 vs 0.49).
#   - The two signals that DO: (1) own-stem narrowband energy at the
#     note's own fundamental - a note whose own pitch is near-silent in
#     its own stem's audio is a bleed/hallucination suspect (this is the
#     defining signal, ~16.5% of notes); (2) AnechoicMa's
#     active_material_probability - suspects average 0.165 vs 0.316 for
#     real notes, a clean 2:1 separation (and the best of the existing
#     annotated-but-unconsumed witnesses).
#
# THE PRECISION GUARD (why this ANDs the two signals): each signal alone
# has overlapping distributions - a threshold on either one drops some
# real notes. Requiring BOTH (weak own-energy AND low activity) is what
# gives precision: on the diagnostic it flagged 6.6% of notes, cleanly
# separated from what it keeps (active 0.047 vs 0.309, own/other energy
# ratio 0.01 vs 43). A note that's energy-weak but acoustically active
# (a real quiet passage) is PROTECTED; a note that's low-activity but
# has real own-pitch energy (a sustained soft note) is PROTECTED. Only
# the intersection - no own-pitch energy AND no acoustic activity - is
# flagged. This is annotation, not deletion: a `note_support` verdict
# the Scribe Engraver may optionally act on at export (§2.2), same as
# every other pass this session.
#
# HONEST CEILINGS (measured, not hand-waved): (a) 6.6% is a real dent,
# not a transformation - clean notes still leave dense polyphony dense;
# (b) the distributions overlap, so the thresholds are a tunable
# precision/recall trade, not a magic cleaner; (c) much of what this
# catches is Demucs bleed, a symptom of imperfect separation this only
# masks post-hoc. Thresholds are the diagnostic's measured operating
# point, revisit-able against a fresh diagnostic - not guessed.
# =================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from core import Note

NOTE_SUPPORT_ANNOTATION_KIND = "note_support"
NOTE_SUPPORT_SAMPLE_RATE = 22050

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"

# Measured operating point from the 3-song diagnostic. A note's own-f0
# energy below this fraction of its stem's median = "weak own-energy";
# active_material below this = "acoustically inactive". Both must hold
# to flag (the precision guard).
WEAK_ENERGY_FRACTION = 0.15
LOW_ACTIVITY_THRESHOLD = 0.15
_BAND_WINDOW = 2048


@dataclass(frozen=True)
class SupportVerdict:
    note_id: str
    verdict: str           # SUPPORTED or UNSUPPORTED
    own_energy: float
    energy_ratio_to_median: float
    active_material: float
    reason: str


def _f0_hz(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def _own_band_energy(samples: np.ndarray, sample_rate: int, onset_ms: float, fund_hz: float) -> float:
    """Narrowband amplitude at the note's own fundamental (+2nd
    harmonic), +/-1 semitone, in a window at the note's onset. Same
    computation the diagnostic validated on - a note whose own f0 is
    near-silent here is playing a pitch that isn't really in its own
    stem's audio."""
    start = int(onset_ms * sample_rate / 1000)
    frame = samples[start:start + _BAND_WINDOW]
    if len(frame) < 256:
        return 0.0
    w = np.hanning(len(frame))
    spec = np.abs(np.fft.rfft(frame * w))
    freqs = np.fft.rfftfreq(len(frame), 1.0 / sample_rate)
    e = 0.0
    for h in (1, 2):
        target = fund_hz * h
        lo, hi = target * 2 ** (-1 / 12), target * 2 ** (1 / 12)
        mask = (freqs >= lo) & (freqs <= hi)
        if np.any(mask):
            e += float(np.sum(spec[mask] ** 2))
    return math.sqrt(e / len(frame))


def evaluate_stem_support(
        notes: Sequence[Note],
        stem_samples: np.ndarray,
        sample_rate: int,
        active_by_note_id: Dict[str, float],
) -> Dict[str, SupportVerdict]:
    """One stem's notes -> per-note support verdict. Own-energy is
    normalized to THIS stem's own median (a loud stem and a quiet stem
    have different absolute band energies; the suspect signal is being
    weak RELATIVE to the stem's own typical note). Requires both weak
    own-energy AND low acoustic activity to flag UNSUPPORTED - either
    one alone leaves the note SUPPORTED (the precision guard)."""
    if not notes or stem_samples is None or len(stem_samples) == 0:
        return {}

    energies = {n.id: _own_band_energy(stem_samples, sample_rate, n.start_ms, _f0_hz(n.pitch)) for n in notes}
    nonzero = [e for e in energies.values() if e > 0]
    median = float(np.median(nonzero)) if nonzero else 1e-9
    weak_threshold = WEAK_ENERGY_FRACTION * median

    verdicts: Dict[str, SupportVerdict] = {}
    for n in notes:
        e_own = energies[n.id]
        active = float(active_by_note_id.get(n.id, 1.0))  # unknown activity -> treat as active (supported)
        weak = e_own < weak_threshold
        inactive = active < LOW_ACTIVITY_THRESHOLD
        unsupported = weak and inactive
        ratio = e_own / (median + 1e-9)
        if unsupported:
            reason = (f"own-f0 energy {ratio:.2f}x stem median (weak) AND active_material "
                      f"{active:.2f} (inactive) - bleed/hallucination candidate")
        else:
            why = []
            if not weak:
                why.append(f"own-f0 energy {ratio:.2f}x median (present)")
            if not inactive:
                why.append(f"active_material {active:.2f} (active)")
            reason = "supported: " + ", ".join(why)
        verdicts[n.id] = SupportVerdict(
            note_id=n.id, verdict=UNSUPPORTED if unsupported else SUPPORTED,
            own_energy=e_own, energy_ratio_to_median=ratio, active_material=active, reason=reason,
        )
    return verdicts


__all__ = [
    "SupportVerdict",
    "evaluate_stem_support",
    "NOTE_SUPPORT_ANNOTATION_KIND",
    "NOTE_SUPPORT_SAMPLE_RATE",
    "SUPPORTED",
    "UNSUPPORTED",
]
