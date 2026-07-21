# =================================================================
# MODULE: rhythm_engine/__init__.py
# Public API surface for Grimlock 6.0's Rhythm Engine (GRIMLOCK_6.0_
# DESIGN_DECISIONS.md §7): tempo/meter/groove analysis + drum
# detection. Produces up to FOUR independent tempo witnesses (librosa,
# madmom, note-onset-pattern, lattice) rather than resolving to one -
# reconciling them is epistemic/referee.py's job (§7 Epistemic:
# scalar-contradiction referee), not this module's. Each audio-only
# witness is also octave-corrected here (octave_correction.py) before
# it ever reaches the referee.
# =================================================================

from rhythm_engine.onsets import OnsetCandidates, detect_onset_candidates, refine_onset_to_nearest
from rhythm_engine.tempo_witness import TempoWitness, run_librosa_tempo, run_madmom_tempo
from rhythm_engine.note_onset_tempo_witness import run_note_onset_tempo_witness
from rhythm_engine.lattice_witness import Anchor, GeometricLattice, run_lattice_witness
from rhythm_engine.pulse_field import PulseFieldResult, run_pulse_field
from rhythm_engine.groove import estimate_groove
from rhythm_engine.groove_field import GrooveType, PhaseDeltaResult, compute_phase_deltas
from rhythm_engine.meter import (
    build_phase_locked_grid, estimate_time_signature, sample_beat_accents,
    fft_meter_candidate, resolve_denominator, PickupResult, detect_pickup,
)
from rhythm_engine.tempo_drift import DriftKind, TempoDriftResult, classify_drift, local_tempo_curve
from rhythm_engine.drums import detect_drums, GM_PITCH

__all__ = [
    "OnsetCandidates",
    "detect_onset_candidates",
    "refine_onset_to_nearest",
    "TempoWitness",
    "run_librosa_tempo",
    "run_madmom_tempo",
    "run_note_onset_tempo_witness",
    "Anchor",
    "GeometricLattice",
    "run_lattice_witness",
    "PulseFieldResult",
    "run_pulse_field",
    "estimate_groove",
    "GrooveType",
    "PhaseDeltaResult",
    "compute_phase_deltas",
    "build_phase_locked_grid",
    "estimate_time_signature",
    "sample_beat_accents",
    "fft_meter_candidate",
    "resolve_denominator",
    "PickupResult",
    "detect_pickup",
    "DriftKind",
    "TempoDriftResult",
    "classify_drift",
    "local_tempo_curve",
    "detect_drums",
    "GM_PITCH",
]
