# =================================================================
# MODULE: rhythm_engine/note_onset_tempo_witness.py
# Ports Symphony's NoteOnsetPatternTempoWitness (agents/analysis/
# tempo_intelligence.py) - a tempo estimate derived from Basic Pitch's
# own detected note onsets, not raw audio. Every other tempo witness in
# this package is audio-only; none of them ever consult what the pitch
# detector actually found. This is the most direct embodiment of "don't
# fight the models" in the whole rhythm stack: it derives tempo
# agreement FROM Basic Pitch's own output rather than an independent
# audio-only guess that might contradict it.
#
# Symphony's own documented motivation: on a real quiet, kick-less song,
# the note pattern independently implied ~193-204 BPM while every
# audio-only witness converged instead on ~107-117 BPM - a real,
# unreconciled split this witness exists to surface.
# =================================================================

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.stats import gaussian_kde

from core import Note, Provenance
from rhythm_engine.tempo_witness import TempoWitness

MIN_TEMPO_BPM = 20.0
MAX_TEMPO_BPM = 300.0
DEDUP_WINDOW_MS = 30.0     # collapses chord/octave-doubled onsets into one event
MIN_IOI_MS = 30.0
MAX_IOI_MS = 2000.0
HISTOGRAM_BIN_WIDTH_MS = 15.0


def _find_modal_ioi(iois: np.ndarray) -> Tuple[float, float]:
    """Returns (modal_interval_ms, concentration) - the tallest peak of a
    KDE over the IOI distribution, not a fixed-width histogram bin.

    A fixed-width histogram is sensitive to where its bin edges happen to
    land: two real, distinct clusters can each straddle a bin boundary
    and get individually undercounted, letting a weaker but
    better-aligned-to-the-bins cluster win the argmax instead. Confirmed
    directly on real audio: a 15ms fixed-bin histogram picked a cluster
    with roughly a fifth the real point density of the true strongest
    cluster, purely from bin-edge placement - not a genuine ambiguity in
    the data. A KDE has no edges to be sensitive to.
    """
    try:
        kde = gaussian_kde(iois)
    except np.linalg.LinAlgError:
        # All (or nearly all) IOIs identical - no real spread for a KDE;
        # the data itself already says exactly what the modal value is.
        peak_x = float(np.median(iois))
        concentration = float(np.mean(np.abs(iois - peak_x) < HISTOGRAM_BIN_WIDTH_MS))
        return peak_x, concentration

    x_range = np.linspace(float(iois.min()), float(iois.max()), 400)
    densities = kde(x_range)
    peaks_idx = np.where((densities[1:-1] > densities[:-2]) & (densities[1:-1] > densities[2:]))[0]

    if len(peaks_idx) == 0:
        peak_x = float(x_range[np.argmax(densities)])
    else:
        peak_positions = x_range[peaks_idx + 1]
        peak_heights = densities[peaks_idx + 1]
        peak_x = float(peak_positions[np.argmax(peak_heights)])

    concentration = float(np.mean(np.abs(iois - peak_x) < HISTOGRAM_BIN_WIDTH_MS))
    return peak_x, concentration


def run_note_onset_tempo_witness(notes: List[Note]) -> TempoWitness:
    """Derives a tempo estimate from the modal inter-onset-interval of
    `notes`' own start times (typically all pitched notes across
    stems). Returns confidence 0.0 (not a claim, just "no evidence")
    when there aren't enough notes to say anything real."""
    onset_times_ms = sorted(n.start_ms for n in notes)
    if len(onset_times_ms) < 8:
        return _null_witness()

    collapsed = [onset_times_ms[0]]
    for t in onset_times_ms[1:]:
        if t - collapsed[-1] > DEDUP_WINDOW_MS:
            collapsed.append(t)
    if len(collapsed) < 8:
        return _null_witness()

    iois = np.diff(collapsed)
    iois = iois[(iois > MIN_IOI_MS) & (iois < MAX_IOI_MS)]
    if len(iois) < 6:
        return _null_witness()

    modal_interval_ms, concentration = _find_modal_ioi(iois)

    # Guard the octave-folding loops below against a degenerate interval.
    # The IOI filter above already bounds values to (MIN_IOI_MS, MAX_IOI_MS)
    # so this cannot fire in practice - but a non-finite or zero interval
    # would make tempo_bpm 0 or NaN, and `while tempo_bpm < MIN` would then
    # never terminate. Cheap insurance on a loop that cannot otherwise exit.
    if not np.isfinite(modal_interval_ms) or modal_interval_ms <= 0:
        return _null_witness()

    tempo_bpm = 60000.0 / modal_interval_ms
    if not np.isfinite(tempo_bpm) or tempo_bpm <= 0:
        return _null_witness()

    while tempo_bpm < MIN_TEMPO_BPM:
        tempo_bpm *= 2.0
    while tempo_bpm > MAX_TEMPO_BPM:
        tempo_bpm /= 2.0

    confidence = min(0.85, 0.3 + concentration * 1.2)
    return TempoWitness(
        tempo_bpm=tempo_bpm,
        confidence=confidence,
        beat_times_ms=(),  # a modal-interval estimate, not a phase-locked grid
        source=Provenance.TEMPO_INTELLIGENCE,
        weight=0.7,
    )


def _null_witness() -> TempoWitness:
    return TempoWitness(tempo_bpm=0.0, confidence=0.0, beat_times_ms=(), source=Provenance.TEMPO_INTELLIGENCE, weight=0.7)


__all__ = ["run_note_onset_tempo_witness"]
