# =================================================================
# MODULE: instrument_attribution/resolve.py
# Step 3 of "fingerprint -> stream -> resolve" (GRIMLOCK_6.0_DESIGN_
# DECISIONS.md §5): the coupling step. Per-note timbre votes are noisy;
# a voice LINE (voice_continuity.py) is the unit that gets ONE resolved
# instrument identity, written back to every note in that line as an
# Annotation - never a mutation of the Note itself.
#
# Separation Engine (with htdemucs_6s) already pulled bass/vocals/
# guitar/piano out as their OWN stems at the audio layer (§4) - for
# those, "which instrument is this" is already answered by which stem
# the note came from; this module's job there is just to run voice
# continuity (so one instrument's phrase doesn't fragment into several
# line identities) and stamp the already-known family.
#
# The OTHER stem is Demucs's genuine residual catch-all (synths,
# strings, horns - whatever htdemucs_6s doesn't split further) - THAT
# is where real per-note fingerprinting earns its keep, classifying
# each resolved line into a small set of deliberately coarse timbre
# buckets. This is a first working classifier, not a tuned rule engine
# like Symphony's 5.x TimbreIntelligence (dozens of iterations against
# real audio) - the bucket names describe acoustic character
# (bright/warm/percussive), not instrument names this hasn't earned
# the right to claim precisely.
#
# Band boundaries are data-driven per stem, not fixed constants. Two
# hardcoded Hz cutoffs (1200/2500) badly skewed real audio - on a 60s
# Hopeful.mp3 clip, 575 of the "other" stem's 587 notes landed in the
# single "warm_sustained" bucket because almost every line's mean
# centroid happened to fall under 1200Hz. A fixed cutoff also breaks
# in the other direction for a piece the training-by-eye guess didn't
# anticipate (an all-bright or all-warm "other" stem would collapse to
# one bucket regardless of what boundary was picked). Boundaries are
# now the tertile split of THIS stem's own observed per-line centroids
# - the same "let the real data set the threshold" discipline meter.py
# already uses for its significance test, applied here to timbre
# instead of rhythm. This does NOT touch voice_continuity.py - lines
# are still built the same way; only how a resolved line gets LABELED
# changes. It also does not attempt genuine multi-feature timbre
# discrimination (see fingerprint.py's unused bandwidth/ZCR/MFCC
# fields) - that's a separate, bigger step, deliberately deferred.
# =================================================================

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from audio_engine import AudioEngine, AudioTrack
from core import AnnotationStore, Annotation, Note, Provenance, StemType
from instrument_attribution.fingerprint import fingerprint_notes
from instrument_attribution.voice_continuity import VoiceLine, stream_into_lines

ANNOTATION_KIND = "instrument_family"

# Stems Separation Engine already resolved identity for at the audio
# layer - Instrument Attribution trusts that, it doesn't re-derive it.
STEM_FAMILY: Dict[StemType, str] = {
    StemType.BASS: "bass",
    StemType.VOCALS: "vocals",
    StemType.GUITAR: "guitar",
    StemType.PIANO: "piano",
    StemType.DRUMS: "drums",
}

# Fallback boundaries when there isn't enough data in this stem to
# trust a percentile split (see _compute_band_boundaries) - the same
# guessed values the fixed-constant version always used.
_DEFAULT_LOW_CUTOFF_HZ = 1200.0
_DEFAULT_HIGH_CUTOFF_HZ = 2500.0
_MIN_LINES_FOR_ADAPTIVE_BANDS = 6   # need enough lines for a tertile split to mean anything
_MIN_BAND_SEPARATION_HZ = 50.0      # below this the distribution is effectively flat - don't invent a boundary
_LOW_PERCENTILE = 100.0 / 3.0
_HIGH_PERCENTILE = 200.0 / 3.0


def _compute_band_boundaries(line_centroids_hz: List[float]) -> Tuple[float, float]:
    """Tertile split of this stem's own observed per-line centroids -
    bottom third -> warm_sustained, middle third -> mid_body, top third
    -> bright_lead. Falls back to the fixed defaults when there are too
    few lines to trust a percentile split, or when the split would be
    degenerate (a near-solo "other" stem whose lines all share nearly
    the same centroid - a tertile of a flat distribution isn't a real
    boundary, it's noise)."""
    if len(line_centroids_hz) < _MIN_LINES_FOR_ADAPTIVE_BANDS:
        return _DEFAULT_LOW_CUTOFF_HZ, _DEFAULT_HIGH_CUTOFF_HZ

    low_cutoff = float(np.percentile(line_centroids_hz, _LOW_PERCENTILE))
    high_cutoff = float(np.percentile(line_centroids_hz, _HIGH_PERCENTILE))

    if high_cutoff - low_cutoff < _MIN_BAND_SEPARATION_HZ:
        return _DEFAULT_LOW_CUTOFF_HZ, _DEFAULT_HIGH_CUTOFF_HZ

    return low_cutoff, high_cutoff


def _classify_other(
        mean_centroid_hz: float, low_cutoff_hz: float, high_cutoff_hz: float,
        band_margin_hz: float = 300.0,
) -> Tuple[str, float]:
    """Coarse band classification against this stem's OWN computed
    boundaries, with a confidence that drops near a band boundary
    (where the call is genuinely uncertain) rather than always
    reporting flat confidence regardless of margin."""
    bands: List[Tuple[float, str]] = [
        (low_cutoff_hz, "warm_sustained"),
        (high_cutoff_hz, "mid_body"),
        (float("inf"), "bright_lead"),
    ]
    for boundary, family in bands:
        if mean_centroid_hz < boundary:
            distance_to_boundary = boundary - mean_centroid_hz if boundary != float("inf") else band_margin_hz * 2
            confidence = float(np.clip(0.5 + 0.5 * min(distance_to_boundary, band_margin_hz) / band_margin_hz, 0.5, 0.95))
            return family, confidence
    return "bright_lead", 0.5


def resolve_instrument_identity(
        engine: AudioEngine,
        track: AudioTrack,
        stem: StemType,
        notes: List[Note],
        annotations: AnnotationStore,
) -> List[VoiceLine]:
    """Streams `notes` into voice lines, resolves ONE instrument family
    per line, and writes it as an Annotation on every note in that
    line. Returns the lines (useful for callers that also want the
    line grouping itself, e.g. Rhythm Engine's phrase-level reasoning)."""
    lines = stream_into_lines(notes)

    known_family = STEM_FAMILY.get(stem)
    if known_family is not None:
        for line in lines:
            for note in line.notes:
                annotations.add(Annotation(
                    note_id=note.id,
                    kind=ANNOTATION_KIND,
                    value=known_family,
                    source=Provenance.TIMBRE_INTELLIGENCE,
                    confidence=1.0,
                    contested=False,
                ))
        return lines

    fingerprints = fingerprint_notes(engine, track, notes)

    # First pass: each line's own mean centroid, so the boundaries below
    # can be computed from THIS stem's real distribution before any line
    # gets labeled.
    line_centroids: Dict[str, float] = {}
    for line in lines:
        line_fps = [fingerprints[n.id] for n in line.notes if n.id in fingerprints]
        if not line_fps:
            continue
        line_centroids[line.line_id] = float(np.mean([fp.spectral_centroid_hz for fp in line_fps]))

    low_cutoff, high_cutoff = _compute_band_boundaries(list(line_centroids.values()))

    for line in lines:
        mean_centroid = line_centroids.get(line.line_id)
        if mean_centroid is None:
            continue
        family, confidence = _classify_other(mean_centroid, low_cutoff, high_cutoff)
        contested = confidence < 0.6
        for note in line.notes:
            annotations.add(Annotation(
                note_id=note.id,
                kind=ANNOTATION_KIND,
                value=family,
                source=Provenance.TIMBRE_INTELLIGENCE,
                confidence=confidence,
                contested=contested,
            ))

    return lines


__all__ = ["resolve_instrument_identity", "STEM_FAMILY", "ANNOTATION_KIND"]
