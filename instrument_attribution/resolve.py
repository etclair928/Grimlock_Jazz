# =================================================================
# MODULE: instrument_attribution/resolve.py
# Step 3 of "fingerprint -> stream -> resolve": the coupling step.
# A voice LINE (voice_continuity.py) is the unit that gets ONE
# instrument identity, written back to every note in that line as an
# Annotation - never a mutation of the Note itself.
#
# WHAT CHANGED (overtone model replaces centroid tertiles)
# The old classifier split a full-spectrum centroid into tertiles.
# That number moves with pitch, register and whoever else is sounding,
# so it is not timbre. Now fingerprint.py measures partials by
# heterodyning, pool_fingerprints() median-pools them over the line,
# and _classify_line() walks a fixed decision ladder:
#
#   1. A clean pre-merge stem vote (stem_attribution.py) wins. Overtones
#      never overrule it.
#   2. Struck string: stiffness B clearly > 0, harmonic centroid falling,
#      no vibrato. Large B = piano. Milder B + pick transient = guitar.
#      Anything in between -> contested.
#   3. B ~ 0 and deep vibrato -> "vibrato_lead" (voice, sax or strings).
#      Telling those apart needs a formant-stability measure that
#      fingerprint.py does not compute yet, so this stays coarse and
#      contested on non-vocal stems.
#   4. B ~ 0 and even partials suppressed -> clarinet_like.
#   5. B ~ 0, bright and stable envelope, no/shallow vibrato -> brass.
#      No trumpet vs trombone: that is mostly register.
#   6. Otherwise a coarse bucket, contested=True.
#
# The VOCALS stem used to be stamped "vocals" at confidence 1.0 for every
# note. Now the line is measured, and if it positively looks like a struck
# string, brass or clarinet, the notes stay "vocals" but are marked
# contested (a leaked horn is flagged, not silently trusted).
#
# All numeric thresholds below are STARTING GUESSES from acoustics, not
# tuned on your audio. Tune them against your labeled songs.
# =================================================================

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from audio_engine import AudioEngine, AudioTrack
from core import AnnotationStore, Annotation, Note, Provenance, StemType
from instrument_attribution.fingerprint import (
    LineTimbre,
    fingerprint_notes,
    pool_fingerprints,
)
from instrument_attribution.voice_continuity import VoiceLine, stream_into_lines

ANNOTATION_KIND = "instrument_family"
VOICE_ANNOTATION_KIND = "voice"

# Stems Separation Engine already resolved identity for at the audio
# layer - Instrument Attribution trusts that, it doesn't re-derive it.
# (VOCALS is special-cased below: it gets a guard, not blind trust.)
STEM_FAMILY: Dict[StemType, str] = {
    StemType.BASS: "bass",
    StemType.VOCALS: "vocals",
    StemType.GUITAR: "guitar",
    StemType.PIANO: "piano",
    StemType.DRUMS: "drums",
}

# Pre-merge stem vote: note_id -> (family, confidence, contested).
# Produced by the conductor from stem_attribution.py BEFORE the stems are summed.
StemVotes = Dict[str, Tuple[str, float, bool]]

# ---------------- thresholds (tune these) ----------------
MIN_LINE_NOTES = 3          # fewer usable notes than this = not enough evidence
B_ZERO = 1e-5               # B below this counts as "no stiffness" (winds, voice, bowed)
B_PIANO = 1.5e-4            # B at/above this on the low notes = piano
CENTROID_FALLING = -0.3     # partials/sec; below this the high partials are dying first
CENTROID_STABLE = 0.3       # abs slope below this = held, stable envelope
VIB_NONE = 6.0              # cents (peak); below this = no vibrato
VIB_DEEP = 20.0             # cents (peak); above this = deep vibrato
EVEN_ODD_CLARINET = 0.4     # RMS even/odd below this = closed pipe
ATTACK_PICK = 0.15          # attack-noise ratio at/above this = pick/hammer-like transient
BRIGHT_DB = -10.0           # median of partials 2..8 above this (dB re partial 1) = bright
FALLBACK_FAMILY = "mid_body"  # neutral bucket the engraver already knows how to render


def _finite(x: float) -> bool:
    return x is not None and np.isfinite(x)


def _bright(timbre: LineTimbre) -> bool:
    vals = [v for v in timbre.partial_log_ratios[:7] if np.isfinite(v)]  # k = 2..8
    return bool(vals) and float(np.median(vals)) > BRIGHT_DB


def _no_vibrato(timbre: LineTimbre) -> bool:
    v = timbre.vibrato_depth_cents
    return (not _finite(v)) or v < VIB_NONE


def _classify_line(timbre: Optional[LineTimbre]) -> Tuple[str, float, bool, bool]:
    """Walk the ladder (steps 2-6). Returns (family, confidence, contested, decided).
    decided=False means the evidence was missing or the ladder found nothing."""
    if timbre is None or timbre.n_notes < MIN_LINE_NOTES:
        return FALLBACK_FAMILY, 0.3, True, False

    b = timbre.inharmonicity_b
    slope = timbre.harmonic_centroid_slope
    vib = timbre.vibrato_depth_cents
    stiff_known = _finite(b)

    # Step 2: struck string
    if stiff_known and b > B_ZERO and _finite(slope) and slope < CENTROID_FALLING and _no_vibrato(timbre):
        if b >= B_PIANO:
            return "piano", 0.8, False, True
        if _finite(timbre.attack_noise_ratio) and timbre.attack_noise_ratio >= ATTACK_PICK:
            return "guitar", 0.7, False, True
        return FALLBACK_FAMILY, 0.4, True, True  # struck, but piano vs guitar is unclear

    # Steps 3-5 need B ~ 0
    if stiff_known and b <= B_ZERO:
        # Step 3: deep vibrato
        if _finite(vib) and vib >= VIB_DEEP:
            return "vibrato_lead", 0.6, True, True
        # Step 4: clarinet-like
        if _finite(timbre.even_odd_ratio) and timbre.even_odd_ratio < EVEN_ODD_CLARINET:
            return "clarinet_like", 0.65, False, True
        # Step 5: brass
        if _bright(timbre) and _finite(slope) and abs(slope) < CENTROID_STABLE and (
                not _finite(vib) or vib < VIB_DEEP):
            return "brass", 0.65, False, True

    # Step 6: nothing fit
    return FALLBACK_FAMILY, 0.35, True, False


def resolve_instrument_identity(
        engine: AudioEngine,
        track: AudioTrack,
        stem: StemType,
        notes: List[Note],
        annotations: AnnotationStore,
        stem_votes: Optional[StemVotes] = None,
) -> List[VoiceLine]:
    """Streams `notes` into voice lines, resolves ONE instrument family
    per line, and writes it as an Annotation on every note in that line.
    Returns the lines.

    `stem_votes` is optional (note_id -> (family, confidence, contested))
    from the pre-merge stem attribution. Without it, only the overtone
    ladder runs."""
    lines = stream_into_lines(notes)

    # Persist voice membership (line_id) for every note, before any
    # family branch, so the engraver can put voices on
