# =================================================================
# MODULE: instrument_attribution/__init__.py
# Public API surface for Grimlock 6.0's Instrument Attribution
# (GRIMLOCK_6.0_DESIGN_DECISIONS.md §5 / §7): timbre + voice-continuity,
# coupled. Fingerprint -> stream -> resolve. Annotation-only; the line
# is the unit of instrument identity.
# =================================================================

from instrument_attribution.fingerprint import TimbreFingerprint, fingerprint_notes
from instrument_attribution.voice_continuity import VoiceLine, stream_into_lines
from instrument_attribution.resolve import resolve_instrument_identity, STEM_FAMILY, VOICE_ANNOTATION_KIND

__all__ = [
    "TimbreFingerprint",
    "fingerprint_notes",
    "VoiceLine",
    "stream_into_lines",
    "resolve_instrument_identity",
    "STEM_FAMILY",
    "VOICE_ANNOTATION_KIND",
]
