# =================================================================
# MODULE: key_intelligence/__init__.py
# Public API surface for Grimlock 6.0's Key Intelligence layer. Not
# named in GRIMLOCK_6.0_DESIGN_DECISIONS.md's original §7 layer list -
# added per explicit user request, closing the gap MusicalFindingsMap's
# own docstring called out ("key/harmonic detection... a future layer
# this build doesn't include yet"). Ports Symphony's KeyDetector
# (agents/detection/harmonic_intelligence.py) only - see
# key_detector.py's module docstring for what was deliberately left
# out (HarmonicValidator, redundant with acoustic_witness/
# schoenberg_mirror.py) and why.
# =================================================================

from key_intelligence.key_detector import (
    KeyResult, detect_key, analyze_key, key_fit,
    KEY_FIT_ANNOTATION_KIND, KEY_CONFIDENCE_THRESHOLD,
)

__all__ = [
    "KeyResult",
    "detect_key",
    "analyze_key",
    "key_fit",
    "KEY_FIT_ANNOTATION_KIND",
    "KEY_CONFIDENCE_THRESHOLD",
]
