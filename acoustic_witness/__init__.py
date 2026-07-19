# =================================================================
# MODULE: acoustic_witness/__init__.py
# Public API surface for Grimlock 6.0's Acoustic Witness layer. Not
# named in GRIMLOCK_6.0_DESIGN_DECISIONS.md's original §7 layer list -
# added per explicit user request to bring Symphony's AnechoicMa
# (silence/resonance/activity evidence) and SchoenbergMirror (harmonic-
# legitimacy audit) into 6.0, adapted to be pure evidence producers:
# neither module mutates a Note or hard-vetoes anything. The Conductor
# writes their output as Annotations; Scribe Engraver decides at
# export time whether to act on them (opt-in, same law as every other
# pass in this codebase).
# =================================================================

from acoustic_witness.anechoic_ma import (
    AnechoicReport, SilenceState, SilenceRegion, RegionType, analyze_stem,
    ACOUSTIC_ACTIVITY_ANNOTATION_KIND,
)
from acoustic_witness.schoenberg_mirror import (
    HarmonicVerdict, HarmonicAudit, HarmonicSeries, PartialTrack, audit_note,
    HARMONIC_LEGITIMACY_ANNOTATION_KIND,
)
from acoustic_witness.note_support import (
    SupportVerdict, evaluate_stem_support, NOTE_SUPPORT_ANNOTATION_KIND,
    NOTE_SUPPORT_SAMPLE_RATE, SUPPORTED, UNSUPPORTED,
)

__all__ = [
    "AnechoicReport",
    "SilenceState",
    "SilenceRegion",
    "RegionType",
    "analyze_stem",
    "ACOUSTIC_ACTIVITY_ANNOTATION_KIND",
    "HarmonicVerdict",
    "HarmonicAudit",
    "HarmonicSeries",
    "PartialTrack",
    "audit_note",
    "HARMONIC_LEGITIMACY_ANNOTATION_KIND",
    "SupportVerdict",
    "evaluate_stem_support",
    "NOTE_SUPPORT_ANNOTATION_KIND",
    "NOTE_SUPPORT_SAMPLE_RATE",
    "SUPPORTED",
    "UNSUPPORTED",
]
