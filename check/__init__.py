# =================================================================
# PACKAGE: check
# The form / self-similarity layer - the top of the perception ladder.
# Finds where the music repeats an idea (SECTIONS and MOTIFS) and
# measures how consistently each repeat was transcribed, so the same
# idea can be written the same way. Annotation-only; never mutates a
# Note. See check/check.py for the full rationale.
# =================================================================

from check.structure import Section, Motif, detect_form, find_motifs
from check.check import (
    run_check, CheckResult,
    SECTION_KIND, REPEAT_GROUP_KIND, MOTIF_KIND,
)

__all__ = [
    "run_check", "CheckResult",
    "Section", "Motif", "detect_form", "find_motifs",
    "SECTION_KIND", "REPEAT_GROUP_KIND", "MOTIF_KIND",
]
