# =================================================================
# MODULE: epistemic/__init__.py
# Public API surface for Grimlock 6.0's Epistemic layer (GRIMLOCK_6.0_
# DESIGN_DECISIONS.md §7): a scalar-contradiction referee, scoped down
# from 5.x's gate-everything EpistemicCouncil. Music_Box and
# MusicalFindingsMap are ambient Core services (core/music_box.py,
# core/musical_findings_map.py) - NOT part of this arbiter, despite
# living in the same conceptual "epistemic" territory in 5.x.
# =================================================================

from epistemic.referee import TempoResolution, MeterResolution, resolve_tempo, resolve_meter, arbitrate_tempo_octave

__all__ = ["TempoResolution", "MeterResolution", "resolve_tempo", "resolve_meter", "arbitrate_tempo_octave"]
