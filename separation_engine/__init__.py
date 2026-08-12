# =================================================================
# MODULE: separation_engine/__init__.py
# Public API surface for Grimlock 6.0's Separation Engine (GRIMLOCK_6.0_
# DESIGN_DECISIONS.md §4 / §7). 6-stem Demucs to pull guitar and piano
# out of the "other" junk-drawer at the audio layer.
# =================================================================

from separation_engine.demucs_engine import separate
from separation_engine.stem_merge import merge_harmonic_stems, load_cached_separation, HARMONIC_STEMS

__all__ = ["separate", "merge_harmonic_stems", "load_cached_separation", "HARMONIC_STEMS"]
