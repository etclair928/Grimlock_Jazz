# =================================================================
# MODULE: core/stem_types.py
# The ONE canonical way to say "which audio stem did this come from."
#
# GRIMLOCK_6.0_DESIGN_DECISIONS.md §9: Symphony had no real stem field on a
# note at all - stem identity was reconstructed by grepping a free-text
# reasoning_chain for a substring like "stem:bass" (seven separate call
# sites did this), while its SourceType enum ALSO carried BASS_STEM/
# MASTER_STEM members that overlapped with a separate StemType enum's own
# BASS/OTHER/VOCALS. Three ways to say the same thing, and the "real" one
# wasn't even a typed field - just a parsed string.
#
# 6.0 rule: stem identity lives ONLY here, as a real typed field on Note
# (see note_types.py). SourceType (source_types.py) never gets a stem
# member again - it answers a different question (which algorithm).
# =================================================================

from __future__ import annotations

from enum import Enum


class StemType(str, Enum):
    """Which isolated audio stem a note/measurement came from.

    Kept minimal and evidence-driven: only members a real separator in
    this project actually produces. Add a member only when a real
    separator ships one - don't speculatively pre-declare stems nothing
    produces yet (that's how StemType and SourceType drifted apart with
    overlapping members in the first place).
    """
    FULL_MIX = "full_mix"      # unseparated master audio
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    OTHER = "other"            # Demucs 4-stem catch-all
    GUITAR = "guitar"          # htdemucs_6s only
    PIANO = "piano"            # htdemucs_6s only
    BRASS = "brass"            # external 7-stem separators only
