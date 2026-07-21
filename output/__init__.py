# =================================================================
# MODULE: output/__init__.py
# Public API surface for Grimlock 6.0's Output layer (GRIMLOCK_6.0_
# DESIGN_DECISIONS.md §7): the Scribe Engraver, the single Compositor/
# commit step that turns Notes + Annotations into a MIDI file.
# =================================================================

from output.scribe_engraver import engrave

__all__ = ["engrave"]
