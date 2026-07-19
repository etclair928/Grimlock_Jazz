# =================================================================
# MODULE: core/__init__.py
# DESCRIPTION: Public API surface for Grimlock 6.0's Core (bedrock) layer.
#
# LAW: import from `core`, not `core.submodule`, outside of core itself -
# one import surface, so a future rename/reshuffle of an internal module
# never touches call sites project-wide.
#
# This file only exports what actually exists. It grows as each part of
# the Core layer gets built (see GRIMLOCK_6.0_DESIGN_DECISIONS.md §7) -
# it is deliberately NOT a re-export of everything Symphony's old core/
# had, since a chunk of that surface was the duplicated-type disease this
# rebuild exists to remove (see §9). The three files that carried that
# disease - order_types.py, testimony.py, contracts.py - are archived at
# docs/archive/legacy_core/ for reference, not imported from here.
#
# VERSION: 6.0 (pre-alpha, Core layer only)
# =================================================================

from core.confidence import Confidence, clamp_confidence
from core.stem_types import StemType
from core.source_types import Provenance
from core.note_types import Note
from core.annotation_types import Annotation, AnnotationStore
from core.tempo_types import TempoMeter
from core.separation_types import Separation
from core.music_box import MusicBox, ForensicRecord
from core.musical_findings_map import MusicalFindingsMap

__all__ = [
    "Confidence",
    "clamp_confidence",
    "StemType",
    "Provenance",
    "Note",
    "Annotation",
    "AnnotationStore",
    "TempoMeter",
    "Separation",
    "MusicBox",
    "ForensicRecord",
    "MusicalFindingsMap",
]
