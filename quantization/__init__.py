# =================================================================
# MODULE: quantization/__init__.py
# Public API surface for Grimlock 6.0's Quantization layer. Not named
# in GRIMLOCK_6.0_DESIGN_DECISIONS.md's original §7 layer list - added
# per explicit user request to bring Symphony's TemporalLattice/
# QuaverIntelligence timing logic into 6.0, adapted to write
# Annotations rather than mutate Notes (§2's frozen-detection-floor law
# holds by construction here, not by convention).
# =================================================================

from quantization.lattice_judge import TemporalLattice, QuantizationMode, build_lattice, QUANTIZATION_ANNOTATION_KIND
from quantization.duration_witness import DurationHypothesis, DurationTestimony, infer_duration
from quantization.sustain_recovery import SustainExtension, propose_sustain_extensions, SUSTAIN_RECOVERY_ANNOTATION_KIND
from quantization.onset_refinement import (
    OnsetRefinement, refine_note_onsets, ONSET_REFINEMENT_ANNOTATION_KIND,
)
from quantization.pitch_wobble_collapse import (
    WobbleGroup, find_wobble_groups, PITCH_WOBBLE_ANNOTATION_KIND,
)
from quantization.micro_note_purge import (
    LegitimacyVerdict, evaluate_legitimacy,
    MICRO_NOTE_PURGE_ANNOTATION_KIND, PURGE_CANDIDATE, LEGITIMATE,
)
from quantization.tie_reconstruction import (
    TieCandidate, find_tie_candidates, TIE_RECONSTRUCTION_ANNOTATION_KIND,
)
from quantization.trouble_map import (
    TroubleMeasure, build_trouble_map,
    TROUBLE_MAP_MIN_CONFIDENCE, TROUBLE_MAP_MAX_FRAGMENTATION_NOTES_PER_BEAT,
)
from quantization.notation_quantizer import (
    NotationTiming, notation_quantize_note, NOTATION_TIMING_ANNOTATION_KIND,
)
from quantization.rhythm_inference import (
    BeatFilling, BeatRhythm, InferredNoteTiming, infer_beat, infer_voice_rhythm,
)
from quantization.note_consolidation import (
    consolidate_fragments, CONSOLIDATION_ANNOTATION_KIND, DEFAULT_MERGE_GAP_MS,
)

__all__ = [
    "TemporalLattice",
    "QuantizationMode",
    "build_lattice",
    "QUANTIZATION_ANNOTATION_KIND",
    "DurationHypothesis",
    "DurationTestimony",
    "infer_duration",
    "SustainExtension",
    "propose_sustain_extensions",
    "SUSTAIN_RECOVERY_ANNOTATION_KIND",
    "OnsetRefinement",
    "refine_note_onsets",
    "ONSET_REFINEMENT_ANNOTATION_KIND",
    "WobbleGroup",
    "find_wobble_groups",
    "PITCH_WOBBLE_ANNOTATION_KIND",
    "LegitimacyVerdict",
    "evaluate_legitimacy",
    "MICRO_NOTE_PURGE_ANNOTATION_KIND",
    "PURGE_CANDIDATE",
    "LEGITIMATE",
    "TieCandidate",
    "find_tie_candidates",
    "TIE_RECONSTRUCTION_ANNOTATION_KIND",
    "TroubleMeasure",
    "build_trouble_map",
    "TROUBLE_MAP_MIN_CONFIDENCE",
    "TROUBLE_MAP_MAX_FRAGMENTATION_NOTES_PER_BEAT",
    "NotationTiming",
    "notation_quantize_note",
    "NOTATION_TIMING_ANNOTATION_KIND",
    "BeatFilling",
    "BeatRhythm",
    "InferredNoteTiming",
    "infer_beat",
    "infer_voice_rhythm",
    "consolidate_fragments",
    "CONSOLIDATION_ANNOTATION_KIND",
    "DEFAULT_MERGE_GAP_MS",
]
