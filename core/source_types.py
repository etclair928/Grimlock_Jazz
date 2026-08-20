# =================================================================
# MODULE: core/source_types.py
# The ONE canonical way to say "which algorithm/witness produced this."
#
# GRIMLOCK_6.0_DESIGN_DECISIONS.md §9: Symphony's SourceType conflated
# "which algorithm" (RHYTHM, DEMUCS, RITORNELLO, ...) with "which stem"
# (BASS_STEM, MASTER_STEM, ...) in one enum. Stem identity now lives only
# in stem_types.StemType. SourceType/Provenance never gets a stem member.
# =================================================================

from __future__ import annotations

from enum import Enum


class Provenance(str, Enum):
    """Which agent/algorithm produced a Note, TempoMeter, Annotation, etc.
    Answers "which witness said this," never "which stem this is about" -
    that's StemType's job, carried alongside on whatever record needs both
    (e.g. Note has both `stem` and `source` fields)."""

    # Pitch detection
    BASIC_PITCH = "basic_pitch"
    CREPE = "crepe"

    # Separation
    DEMUCS = "demucs"

    # Rhythm / tempo
    TEMPO_INTELLIGENCE = "tempo_intelligence"
    RHYTHM_ENGINE = "rhythm_engine"
    DRUM_INTELLIGENCE = "drum_intelligence"
    GROOVE_FIELD = "groove_field"
    PULSE_FIELD = "pulse_field"

    # Instrument attribution (timbre + voice-continuity, coupled - §5)
    TIMBRE_INTELLIGENCE = "timbre_intelligence"
    VOICE_CONTINUITY = "voice_continuity"

    # Presentation / quantization (annotation-only passes)
    TEMPORAL_LATTICE = "temporal_lattice"
    RITORNELLO = "ritornello"
    CONSOLIDATION = "consolidation"   # merges Basic Pitch's fragmented same-pitch runs (note-level)

    # Acoustic Witness (annotation-only passes, §7-adjacent - see
    # acoustic_witness/ module docstring)
    ANECHOIC_MA = "anechoic_ma"
    SCHOENBERG_MIRROR = "schoenberg_mirror"
    OCTAVE_STACK = "octave_stack"

    # Key Intelligence (annotation-only, §7-adjacent - see
    # key_intelligence/ module docstring)
    KEY_INTELLIGENCE = "key_intelligence"

    # Check - form/motif self-similarity layer (annotation-only; detects
    # repeated sections + motifs and flags where repeats disagree so the
    # same idea can be transcribed consistently - see check/ module docstring)
    CHECK = "check"

    # Scalar-contradiction referee (§7 Epistemic layer - not a note-level gate)
    CONSENSUS = "consensus"

    # User-supplied - a hard lock, never re-arbitrated (§2.7 "guided means guided")
    GUIDED = "guided"

    # Grimlock University - the pattern-study layer (GRIMLOCK_UNIVERSITY.md).
    # OFF by default; STUDY observes and logs; APPLY additionally lets the
    # NOTATION view honor what was found. Never mutates a Note in any mode.
    GRIMLOCK_UNIVERSITY = "grimlock_university"
