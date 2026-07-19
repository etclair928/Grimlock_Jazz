# =================================================================
# MODULE: pitch_engine/__init__.py
# Public API surface for Grimlock 6.0's Pitch Engine (GRIMLOCK_6.0_
# DESIGN_DECISIONS.md §3 / §7). Basic Pitch per stem is the
# transcription; CREPE is a bass-only specialist read alongside it.
# No council, no voting, no median-blend, no mid-stream refinement.
# =================================================================

from pitch_engine.basic_pitch_engine import (
    transcribe as transcribe_basic_pitch,
    transcribe_with_posteriorgram as transcribe_basic_pitch_with_posteriorgram,
    BasicPitchPosteriorgram,
    BASIC_PITCH_SAMPLE_RATE,
)
from pitch_engine.crepe_bass import transcribe_bass as transcribe_crepe_bass

__all__ = [
    "transcribe_basic_pitch", "transcribe_basic_pitch_with_posteriorgram",
    "BasicPitchPosteriorgram", "BASIC_PITCH_SAMPLE_RATE", "transcribe_crepe_bass",
]
