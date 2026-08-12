# =================================================================
# MODULE: university/__init__.py
# Grimlock University - the pattern-study layer (GRIMLOCK_UNIVERSITY.md).
#
# A university studies, publishes and is peer-reviewed; it does not run
# the factory floor. OFF (default) = Grimlock Jazz exactly as it is.
# STUDY = observe + log, output unchanged. APPLY = the page additionally
# honors what was found. Frozen Notes are never mutated in any mode.
# =================================================================

from university.observation_types import (
    PATTERN_STUDY_ANNOTATION_KIND, PAGE_SUPPRESS_ANNOTATION_KIND,
    VOICE_COHESION_ANNOTATION_KIND, PatternObservation, StudyReport,
    UniversityMode,
)
from university.study import study, write_corpus
from university.apply import (
    apply_observations, page_suppression_map, write_study_annotations,
)
from university.null_model import run_null_model

__all__ = [
    "UniversityMode", "PatternObservation", "StudyReport",
    "study", "write_corpus",
    "write_study_annotations", "apply_observations", "page_suppression_map",
    "run_null_model",
    "PATTERN_STUDY_ANNOTATION_KIND", "PAGE_SUPPRESS_ANNOTATION_KIND",
    "VOICE_COHESION_ANNOTATION_KIND",
]
