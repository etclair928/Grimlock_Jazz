# =================================================================
# MODULE: orchestration/__init__.py
# Public API surface for Grimlock 6.0's Orchestration Conductor
# (GRIMLOCK_6.0_DESIGN_DECISIONS.md §7): the linear, deterministic
# pipeline that ties every other layer together.
# =================================================================

from orchestration.conductor import transcribe_file, PipelineResult, DEFAULT_SEPARATION_MODEL

__all__ = ["transcribe_file", "PipelineResult", "DEFAULT_SEPARATION_MODEL"]
