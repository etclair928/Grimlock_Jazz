# =================================================================
# MODULE: model_registry/basic_pitch_model.py
# Registers Basic Pitch's Model with the shared ModelRegistry. Basic
# Pitch's predict() accepts either a loaded Model object or a path/str
# for model_or_model_path - passing a path makes it reload and rebuild
# from disk on EVERY call, so callers must always go through
# get_basic_pitch_model() rather than passing ICASSP_2022_MODEL_PATH
# directly to predict().
# =================================================================

from __future__ import annotations

from model_registry.registry import ModelRegistry, get_registry

BASIC_PITCH_MODEL_KEY = "basic_pitch"


def _load_basic_pitch_model():
    from basic_pitch.inference import Model
    from basic_pitch import ICASSP_2022_MODEL_PATH
    return Model(ICASSP_2022_MODEL_PATH)


def register_basic_pitch(registry: ModelRegistry) -> None:
    registry.register(BASIC_PITCH_MODEL_KEY, _load_basic_pitch_model)


def get_basic_pitch_model():
    """Convenience accessor against the process-wide shared registry -
    registers the factory (idempotent, does not reload if already
    cached) so callers don't have to remember to call
    register_basic_pitch() themselves."""
    registry = get_registry()
    register_basic_pitch(registry)
    return registry.get(BASIC_PITCH_MODEL_KEY)


__all__ = ["BASIC_PITCH_MODEL_KEY", "register_basic_pitch", "get_basic_pitch_model"]
