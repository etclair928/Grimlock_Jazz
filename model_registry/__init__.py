# =================================================================
# MODULE: model_registry/__init__.py
# Public API surface for Grimlock 6.0's Model Registry (GRIMLOCK_6.0_
# DESIGN_DECISIONS.md §7). One home for every model wrapper's lifetime -
# lazy-load, cache, unload under memory pressure.
# =================================================================

from model_registry.registry import ModelRegistry, get_registry
from model_registry.basic_pitch_model import BASIC_PITCH_MODEL_KEY, get_basic_pitch_model, register_basic_pitch
from model_registry.crepe_model import CREPE_MODEL_KEY_PREFIX, get_crepe_model, register_crepe
from model_registry.demucs_model import DEMUCS_MODEL_KEY_PREFIX, get_demucs_model, register_demucs, unload_demucs

__all__ = [
    "ModelRegistry",
    "get_registry",
    "BASIC_PITCH_MODEL_KEY",
    "get_basic_pitch_model",
    "register_basic_pitch",
    "CREPE_MODEL_KEY_PREFIX",
    "get_crepe_model",
    "register_crepe",
    "DEMUCS_MODEL_KEY_PREFIX",
    "get_demucs_model",
    "register_demucs",
    "unload_demucs",
]
