# =================================================================
# MODULE: model_registry/crepe_model.py
# Registers CREPE's keras model with the shared ModelRegistry. CREPE
# already keeps its own internal cache (a module-level dict in
# crepe.core keyed by model_capacity) - registering it here doesn't
# replace that, it gives the rest of 6.0 a uniform get/unload interface
# that also reaches into CREPE's own cache on unload, so a
# memory-pressure unload through the registry actually frees the
# weights instead of leaving them alive in crepe's private dict.
# =================================================================

from __future__ import annotations

from model_registry.registry import ModelRegistry, get_registry

CREPE_MODEL_KEY_PREFIX = "crepe:"


def _crepe_model_key(capacity: str) -> str:
    return f"{CREPE_MODEL_KEY_PREFIX}{capacity}"


def _load_crepe_model(capacity: str):
    import crepe.core
    return crepe.core.build_and_load_model(capacity)


def _unload_crepe_model(capacity: str) -> None:
    import crepe.core
    crepe.core.models[capacity] = None


def register_crepe(registry: ModelRegistry, capacity: str = "small") -> None:
    key = _crepe_model_key(capacity)
    registry.register(
        key,
        factory=lambda: _load_crepe_model(capacity),
        unloader=lambda: _unload_crepe_model(capacity),
    )


def get_crepe_model(capacity: str = "small"):
    """Forces CREPE's model for `capacity` to be loaded (and registered
    for unload) before crepe.predict() is called, so the registry - not
    crepe's own lazy-load-on-first-predict behavior - is what's actually
    holding the reference."""
    registry = get_registry()
    register_crepe(registry, capacity)
    return registry.get(_crepe_model_key(capacity))


__all__ = ["CREPE_MODEL_KEY_PREFIX", "register_crepe", "get_crepe_model"]
