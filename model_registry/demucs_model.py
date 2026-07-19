# =================================================================
# MODULE: model_registry/demucs_model.py
# Registers a Demucs model (htdemucs / htdemucs_6s / ...) with the
# shared ModelRegistry. Loading is real work (network fetch on first
# use, GPU/CPU placement, eval mode) - exactly the kind of lifetime
# Model Registry exists to own instead of Separation Engine keeping its
# own private singleton.
# =================================================================

from __future__ import annotations

from model_registry.registry import ModelRegistry, get_registry

DEMUCS_MODEL_KEY_PREFIX = "demucs:"


def _demucs_model_key(model_name: str, device: str) -> str:
    return f"{DEMUCS_MODEL_KEY_PREFIX}{model_name}@{device}"


def _load_demucs_model(model_name: str, device: str):
    import torch
    from demucs import pretrained

    model = pretrained.get_model(model_name)
    model.to(torch.device(device))
    model.eval()
    return model


def register_demucs(registry: ModelRegistry, model_name: str = "htdemucs_6s", device: str = "cpu") -> None:
    key = _demucs_model_key(model_name, device)
    registry.register(key, factory=lambda: _load_demucs_model(model_name, device))


def get_demucs_model(model_name: str = "htdemucs_6s", device: str = "cpu"):
    registry = get_registry()
    register_demucs(registry, model_name, device)
    return registry.get(_demucs_model_key(model_name, device))


def unload_demucs(model_name: str = "htdemucs_6s", device: str = "cpu") -> None:
    """Drops the cached Demucs model so its weights (typically the
    single largest resident model in the whole pipeline) become
    eligible for garbage collection once Separation Engine is done with
    it - callers that only separate once per run gain nothing from
    keeping it loaded through Pitch/Rhythm Engine's later stages."""
    get_registry().unload(_demucs_model_key(model_name, device))


__all__ = ["DEMUCS_MODEL_KEY_PREFIX", "register_demucs", "get_demucs_model", "unload_demucs"]
