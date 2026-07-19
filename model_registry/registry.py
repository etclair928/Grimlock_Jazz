# =================================================================
# MODULE: model_registry/registry.py
# Model Registry (GRIMLOCK_6.0_DESIGN_DECISIONS.md §7): one home for
# every model wrapper's lifetime. Lazy-load, cache across calls, unload
# under memory pressure - "most real memory management actually lives
# here" per the design doc's Memory section (§7 Memory: the registry is
# one of the three object owners the memory POLICY is distributed to,
# alongside Audio Engine's decoded audio and the transform LRU cache).
#
# This is deliberately a thin, generic mechanism - one dict of factories,
# one dict of loaded instances, one lock per key so loading model A never
# blocks a concurrent request for model B. Model-specific load/unload
# behavior (Basic Pitch's Model class, CREPE's own internal cache dict)
# lives in each model's own registration module, not here.
# =================================================================

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional


class ModelRegistry:
    """Owns model lifetime. Nothing outside this class should hold a
    long-lived reference to a loaded model - callers ask for it by key
    each time they need it; the registry decides whether that means a
    fresh load or a cache hit."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._unloaders: Dict[str, Callable[[], None]] = {}
        self._instances: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def register(
            self,
            key: str,
            factory: Callable[[], Any],
            unloader: Optional[Callable[[], None]] = None,
    ) -> None:
        """Declares how to load (and optionally how to specially unload)
        the model at `key`. Registering again with the same key replaces
        the factory but does not touch an already-loaded instance."""
        with self._registry_lock:
            self._factories[key] = factory
            if unloader is not None:
                self._unloaders[key] = unloader
            if key not in self._locks:
                self._locks[key] = threading.Lock()

    def get(self, key: str) -> Any:
        """Lazily loads `key` exactly once, caches it, returns it on every
        subsequent call until `unload`/`unload_all` drops it."""
        if key in self._instances:
            return self._instances[key]

        if key not in self._factories:
            raise KeyError(f"ModelRegistry: no factory registered for '{key}'")

        lock = self._locks[key]
        with lock:
            if key not in self._instances:
                self._instances[key] = self._factories[key]()
        return self._instances[key]

    def is_loaded(self, key: str) -> bool:
        return key in self._instances

    def loaded_keys(self) -> List[str]:
        return list(self._instances.keys())

    def unload(self, key: str) -> None:
        """Drops the cached instance (and runs any model-specific
        unloader, e.g. clearing a library's own internal cache) so the
        weights become eligible for garbage collection. The next `get()`
        for this key reloads from scratch."""
        if key not in self._instances:
            return
        lock = self._locks.get(key)
        if lock is not None:
            with lock:
                self._instances.pop(key, None)
                unloader = self._unloaders.get(key)
                if unloader is not None:
                    unloader()
        else:
            self._instances.pop(key, None)

    def unload_all(self) -> None:
        for key in list(self._instances.keys()):
            self.unload(key)


_GLOBAL_REGISTRY: Optional[ModelRegistry] = None
_GLOBAL_REGISTRY_LOCK = threading.Lock()


def get_registry() -> ModelRegistry:
    """The process-wide shared registry. Every model wrapper (Pitch
    Engine, Separation Engine, ...) asks for models through this same
    instance, so a memory-pressure unload of one model is visible to
    every caller, not just the one that happened to load it."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _GLOBAL_REGISTRY_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = ModelRegistry()
    return _GLOBAL_REGISTRY


__all__ = ["ModelRegistry", "get_registry"]
