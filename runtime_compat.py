# =================================================================
# MODULE: runtime_compat.py
# The ONE place runtime compatibility aliases live.
#
# Grimlock's stack spans libraries written for different numpy
# generations, and the band that satisfies all of them is narrow
# (see requirements.txt): numpy 1.26.x, TensorFlow 2.15 (Basic
# Pitch + CREPE), madmom 0.16.1, scipy >= 1.13. Two of those
# libraries call names their platform has since deleted:
#
#   - madmom 0.16.1 uses np.float / np.int / np.complex / np.bool /
#     np.object / np.str throughout (dtype construction in
#     madmom/io/__init__.py is only the first). numpy 1.24 removed
#     those aliases, so on numpy 1.26 `import madmom` raises
#     AttributeError before any witness exists - and the tempo,
#     downbeat, and onset witnesses die with it. Restoring the
#     names on the numpy module fixes every madmom file at once,
#     because they all look the names up at runtime. Patching one
#     dtype line would not be enough.
#   - Basic Pitch 0.3.0 calls scipy.signal.gaussian, which scipy
#     >= 1.13 moved to scipy.signal.windows. Pure re-export, same
#     function under its old name. (This block used to be copied
#     into pitch_engine/basic_pitch_engine.py and four tools/
#     scripts; the copies are gone - this module is the original.)
#   - Python >= 3.10 removed the ABCs from `collections`
#     (collections.abc since 3.3); madmom/processors.py still does
#     `from collections import MutableSequence`. On Python >= 3.10
#     this wall is hit BEFORE np.float, so the aliases restore the
#     ABC names too. (Found by testing on Python 3.12; the original
#     bug report only named the numpy layer.)
#
# Do NOT fork madmom (unmaintained; the alias is the whole
# compatibility surface), do NOT downgrade numpy to 1.23 for it
# (pulls TensorFlow and librosa apart), and do NOT upgrade to
# numpy 2 (the TF 2.15 wheel will not load).
#
# CALL IT AT PROCESS START. A failed import is cached by the
# interpreter: if madmom is imported before the aliases are
# installed, it stays broken until the process restarts. Current
# call sites: orchestration/conductor.py (module top, before any
# Grimlock import), rhythm_engine/__init__.py, pitch_engine/
# basic_pitch_engine.py, and the tools/ entry points - each calls
# it before anything can import madmom or Basic Pitch.
# Idempotent: every name is only set when missing.
# =================================================================

from __future__ import annotations

__all__ = ["install_runtime_aliases"]


def install_runtime_aliases() -> None:
    """Restore the names the platform deleted out from under the
    pinned libraries:

    - numpy 1.24 removed np.float / np.int / np.complex / np.bool /
      np.object / np.str; madmom 0.16.1 looks them up at runtime.
    - Python 3.10 removed the abstract base classes from
      `collections` (they live in collections.abc since 3.3);
      madmom/processors.py still does
      `from collections import MutableSequence`. This wall is hit
      BEFORE the numpy one on Python >= 3.10 - the numpy aliases
      alone do not get `import madmom` past madmom/audio.
    - scipy 1.13 moved scipy.signal.gaussian to
      scipy.signal.windows; Basic Pitch 0.3.0 still calls the old
      path (pure re-export)."""
    import numpy as np

    for name, val in (("float", float), ("int", int), ("complex", complex),
                      ("bool", bool), ("object", object), ("str", str)):
        if not hasattr(np, name):
            setattr(np, name, val)

    import collections
    import collections.abc
    for name in ("Callable", "Hashable", "Iterable", "Iterator",
                 "Mapping", "MutableMapping", "MutableSequence",
                 "MutableSet", "Sequence", "Set"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))

    import scipy.signal as sps
    if not hasattr(sps, "gaussian"):
        from scipy.signal.windows import gaussian
        sps.gaussian = gaussian
