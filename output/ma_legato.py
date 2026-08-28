# =================================================================
# MODULE: output/ma_legato.py
# ANECHOIC MA AS A GATE ON EVERY REST THE PAGE WRITES.
#
# THE PROBLEM IT EXISTS FOR. A rest is a claim that nobody is playing. Basic
# Pitch's note ends are cut-offs, not releases, so the page filled the leftover
# of each beat with rests nobody performs: on Clocks, 2767 rests against the
# published transcription's 1037, and 1347 sixteenth rests against its ZERO.
#
# THE EVIDENCE WAS ALREADY BEING MEASURED. acoustic_witness/anechoic_ma.py
# queries the gap AFTER every note and reports whether the stem is still
# ringing there (resonance) or genuinely silent (rhythmic void). It runs on
# every pitched note of every run - 100% coverage on bass, vocals and the
# harmonic stems - and only output/piano_reduction.py ever read it, for the
# piano staff alone. Every other part wrote its rests blind.
#
# WHY PERCENTILES AND NOT THRESHOLDS - a lesson this project already paid for
# once, recorded at piano_reduction.py:100. `resonance_probability` reads as a
# probability and is not one: measured over 11,126 gaps it occupies roughly
# [0.24, 0.62], so the natural-looking gate `resonance > 0.60` sits above its
# 99th percentile and fires on ~0.1% of gaps. `trailing_void` bottoms out at
# 0.45, so `void > 0.65` passes almost everything. Both absolute gates were
# inert. Confirmed again on Clocks: resonance [0.233, 0.722], void [0.477,
# 0.855]. The scale also drifts between songs, so no constant can mean the same
# thing twice. Percentiles of the material's own distribution can.
#
# THE POLICY IS THREE-WAY, and deliberately so. Evidence MODULATES the ordinary
# fill window, it never silently replaces it:
#
#   clearly ringing   -> reach further; the gap is decay, not silence
#   clearly silent    -> blocked; the rest is real and must stay
#   anything between  -> the ordinary window, exactly as before
#
# The middle case is what keeps this safe. A witness that is only sometimes
# informative must not be allowed to decide on its own when it is not.
#
# WHAT IT WILL NOT DO. It never crosses a barline and never runs a note into
# the next attack in its own voice - it closes the gap up to that attack and no
# further. Lengthening past either is how a readable page becomes a wrong one,
# and this module can only ever close a gap that already exists.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Calibrated in output/piano_reduction.py against 11,126 trailing-gap
# observations and re-confirmed on Clocks. Shared, not re-derived, so the two
# consumers cannot drift apart.
RINGING_PERCENTILE = 25.0    # at/above this percentile of resonance = sounding
VOID_PERCENTILE = 10.0       # at/below this percentile of void = a real rest
RINGING_REACH = 2.0          # a ringing gap may be closed this much further
MIN_SAMPLES = 30             # below this, percentiles are noise - stay blind


@dataclass(frozen=True)
class MaGate:
    """A resonance/void gate calibrated against one score's own distribution."""
    ring_at: Optional[float] = None
    void_at: Optional[float] = None

    @property
    def blind(self) -> bool:
        return self.ring_at is None

    @classmethod
    def calibrate(cls, resonance: Sequence[float],
                  void: Sequence[float]) -> "MaGate":
        res = [r for r in resonance if r is not None]
        vd = [v for v in void if v is not None]
        if len(res) < MIN_SAMPLES:
            return cls()          # not enough evidence to say anything
        return cls(
            ring_at=float(np.percentile(res, RINGING_PERCENTILE)),
            void_at=(float(np.percentile(vd, 100.0 - VOID_PERCENTILE))
                     if len(vd) >= MIN_SAMPLES else None),
        )

    def limit(self, resonance: Optional[float], void: Optional[float],
              base: float) -> float:
        """How far this particular gap may be closed.

        `base` is the ordinary window the caller would have used with no
        evidence at all, and is what comes back whenever the witness has
        nothing confident to say - which is the point of the middle case.
        """
        if self.blind or resonance is None:
            return base
        if self.void_at is not None and void is not None and void >= self.void_at:
            return 0.0                       # the audio says: genuine silence
        if resonance >= self.ring_at:
            return base * RINGING_REACH      # still sounding: reach further
        return base


def calibrate_from_notes(notes: Sequence) -> MaGate:
    """Calibrates against whatever trailing evidence a set of NotationNotes
    carries. Notes without it (drums, which have no Ma report by design) simply
    do not contribute observations."""
    return MaGate.calibrate(
        [getattr(n, "trailing_resonance", None) for n in notes],
        [getattr(n, "trailing_void", None) for n in notes],
    )


__all__ = ["MaGate", "calibrate_from_notes", "RINGING_PERCENTILE",
           "VOID_PERCENTILE", "RINGING_REACH", "MIN_SAMPLES"]
