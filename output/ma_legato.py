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

# THE HOLD TIER. Measured against Klangio on Educated Heart, working from the
# same audio: their accompaniment parts are 52-79% HALF notes and ours were
# 0-17%. They hold a chord until the next attack; we wrote the attack and then
# rests, which is why 41% of our page was rests and 0.4-12.8% of theirs were
# short values against our 12.9-36.3%. Our onset placement is not the problem -
# we move a note a median 50ms from where it was played against their 45ms.
# The whole gap is duration policy.
#
# But "sustain more" cannot mean "sustain always". RINGING_PERCENTILE is the
# 25th, so three gaps in four already count as ringing, and letting all of them
# run to the next attack would over-sustain exactly as badly in the other
# direction. So the reach is GRADED: only the strongest quartile of the
# material's own resonance earns a hold to the next attack, and even that is
# capped. Everything else keeps the window it already had.
HOLD_PERCENTILE = 75.0       # at/above this percentile of resonance = hold on
# CAPPED AT WHAT THE REFERENCE ACTUALLY WRITES. Measured on Educated Heart the
# gap to the next attack in a voice has median 3.5 quarters, 90th percentile
# 11.5 and a maximum of 74 - voices are sparse, so "hold to the next attack"
# without a cap writes absurdities. Klangio's dominant accompaniment value is a
# half note, which on our clock is 2.0 quarters, and that is the cap.
MAX_HOLD_QL = 2.0            # never hold further than this, whatever the gap
MIN_SAMPLES = 30             # below this, percentiles are noise - stay blind


@dataclass(frozen=True)
class MaGate:
    """A resonance/void gate calibrated against one score's own distribution."""
    ring_at: Optional[float] = None
    void_at: Optional[float] = None
    hold_at: Optional[float] = None

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
            hold_at=float(np.percentile(res, HOLD_PERCENTILE)),
        )

    def limit(self, resonance: Optional[float], void: Optional[float],
              base: float, gap: Optional[float] = None,
              max_hold: Optional[float] = None) -> float:
        """How far this particular gap may be closed.

        `base` is the ordinary window the caller would have used with no
        evidence at all, and is what comes back whenever the witness has
        nothing confident to say - which is the point of the middle case.

        `gap` is the distance to the next attack in this voice. Supplying it
        enables the hold tier: on the strongest quartile of resonance the note
        may run all the way to that attack, which is what a held chord IS and
        what the comparison against Klangio says we were failing to write.
        Without `gap` the behaviour is exactly what it was.

        Four outcomes, most specific first, and every one of them bounded:
          genuine silence  -> 0, the rest is real
          strongly ringing -> to the next attack, capped at `max_hold`
          ringing          -> the ordinary window, reached further
          anything else    -> the ordinary window
        """
        if self.blind or resonance is None:
            return base
        if self.void_at is not None and void is not None and void >= self.void_at:
            return 0.0                       # the audio says: genuine silence
        if (gap is not None and self.hold_at is not None
                and resonance >= self.hold_at):
            # Read at CALL time, not bound as a default argument. A default is
            # evaluated when the function is defined, so a swept constant never
            # reached it and every row of the first sweep silently used the
            # same cap - which is what made the tier look inert.
            cap = MAX_HOLD_QL if max_hold is None else max_hold
            return max(base * RINGING_REACH, min(float(gap), cap))
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
           "VOID_PERCENTILE", "RINGING_REACH", "MIN_SAMPLES",
           "HOLD_PERCENTILE", "MAX_HOLD_QL"]
