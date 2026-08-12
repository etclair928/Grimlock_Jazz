# =================================================================
# MODULE: core/musical_time.py
# THE MAP BETWEEN CLOCK TIME AND MUSICAL TIME.
#
# A note is not "at 82.3 seconds". It is "the second sixteenth of beat 3 of bar
# 41". The seconds are an accident of how fast the performer happened to be
# moving right then. Everything in this project was built in absolute
# milliseconds and converted to musical position at the last moment by dividing
# by 60000/bpm - which is correct only if the performer is a metronome.
#
# MEASURED, on Rubinstein's Chopin Op.62/1 (2026-08-11):
#   * madmom tracks 222 beats with CV 0.168 and local tempo 53-133 BPM
#   * an isochronous grid drifts up to 6.99s from those beats and is >0.35s
#     off for 93% of them
#   * recall against the published edition rises 0.46 -> 0.78 -> 0.94 as the
#     matching tolerance goes 0.35s -> 1.0s -> 5.0s, which is that same drift
#     distribution seen from the other side
#   * fixing ONE of the three conversion sites moved recall +9.6%; the other
#     two (musicxml_exporter._offset_quarter_length and ._quarter_length)
#     divide by a constant and silently revert it
#
# So the rule this module exists to enforce: THIS IS THE ONLY CURRENCY between
# clock time and musical position. Every conversion goes through it or the fix
# is undone one layer down. `tempo_bpm` becomes a display statistic derived
# from the local slope, not an input to anything.
#
# HONEST STATUS. We have NOT established the curve is correct - only that it is
# not a straight line. Validating it against a score-performance alignment was
# attempted and the DTW-derived reference proved too noisy per-beat (its own CV
# was 0.602 against the tracker's 0.168). Both sane correspondences scored at
# or BELOW a density-matched null. That is inconclusive rather than negative,
# but it is why `confidence()` exists and why callers are expected to quantise
# LOOSELY where it is low: a rigid grid fails diffusely, a wrong map fails
# confidently, and the second is worse.
# =================================================================

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

DEFAULT_BEAT_MS = 500.0
# A beat interval this far from its neighbours is a tracker glitch, not rubato -
# real performers do not double their tempo for exactly one beat.
GLITCH_RATIO = 2.5


@dataclass(frozen=True)
class MusicalTime:
    """Monotonic, invertible map between performance ms and beat position.

    `beats_ms` are the tracked beat instants. Outside their span the map
    extrapolates at the local rate rather than snapping to a global tempo,
    so a pickup before beat one still lands sensibly."""

    beats_ms: Tuple[float, ...]
    fallback_beat_ms: float = DEFAULT_BEAT_MS

    # -- construction ------------------------------------------------------

    @classmethod
    def from_beats(cls, beats_ms: Sequence[float],
                   fallback_beat_ms: float = DEFAULT_BEAT_MS) -> "MusicalTime":
        clean = sorted(float(b) for b in (beats_ms or ()))
        # drop duplicates and non-monotonic entries; the map must be invertible
        out = []
        for b in clean:
            if not out or b - out[-1] > 1.0:
                out.append(b)
        return cls(tuple(out), float(fallback_beat_ms or DEFAULT_BEAT_MS))

    @classmethod
    def isochronous(cls, tempo_bpm: float, origin_ms: float = 0.0,
                    duration_ms: float = 0.0) -> "MusicalTime":
        """The old behaviour, made explicit. Only for callers with no tracked
        beats - never as a 'clean up' of beats we do have."""
        beat = 60000.0 / max(tempo_bpm, 1e-6)
        n = max(2, int(duration_ms / beat) + 1) if duration_ms > 0 else 2
        return cls(tuple(origin_ms + i * beat for i in range(n)), beat)

    # -- the map -----------------------------------------------------------

    @property
    def usable(self) -> bool:
        return len(self.beats_ms) >= 2

    def _local_beat_ms(self, i: int) -> float:
        b = self.beats_ms
        if len(b) < 2:
            return self.fallback_beat_ms
        i = max(0, min(i, len(b) - 2))
        return max(b[i + 1] - b[i], 1.0)

    def to_beats(self, t_ms: float) -> float:
        """Performance ms -> continuous beat position."""
        b = self.beats_ms
        n = len(b)
        if n == 0:
            return t_ms / self.fallback_beat_ms
        if n == 1:
            return (t_ms - b[0]) / self.fallback_beat_ms
        if t_ms <= b[0]:
            return (t_ms - b[0]) / self._local_beat_ms(0)
        if t_ms >= b[-1]:
            return (n - 1) + (t_ms - b[-1]) / self._local_beat_ms(n - 2)
        i = bisect.bisect_right(b, t_ms) - 1
        return i + (t_ms - b[i]) / self._local_beat_ms(i)

    def to_ms(self, pos: float) -> float:
        """Beat position -> performance ms. Exact inverse of to_beats."""
        b = self.beats_ms
        n = len(b)
        if n == 0:
            return pos * self.fallback_beat_ms
        if n == 1:
            return b[0] + pos * self.fallback_beat_ms
        if pos <= 0.0:
            return b[0] + pos * self._local_beat_ms(0)
        if pos >= n - 1:
            return b[-1] + (pos - (n - 1)) * self._local_beat_ms(n - 2)
        i = int(pos)
        return b[i] + (pos - i) * self._local_beat_ms(i)

    def snap(self, t_ms: float, subdivisions: int) -> Tuple[float, float]:
        """Snap an onset to the nearest subdivision OF THE BEAT IT BELONGS TO.
        Returns (snapped_ms, residual_beats) - the residual is the expressive
        microtiming, which is the performance information the old grid destroyed
        rather than recorded."""
        pos = self.to_beats(t_ms)
        snapped = round(pos * subdivisions) / subdivisions
        return self.to_ms(snapped), pos - snapped

    # -- confidence --------------------------------------------------------

    def local_tempo_bpm(self, t_ms: float) -> float:
        i = max(0, min(int(self.to_beats(t_ms)), len(self.beats_ms) - 2)) \
            if self.usable else 0
        return 60000.0 / self._local_beat_ms(i)

    def confidence(self, t_ms: float) -> float:
        """How much to trust the map here, in [0,1]. Low where the beat
        interval jumps relative to its neighbours - a tracker that suddenly
        doubles its period for one beat is glitching, not following rubato.
        Callers should quantise LOOSELY (or not at all) where this is low."""
        if not self.usable:
            return 0.0
        i = max(0, min(int(self.to_beats(t_ms)), len(self.beats_ms) - 2))
        here = self._local_beat_ms(i)
        prev = self._local_beat_ms(max(i - 1, 0))
        nxt = self._local_beat_ms(min(i + 1, len(self.beats_ms) - 2))
        ratio = max(here / max(prev, 1.0), here / max(nxt, 1.0),
                    max(prev, 1.0) / here, max(nxt, 1.0) / here)
        if ratio >= GLITCH_RATIO:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - (ratio - 1.0) / (GLITCH_RATIO - 1.0))))

    def summary_bpm(self) -> float:
        """The single number the rest of the world still asks for. DERIVED -
        never an input."""
        if not self.usable:
            return 60000.0 / self.fallback_beat_ms
        span = self.beats_ms[-1] - self.beats_ms[0]
        return 60000.0 * (len(self.beats_ms) - 1) / max(span, 1.0)


__all__ = ["MusicalTime", "DEFAULT_BEAT_MS"]
