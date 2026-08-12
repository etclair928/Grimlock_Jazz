# =================================================================
# Synthetic +/- tests for every detector (GRIMLOCK_UNIVERSITY.md §7).
# A hand-written figure MUST fire; unstructured material must NOT.
# This is the cheap guard that would have caught four of the five dead
# detectors in the pasted proposal (its Alberti test compared a 4-element
# list to a 3-element list and could never return True).
#   python university/test_detectors.py
# =================================================================

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from university.detectors import (  # noqa: E402
    detect_alberti, detect_arpeggios, detect_ostinato, detect_pedal,
    detect_rearticulation, detect_scale_runs, detect_sequences,
)

BEAT = 500.0


class N:
    """Note stand-in: the detectors only read pitch/start/end/id/confidence."""
    def __init__(self, i, pitch, start, dur=240.0, confidence=0.9):
        self.id = f"n{i}"
        self.pitch = pitch
        self.start_ms = start
        self.end_ms = start + dur
        self.confidence = confidence


def seq(pitches, step=250.0, dur=240.0, confidence=0.9):
    return [N(i, p, i * step, dur, confidence) for i, p in enumerate(pitches)]


def noise(n=24, seed=3):
    rng = random.Random(seed)
    return [N(i, rng.randint(48, 84), i * 250.0) for i in range(n)]


RESULTS = []


def check(name, condition):
    RESULTS.append((name, bool(condition)))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")


print("scale_run")
check("+ ascending C major run fires",
      detect_scale_runs(seq([60, 62, 64, 65, 67, 69]), BEAT, "other"))
check("- arpeggio does not fire as a scale",
      not detect_scale_runs(seq([60, 64, 67, 72]), BEAT, "other"))

print("arpeggio")
check("+ C major broken chord fires",
      detect_arpeggios(seq([60, 64, 67, 72]), BEAT, "other"))
check("- stepwise run does not fire as arpeggio",
      not detect_arpeggios(seq([60, 62, 64, 65]), BEAT, "other"))

print("sequence")
check("+ cell repeated a step higher fires",
      detect_sequences(seq([60, 62, 64, 62, 64, 66]), BEAT, "other"))
check("- same cell repeated at SAME pitch is not a sequence",
      not detect_sequences(seq([60, 62, 64, 60, 62, 64]), BEAT, "other"))

print("ostinato")
check("+ 3x identical cell fires",
      detect_ostinato(seq([60, 64, 60, 64, 60, 64]), BEAT, "other"))
check("- transposed repeat is not an ostinato",
      not detect_ostinato(seq([60, 64, 62, 66, 64, 68]), BEAT, "other"))

print("pedal_point")
check("+ one pitch repeated over 2+ beats fires",
      detect_pedal(seq([48, 48, 48, 48], step=500.0, dur=480.0), BEAT, "bass"))
check("- moving bass does not fire",
      not detect_pedal(seq([48, 50, 52, 53], step=500.0, dur=480.0), BEAT, "bass"))

print("rearticulation")
check("+ same pitch re-struck fast fires",
      detect_rearticulation(seq([60, 60, 60, 60], step=100.0, dur=80.0), BEAT, "other"))
check("- same pitch spread over beats does NOT fire",
      not detect_rearticulation(seq([60, 60, 60, 60], step=600.0, dur=480.0), BEAT, "other"))

print("alberti_bass  (the one the proposal could never fire)")
check("+ C-G-E-G x2 fires",
      detect_alberti(seq([60, 67, 64, 67, 60, 67, 64, 67]), BEAT, "other"))
check("- single cycle does not fire",
      not detect_alberti(seq([60, 67, 64, 67]), BEAT, "other"))
check("- scale does not fire as alberti",
      not detect_alberti(seq([60, 62, 64, 65, 67, 69, 71, 72]), BEAT, "other"))

print("noise sanity (unstructured input should be mostly quiet)")
nz = noise()
fired = sum(len(d(nz, BEAT, "other")) for d in
            (detect_scale_runs, detect_ostinato, detect_alberti, detect_pedal))
check(f"random pitches fire few structured patterns (got {fired})", fired <= 2)

failed = [n for n, ok in RESULTS if not ok]
print("\n" + "=" * 60)
print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
