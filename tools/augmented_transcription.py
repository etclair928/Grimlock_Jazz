#!/usr/bin/env python3
# =================================================================
# TOOL: tools/augmented_transcription.py
# TEST-TIME AUGMENTATION probe: does Basic Pitch hear MORE when the audio is
# played slower / in a different register, with the result mapped back so the
# output is as if we had never altered the source?
#
# WHY RESAMPLING AND NOT A PITCH SHIFTER. Shifting pitch at constant tempo (or
# stretching time at constant pitch) needs a phase vocoder, which SMEARS
# TRANSIENTS - precisely the onset sharpness the "slower is rhythmically
# clearer" hypothesis depends on. A measured loss would then be unattributable:
# the idea failing, or the stretcher damaging the signal? Plain resampling
# changes pitch and tempo TOGETHER and is bit-exact - no interpolation
# artifacts at all, because we do not resample the samples. We simply declare a
# different sample rate, and let Basic Pitch's own front end do the one
# resample it was always going to do.
#
# The happy arithmetic: 2^(-5/12) = 0.7492. Playing at ~75% speed IS dropping a
# perfect fourth. "Slow it down" and "shift it down a fourth" are the same
# operation, so one artifact-free pass tests both hypotheses at once.
#
# BOTH INVERSES ARE EXACT on symbolic output, which is what makes this honest:
#   time_original  = time_detected * ratio
#   pitch_original = pitch_detected - 12*log2(ratio)      (whole semitones)
# No estimation, no drift. The emitted notes are directly comparable to a
# baseline run - the augmentation is invisible in the output by construction.
#
# NOT part of the pipeline. A standalone measurement tool.
#   python tools/augmented_transcription.py Input/End_Transmission/stems \
#          --stems other bass vocals
# =================================================================

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import numpy as np

try:  # Basic Pitch 0.3.0 + modern scipy (same shim the rest of the repo carries)
    import scipy.signal as _sps
    if not hasattr(_sps, "gaussian"):
        from scipy.signal.windows import gaussian as _gaussian
        _sps.gaussian = _gaussian
except Exception:
    pass

SEMITONE = 2.0 ** (1.0 / 12.0)
# (label, nominal semitone shift). The speed ratio is derived, so the two are
# guaranteed consistent - there is no way to state a tempo that disagrees with
# the pitch.
CONDITIONS = [("baseline", 0), ("down_P4", -5), ("up_P4", +5)]


def condition_ratio(semitones: int) -> float:
    return SEMITONE ** semitones


def write_rescaled(samples: np.ndarray, sr: int, ratio: float, path: str) -> float:
    """Write `samples` with a DECLARED sample rate of sr*ratio. Playing the same
    samples at a different rate scales pitch and duration by exactly that ratio
    with no interpolation. Returns the ratio actually achieved (the declared
    rate must be an integer, so it can differ from the nominal by a fraction of
    a cent - we return the truth rather than the intention)."""
    import soundfile as sf
    declared = int(round(sr * ratio))
    sf.write(path, samples, declared, subtype="FLOAT")
    return declared / float(sr)


def run_basic_pitch(model, path: str) -> List[tuple]:
    from basic_pitch.inference import predict
    _mo, _midi, note_events = predict(path, model)
    return note_events


def invert(note_events: List[tuple], ratio: float) -> List[Tuple[float, float, int, float]]:
    """Map detections on the rescaled audio back onto the original timeline and
    register. Exact: times scale by the ratio, pitch shifts by the whole number
    of semitones that ratio represents."""
    semis = int(round(12.0 * math.log2(ratio))) if ratio != 1.0 else 0
    out = []
    for ev in note_events:
        start, end, pitch, amp = ev[0], ev[1], ev[2], ev[3]
        out.append((start * ratio, end * ratio, int(pitch) - semis, float(amp)))
    return out


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def sounding_time(notes) -> float:
    return sum(e - s for s, e, _p, _a in notes)


def pitch_class_profile(notes) -> np.ndarray:
    v = np.zeros(12)
    for _s, _e, p, _a in notes:
        v[int(p) % 12] += 1
    return v / v.sum() if v.sum() else v


def onset_grid_error(notes, beat_s: float) -> float:
    """Mean absolute distance from the nearest grid line, as a FRACTION of a
    subdivision. This is the direct test of 'slower audio -> cleaner rhythm':
    if the hypothesis is right, augmented onsets should sit nearer the grid."""
    if not notes or beat_s <= 0:
        return float("nan")
    sub = beat_s / 4.0          # sixteenth-note lattice
    errs = [abs(((s / sub) - round(s / sub))) for s, _e, _p, _a in notes]
    return float(np.mean(errs))


def match_rate(a, b, onset_tol: float = 0.05) -> float:
    """Fraction of `a` that finds a same-pitch partner in `b` within tolerance -
    how much of the baseline the augmented run reproduces (and vice versa)."""
    if not a or not b:
        return 0.0
    by_pitch: Dict[int, List[float]] = {}
    for s, _e, p, _amp in b:
        by_pitch.setdefault(int(p), []).append(s)
    for v in by_pitch.values():
        v.sort()
    hit = 0
    for s, _e, p, _amp in a:
        cands = by_pitch.get(int(p))
        if not cands:
            continue
        i = np.searchsorted(cands, s)
        for j in (i - 1, i):
            if 0 <= j < len(cands) and abs(cands[j] - s) <= onset_tol:
                hit += 1
                break
    return hit / len(a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stems_dir")
    ap.add_argument("--stems", nargs="*", default=["other", "bass", "vocals"])
    ap.add_argument("--beat-seconds", type=float, default=0.400,
                    help="beat period of the ORIGINAL audio, for the grid-error metric")
    ap.add_argument("--out", default=None, help="write inverted MIDI here")
    args = ap.parse_args()

    import soundfile as sf
    from basic_pitch.inference import Model
    from basic_pitch import ICASSP_2022_MODEL_PATH
    model = Model(ICASSP_2022_MODEL_PATH)

    results: Dict[str, Dict[str, list]] = {}
    for stem in args.stems:
        path = os.path.join(args.stems_dir, stem + ".wav")
        if not os.path.exists(path):
            print(f"  ! missing {path}")
            continue
        samples, sr = sf.read(path, dtype="float32", always_2d=True)
        results[stem] = {}
        for label, semis in CONDITIONS:
            nominal = condition_ratio(semis)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                tmp = fh.name
            try:
                actual = write_rescaled(samples, sr, nominal, tmp)
                t0 = time.time()
                ev = run_basic_pitch(model, tmp)
                notes = invert(ev, actual)
                dt = time.time() - t0
                results[stem][label] = notes
                print(f"  {stem:8s} {label:9s} ratio={actual:.6f} "
                      f"({12*math.log2(actual):+.3f} semitones)  "
                      f"notes={len(notes):5d}  {dt:.0f}s", flush=True)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    print("\n=== COMPARISON (all mapped back to the original key and tempo) ===")
    hdr = (f"{'stem':8s} {'condition':9s} {'notes':>6s} {'sounding_s':>11s} "
           f"{'grid_err':>9s} {'keeps_base':>11s} {'base_keeps':>11s} {'pc_r':>6s}")
    print(hdr); print("-" * len(hdr))
    for stem, by_cond in results.items():
        base = by_cond.get("baseline", [])
        bpp = pitch_class_profile(base)
        for label, _ in CONDITIONS:
            n = by_cond.get(label)
            if n is None:
                continue
            pc = pitch_class_profile(n)
            r = float(np.corrcoef(bpp, pc)[0, 1]) if base and n else float("nan")
            print(f"{stem:8s} {label:9s} {len(n):6d} {sounding_time(n):11.1f} "
                  f"{onset_grid_error(n, args.beat_seconds):9.4f} "
                  f"{match_rate(base, n):11.3f} {match_rate(n, base):11.3f} {r:6.3f}")
        print()

    if args.out:
        import pretty_midi
        os.makedirs(args.out, exist_ok=True)
        for stem, by_cond in results.items():
            for label, notes in by_cond.items():
                pm = pretty_midi.PrettyMIDI()
                inst = pretty_midi.Instrument(program=0)
                for s, e, p, a in notes:
                    if e <= s or not (0 <= p < 128):
                        continue
                    inst.notes.append(pretty_midi.Note(
                        velocity=int(np.clip(a * 127, 1, 127)), pitch=int(p),
                        start=float(s), end=float(e)))
                pm.instruments.append(inst)
                pm.write(os.path.join(args.out, f"{stem}_{label}.mid"))
        print(f"MIDI written to {args.out} (already mapped back to original key/tempo)")


if __name__ == "__main__":
    main()
