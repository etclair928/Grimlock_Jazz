#!/usr/bin/env python3
# =================================================================
# TOOL: tools/meter_harness.py
# Meter detection only, across the library, against KNOWN-CORRECT answers.
#
# WHY THIS EXISTS. Meter is the weakest scalar in the pipeline (§XVII.3), and
# every attempt to improve it so far has been judged on one song at a time -
# which is how the downbeat witness came to fix Hopeful and No Pasaran while
# nobody noticed what it does to a rubato solo piano piece. Meter is also cheap
# to evaluate compared with a full run: decode, onsets, tempo, accents, vote.
# Minutes, not hours. There is no reason to judge it one song at a time.
#
# WHAT IT MEASURES. For each song, the three meter witnesses are computed on
# BOTH candidate grids, and every vote rule under consideration is applied to
# the same candidates, so the grid question and the vote question are separated
# instead of confounded:
#
#   grids   isochronous  - build_phase_locked_grid(resolved_tempo): one scalar
#                          tempo extrapolated across the track.
#           tracked      - the anchor witness's own per-beat grid, which does
#                          not accumulate tempo error because nothing is
#                          extrapolated.
#
#   votes   sum          - the shipped rule: add the confidences.
#           agreement    - rank by HOW MANY independent witnesses chose that
#                          numerator first, summed confidence only as the
#                          tie-break.
#
# The `sum` rule adds numbers from three different scales - a chance-corrected
# salience, a relative spectral power, and a DBN's own confidence - as though
# they were commensurable votes. They are not, and on Chopin that lets one
# witness at 0.384 outvote two agreeing witnesses at 0.159 + 0.051.
#
#   python tools/meter_harness.py [song ...]
# =================================================================

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

# Ground truth. Answer-key entries are authoritative (read out of a published
# edition); doc entries are recorded elsewhere in the project and are marked as
# such so nobody mistakes one for the other.
GROUND_TRUTH: Dict[str, Tuple[Optional[int], str]] = {
    "Chopin_Nocturne_62_1": (4, "answer key (Chopin_ANSWER_KEY.musicxml)"),
    "Heavy_Rotation_Vibez": (3, "answer key (HRV_ANSWER_KEY.musicxml)"),
    "Ellington_Reflections_in_D": (None, "answer key: MIXED 3/2,4/2,4/4,5/4,6/4 - not scored"),
    "Hopeful": (6, "OPEN_PROBLEMS §XVII.3 'confirmed 6/4' (Klangio reads the same family as 3/4)"),
    "No_Pasaran": (4, "OPEN_PROBLEMS §536 'No Pasarán->4/4'"),
}


def candidates_for(engine, track, beats: Sequence[float], ratio_family: Optional[str],
                   downbeat) -> List[Tuple[str, int, int, float]]:
    """The three meter witnesses on one grid, tagged with which produced them."""
    from rhythm_engine import (estimate_time_signature_with_phase, sample_beat_accents,
                               fft_meter_candidate, resolve_denominator)
    out: List[Tuple[str, int, int, float]] = []
    n, d, c, _phase = estimate_time_signature_with_phase(engine, track, beats,
                                                         ratio_family=ratio_family)
    out.append(("accent", n, d, c))
    fft = fft_meter_candidate(sample_beat_accents(engine, track, beats))
    if fft is not None:
        out.append(("fft", fft[0], resolve_denominator(fft[0], ratio_family), fft[1]))
    if downbeat is not None:
        out.append(("downbeat", downbeat.beats_per_bar,
                    resolve_denominator(downbeat.beats_per_bar, ratio_family),
                    downbeat.confidence))
    return out


def vote_sum(cands: Sequence[Tuple[str, int, int, float]]) -> Tuple[int, int]:
    """The shipped rule: add the confidences and take the largest."""
    votes: Dict[Tuple[int, int], float] = {}
    for _src, n, d, c in cands:
        votes[(n, d)] = votes.get((n, d), 0.0) + c
    return max(votes, key=lambda k: votes[k])


def vote_agreement(cands: Sequence[Tuple[str, int, int, float]]) -> Tuple[int, int]:
    """Corroboration first: the numerator the most INDEPENDENT witnesses chose
    wins, and summed confidence only breaks a tie. Two methods that disagree
    about everything else landing on the same bar length is a stronger signal
    than one method's self-reported confidence, and unlike a sum it does not
    require the three scales to be commensurable."""
    agree: Dict[Tuple[int, int], List[float]] = {}
    for _src, n, d, c in cands:
        agree.setdefault((n, d), []).append(c)
    return max(agree, key=lambda k: (len(agree[k]), sum(agree[k])))


def evaluate(song: str, audio: str) -> None:
    from audio_engine import AudioEngine
    from rhythm_engine import (run_librosa_tempo, run_madmom_tempo, run_lattice_witness,
                               run_madmom_downbeat, detect_onset_candidates,
                               build_phase_locked_grid)
    from epistemic import resolve_tempo

    truth, source = GROUND_TRUTH.get(song, (None, "unknown"))
    engine = AudioEngine()
    track = engine.decode(audio)

    witnesses = [run_librosa_tempo(engine, track), run_madmom_tempo(engine, track)]
    lattice_w, lattice = run_lattice_witness(engine, track)
    if lattice_w is not None:
        witnesses.append(lattice_w)
    ratio_family = lattice.ratio_family if lattice is not None else None
    resolution = resolve_tempo(witnesses)
    tempo_meter = resolution.tempo_meter

    downbeat = run_madmom_downbeat(engine, track)
    onsets = detect_onset_candidates(engine, track)
    duration_ms = track.duration_seconds * 1000.0

    grids = {
        "isochronous": list(build_phase_locked_grid(tempo_meter.tempo_bpm,
                                                    onsets.combined_ms, duration_ms)),
        "tracked": list(tempo_meter.beat_times_ms or ()),
    }

    print(f"\n{'='*78}\n{song}   truth = {truth if truth else 'MIXED/unknown'}   [{source}]")
    print(f"  resolved tempo {tempo_meter.tempo_bpm:.1f} BPM   ratio_family={ratio_family}")
    for gname, beats in grids.items():
        if len(beats) < 8:
            print(f"  {gname:12s}: too few beats ({len(beats)})")
            continue
        cands = candidates_for(engine, track, beats, ratio_family, downbeat)
        detail = "  ".join(f"{src}={n}/{d}@{c:.3f}" for src, n, d, c in cands)
        s_n, _s_d = vote_sum(cands)
        a_n, _a_d = vote_agreement(cands)
        def mark(v):
            if truth is None:
                return "  -"
            return "  OK " if v == truth else "  XX "
        print(f"  {gname:12s} [{len(beats):4d} beats]  {detail}")
        print(f"  {'':12s}   vote:sum -> {s_n}{mark(s_n)}    vote:agreement -> {a_n}{mark(a_n)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("songs", nargs="*", default=None)
    args = ap.parse_args()

    root = os.path.join(JAZZ, "Input")
    songs = args.songs or list(GROUND_TRUTH)
    for song in songs:
        folder = os.path.join(root, song)
        if not os.path.isdir(folder):
            print(f"\n{song}: no Input/ folder, skipped")
            continue
        audio = None
        for ext in (".wav", ".mp3"):
            p = os.path.join(folder, song + ext)
            if os.path.exists(p):
                audio = p
                break
        if audio is None:
            print(f"\n{song}: no audio found, skipped")
            continue
        t0 = time.time()
        try:
            evaluate(song, audio)
        except Exception as exc:
            print(f"\n{song}: FAILED {type(exc).__name__}: {exc}")
        print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
