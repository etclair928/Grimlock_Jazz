#!/usr/bin/env python3
# =================================================================
# TOOL: tools/score_vs_answer_key.py
# Grades a transcription against a PUBLISHED SCORE of the same music.
#
# WHY THIS IS DIFFERENT FROM EVERY OTHER METRIC WE HAVE. readability.py scores
# the SHAPE of a page (rests, voices, chords) and cannot tell whether the notes
# are right. The Klangio head-to-heads compare us to another estimate, not to
# truth. This compares against an actual edition - the first note-level ground
# truth in the project for real repertoire (Bach chorales gave labelled VOICES,
# but not for music we target).
#
# THE HARD PART IS ALIGNMENT, NOT COMPARISON. A published score is in SCORE
# time (measures/beats); a performance is in PERFORMANCE time, and a Chopin
# nocturne is nothing but rubato - the edition itself carries 26 tempo marks.
# A fixed BPM mapping would misalign within a few bars and score a perfect
# transcription as garbage. So:
#
#   1. Compare on PITCH CONTENT first (histograms, ranges, density). These need
#      no alignment at all and cannot be faked by getting the timing wrong.
#   2. Align with DTW over pitch-class chroma sequences, which tolerates rubato
#      by construction, then report note-level precision/recall on the aligned
#      region ONLY - and report how much of the score that region covers, so a
#      good score on 20% of the music cannot masquerade as a good score.
#
# HONEST LIMIT: an edition is not a performance. Rubinstein adds and omits
# things, rolls chords, and the recording's first 3 minutes cover only part of
# the piece. Numbers here are "agreement with the edition", not "correctness".
#
#   python tools/score_vs_answer_key.py <ours.musicxml|.mid> <answer_key.mxl> [--audio-seconds 180]
# =================================================================

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

Note = Tuple[float, float, int]      # (start, end, midi) - units differ per source


def load_answer_key(path: str) -> Tuple[List[Note], float]:
    """Returns (notes in QUARTER-NOTE units, total quarters)."""
    from music21 import converter
    s = converter.parse(path)
    out = []
    for el in s.flatten().notes:
        off = float(el.offset)
        dur = float(el.quarterLength)
        # Skip <unpitched> percussion. Drum noteheads carry a staff POSITION,
        # not a pitch, so el.pitch raises on them - and comparing them to our
        # pitched output would be meaningless anyway (we exclude drums on our
        # side too). Klangio notates its drum staff this way, correctly.
        try:
            pitches = el.pitches if el.isChord else [el.pitch]
        except AttributeError:
            continue
        for p in pitches:
            out.append((off, off + dur, int(p.midi)))
    out.sort()
    return out, (max(e for _s, e, _p in out) if out else 0.0)


def load_ours(path: str) -> List[Note]:
    """Returns notes in SECONDS."""
    if path.lower().endswith((".mid", ".midi")):
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(path)
        return sorted((n.start, n.end, n.pitch)
                      for i in pm.instruments if not i.is_drum for n in i.notes)
    import xml.etree.ElementTree as ET
    STEP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    root = ET.parse(path).getroot()
    div = int(next(root.iter("divisions")).text)
    tempo = 120.0
    for m in root.iter("per-minute"):
        tempo = float(m.text); break
    qsec = 60.0 / tempo
    names = {sp.get("id"): (sp.find("part-name").text or "")
             for sp in root.findall(".//score-part")}
    out = []
    for part in root.findall(".//part"):
        if "drum" in names.get(part.get("id"), "").lower():
            continue
        t = 0.0
        for m in part.findall("measure"):
            cur, longest, last = {}, 0.0, {}
            for n in m.findall("note"):
                v = n.find("voice"); key = v.text if v is not None else "1"
                d = int(n.find("duration").text) if n.find("duration") is not None else 0
                dq = d / div
                if n.find("chord") is not None:
                    onset = last.get(key, cur.get(key, 0.0))
                else:
                    onset = cur.get(key, 0.0); last[key] = onset
                    cur[key] = onset + dq
                p = n.find("pitch")
                if n.find("rest") is not None or p is None:
                    continue
                alt = int(p.find("alter").text) if p.find("alter") is not None else 0
                midi = (int(p.find("octave").text) + 1) * 12 + STEP[p.find("step").text] + alt
                out.append(((t + onset) * qsec, (t + onset + dq) * qsec, midi))
                longest = max(longest, cur[key])
            t += longest if longest else 4.0
    return sorted(out)


def chroma_seq(notes: Sequence[Note], t_end: float, steps: int) -> np.ndarray:
    """Pitch-class histogram per time step - the alignment feature. Chroma is
    transposition-stable and, more importantly here, insensitive to octave
    errors and to how many notes of a chord were detected."""
    grid = np.zeros((steps, 12))
    if t_end <= 0:
        return grid
    for s, e, p in notes:
        a = int(np.clip(s / t_end * steps, 0, steps - 1))
        b = int(np.clip(e / t_end * steps, 0, steps - 1))
        grid[a:b + 1, p % 12] += 1.0
    n = np.linalg.norm(grid, axis=1, keepdims=True)
    return grid / np.where(n > 0, n, 1.0)


def dtw_path(A: np.ndarray, B: np.ndarray, open_end: bool = False):
    """A = answer key, B = ours.

    open_end=True does SUBSEQUENCE alignment: B may match a PREFIX of A rather
    than all of it. Classic DTW pins both endpoints, which is wrong whenever the
    recording covers only part of the piece - it stretches the excerpt across
    the whole score to satisfy the endpoint constraint. MEASURED symptom: a
    180s excerpt of a 411s performance reported as covering "380 of 380
    quarters, 100% of the piece", a ~2.3x stretch, after which every note-level
    number is meaningless."""
    na, nb = len(A), len(B)
    cost = 1.0 - A @ B.T
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    if open_end:
        # end anywhere in the key, but consume all of ours; normalise by path
        # length so short prefixes are not trivially cheap
        best_i = int(np.argmin([D[i, nb] / max(i, 1) for i in range(1, na + 1)])) + 1
        i, j, path = best_i, nb, []
    else:
        i, j, path = na, nb, []
    end_i, end_j = i, j
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        if step == D[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif step == D[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return path[::-1], D[end_i, end_j] / max(len(path), 1)


# =====================================================================
# CONTINUOUS TIMING METRICS - no threshold, no cliff
# =====================================================================
# WHY. A hard tolerance turns one number into two stories: on Chopin the same
# transcription scored recall 0.645 at tau=0.25s and 0.903 at tau=0.75s. Every
# conclusion drawn from a single tau is really a conclusion about tau. Worse,
# the cliff hides its own cause - the tool's DTW step was 0.20s while it graded
# at 0.25s, so a quarter of the "error" was the ruler.
#
# THE KERNEL IS SCORED OVER EVERY REFERENCE NOTE, NOT OVER MATCHED PAIRS.
# Scoring only matched pairs smuggles the threshold back in: you need a
# tolerance to decide what "matched" means, and the resulting average is
# conditioned on it. Here each reference note contributes exp(-dt^2/2s^2) for
# its nearest same-pitch candidate, and 0 when there is none at all - so a
# missed note costs a real zero and the statistic is a soft RECALL that no
# threshold touches.

def gaussian_timing_score(reference_times, ours_by_pitch, sigma=0.100,
                          search_cap=2.0):
    """Soft recall in [0,1]: 1.0 only if every reference note has a same-pitch
    neighbour at exactly its time. `sigma` sets how fast credit decays - it is a
    RULER, not a tolerance, and nothing is ever excluded by it."""
    import math
    total = 0.0
    errors = []
    used = set()
    for t, pitch in reference_times:
        best, bd = None, search_cap
        for st, idx in ours_by_pitch.get(pitch, ()):
            if idx in used:
                continue
            if abs(st - t) < bd:
                best, bd = idx, abs(st - t)
        if best is None:
            continue                      # contributes 0.0
        used.add(best)
        errors.append(bd)
        total += math.exp(-(bd ** 2) / (2.0 * sigma ** 2))
    return total / max(len(reference_times), 1), errors


def timing_moments(errors_signed):
    """delta (bias) and sigma (spread) in ms, plus the ratio that decides
    whether calibrating the bias is worth anything: below ~1 the error is
    variance-dominated and removing delta buys almost nothing."""
    import math
    if not errors_signed:
        return {"delta_ms": float("nan"), "sigma_ms": float("nan"),
                "rmse_ms": float("nan"), "delta_over_sigma": float("nan")}
    n = len(errors_signed)
    mean = sum(errors_signed) / n
    var = sum((e - mean) ** 2 for e in errors_signed) / n
    sigma = math.sqrt(var)
    rmse = math.sqrt(sum(e * e for e in errors_signed) / n)
    return {"delta_ms": mean * 1000.0, "sigma_ms": sigma * 1000.0,
            "rmse_ms": rmse * 1000.0,
            "delta_over_sigma": (abs(mean) / sigma) if sigma > 0 else float("inf")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ours")
    ap.add_argument("answer_key")
    ap.add_argument("--audio-seconds", type=float, default=180.0)
    # Alignment resolution. At 300 steps over a 180s excerpt each step is 0.6s -
    # COARSER than the +/-0.35s tolerance we grade at, so that column was partly
    # measuring this tool rather than the transcription. 900 gives 0.2s, which
    # is STILL coarser than the 0.25s we grade at.
    #
    # MEASURED on Chopin (2026-08-19), sweeping this parameter alone with the
    # transcription held fixed:
    #     steps   s/step   timing sigma   within 0.25s
    #       450    0.397        0.315s        59.8%
    #       900    0.199        0.265s        69.2%
    #      1800    0.099        0.226s        79.1%
    #      3600    0.050        0.216s        80.8%
    # A third of the apparent error was this tool's own resolution, and 11
    # points of "recall" were recovered by measuring more carefully rather than
    # by transcribing better. It plateaus by 1800, so that is the default; the
    # residual ~0.216s is real and ours.
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--onset-tol", type=float, default=0.25,
                    help="onset tolerance as a FRACTION of the local step")
    args = ap.parse_args()

    key, key_end_q = load_answer_key(args.answer_key)
    ours = load_ours(args.ours)
    ours_end = max(e for _s, e, _p in ours) if ours else 0.0
    print(f"answer key: {len(key)} noteheads over {key_end_q:.0f} quarters")
    print(f"ours:       {len(ours)} noteheads over {ours_end:.1f}s "
          f"({args.audio_seconds:.0f}s of audio)")

    # ---- content comparison, NO alignment needed ----
    print("\n=== CONTENT (alignment-free) ===")
    kc = Counter(p % 12 for _s, _e, p in key)
    oc = Counter(p % 12 for _s, _e, p in ours)
    kv = np.array([kc[i] for i in range(12)], float); kv /= kv.sum()
    ov = np.array([oc[i] for i in range(12)], float); ov /= max(ov.sum(), 1)
    PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    print(f"  pitch-class correlation r = {np.corrcoef(kv, ov)[0,1]:.4f}")
    print("    key : " + " ".join(f"{PC[i]}:{kv[i]*100:.0f}" for i in np.argsort(kv)[::-1][:6]))
    print("    ours: " + " ".join(f"{PC[i]}:{ov[i]*100:.0f}" for i in np.argsort(ov)[::-1][:6]))
    kp = [p for _s, _e, p in key]; op = [p for _s, _e, p in ours]
    print(f"  range   key {min(kp)}-{max(kp)}   ours {min(op)}-{max(op)}")
    print(f"  median pitch  key {np.median(kp):.0f}   ours {np.median(op):.0f}")

    # ---- alignment ----
    A = chroma_seq(key, key_end_q, args.steps)
    B = chroma_seq(ours, ours_end, args.steps)
    path, cost = dtw_path(A, B, open_end=True)
    print(f"\n=== ALIGNMENT (DTW over chroma, rubato-tolerant) ===")
    print(f"  mean path cost {cost:.4f}  (0 = identical chroma, 1 = orthogonal)")
    last_key_step = max(i for i, _j in path)
    covered_q = (last_key_step + 1) / args.steps * key_end_q
    print(f"  our {args.audio_seconds:.0f}s maps to roughly the first "
          f"{covered_q:.0f} of {key_end_q:.0f} quarters "
          f"({100*covered_q/key_end_q:.0f}% of the piece)")

    # map key quarters -> our seconds via the path, then score notes
    kq = np.array([i for i, _j in path], float) / args.steps * key_end_q
    osec = np.array([j for _i, j in path], float) / args.steps * ours_end

    def key_q_to_sec(q):
        return float(np.interp(q, kq, osec))

    # only the portion of the score our excerpt actually reaches
    covered_end_q = float(kq.max())
    in_range = [n for n in key if n[0] <= covered_end_q]
    tol = args.onset_tol

    matched = set()
    hit = 0
    by_pitch = {}
    for idx, (s, _e, p) in enumerate(ours):
        by_pitch.setdefault(p, []).append((s, idx))
    for v in by_pitch.values():
        v.sort()
    for s, _e, p in in_range:
        t = key_q_to_sec(s)
        cands = by_pitch.get(p, [])
        best, bd = None, tol
        for st, idx in cands:
            if idx in matched:
                continue
            d = abs(st - t)
            if d <= bd:
                best, bd = idx, d
        if best is not None:
            matched.add(best); hit += 1

    rec = hit / max(len(in_range), 1)
    prec = hit / max(len(ours), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\n=== NOTE-LEVEL AGREEMENT (tolerance {tol:.2f}s, rubato-aligned) ===")
    print(f"  answer-key notes in our window: {len(in_range)}")
    print(f"  recall    {rec:.3f}   (of the edition's notes, how many we found)")
    print(f"  precision {prec:.3f}   (of our notes, how many the edition has)")
    print(f"  F1        {f1:.3f}")

    # ---- threshold-free view of the same data --------------------------
    ref_times = [(key_q_to_sec(s0), p) for s0, _e0, p in in_range]
    soft, abs_errs = gaussian_timing_score(ref_times, by_pitch)
    signed = []
    used2 = set()
    for t, pitch in ref_times:
        best, bd = None, 2.0
        for st, idx in by_pitch.get(pitch, ()):
            if idx in used2:
                continue
            if abs(st - t) < bd:
                best, bd = idx, abs(st - t)
        if best is not None:
            used2.add(best)
            signed.append(ours[best][0] - t)
    m = timing_moments(signed)
    step_s = ours_end / max(args.steps, 1)
    print(f"\n=== CONTINUOUS TIMING (no threshold) ===")
    print(f"  alignment resolution      {step_s*1000:6.1f} ms/step "
          f"({'FINER' if step_s < tol else 'COARSER'} than the {tol:.2f}s tolerance above)")
    print(f"  soft recall (gaussian)    {soft:.3f}   sigma=100ms, scored over every "
          f"reference note")
    print(f"  RMSE                      {m['rmse_ms']:6.1f} ms")
    print(f"  bias  delta               {m['delta_ms']:+6.1f} ms")
    print(f"  spread sigma              {m['sigma_ms']:6.1f} ms")
    print(f"  delta/sigma               {m['delta_over_sigma']:.3f}   "
          f"({'BIAS-dominated: calibration would pay' if m['delta_over_sigma'] > 1.0 else 'VARIANCE-dominated: calibrating the bias buys ~nothing'})")
    print("\n  NB precision is depressed by anything Rubinstein plays that the")
    print("  edition does not notate (rolled chords, pedal resonance) and by our")
    print("  own extra octaves. Recall is the cleaner number.")


if __name__ == "__main__":
    main()
