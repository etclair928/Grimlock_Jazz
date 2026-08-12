# =================================================================
# TOOL: tools/separation_quality.py
# An INTRINSIC scorecard for a set of stems - no ground truth required.
#
# WHY INTRINSIC. Comparing two separators (SUNO vs htdemucs_6s) the obvious
# way needs reference stems nobody has. What we CAN measure, from the stems
# alone, is whether each stem contains the KIND of signal its label claims
# and how much of its neighbours leaked in. Both separators get scored by
# the identical procedure, so the numbers are comparable even though
# neither is scored against truth.
#
# Deliberately not a single number: separators trade leakage against
# artifacts, and a scalar would hide which way a given one traded.
#
# THE AXES
#   percussive_ratio  HPSS energy split. A drums stem should be percussive-
#                     dominant; bass/brass/piano harmonic-dominant. A drums
#                     stem full of harmonic energy is carrying pitched bleed.
#   onset_leak        Fraction of THIS stem's onsets that coincide (<=30ms)
#                     with a drums onset. High on a pitched stem = kick and
#                     snare transients bled through.
#   spec_corr         Correlation of log-mel spectrograms between each pair.
#                     Two stems of genuinely different sources decorrelate;
#                     high correlation means shared content in both.
#   band profile      Energy per octave band. Catches a "bass" stem with
#                     cymbal energy, or a lead stem with sub content.
#   silence           Fraction of frames below -50 dBFS, plus whether the
#                     stem is empty outright (an unfilled taxonomy slot).
# =================================================================

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

SR = 22050
HOP = 512
ONSET_TOLERANCE_S = 0.030
SILENCE_DBFS = -50.0
BANDS = [(20, 80), (80, 250), (250, 800), (800, 2500), (2500, 8000), (8000, 11000)]
BAND_LABELS = ["sub", "low", "lomid", "mid", "hi", "air"]


def _load(directory: str) -> Dict[str, np.ndarray]:
    import librosa
    stems = {}
    for f in sorted(os.listdir(directory)):
        if not f.lower().endswith((".wav", ".mp3", ".flac")):
            continue
        name = os.path.splitext(f)[0].lower()
        y, _ = librosa.load(os.path.join(directory, f), sr=SR, mono=True)
        stems[name] = y
    return stems


def measure(directory: str) -> Dict[str, dict]:
    import librosa
    stems = _load(directory)
    if not stems:
        raise SystemExit(f"no audio files in {directory}")

    drums = stems.get("drums")
    drum_onsets = (librosa.onset.onset_detect(y=drums, sr=SR, hop_length=HOP, units="time")
                   if drums is not None else np.array([]))

    out: Dict[str, dict] = {}
    mels: Dict[str, np.ndarray] = {}
    for name, y in stems.items():
        rms = float(np.sqrt(np.mean(y ** 2)))
        frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
        silent = float(np.mean(20 * np.log10(frame_rms + 1e-12) < SILENCE_DBFS))
        empty = float(np.abs(y).max()) < 1e-3

        rec = {"rms": rms, "silence": silent, "empty": empty,
               "dur": len(y) / SR}

        if not empty:
            h, p = librosa.effects.hpss(y)
            he, pe = float(np.sum(h ** 2)), float(np.sum(p ** 2))
            rec["percussive_ratio"] = pe / (he + pe + 1e-12)

            on = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP, units="time")
            if len(on) and len(drum_onsets) and name != "drums":
                near = [np.min(np.abs(drum_onsets - t)) for t in on]
                rec["onset_leak"] = float(np.mean(np.array(near) <= ONSET_TOLERANCE_S))
            rec["onsets"] = int(len(on))

            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
            freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
            tot = S.sum() + 1e-12
            rec["bands"] = [float(S[(freqs >= lo) & (freqs < hi)].sum() / tot)
                            for lo, hi in BANDS]
            mels[name] = librosa.power_to_db(
                librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=64))
        out[name] = rec

    # pairwise spectral correlation
    pairs = {}
    keys = sorted(mels)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            L = min(mels[a].shape[1], mels[b].shape[1])
            x = mels[a][:, :L].ravel(); z = mels[b][:, :L].ravel()
            x = x - x.mean(); z = z - z.mean()
            pairs[f"{a}|{b}"] = float(np.dot(x, z) / (np.linalg.norm(x) * np.linalg.norm(z) + 1e-12))
    return {"stems": out, "pairs": pairs}


def format_report(label: str, res: dict) -> str:
    lines = [f"=== {label} ==="]
    lines.append(f"{'stem':11s} {'dur':>7s} {'rms':>7s} {'silent':>7s} {'perc%':>6s} "
                 f"{'onsets':>7s} {'drmleak':>8s}   band profile (sub low lomid mid hi air)")
    for name, r in sorted(res["stems"].items()):
        if r["empty"]:
            lines.append(f"{name:11s} {r['dur']:7.1f} {'--':>7s} {'EMPTY (unfilled slot)':>7s}")
            continue
        bands = " ".join(f"{b*100:4.1f}" for b in r["bands"])
        leak = f"{r['onset_leak']*100:7.1f}%" if "onset_leak" in r else "      -"
        lines.append(f"{name:11s} {r['dur']:7.1f} {r['rms']:7.4f} {r['silence']*100:6.1f}% "
                     f"{r['percussive_ratio']*100:5.1f}% {r['onsets']:7d} {leak:>8s}   {bands}")
    lines.append("\n  pairwise log-mel correlation (high = shared content in both):")
    for k, v in sorted(res["pairs"].items(), key=lambda kv: -kv[1]):
        flag = "  <-- BLEED" if v > 0.5 else ""
        lines.append(f"    {k:28s} {v:+.3f}{flag}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="one or more stem directories to score")
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.dirs]
    for d, lab in zip(args.dirs, labels):
        print(format_report(lab, measure(d)))
        print()


if __name__ == "__main__":
    main()
