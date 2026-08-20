#!/usr/bin/env python3
# =================================================================
# TOOL: tools/solo_detector_calibrate.py
# Does the pre-Demucs solo test actually separate solo from ensemble?
#
# The asymmetry that matters: a false SOLO silently throws away real
# instruments and nothing downstream can notice, while a false ENSEMBLE only
# costs time. So the number to watch is not accuracy - it is whether any
# genuine ensemble is ever called solo.
#
# Isolated stems are included as positive controls with known answers: a
# guitar stem MUST read solo_guitar, a drums stem MUST read ensemble. Real
# mixes are the cases that count.
# =================================================================
import os, sys, warnings
warnings.filterwarnings("ignore")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)
import librosa
from separation_engine.solo_detector import analyze_solo, ANALYSIS_SAMPLE_RATE

CASES = [
    ("Chopin FULL (solo piano)",  "Input/Chopin_Full/Chopin_Full.mp3",                 "solo_piano"),
    ("Chopin clip (solo piano)",  "Input/Chopin_Nocturne_62_1/Chopin_Nocturne_62_1.wav","solo_piano"),
    ("Hopeful (full band)",       "Input/Hopeful/Hopeful.mp3",                          "ensemble"),
    ("HRV (full band)",           "Input/Heavy_Rotation_Vibez/Heavy_Rotation_Vibez.mp3","ensemble"),
    # Demucs OUTPUT, not clean recordings. Kept because the finding is worth
    # having: separation artifacts inflate the crescendo measure to 0.38 on
    # every stem, so these read ENSEMBLE. The witness works on real audio and
    # misfires on separated audio - which is fine, since it only ever runs
    # BEFORE separation, but it means stems cannot calibrate it.
    ("-- Demucs stems (not clean audio) --", None,                                       None),
    ("Hopeful guitar stem",       "Input/Hopeful/stems/guitar.wav",                     "solo_guitar"),
    ("Hopeful piano stem",        "Input/Hopeful/stems/piano.wav",                      "solo_piano"),
    ("Hopeful drums stem",        "Input/Hopeful/stems/drums.wav",                      "ensemble"),
    ("Hopeful vocals stem",       "Input/Hopeful/stems/vocals.wav",                     "ensemble"),
]

print(f"{'source':30} {'verdict':>13} {'exp':>13} {'kitHF':>7} {'cresc':>7} {'low':>7}")
print("-" * 84)
wrong_solo = 0
for label, path, expect in CASES:
    if path is None:
        print(f"{label}")
        continue
    full = os.path.join(JAZZ, path)
    if not os.path.exists(full):
        print(f"{label:30} (missing)"); continue
    y, sr = librosa.load(full, sr=ANALYSIS_SAMPLE_RATE, mono=True)
    v = analyze_solo(y, sr)
    ok = "ok" if v.verdict == expect else "**"
    if v.is_solo and expect == "ensemble":
        wrong_solo += 1
        ok = "DANGEROUS"
    print(f"{label:30} {v.verdict:>13} {expect:>13} {v.kit_hf_fraction:>7.3f} "
          f"{v.crescendo_fraction:>7.3f} {v.low_energy_fraction:>7.3f}  {ok}")
print()
print(f"  ensembles wrongly called solo: {wrong_solo}  "
      f"(this is the only number that can lose real music)")
