# =================================================================
# TOOL: tools/run_hrv_analysis.py
# Unattended driver for the Heavy Rotation Vibez comparison set.
# Serialized on purpose - two heavy pipelines at once exhausted RAM
# before (madmom RNN ArrayMemoryError), so nothing here overlaps.
#
# Produces, in order:
#   1. Input/Heavy_Rotation_Vibez/stems/*.wav   (htdemucs_6s, seed 0)
#   2. naked Basic Pitch on the full mix AND on each stem
#   3. ONE full Jazz pipeline run with University=APPLY
#   4. the University-OFF page, re-exported from that run's intermediate
#
# WHY ONE PIPELINE RUN FOR BOTH JAZZ AND JAZZ+UNIVERSITY:
# the University stage runs LAST (after consolidation, before export) and
# only ADDS annotations of its own kinds. The notes, every upstream
# annotation, and the MIDI are therefore identical whether it ran or not -
# already verified this session (OFF and STUDY produce a byte-identical
# page). So the honest OFF page is the same intermediate exported with
# honor_university=False, and paying ~80 minutes to re-derive identical
# notes would buy nothing. Both pages come from the same frozen notes,
# which is exactly what makes them comparable.
# =================================================================
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

JAZZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JAZZ))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SONG = "Heavy_Rotation_Vibez"
AUDIO = JAZZ / "Input" / SONG / f"{SONG}.mp3"
STEMS = JAZZ / "Input" / SONG / "stems"
OUT = JAZZ / "transcriptions"
PY = sys.executable


def run(label, args):
    print(f"\n{'=' * 70}\n[{label}] {' '.join(str(a) for a in args)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + [str(a) for a in args], cwd=str(JAZZ))
    print(f"[{label}] exit={r.returncode} in {time.time() - t0:.0f}s", flush=True)
    return r.returncode


# ---- 1. stems -------------------------------------------------------
if not STEMS.exists() or not any(STEMS.glob("*.wav")):
    run("stems", [JAZZ / "tools" / "build_stem_cache.py", SONG])
else:
    print(f"[stems] already present: {sorted(p.name for p in STEMS.glob('*.wav'))}", flush=True)

# ---- 2. naked Basic Pitch (mix + every stem) ------------------------
naked_inputs = [AUDIO]
if STEMS.exists():
    naked_inputs += sorted(STEMS.glob("*.wav"))
run("naked_bp", [JAZZ / "tools" / "naked_basic_pitch.py"] + naked_inputs + ["--overwrite"])

# ---- 3. full pipeline, University APPLY ------------------------------
run("jazz+university", [JAZZ / "tools" / "run_full.py", AUDIO, f"{SONG}_UNI", "apply"])

# ---- 4. University-OFF page from the SAME intermediate ---------------
pkl = OUT / f"{SONG}_UNI_FULLRUN.pkl"
if pkl.exists():
    import pickle
    from output.notation_score import build_routed_score
    from output.musicxml_exporter import export_musicxml

    d = pickle.load(open(pkl, "rb"))
    score = build_routed_score(
        d["all_notes"], d["annotations"], tempo_bpm=d["tempo_bpm"],
        time_signature=d["time_signature"], key=d["key"],
        use_consolidation=True, ratio_family=d["ratio_family"],
        honor_university=False,          # <- Jazz alone
    )
    out = OUT / f"{SONG}_JAZZ_ONLY.musicxml"
    export_musicxml(score, str(out))
    print(f"\n[jazz-only page] parts={len(score.parts)} notes={score.total_notes} -> {out}", flush=True)
else:
    print(f"[jazz-only page] MISSING intermediate {pkl}", file=sys.stderr)

print("\nALL DONE", flush=True)
