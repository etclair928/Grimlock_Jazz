# =================================================================
# TOOL: tools/run_hrv_pipeline_only.py
# Lean retry for Heavy Rotation Vibez: ONLY the full Jazz pipeline
# (University=APPLY) + the University-OFF page from the same run.
# Stems and naked Basic Pitch are already done, so this skips them.
#
# The first attempt died with numpy ArrayMemoryError inside
# rhythm_engine.drums (a 334 MiB complex128 STFT) - not because the song
# is long (4.1 min, SHORTER than Hopeful's 4.9) but because only 0.6 GB
# of 13.9 GB was free. It needs headroom, not a code change.
# =================================================================
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

JAZZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JAZZ))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SONG = "Heavy_Rotation_Vibez"
AUDIO = JAZZ / "Input" / SONG / f"{SONG}.mp3"
OUT = JAZZ / "transcriptions"

try:
    import psutil
    avail = psutil.virtual_memory().available / 2 ** 30
    print(f"[mem] {avail:.1f} GB available", flush=True)
    if avail < 3.0:
        print("[mem] WARNING: under 3 GB free - the drums stage needs headroom "
              "and will likely fail again. Free some memory and re-run.", flush=True)
except Exception:
    pass

t0 = time.time()
r = subprocess.run([sys.executable, str(JAZZ / "tools" / "run_full.py"),
                    str(AUDIO), f"{SONG}_UNI", "apply"], cwd=str(JAZZ))
print(f"[pipeline] exit={r.returncode} in {time.time() - t0:.0f}s", flush=True)

pkl = OUT / f"{SONG}_UNI_FULLRUN.pkl"
if r.returncode == 0 and pkl.exists():
    from output.notation_score import build_routed_score
    from output.musicxml_exporter import export_musicxml
    d = pickle.load(open(pkl, "rb"))
    score = build_routed_score(
        d["all_notes"], d["annotations"], tempo_bpm=d["tempo_bpm"],
        time_signature=d["time_signature"], key=d["key"],
        use_consolidation=True, ratio_family=d["ratio_family"],
        honor_university=False,          # Jazz alone
    )
    out = OUT / f"{SONG}_JAZZ_ONLY.musicxml"
    export_musicxml(score, str(out))
    print(f"[jazz-only] parts={len(score.parts)} notes={score.total_notes} -> {out}", flush=True)
else:
    print("[jazz-only] skipped - pipeline did not produce an intermediate", file=sys.stderr)
print("DONE", flush=True)
