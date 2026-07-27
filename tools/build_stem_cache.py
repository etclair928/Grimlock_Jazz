#!/usr/bin/env python3
# =================================================================
# MODULE: tools/build_stem_cache.py
# One-off TEST-ASSET builder: pre-separates every song in Input/<name>/
# into its 6 htdemucs_6s stems, saved to Input/<name>/stems/. Purely a
# convenience cache so we don't re-run Demucs by hand for stem-level
# analysis - it does NOT feed the pipeline and changes nothing about how
# the pipeline behaves (the Conductor still separates from scratch on every
# real run). Uses the SAME separator + seed as the pipeline (htdemucs_6s,
# seed=0) so the cached stems match what a run would produce.
#
# Run:  python tools/build_stem_cache.py            # all songs, skip done
#       python tools/build_stem_cache.py Hopeful    # just one
# =================================================================
from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path

JAZZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JAZZ))
INPUT_DIR = JAZZ / "Input"
STEM_SR = 44100  # htdemucs native rate; store faithful full-rate stems

import numpy as np
import soundfile as sf
from audio_engine import AudioEngine
from separation_engine.demucs_engine import separate

EXPECTED_STEMS = 6


def _song_dirs(argv):
    if argv:
        return [INPUT_DIR / a for a in argv]
    return sorted(d for d in INPUT_DIR.iterdir() if d.is_dir())


def _source_audio(song_dir: Path):
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        hits = sorted(song_dir.glob(f"*{ext}"))
        if hits:
            return hits[0]
    return None


def build_one(engine, song_dir: Path) -> bool:
    stems_dir = song_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    existing = list(stems_dir.glob("*.wav"))
    if len(existing) >= EXPECTED_STEMS:
        print(f"= {song_dir.name}: {len(existing)} stems already cached, skipping")
        return True
    src = _source_audio(song_dir)
    if src is None:
        print(f"! {song_dir.name}: no source audio found, skipping")
        return False

    print(f"+ {song_dir.name}: separating {src.name} ...", flush=True)
    t0 = time.time()
    track = engine.decode(str(src))
    sep = separate(engine, track, model_name="htdemucs_6s", device="cpu", seed=0)
    for stem_type, stem_track in sep.stems.items():
        name = getattr(stem_type, "value", str(stem_type)).lower()
        view = engine.view(stem_track, STEM_SR)
        sf.write(str(stems_dir / f"{name}.wav"), np.asarray(view.samples), STEM_SR)
    dt = time.time() - t0
    print(f"  -> {len(sep.stems)} stems in {stems_dir}  ({dt/60:.1f} min)", flush=True)
    return True


def main(argv):
    dirs = _song_dirs(argv)
    print(f"Stem cache build | {len(dirs)} song(s) | seed=0 htdemucs_6s -> Input/<name>/stems/")
    engine = AudioEngine()
    ok = 0
    for d in dirs:
        try:
            if build_one(engine, d):
                ok += 1
        except Exception as e:
            print(f"! FAILED {d.name}: {type(e).__name__}: {e}", flush=True)
    print(f"\nDone: {ok}/{len(dirs)} songs cached.")
    return 0 if ok == len(dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
