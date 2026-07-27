#!/usr/bin/env python3
# =================================================================
# MODULE: tools/naked_basic_pitch.py
# "Naked" Basic Pitch: audio -> MIDI with ZERO Grimlock. No separation,
# no CREPE, no rhythm engine, no instrument attribution, no quantization,
# no acoustic witnesses - just Spotify's Basic Pitch model run straight on
# the whole mix, its own note post-processor, its own MIDI writer.
#
# This exists as an A/B REFERENCE: it shows what Basic Pitch alone hears,
# so the value the full 6.0 pipeline adds (or subtracts) is visible by
# comparison. Output files are deliberately tagged `_naked_basic_pitch`
# so they can never be confused with a real pipeline transcription
# (`*_jazz*.mid`, `*_6.0*.mid`).
#
# NOT part of the pipeline. NOT imported by anything. A standalone tool.
# Run:  python tools/naked_basic_pitch.py <file-or-folder> [...]
# =================================================================

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# scipy >= 1.13 removed the top-level scipy.signal.gaussian re-export, which
# Basic Pitch 0.3.0's note_creation.get_pitch_bends still calls - without this
# shim every predict() raises AttributeError on this machine's scipy. Pure
# re-export, behaviour-identical. (Same shim Jazz's pitch_engine carries.)
try:  # pragma: no cover - trivial compat shim
    import scipy.signal as _sps
    if not hasattr(_sps, "gaussian"):
        from scipy.signal.windows import gaussian as _gaussian
        _sps.gaussian = _gaussian
except Exception:
    pass

OUTPUT_SUFFIX = "_naked_basic_pitch"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}
# Default output dir = Jazz/transcriptions (sibling of tools/), resolved from
# THIS file so it works no matter what the current working directory is.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "transcriptions"


def _gather_inputs(paths: List[str]) -> List[Path]:
    """Expand each argument (a file or a directory) into a sorted, de-duped
    list of audio files. Directories are scanned one level deep, not
    recursively - keep the tool predictable."""
    found: List[Path] = []
    seen = set()
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            candidates = sorted(c for c in p.iterdir() if c.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            candidates = [p]
        else:
            print(f"  ! not found, skipping: {p}", file=sys.stderr)
            continue
        for c in candidates:
            key = c.resolve()
            if key not in seen:
                seen.add(key)
                found.append(c)
    return found


def _load_model():
    """Load Basic Pitch's ICASSP-2022 model once and reuse it across every
    file in the batch (passing a path to predict() reloads it every call)."""
    from basic_pitch.inference import Model
    from basic_pitch import ICASSP_2022_MODEL_PATH
    return Model(ICASSP_2022_MODEL_PATH)


def transcribe_one(model, audio_path: Path, out_dir: Path, args) -> Path | None:
    """Run Basic Pitch on one file and write <stem>_naked_basic_pitch.mid."""
    from basic_pitch.inference import predict

    out_path = out_dir / f"{audio_path.stem}{OUTPUT_SUFFIX}.mid"
    if out_path.exists() and not args.overwrite:
        print(f"  = exists, skipping (use --overwrite): {out_path.name}")
        return out_path

    t0 = time.time()
    _model_output, midi_data, note_events = predict(
        str(audio_path),
        model,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        minimum_note_length=args.min_note_ms,
        minimum_frequency=args.min_freq,
        maximum_frequency=args.max_freq,
        midi_tempo=args.midi_tempo,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_data.write(str(out_path))
    dt = time.time() - t0
    print(f"  + {audio_path.name}  ->  {out_path.name}   "
          f"({len(note_events)} notes, {dt:.1f}s)")
    return out_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="naked_basic_pitch",
        description="Run audio through Basic Pitch ONLY (no Grimlock) -> MIDI. "
                    "Outputs are tagged _naked_basic_pitch so they are never "
                    "confused with full-pipeline transcriptions.",
    )
    parser.add_argument("inputs", nargs="+",
                        help="Audio file(s) and/or folder(s) of audio.")
    parser.add_argument("-o", "--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"Output folder (default: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-transcribe even if the output MIDI exists.")
    # Basic Pitch's own knobs, defaulted to its library defaults so "naked"
    # really means naked - no ensemble-tuned overrides.
    parser.add_argument("--onset-threshold", type=float, default=0.5,
                        help="Basic Pitch note-onset threshold (default 0.5).")
    parser.add_argument("--frame-threshold", type=float, default=0.3,
                        help="Basic Pitch frame/sustain threshold (default 0.3).")
    parser.add_argument("--min-note-ms", type=float, default=127.7,
                        help="Minimum note length in ms (default 127.7).")
    parser.add_argument("--min-freq", type=float, default=None,
                        help="Minimum detected frequency in Hz (default: none).")
    parser.add_argument("--max-freq", type=float, default=None,
                        help="Maximum detected frequency in Hz (default: none).")
    parser.add_argument("--midi-tempo", type=float, default=120.0,
                        help="Tempo written into the MIDI header (default 120; "
                             "Basic Pitch does not detect tempo).")
    args = parser.parse_args(argv)

    inputs = _gather_inputs(args.inputs)
    if not inputs:
        print("No audio files found.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    print(f"Naked Basic Pitch  |  {len(inputs)} file(s)  ->  {out_dir}")
    print("Loading Basic Pitch model...")
    model = _load_model()

    ok = 0
    for audio_path in inputs:
        try:
            if transcribe_one(model, audio_path, out_dir, args) is not None:
                ok += 1
        except Exception as e:  # keep the batch going if one file is bad
            print(f"  ! FAILED {audio_path.name}: {type(e).__name__}: {e}",
                  file=sys.stderr)

    print(f"\nDone: {ok}/{len(inputs)} written to {out_dir}")
    return 0 if ok == len(inputs) else (2 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
