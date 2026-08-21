# =================================================================
# MODULE: app/worker.py
# The subprocess a run actually happens in.
#
# Deliberately tiny and deliberately separate. It exists so the pipeline gets
# its own process - a Demucs OOM kills this, not the window - and so the UI
# never imports torch, librosa or music21 into its own address space.
#
# Its only job: build the tee'd MusicBox + Window Pane, call transcribe_file,
# and write a JSON result. Every decision the engine logs is mirrored to the
# pane's JSONL stream as it happens, which is what the UI tails.
#
#   python -m app.worker --config cfg.json --stream pane.jsonl --result out.json
# =================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import WindowPane                                   # noqa: E402
from app.runner import PaneMusicBox, RunConfig                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--result", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = RunConfig.from_json(fh.read())
    paths = cfg.paths()

    pane = WindowPane(stream_path=args.stream)
    # buffer_size=1 so the forensic ledger is also current on disk; the run is
    # long enough that a crash at minute forty should not lose minute one.
    box = PaneMusicBox(buffer_size=1, pane=pane)

    result = {"ok": False, "paths": paths}
    try:
        pane.emit("run_start", "conductor", audio=cfg.audio_path,
                  guided_separation=cfg.guided_separation)
        from orchestration.conductor import transcribe_file

        ts = tuple(cfg.guided_time_signature) if cfg.guided_time_signature else None
        res = transcribe_file(
            cfg.audio_path, paths["midi"],
            output_musicxml_path=paths["musicxml"],
            save_intermediate_path=paths["pkl"],
            use_notation_timing=cfg.use_notation_timing,
            use_consolidated_timing=cfg.use_consolidated_timing,
            drop_purge_candidates=cfg.drop_purge_candidates,
            university_mode=cfg.university_mode,
            stem_cache_dir=cfg.stem_cache_dir,
            guided_separation=cfg.guided_separation,
            guided_tempo_bpm=cfg.guided_tempo_bpm,
            guided_time_signature=ts,
            guided_key=cfg.guided_key,
            music_box=box,
        )
        result.update({
            "ok": True,
            "separation_model": res.separation_model,
            "tempo_bpm": res.tempo_bpm,
            "time_signature": list(res.time_signature),
            "note_counts_by_stem": dict(res.note_counts_by_stem),
            "total_notes_exported": res.total_notes_exported,
            "elapsed_seconds": res.elapsed_seconds,
            "key": getattr(res.findings, "key", None) if res.findings else None,
        })
        pane.emit("run_end", "conductor", ok=True,
                  notes=res.total_notes_exported)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        pane.emit("run_end", "conductor", ok=False, error=result["error"])
        traceback.print_exc()
    finally:
        try:
            box.flush()
        except Exception:
            pass
        pane.stop()
        with open(args.result, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
