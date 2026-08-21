# Re-export notation from a saved pipeline intermediate (notes+annotations)
# WITHOUT re-running separation/Basic Pitch/witnesses. Seconds, not an hour.
#   python tools/reexport_notation.py <intermediate.pkl> <out.musicxml> [--family]
import sys, os, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from output.musicxml_exporter import export_musicxml

ap = argparse.ArgumentParser()
ap.add_argument("pkl")
ap.add_argument("out")
ap.add_argument("--family", action="store_true",
                help="Use the old one-staff-per-family layout instead of per-stem routing.")
args = ap.parse_args()

with open(args.pkl, "rb") as fh:
    d = pickle.load(fh)

if args.family:
    from output.notation_score import build_notation_score
    score = build_notation_score(
        d["all_notes"], d["annotations"], tempo_bpm=d["tempo_bpm"],
        time_signature=d["time_signature"], key=d["key"],
        use_voices=False, use_consolidation=True, ratio_family=d["ratio_family"])
else:
    # THE GRID AND THE BAR ORIGIN MATTER, and leaving them out made this tool
    # quietly produce a DIFFERENT score from the one the pipeline ships. On
    # Chopin the pipeline's own file had 2310 onsets and this tool's re-export
    # of its own pkl had 2602 - so every notation number measured through here
    # was measured on a path the pipeline does not use. Intermediates written
    # before 2026-08-21 do not carry these keys; the warning below says so
    # rather than silently reverting to the old, divergent behaviour.
    from core.musical_time import MusicalTime
    from output.notation_score import build_routed_score

    beats = d.get("beat_times_ms")
    musical_time = MusicalTime.from_beats(beats) if beats else None
    bar_origin_ms = d.get("bar_origin_ms")
    if musical_time is None:
        print("  WARNING: this intermediate predates beat_times_ms/bar_origin_ms. "
              "The re-export will use a flat clock from the first note and will "
              "NOT match what the pipeline engraves. Re-run to compare fairly.")

    score = build_routed_score(
        d["all_notes"], d["annotations"], tempo_bpm=d["tempo_bpm"],
        time_signature=d["time_signature"], key=d["key"],
        use_consolidation=True, ratio_family=d["ratio_family"],
        musical_time=musical_time, bar_origin_ms=bar_origin_ms)

export_musicxml(score, args.out)
print(f"parts={len(score.parts)} notes={score.total_notes} -> {args.out}")
for p in score.parts:
    print(f"  {p.family} {p.voice_id}: {len(p.notes)} notes")
