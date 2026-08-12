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
    from output.notation_score import build_routed_score
    score = build_routed_score(
        d["all_notes"], d["annotations"], tempo_bpm=d["tempo_bpm"],
        time_signature=d["time_signature"], key=d["key"],
        use_consolidation=True, ratio_family=d["ratio_family"])

export_musicxml(score, args.out)
print(f"parts={len(score.parts)} notes={score.total_notes} -> {args.out}")
for p in score.parts:
    print(f"  {p.family} {p.voice_id}: {len(p.notes)} notes")
