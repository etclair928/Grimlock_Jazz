# =================================================================
# TOOL: tools/university_report.py
# Runs Grimlock University over a saved pipeline intermediate (.pkl) -
# seconds, no re-transcription - and reports:
#   1. what the curriculum found (STUDY)
#   2. the null-model grade per detector (does it know anything?)
#   3. OFF vs STUDY vs APPLY page deltas, incl. the naked-BP baseline
#      guardrail if a baseline MIDI is supplied
#
#   python tools/university_report.py transcriptions/Hopeful_FULLRUN.pkl \
#          [--baseline transcriptions/..._naked_basic_pitch.mid] [--trials 5]
# =================================================================

from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from university import (  # noqa: E402
    UniversityMode, apply_observations, run_null_model, study,
    write_study_annotations,
)
from output.notation_score import build_routed_score  # noqa: E402


def page_stats(score):
    notes = sum(len(p.notes) for p in score.parts)
    sounding_s = sum(max(0.0, n.end_ms - n.start_ms) for p in score.parts
                     for n in p.notes) / 1000.0
    return {"parts": len(score.parts), "notes": notes,
            "sounding_s": round(sounding_s, 1)}


def build(d, honor):
    return build_routed_score(
        d["all_notes"], d["annotations"], tempo_bpm=d["tempo_bpm"],
        time_signature=d["time_signature"], key=d["key"],
        use_consolidation=True, ratio_family=d["ratio_family"],
        honor_university=honor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--baseline", default=None, help="naked Basic Pitch MIDI")
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    with open(args.pkl, "rb") as fh:
        d = pickle.load(fh)
    song = os.path.basename(args.pkl).replace("_FULLRUN.pkl", "")
    print("=" * 78)
    print(f"{song}   {len(d['all_notes'])} notes   {d['tempo_bpm']:.0f}bpm "
          f"{d['time_signature'][0]}/{d['time_signature'][1]}   key={d['key']}")

    # ---------- OFF (control) ----------
    off = build(copy.deepcopy(d), honor=False)
    print(f"\nOFF   (Grimlock Jazz as-is): {page_stats(off)}")

    # ---------- STUDY ----------
    d_study = copy.deepcopy(d)
    rep = study(d_study["all_notes"], d_study["annotations"],
                tempo_bpm=d_study["tempo_bpm"], mode=UniversityMode.STUDY)
    write_study_annotations(rep, d_study["annotations"])
    st = build(d_study, honor=False)
    print(f"STUDY (observe + log)      : {page_stats(st)}")
    print(f"      observations={len(rep.observations)}  "
          f"coverage={rep.coverage:.1%}  by_pattern={rep.by_pattern()}")
    same = page_stats(st) == page_stats(off)
    print(f"      OUTPUT UNCHANGED vs OFF: {'YES' if same else 'NO  <-- BUG'}")

    # ---------- APPLY ----------
    d_apply = copy.deepcopy(d)
    rep2 = study(d_apply["all_notes"], d_apply["annotations"],
                 tempo_bpm=d_apply["tempo_bpm"], mode=UniversityMode.APPLY)
    write_study_annotations(rep2, d_apply["annotations"])
    counts = apply_observations(rep2, d_apply["annotations"],
                                {n.id: n for n in d_apply["all_notes"]})
    ap_score = build(d_apply, honor=True)
    print(f"APPLY (page honors patterns): {page_stats(ap_score)}   {counts}")

    off_s, ap_s = page_stats(off), page_stats(ap_score)
    if off_s["notes"]:
        delta = (ap_s["notes"] - off_s["notes"]) / off_s["notes"] * 100
        print(f"      page noteheads {off_s['notes']} -> {ap_s['notes']} ({delta:+.1f}%)")

    # ---------- naked-BP baseline guardrail ----------
    if args.baseline and os.path.exists(args.baseline):
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(args.baseline)
        bp_notes = [(n.start, n.end) for inst in pm.instruments for n in inst.notes]
        bp_n = len(bp_notes)
        bp_sound = sum(e - s for s, e in bp_notes)
        print(f"\nBASELINE naked BP: {bp_n} notes, {bp_sound:.0f} sounding-sec")
        for label, sc in (("OFF", off), ("APPLY", ap_score)):
            s = page_stats(sc)
            print(f"  {label:5s} n/BP={s['notes']/bp_n:.2f}  "
                  f"sndT/BP={s['sounding_s']/bp_sound:.2f}")

    # ---------- null model: does each detector know anything? ----------
    print(f"\nNULL MODEL ({args.trials} trials/detector) - "
          f"does the detector beat shuffled notes?")
    grades = run_null_model(d["all_notes"], d["annotations"],
                            tempo_bpm=d["tempo_bpm"], trials=args.trials)
    print(f"  {'detector':16s} {'real':>7} {'null_p':>7} {'null_t':>7} {'lift':>6}  verdict")
    for name, g in sorted(grades.items(), key=lambda kv: -(kv[1]["real"])):
        lift = "  -  " if g["lift"] is None else f"{g['lift']:.2f}"
        print(f"  {name:16s} {g['real']:7.3f} {g['null_pitch_shuffle']:7.3f} "
              f"{g['null_time_shuffle']:7.3f} {lift:>6}  {g['verdict']}")


if __name__ == "__main__":
    main()
