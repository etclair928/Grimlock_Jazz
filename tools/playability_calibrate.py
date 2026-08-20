#!/usr/bin/env python3
# =================================================================
# TOOL: tools/playability_calibrate.py
# THE ACCEPTANCE TEST FOR output/playability.py.
#
# A published edition is BY DEFINITION playable. So the model has exactly
# one non-negotiable property: run it on the edition and it must call
# almost nothing impossible. Every percent it flags there is a percent of
# false positives it would contribute anywhere else, and a filter built on
# top of a model with a floor deletes real music at that rate forever.
#
# This is the same discipline that caught three bugs in tools/tuplet_audit.py
# by making it fail the published edition first. It is run against three
# sources so the number has a scale:
#
#   ANSWER KEY  the floor. Should be ~0%.
#   KLANGIO     a competent commercial transcriber, for context.
#   OURS        whatever excess we show over the edition is ours to explain.
#
# Both timeline modes are printed side by side, because the gap between
# them IS the pedal argument: notes ringing together are not notes held
# together, and a model that cannot tell the difference reports the
# sustain pedal as a hand injury.
#
#   python tools/playability_calibrate.py
# =================================================================

from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JAZZ)

from output.playability import PROFILES, IMPOSSIBLE, ROLLED, assess  # noqa: E402


def load_xml(path):
    """(start, end, pitch) in QUARTER LENGTHS - the module's native unit."""
    from music21 import converter
    score = converter.parse(path)
    out = []
    for n in score.flatten().notes:
        start = float(n.offset)
        end = start + max(float(n.quarterLength), 0.01)
        for p in (n.pitches if n.isChord else [n.pitch]):
            out.append((start, end, int(p.midi)))
    return out


def load_pkl(path):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    bpm = float(data.get("tempo_bpm") or 120.0)
    ms_per_beat = 60000.0 / bpm
    return [(n.start_ms / ms_per_beat, n.end_ms / ms_per_beat, int(n.pitch))
            for n in data["all_notes"] if n.stem.value != "drums"], bpm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answer-key",
                    default=os.path.join(JAZZ, "transcriptions", "Chopin_ANSWER_KEY.musicxml"))
    ap.add_argument("--klangio",
                    default=os.path.join(JAZZ, "transcriptions", "Chopin_KLANGIO.musicxml"))
    ap.add_argument("--ours",
                    default=os.path.join(JAZZ, "transcriptions", "Chopin_44",
                                         "Chopin_Nocturne_62_1.pkl"))
    args = ap.parse_args()

    sources = []
    tempo_bpm = 120.0
    if os.path.exists(args.answer_key):
        sources.append(("ANSWER KEY (edition)", load_xml(args.answer_key)))
    if os.path.exists(args.klangio):
        sources.append(("KLANGIO", load_xml(args.klangio)))
    if os.path.exists(args.ours):
        notes, bpm = load_pkl(args.ours)
        tempo_bpm = bpm
        sources.append((f"OURS (raw, {bpm:.0f}bpm)", notes))

    # The repetition ceiling is expressed in Hz, so it needs a real tempo.
    # All three sources are the SAME PIECE, so our detected tempo is the
    # fair nominal for the two scores too - an edition has no performance
    # tempo of its own, and comparing at different tempi would compare the
    # tempi rather than the writing.
    bps = tempo_bpm / 60.0

    for profile in ("comfortable", "standard", "virtuoso"):
        cfg = dataclasses.replace(PROFILES[profile], beats_per_second=bps)
        print(f"\n=== profile '{profile}': span <= {cfg.max_span_semitones}st, "
              f"inner gap <= {cfg.max_adjacent_gap_semitones}st, "
              f"crossings <= {cfg.max_crossings} ===")
        print(f"{'source':24} {'STRUCK':>22} {'SOUNDING':>14}")
        print(f"{'':24} {'impossible':>11} {'rolled':>10} {'impossible':>14}")
        print("-" * 64)
        for label, notes in sources:
            struck = assess(notes, cfg, mode="struck")
            sounding = assess(notes, cfg, mode="sounding")
            bad_s = struck.sounding_instants - struck.playable_instants
            bad_d = sounding.sounding_instants - sounding.playable_instants
            print(f"{label:24} {100.0 * bad_s / max(struck.sounding_instants, 1):>10.1f}% "
                  f"{100.0 * struck.rolled_instants / max(struck.sounding_instants, 1):>9.1f}% "
                  f"{100.0 * bad_d / max(sounding.sounding_instants, 1):>13.1f}%")

    print("\n--- difficulty (hard, NOT impossible) at profile 'virtuoso' ---")
    cfg = dataclasses.replace(PROFILES["virtuoso"], beats_per_second=bps)
    print(f"  (repetition ceiling {PROFILES['virtuoso'].max_repetition_hz:.0f}Hz "
          f"evaluated at {tempo_bpm:.0f}bpm for every source)")
    print(f"{'source':24} {'leaps':>8} {'repetitions':>13} {'strikes':>9}")
    for label, notes in sources:
        rep = assess(notes, cfg, mode="struck")
        print(f"{label:24} {len(rep.leaps):>8} {len(rep.repetitions):>13} "
              f"{rep.sounding_instants:>9}")

    print("\n  ACCEPTANCE: the edition's STRUCK impossibility is the model's")
    print("  false-positive floor. If it is not near zero, nothing built on")
    print("  this model may be used to remove a note.")


if __name__ == "__main__":
    main()
