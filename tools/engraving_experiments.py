# =================================================================
# TOOL: tools/engraving_experiments.py
# Slice 1 experiments (GRIMLOCK_6.0_PERFORMANCE_ENGRAVING.md §11-§13).
#
# Tests the levers the Slice-0 measurement PROMOTED (§12.7): does turning
# the harmonic "junk drawer" into a two-staff GRAND STAFF by register -
# instead of N brightness-family staves - actually read better?
#
# Method: extract the harmonic note stream from a produced page ONCE, then
# re-engrave it three ways through the SAME production exporter
# (output.musicxml_exporter), so chord-aggregation, <=4 voicing, ties and
# voice-legality are identical across candidates and only the STAFF
# ASSIGNMENT differs:
#
#   A  family      - one part per brightness family (today's default)   [control]
#   B  split_fixed - two staves, boundary fixed at middle C (60)        [strawman]
#   C  split_hyst  - two staves, boundary chosen by a Viterbi with a
#                    switch penalty (hysteresis) that keeps each hand
#                    playable and stops the split churning              [proposal]
#
# Then it measures each written page identically: rest/note ratio, staff
# count, max voices/staff, chord rate, and two-hand playability (via
# output.playability). Baselines from §12.7 are the reference.
#
#   python tools/engraving_experiments.py transcriptions/Hopeful_notation.musicxml
#
# It writes the candidate pages next to the source as <stem>_EXP_<X>.musicxml
# so they can be opened, and prints a comparison table. It changes NO
# pipeline code - it is a measurement, per the Slice discipline.
# =================================================================

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import StemType                                    # noqa: E402
from output.notation_score import NotationNote, NotationPart, NotationScore  # noqa: E402
from output.musicxml_exporter import export_musicxml         # noqa: E402
from output.playability import PlayabilityConfig, assess     # noqa: E402
from output.piano_reduction import build_grand_staff         # noqa: E402

_HARMONIC_EXCLUDE = ("drums", "bass", "vocals")


# ---------------------------------------------------------------------------
# The BASELINE: raw stems -> Basic Pitch, no Grimlock. This is ground truth
# for "what the audio actually contains." Every candidate is juxtaposed
# against it; large deviation = we corrupted the transcription, not improved
# it. (This is the standing rule, not a one-off - see doc §17.)
# ---------------------------------------------------------------------------

def load_naked_bp(path: str) -> List[Tuple[float, float, int]]:
    """Load a naked-Basic-Pitch MIDI as (start_s, end_s, midi) in real
    seconds (pretty_midi note times are already real time)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(path)
    return [(float(n.start), float(n.end), int(n.pitch))
            for inst in pm.instruments for n in inst.notes]


def _content_stats(notes_sec: List[Tuple[float, float, int]]) -> Tuple[int, float]:
    """(#notes, total sounding-seconds = sum of durations). Sounding-time is
    the sustain-invention guardrail: legato that over-holds inflates it far
    past the baseline; dropping notes deflates it."""
    n = len(notes_sec)
    sound = sum(max(0.0, e - s) for s, e, _ in notes_sec)
    return n, sound


def _pitch_hist(notes_sec: List[Tuple[float, float, int]]) -> Dict[int, float]:
    from collections import Counter
    c = Counter(p for _, _, p in notes_sec)
    tot = sum(c.values()) or 1
    return {k: v / tot for k, v in c.items()}


def _hist_l1(a: Dict[int, float], b: Dict[int, float]) -> float:
    """0 = identical pitch-content distribution, 1 = disjoint. Timing-robust:
    catches added/dropped pitch content regardless of quantization."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _score_notes_sec(score: NotationScore) -> List[Tuple[float, float, int]]:
    return [(n.start_ms / 1000.0, n.end_ms / 1000.0, n.pitch)
            for p in score.parts for n in p.notes]


@dataclass
class SrcNote:
    start_ms: float
    end_ms: float
    midi: int
    velocity: int
    family: str


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _family_of(part_name: str) -> str:
    return (part_name or "unknown").split(" [")[0].strip() or "unknown"


def extract_harmonic_notes(path: str) -> Tuple[List[SrcNote], float, Tuple[int, int], Optional[str]]:
    """Pull the harmonic parts' notes (absolute beats -> ms via the page's
    own tempo). Returns (notes, tempo_bpm, time_sig, key_str)."""
    from music21 import converter, tempo as m21tempo, meter as m21meter, key as m21key

    score = converter.parse(path)

    bpm = 120.0
    for mm in score.recurse().getElementsByClass(m21tempo.MetronomeMark):
        if mm.number:
            bpm = float(mm.number)
            break
    ts = (4, 4)
    for t in score.recurse().getElementsByClass(m21meter.TimeSignature):
        ts = (t.numerator, t.denominator)
        break
    key_str = None
    for k in score.recurse().getElementsByClass(m21key.Key):
        key_str = k.name
        break

    ms_per_beat = 60000.0 / max(bpm, 1.0)
    notes: List[SrcNote] = []
    for part in score.parts:
        pn = (part.partName or "").lower()
        if any(tok in pn for tok in _HARMONIC_EXCLUDE):
            continue
        fam = _family_of(part.partName)
        for el in part.flatten().notes:
            ql = float(el.quarterLength)
            if ql <= 0:
                continue
            start_ms = float(el.offset) * ms_per_beat
            end_ms = start_ms + ql * ms_per_beat
            vel = 64
            try:
                if el.volume.velocity:
                    vel = int(el.volume.velocity)
            except Exception:
                pass
            for p in (el.pitches if hasattr(el, "pitches") else [el.pitch]):
                notes.append(SrcNote(start_ms, end_ms, int(p.midi), vel, fam))
    return notes, bpm, ts, key_str


# ---------------------------------------------------------------------------
# Register split
# ---------------------------------------------------------------------------

def _to_notation_note(n: SrcNote, i: int) -> NotationNote:
    return NotationNote(
        pitch=n.midi, start_ms=n.start_ms, end_ms=n.end_ms,
        velocity=n.velocity, source_note_id=f"exp_{i}",
    )


def split_fixed(notes: List[SrcNote], boundary: int = 60) -> Dict[str, List[SrcNote]]:
    hands: Dict[str, List[SrcNote]] = {"treble": [], "bass": []}
    for n in notes:
        hands["treble" if n.midi >= boundary else "bass"].append(n)
    return hands


def split_hysteresis(
        notes: List[SrcNote], bpm: float,
        cfg: PlayabilityConfig,
        switch_penalty: float = 0.4,
        b_lo: int = 52, b_hi: int = 74, step: int = 2,
) -> Tuple[Dict[str, List[SrcNote]], List[int]]:
    """Viterbi over a per-beat split boundary. Emission cost = how much a
    boundary makes either hand unplayable (span/finger excess) in that
    beat-window; transition cost = switch_penalty * |Δboundary|, which is
    the hysteresis that stops the split churning."""
    ms_per_beat = 60000.0 / max(bpm, 1.0)
    win_pitches: Dict[int, List[int]] = defaultdict(list)
    for n in notes:
        w0 = int(math.floor(n.start_ms / ms_per_beat))
        w1 = max(w0 + 1, int(math.ceil(n.end_ms / ms_per_beat)))
        for w in range(w0, w1):
            win_pitches[w].append(n.midi)
    windows = sorted(win_pitches.keys())
    if not windows:
        return {"treble": [], "bass": list(notes)}, []
    boundaries = list(range(b_lo, b_hi + 1, step))

    def emit(w: int, b: int) -> float:
        c = 0.0
        for g in ([p for p in win_pitches[w] if p < b],
                  [p for p in win_pitches[w] if p >= b]):
            if not g:
                continue
            span = max(g) - min(g)
            c += max(0, span - cfg.max_span_semitones) * 1.0
            c += max(0, len(set(g)) - cfg.max_fingers_per_hand) * 3.0
        return c

    INF = float("inf")
    dp = {b: emit(windows[0], b) for b in boundaries}
    back: List[Dict[int, int]] = [{}]
    for wi in range(1, len(windows)):
        w = windows[wi]
        ndp: Dict[int, float] = {}
        nback: Dict[int, int] = {}
        for b in boundaries:
            ec = emit(w, b)
            best, bestp = INF, boundaries[0]
            for pb in boundaries:
                cost = dp[pb] + switch_penalty * abs(b - pb) + ec
                if cost < best:
                    best, bestp = cost, pb
            ndp[b], nback[b] = best, bestp
        dp, = (ndp,)
        back.append(nback)

    # backtrack
    b_end = min(dp, key=dp.get)
    chosen = [0] * len(windows)
    chosen[-1] = b_end
    for wi in range(len(windows) - 1, 0, -1):
        chosen[wi - 1] = back[wi][chosen[wi]]
    win_boundary = {w: chosen[i] for i, w in enumerate(windows)}

    hands: Dict[str, List[SrcNote]] = {"treble": [], "bass": []}
    for n in notes:
        w = int(math.floor(n.start_ms / ms_per_beat))
        b = win_boundary.get(w) or win_boundary.get(min(windows, key=lambda x: abs(x - w)))
        hands["treble" if n.midi >= b else "bass"].append(n)
    return hands, chosen


# ---------------------------------------------------------------------------
# Build NotationScore variants
# ---------------------------------------------------------------------------

def _score_from_parts(part_notes: Dict[str, List[SrcNote]], bpm, ts, key,
                      order: List[str]) -> NotationScore:
    parts: List[NotationPart] = []
    i = 0
    for label in order:
        ns = part_notes.get(label, [])
        if not ns:
            continue
        nnotes = []
        for n in ns:
            nnotes.append(_to_notation_note(n, i))
            i += 1
        nnotes.sort(key=lambda x: x.start_ms)
        parts.append(NotationPart(
            family="piano", voice_id=label, stem=StemType.OTHER, notes=nnotes, is_drum=False))
    return NotationScore(parts=parts, tempo_bpm=bpm, time_signature=ts, key=key, ratio_family=None)


def candidate_family(notes, bpm, ts, key) -> NotationScore:
    by_fam: Dict[str, List[SrcNote]] = defaultdict(list)
    for n in notes:
        by_fam[n.family].append(n)
    order = sorted(by_fam.keys(), key=lambda f: -sum(x.midi for x in by_fam[f]) / max(1, len(by_fam[f])))
    return _score_from_parts(by_fam, bpm, ts, key, order)


def candidate_split(hands, bpm, ts, key) -> NotationScore:
    return _score_from_parts(hands, bpm, ts, key, ["treble", "bass"])


def candidate_grandstaff(notes, bpm, ts, key, smooth: bool, fill_max_beats: float = 1.0) -> NotationScore:
    """The implemented production path: register split (hysteresis) + the
    <=4 rhythmic-independence voicer (output.piano_reduction). `smooth`
    adds the §16 de-chop pass; `fill_max_beats` sets legato aggressiveness
    (0 = merge same-pitch only, no gap fill)."""
    nnotes = [_to_notation_note(n, i) for i, n in enumerate(notes)]
    return build_grand_staff(nnotes, bpm, ts, key, smooth=smooth, fill_max_beats=fill_max_beats)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class Measure:
    parts: int
    notes: int
    rests: int
    chord_member_notes: int          # noteheads stacked into a chord (proxy for aggregation)
    max_voices: int
    play_time: float                 # combined two-hand playability (by time)
    play_worst_staff: float          # per-staff worst playability (by time)

    @property
    def rest_ratio(self) -> float:
        return self.rests / self.notes if self.notes else 0.0

    @property
    def chord_share(self) -> float:
        return self.chord_member_notes / self.notes if self.notes else 0.0


def measure(path: str, staff_triples: Dict[str, List[Tuple[float, float, int]]]) -> Measure:
    """Counts from the written MusicXML (ElementTree = fast, robust);
    playability computed from the in-memory per-staff assignments (no
    re-parse) - worst single staff, and the union of all staves."""
    root = ET.parse(path).getroot()
    parts = root.findall(".//part")
    total_notes = total_rests = chord_members = 0
    max_voices = 0
    for p in parts:
        pnotes = p.findall(".//note")
        total_rests += sum(1 for n in pnotes if n.find("rest") is not None)
        pitched = [n for n in pnotes if n.find("rest") is None]
        total_notes += len(pitched)
        chord_members += sum(1 for n in pitched if n.find("chord") is not None)
        max_voices = max(max_voices, len(set(v.text for v in p.iter("voice") if v.text)))

    cfg = PlayabilityConfig(max_span_semitones=14)
    worst = 1.0
    union: List[Tuple[float, float, int]] = []
    for trips in staff_triples.values():
        if trips:
            worst = min(worst, assess(trips, cfg).time_pass_rate)
            union.extend(trips)
    combined = assess(union, cfg).time_pass_rate if union else 1.0

    return Measure(parts=len(parts), notes=total_notes, rests=total_rests,
                   chord_member_notes=chord_members, max_voices=max_voices,
                   play_time=combined, play_worst_staff=worst)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_file(path: str, baseline_notes: Optional[List[Tuple[float, float, int]]] = None) -> None:
    notes, bpm, ts, key = extract_harmonic_notes(path)
    stem = os.path.splitext(path)[0]
    print("=" * 100)
    print(f"{os.path.basename(path)}  |  {len(notes)} harmonic note-events  |  {bpm:.0f}bpm  {ts[0]}/{ts[1]}  key={key}")

    bp_n = bp_sound = 0.0
    bp_hist: Dict[int, float] = {}
    if baseline_notes:
        bp_n, bp_sound = _content_stats(baseline_notes)
        bp_hist = _pitch_hist(baseline_notes)
        print(f"  BASELINE (naked Basic Pitch on stem): {bp_n} notes, {bp_sound:.0f} sounding-sec  "
              f"<- ground truth; deviation from this = off")

    cfg = PlayabilityConfig(max_span_semitones=14)
    variants = {
        "A_family": candidate_family(notes, bpm, ts, key),
        "B_split_fixed": candidate_split(split_fixed(notes, 60), bpm, ts, key),
        "C_split_hyst": candidate_split(split_hysteresis(notes, bpm, cfg)[0], bpm, ts, key),
        "D_grandstaff": candidate_grandstaff(notes, bpm, ts, key, smooth=False),
        "E_smooth_1.0": candidate_grandstaff(notes, bpm, ts, key, smooth=True, fill_max_beats=1.0),
        "F_smooth_0.5": candidate_grandstaff(notes, bpm, ts, key, smooth=True, fill_max_beats=0.5),
        "G_merge_only": candidate_grandstaff(notes, bpm, ts, key, smooth=True, fill_max_beats=0.0),
    }

    header = (f"  {'candidate':16s} {'staves':>6} {'notes':>6} {'rest/note':>9} {'maxV':>4} "
              f"{'play(all)':>9}")
    if baseline_notes:
        header += f"  |  {'n/BP':>6} {'sndT/BP':>7} {'pitchL1':>8}"
    print(header)
    for name, score in variants.items():
        out = f"{stem}_EXP_{name}.musicxml"
        try:
            staff_triples = {
                p.voice_id: [(n.start_ms, n.end_ms, n.pitch) for n in p.notes]
                for p in score.parts
            }
            export_musicxml(score, out)
            m = measure(out, staff_triples)
            line = (f"  {name:16s} {m.parts:6d} {m.notes:6d} {m.rest_ratio:9.2f} {m.max_voices:4d} "
                    f"{m.play_time*100:8.1f}%")
            if baseline_notes:
                cn, csound = _content_stats(_score_notes_sec(score))
                dpitch = _hist_l1(_pitch_hist(_score_notes_sec(score)), bp_hist)
                n_ratio = cn / bp_n if bp_n else 0.0
                s_ratio = csound / bp_sound if bp_sound else 0.0
                flag = "  <-- OFF" if (s_ratio > 1.6 or s_ratio < 0.5 or dpitch > 0.35) else ""
                line += f"  |  {n_ratio:6.2f} {s_ratio:7.2f} {dpitch:8.2f}{flag}"
            print(line)
        except Exception as e:
            print(f"  {name:16s}  ERROR: {e}")
    print()


def main() -> None:
    import glob
    ap = argparse.ArgumentParser(description="Grand-staff register-split experiments.")
    ap.add_argument("paths", nargs="+", help="Source MusicXML page(s) / glob(s).")
    ap.add_argument("--baseline", default=None,
                    help="Naked Basic Pitch MIDI of the source stem - the ground-truth "
                         "baseline every candidate is juxtaposed against (doc §17).")
    args = ap.parse_args()

    baseline_notes = None
    if args.baseline and os.path.exists(args.baseline):
        baseline_notes = load_naked_bp(args.baseline)

    files: List[str] = []
    for pat in args.paths:
        matched = glob.glob(pat)
        files.extend(matched if matched else [pat])
    for f in sorted(set(files)):
        if os.path.exists(f):
            run_file(f, baseline_notes)
        else:
            print(f"(missing) {f}", file=sys.stderr)


if __name__ == "__main__":
    main()
