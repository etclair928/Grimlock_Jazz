# =================================================================
# MODULE: app/lab.py
# Lab mode's library layer (GRIMLOCK_6.0_FRONTEND_DESIGN.md §4).
#
# The question this half of the app exists to answer is not "is this chart
# good" but "did that change actually move anything" - which is the question
# this project keeps getting wrong by hand. Five separate times in one working
# session a verdict was reversed by re-measuring, and the cause was the same
# every time: a measurement path quietly differing from the shipping path.
#
# So the rules here are narrow and deliberate.
#
#   ONE SOURCE PER NUMBER. Everything comes from app/diagnostics.py, which
#   delegates to the shipped functions. Nothing in this file re-derives a rule.
#
#   A COMPARISON IS BETWEEN TWO RUNS, NOT TWO PATHS. compare() reads two
#   intermediates and reports per-check deltas. It refuses to compare a run
#   whose intermediate predates beat_times_ms against one that has it, because
#   that comparison was made by hand and produced three wrong numbers - the
#   older path engraves on a flat clock and is a different score.
#
#   UI-FREE. tkinter appears nowhere below.
# =================================================================

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.diagnostics import FAIL, INFO, PASS, HealthReport, diagnose
from app.runner import JAZZ_ROOT

TRANSCRIPTIONS = os.path.join(JAZZ_ROOT, "transcriptions")


@dataclass
class RunRecord:
    """One completed run on disk, summarised without loading the whole thing."""
    stem: str
    pkl_path: str
    musicxml_path: Optional[str] = None
    midi_path: Optional[str] = None
    modified: float = 0.0
    tempo_bpm: float = 0.0
    time_signature: Tuple[int, int] = (4, 4)
    key: str = ""
    notes: int = 0
    stems: Dict[str, int] = field(default_factory=dict)
    # Whether the intermediate can reproduce the score the pipeline wrote.
    # False means a re-export builds on a flat clock and will NOT match.
    reproducible: bool = False
    error: str = ""

    @property
    def has_score(self) -> bool:
        return bool(self.musicxml_path and os.path.exists(self.musicxml_path))


def list_runs(folder: str = TRANSCRIPTIONS) -> List[RunRecord]:
    """Every intermediate in `folder`, newest first.

    Reads the pickle for its summary fields only. A pkl that cannot be read is
    listed with its error rather than skipped - a run that half-failed is
    exactly the one someone is looking for.
    """
    out: List[RunRecord] = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".pkl"):
            continue
        stem = name[:-4]
        path = os.path.join(folder, name)
        record = RunRecord(stem=stem, pkl_path=path,
                           modified=os.path.getmtime(path))
        for attr, ext in (("musicxml_path", ".musicxml"), ("midi_path", ".mid")):
            candidate = os.path.join(folder, stem + ext)
            if os.path.exists(candidate):
                setattr(record, attr, candidate)
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            record.tempo_bpm = float(data.get("tempo_bpm") or 0.0)
            ts = data.get("time_signature") or (4, 4)
            record.time_signature = (int(ts[0]), int(ts[1]))
            record.key = str(data.get("key") or "")
            notes = data.get("all_notes") or ()
            record.notes = len(notes)
            counts: Dict[str, int] = {}
            for n in notes:
                counts[n.stem.value] = counts.get(n.stem.value, 0) + 1
            record.stems = counts
            record.reproducible = bool(data.get("beat_times_ms"))
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
        out.append(record)
    out.sort(key=lambda r: r.modified, reverse=True)
    return out


def health_of(record: RunRecord) -> HealthReport:
    return diagnose(record.pkl_path, record.musicxml_path)


# ---------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------

@dataclass
class CheckDelta:
    name: str
    left: Any
    right: Any
    changed: bool
    numeric_delta: Optional[float] = None
    status_left: str = INFO
    status_right: str = INFO


@dataclass
class Comparison:
    left: str
    right: str
    deltas: List[CheckDelta] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def changed(self) -> List[CheckDelta]:
        return [d for d in self.deltas if d.changed]


def compare(left: RunRecord, right: RunRecord) -> Comparison:
    """Per-check deltas between two runs.

    THE WARNING IS THE POINT. Comparing a pre-2026-08-21 intermediate against a
    later one compares two ENGRAVINGS, not two runs: without beat_times_ms a
    re-export builds on a flat isochronous clock from the first note, which on
    Chopin gave 2602 onsets where the pipeline's own file had 2310. Three
    numbers were reported wrongly that way before anyone noticed, so the
    mismatch is surfaced rather than left for the reader to spot.
    """
    out = Comparison(left=left.stem, right=right.stem)
    if left.reproducible != right.reproducible:
        stale = left.stem if not left.reproducible else right.stem
        out.warnings.append(
            f"{stale}'s intermediate predates beat_times_ms, so its score is "
            f"engraved on a flat clock and is NOT the same notation path as "
            f"the other. Re-run it before trusting any notation delta below.")

    lh, rh = health_of(left), health_of(right)
    names = [c.name for c in lh.checks]
    names += [c.name for c in rh.checks if c.name not in names]
    for name in names:
        a, b = lh.get(name), rh.get(name)
        av = a.value if a else None
        bv = b.value if b else None
        delta = None
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)) \
                and not isinstance(av, bool) and not isinstance(bv, bool):
            delta = float(bv) - float(av)
        out.deltas.append(CheckDelta(
            name=name, left=av, right=bv, changed=(av != bv),
            numeric_delta=delta,
            status_left=(a.status if a else INFO),
            status_right=(b.status if b else INFO)))
    return out


# ---------------------------------------------------------------------
# re-export, the fast loop
# ---------------------------------------------------------------------

def reexport(record: RunRecord, out_path: Optional[str] = None,
             **score_options: Any) -> Tuple[str, List[str]]:
    """Rebuild notation from an intermediate. Seconds, not an hour.

    Returns (path, warnings). Refuses nothing, but says plainly when the
    intermediate cannot reproduce the pipeline's own engraving.
    """
    from core.musical_time import MusicalTime
    from output.musicxml_exporter import export_musicxml
    from output.notation_score import build_routed_score

    warnings_out: List[str] = []
    with open(record.pkl_path, "rb") as fh:
        data = pickle.load(fh)

    beats = data.get("beat_times_ms")
    if not beats:
        warnings_out.append(
            "this intermediate predates beat_times_ms: the re-export uses a "
            "flat clock from the first note and will NOT match what the "
            "pipeline engraved. Re-run to compare fairly.")

    score = build_routed_score(
        data["all_notes"], data["annotations"],
        tempo_bpm=data["tempo_bpm"], time_signature=data["time_signature"],
        key=data["key"], use_consolidation=True,
        ratio_family=data.get("ratio_family"),
        musical_time=(MusicalTime.from_beats(beats) if beats else None),
        bar_origin_ms=data.get("bar_origin_ms"),
        **score_options)
    path = out_path or record.musicxml_path or (record.pkl_path[:-4] + ".musicxml")
    export_musicxml(score, path)
    return path, warnings_out


# ---------------------------------------------------------------------
# scoring against a reference
# ---------------------------------------------------------------------

def score_against(record: RunRecord, answer_key_path: str,
                  onset_tolerance_s: float = 0.25,
                  steps: int = 1800) -> Dict[str, Any]:
    """Precision / recall / F1 against a reference transcription.

    Uses tools/score_vs_answer_key.py's own alignment rather than a second
    implementation, for the reason this whole module exists.
    """
    import sys
    tools = os.path.join(JAZZ_ROOT, "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import numpy as np
    from score_vs_answer_key import chroma_seq, dtw_path, load_answer_key

    with open(record.pkl_path, "rb") as fh:
        data = pickle.load(fh)
    notes = [n for n in data["all_notes"] if n.stem.value != "drums"]
    if not notes:
        return {"error": "no pitched notes"}

    key, key_end_q = load_answer_key(answer_key_path)
    ours = [(n.start_ms / 1000.0, n.end_ms / 1000.0, int(n.pitch)) for n in notes]
    ours_end = max(e for _s, e, _p in ours)
    path, _cost = dtw_path(chroma_seq(key, key_end_q, steps),
                           chroma_seq(ours, ours_end, steps), open_end=True)
    kq = np.array([i for i, _j in path], float) / steps * key_end_q
    osec = np.array([j for _i, j in path], float) / steps * ours_end
    covered = float(kq.max())
    ref = [(float(np.interp(s, kq, osec)), p) for s, _e, p in key if s <= covered]

    by_pitch: Dict[int, List[float]] = {}
    for n in sorted(notes, key=lambda n: n.start_ms):
        by_pitch.setdefault(int(n.pitch), []).append(n.start_ms / 1000.0)
    used, hits = set(), 0
    for t, p in ref:
        best, bd = None, onset_tolerance_s
        for j, st in enumerate(by_pitch.get(p, ())):
            if (p, j) in used:
                continue
            if abs(st - t) < bd:
                best, bd = j, abs(st - t)
        if best is not None:
            used.add((p, best))
            hits += 1
    precision = hits / max(len(notes), 1)
    recall = hits / max(len(ref), 1)
    return {
        "hits": hits, "ours": len(notes), "reference": len(ref),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
    }


__all__ = ["RunRecord", "list_runs", "health_of", "CheckDelta", "Comparison",
           "compare", "reexport", "score_against", "TRANSCRIPTIONS"]
