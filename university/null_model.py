# =================================================================
# MODULE: university/null_model.py
# Falsification (GRIMLOCK_UNIVERSITY.md §7).
#
#   "A pattern detector that fires as often on shuffled notes as on real
#    music is detecting nothing."
#
# This is the layer's answer to the project's oldest open problem (§VI:
# "we are optimizing without an accessible loss function"). It needs NO
# ground truth and NO labels: destroy the structure a detector claims to
# find, keep everything else, and see whether it still fires.
#
#   pitch_shuffle - permute pitches, keep rhythm  -> kills melodic shape
#   time_shuffle  - permute onsets, keep pitches  -> kills rhythmic shape
#
# Outcome taxonomy (the point of the whole exercise):
#   real ~ 0        -> DEAD detector (the pasted proposal shipped four)
#   real ~ null     -> NOISE: it fires on anything. Delete it.
#   real >> null    -> REAL structure. It earned its place.
#   real ~ 1.0      -> TOO LOOSE: it covers everything. Tighten it.
#
# The shuffled note stand-ins built here are throwaway measurement
# objects - never Notes, never returned to the pipeline, never annotated.
# =================================================================

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from university.detectors import ALL_DETECTOR_NAMES, DETECTORS, VERTICAL_DETECTORS

# Detectors not in the enrolled curriculum but still graded, so the
# demotion/diagnostic decisions can be re-checked on new material.
_EXTRA_GRADED = {'pedal_point', 'rearticulation'}

# WHICH NULL IS VALID FOR WHICH DETECTOR.
#
# A null model is only evidence if it DESTROYS THE STRUCTURE THE DETECTOR
# CLAIMS. Time-shuffle permutes rhythm and leaves the pitch sequence
# intact, so it cannot break a purely melodic pattern - a scale run
# survives it by construction. Scoring scale_run against time-shuffle
# therefore always yields lift 1.0 and a bogus "NOISE" verdict (observed
# 2026-08-06 on Hopeful: real 0.034 vs pitch-shuffle 0.004 = 8.5x REAL,
# but time-shuffle 0.034 = 1.0 masked it). So each detector declares the
# nulls that actually apply to it, and is graded only against those.
_VALID_NULLS = {
    "neighbor_tone":  ("pitch",),
    "gap_fill":       ("pitch",),
    "chord":          ("pitch", "time"),
    "scale_run":      ("pitch",),          # melodic contour only
    "arpeggio":       ("pitch",),          # melodic/harmonic shape only
    "sequence":       ("pitch",),          # interval contour only
    "alberti_bass":   ("pitch",),          # pitch-shape figure
    "ostinato":       ("pitch",),          # exact pitch repetition
    "pedal_point":    ("pitch", "time"),   # same pitch AND persistence in beats
    "rearticulation": ("pitch", "time"),   # same pitch AND re-strike rate
}


@dataclass
class _Stand:
    """A minimal note-shaped object for null trials. Deliberately NOT a
    core.Note: nothing here may ever be mistaken for pipeline data."""
    id: str
    pitch: int
    start_ms: float
    end_ms: float
    confidence: float = 1.0


def _to_stands(stream: Sequence) -> List[_Stand]:
    return [_Stand(id=n.id, pitch=int(n.pitch), start_ms=float(n.start_ms),
                   end_ms=float(n.end_ms),
                   confidence=float(getattr(n, "confidence", 1.0)))
            for n in stream]


def pitch_shuffle(stream: List[_Stand], rng: random.Random) -> List[_Stand]:
    """Keep every onset/duration; permute which pitch sits where."""
    pitches = [s.pitch for s in stream]
    rng.shuffle(pitches)
    return [_Stand(s.id, p, s.start_ms, s.end_ms, s.confidence)
            for s, p in zip(stream, pitches)]


def time_shuffle(stream: List[_Stand], rng: random.Random) -> List[_Stand]:
    """Keep the pitch sequence; permute the inter-onset intervals and
    durations, so rhythmic shape is destroyed but melody survives."""
    if len(stream) < 2:
        return list(stream)
    iois = [stream[i + 1].start_ms - stream[i].start_ms for i in range(len(stream) - 1)]
    durs = [s.end_ms - s.start_ms for s in stream]
    rng.shuffle(iois)
    rng.shuffle(durs)
    out: List[_Stand] = []
    t = stream[0].start_ms
    for i, s in enumerate(stream):
        d = max(1.0, durs[i])
        out.append(_Stand(s.id, s.pitch, t, t + d, s.confidence))
        if i < len(iois):
            t += max(1.0, iois[i])
    return out


def _coverage(observations, total: int) -> float:
    if not total:
        return 0.0
    covered = set()
    for obs in observations:
        covered.update(obs.note_ids)
    return len(covered) / total


def run_null_model(
        notes: List,
        annotations,
        tempo_bpm: float,
        trials: int = 5,
        seed: int = 0,
        use_consolidation: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Per-detector real-vs-null coverage. Returns
    {detector: {real, null_pitch, null_time, lift, verdict}}."""
    from university.study import _consolidation_absorbed, _departments_for, _family_of
    from university.streams import extract_lines

    rng = random.Random(seed)
    beat_ms = 60000.0 / max(float(tempo_bpm), 1.0)

    by_stem: Dict[Any, List] = defaultdict(list)
    for note in notes:
        stem_value = getattr(note.stem, "value", str(note.stem))
        if stem_value == "drums":
            continue
        if use_consolidation and _consolidation_absorbed(annotations, note.id):
            continue
        by_stem[note.stem].append(note)

    # (stream, allowed detector names) pairs, built once and reused for
    # every trial so real and null see identical stream segmentation.
    prepared: List = []
    prepared_stems: List = []          # whole-stem stands, for VERTICAL detectors
    for stem, stem_notes in by_stem.items():
        if len(stem_notes) < 4:
            continue
        stem_value = getattr(stem, "value", str(stem))
        family = _family_of(annotations, stem_notes[0].id)
        names = _departments_for(stem_value, family)
        if len(stem_notes) >= 4:
            prepared_stems.append(
                (_to_stands(sorted(stem_notes, key=lambda n: n.start_ms)), stem_value))
        for line in extract_lines(stem_notes, beat_ms):
            stream = sorted(line.notes, key=lambda n: n.start_ms)
            if len(stream) >= 4:
                prepared.append((_to_stands(stream), names, stem_value))

    total_notes = sum(len(s) for s, _, _ in prepared)
    total_stem_notes = sum(len(s) for s, _ in prepared_stems)
    # Grade EVERYTHING, including demoted/diagnostic detectors - the
    # demotion decision must stay re-measurable, not baked in.
    all_names = sorted(set(ALL_DETECTOR_NAMES))
    results: Dict[str, Dict[str, Any]] = {}

    for name in all_names:
        detector = DETECTORS[name]

        is_vertical = name in VERTICAL_DETECTORS

        def sweep(transform=None) -> float:
            obs = []
            if is_vertical:
                # A vertical detector reads simultaneity across a whole stem;
                # grading it on monophonic lines would always score zero and
                # silently pass an ungraded detector into the curriculum.
                for stem_stream, stem_value in prepared_stems:
                    s = transform(stem_stream, rng) if transform else stem_stream
                    try:
                        obs.extend(detector(s, beat_ms, stem_value))
                    except Exception:
                        continue
                return _coverage(obs, total_stem_notes)
            for stream, names, stem_value in prepared:
                if name not in names and name not in _EXTRA_GRADED:
                    continue
                s = transform(stream, rng) if transform else stream
                try:
                    obs.extend(detector(s, beat_ms, stem_value))
                except Exception:
                    continue
            return _coverage(obs, total_notes)

        real = sweep()
        null_p = sum(sweep(pitch_shuffle) for _ in range(trials)) / max(1, trials)
        null_t = sum(sweep(time_shuffle) for _ in range(trials)) / max(1, trials)

        # Grade only against nulls that genuinely destroy this detector's
        # claimed structure (see _VALID_NULLS). Among those, the STRONGEST
        # is the honest bar.
        valid = _VALID_NULLS.get(name, ("pitch", "time"))
        candidates = ([null_p] if "pitch" in valid else []) + \
                     ([null_t] if "time" in valid else [])
        null = max(candidates) if candidates else max(null_p, null_t)

        if real < 1e-6:
            verdict = "DEAD"
        elif real > 0.95:
            verdict = "TOO_LOOSE"
        elif null > 1e-9 and real / null < 1.5:
            verdict = "NOISE"
        else:
            verdict = "REAL"

        results[name] = {
            "real": round(real, 4),
            "null_pitch_shuffle": round(null_p, 4),
            "null_time_shuffle": round(null_t, 4),
            "graded_against": "+".join(valid),
            "lift": round(real / null, 2) if null > 1e-9 else None,
            "verdict": verdict,
        }

    return results


__all__ = ["run_null_model", "pitch_shuffle", "time_shuffle"]
