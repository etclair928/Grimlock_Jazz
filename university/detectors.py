# =================================================================
# MODULE: university/detectors.py
# The curriculum: Department 1 (universal) + a small keyboard and
# rhythmic department. GRIMLOCK_UNIVERSITY.md §6.
#
# LAWS every detector here obeys:
#   1. Reads Notes; NEVER constructs, mutates or replaces one.
#   2. Reports REAL note ids (observation_types.PatternObservation).
#   3. Rhythm is expressed BEAT-RELATIVE (a "rapid" rearticulation is a
#      fraction of a beat, not a hard-coded 500ms as the pasted proposal
#      assumed) - so a detector means the same thing at 62 and 150 bpm.
#   4. Ships a `falsifier`: what would prove this firing wrong.
#   5. Has a synthetic positive AND negative in the test file. Four of
#      the five detectors in the pasted proposal were provably dead
#      (the Alberti test compared a 4-element list to a 3-element list
#      and could never return True); a positive test catches that class
#      of bug before it ever meets audio.
#
# Detectors take a monophonic-ish STREAM (a list of Notes ordered in
# time, no two overlapping by construction upstream) so "the next note"
# is meaningful. Streams come from instrument_attribution.voice_
# continuity.stream_into_lines - a real Jazz module, not a re-implemented
# gap heuristic.
# =================================================================

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from university.observation_types import PatternObservation

# --- thresholds, all justified in-line -------------------------------
STEP_MAX_SEMITONES = 2          # a "step" is a semitone or whole tone
SCALE_MIN_NOTES = 4             # 3 stepwise notes is a turn, not a run
# Raised 3 -> 4 and given a direction test after the null model measured the
# 3-note version firing MORE on pitch-shuffled notes than on real music
# (Hopeful: real 0.111 vs null 0.153). Any three leaping pitches have a good
# chance of forming some triad under some rotation, so "3 leaps + chord-shaped"
# was close to a coin flip. A real arpeggio also OUTLINES the chord - it walks
# through it rather than zig-zagging randomly.
ARPEGGIO_MIN_NOTES = 4
ARPEGGIO_MIN_LEAP = 3           # below a minor 3rd it's stepwise, not arpeggiated
ARPEGGIO_MIN_DIRECTION_RUN = 0.75   # fraction of steps sharing the dominant direction
SEQUENCE_MIN_CELL = 3
SEQUENCE_MAX_CELL = 5
OSTINATO_MIN_CELL = 2
OSTINATO_MIN_REPEATS = 3
PEDAL_MIN_REPEATS = 3
PEDAL_MIN_BEATS = 2.0           # a drone has to actually persist
# Same pitch re-STRUCK within ~1/3 beat. Measured onset-to-onset (IOI),
# NOT as a gap: a sustained note's gap stays tiny even when its onsets are
# a beat apart, so a gap test called a slow repeated note a machine-gun
# stutter (caught by test_detectors.py's negative case). Re-strike RATE is
# an onset property.
REARTIC_MAX_IOI_BEATS = 0.30
REARTIC_MIN_NOTES = 3
ALBERTI_MIN_CYCLES = 2          # one C-G-E-G is a coincidence; two is a figure

# Triad templates (root-relative pitch classes) for chord-shaped tests.
_TRIADS = ((0, 4, 7), (0, 3, 7), (0, 3, 6), (0, 4, 8))
_TETRADS = ((0, 4, 7, 10), (0, 3, 7, 10), (0, 4, 7, 11), (0, 3, 6, 10), (0, 3, 6, 9))


def _pitches(stream: Sequence) -> List[int]:
    return [int(n.pitch) for n in stream]


def _ids(stream: Sequence) -> Tuple[str, ...]:
    return tuple(n.id for n in stream)


def _is_chord_shaped(pitch_classes: Sequence[int]) -> Optional[Tuple[int, ...]]:
    """Do these pitch classes form a triad/tetrad under some rotation?"""
    uniq = sorted(set(int(p) % 12 for p in pitch_classes))
    if not 3 <= len(uniq) <= 4:
        return None
    for template in (_TRIADS + _TETRADS):
        if len(template) != len(uniq):
            continue
        for root in uniq:
            shifted = tuple(sorted((p - root) % 12 for p in uniq))
            if shifted == tuple(sorted(template)):
                return template
    return None


# =====================================================================
# Department: UNIVERSAL
# =====================================================================

def detect_scale_runs(stream: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """Maximal stepwise runs in one direction. A scale run is the single
    most reliable melodic object and it is what tells a beamer how to
    group - it earns its place first."""
    out: List[PatternObservation] = []
    n = len(stream)
    i = 0
    while i < n - 1:
        direction = 0
        j = i
        while j < n - 1:
            interval = int(stream[j + 1].pitch) - int(stream[j].pitch)
            if interval == 0 or abs(interval) > STEP_MAX_SEMITONES:
                break
            step_dir = 1 if interval > 0 else -1
            if direction == 0:
                direction = step_dir
            elif step_dir != direction:
                break
            j += 1
        run = stream[i:j + 1]
        if len(run) >= SCALE_MIN_NOTES:
            out.append(PatternObservation(
                pattern="scale_run", department="universal", note_ids=_ids(run),
                start_ms=run[0].start_ms, end_ms=run[-1].end_ms,
                confidence=min(0.95, 0.55 + 0.05 * len(run)),
                evidence={"length": len(run), "direction": "up" if direction > 0 else "down",
                          "span_semitones": abs(int(run[-1].pitch) - int(run[0].pitch))},
                falsifier="a non-stepwise interval, or a direction change, inside the span",
                stem=stem,
            ))
            i = j
        else:
            i += 1
    return out


def detect_arpeggios(stream: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """Consecutive leaps whose pitch classes form a chord - a broken
    chord. Distinguished from a scale run by interval size, and from a
    random leap sequence by chord-shape."""
    out: List[PatternObservation] = []
    n = len(stream)
    i = 0
    while i < n - 1:
        j = i
        while j < n - 1:
            interval = abs(int(stream[j + 1].pitch) - int(stream[j].pitch))
            if interval < ARPEGGIO_MIN_LEAP or interval > 12:
                break
            j += 1
        span = stream[i:j + 1]
        if len(span) >= ARPEGGIO_MIN_NOTES:
            template = _is_chord_shaped(_pitches(span))
            # A real arpeggio OUTLINES the chord (mostly one direction);
            # random leaps that happen to be chord-shaped zig-zag.
            sp = _pitches(span)
            steps = [1 if sp[k + 1] > sp[k] else -1 for k in range(len(sp) - 1)]
            dominant = max(steps.count(1), steps.count(-1)) / len(steps) if steps else 0.0
            if template is not None and dominant >= ARPEGGIO_MIN_DIRECTION_RUN:
                out.append(PatternObservation(
                    pattern="arpeggio", department="universal", note_ids=_ids(span),
                    start_ms=span[0].start_ms, end_ms=span[-1].end_ms,
                    confidence=min(0.9, 0.5 + 0.08 * len(span)),
                    evidence={"length": len(span), "chord_template": list(template),
                              "direction_consistency": round(dominant, 2),
                              "pitch_classes": sorted(set(p % 12 for p in _pitches(span)))},
                    falsifier="a pitch class outside the matched chord template, or "
                              "zig-zag motion that does not outline the chord",
                    stem=stem,
                ))
                i = j
                continue
        i += 1
    return out


def detect_sequences(stream: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """A melodic cell immediately repeated at a DIFFERENT pitch level with
    the same interval contour - the engine of Baroque/Classical writing,
    and a strong signal two spans are the same musical idea."""
    out: List[PatternObservation] = []
    pitches = _pitches(stream)
    n = len(pitches)
    i = 0
    while i < n:
        matched = False
        for cell in range(SEQUENCE_MAX_CELL, SEQUENCE_MIN_CELL - 1, -1):
            if i + 2 * cell > n:
                continue
            a = pitches[i:i + cell]
            b = pitches[i + cell:i + 2 * cell]
            ia = [a[k + 1] - a[k] for k in range(cell - 1)]
            ib = [b[k + 1] - b[k] for k in range(cell - 1)]
            transposition = b[0] - a[0]
            if ia == ib and transposition != 0 and abs(transposition) <= 12:
                span = stream[i:i + 2 * cell]
                out.append(PatternObservation(
                    pattern="sequence", department="universal", note_ids=_ids(span),
                    start_ms=span[0].start_ms, end_ms=span[-1].end_ms,
                    confidence=0.8,
                    evidence={"cell_length": cell, "transposition": transposition,
                              "contour": ia},
                    falsifier="the second cell's interval contour differing from the first",
                    stem=stem,
                ))
                i += 2 * cell
                matched = True
                break
        if not matched:
            i += 1
    return out


def detect_ostinato(stream: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """A cell repeated at the SAME pitches, 3+ times - accompaniment
    figure / riff. (A sequence transposes; an ostinato does not.)"""
    out: List[PatternObservation] = []
    pitches = _pitches(stream)
    n = len(pitches)
    i = 0
    while i < n:
        matched = False
        for cell in range(SEQUENCE_MAX_CELL, OSTINATO_MIN_CELL - 1, -1):
            if i + cell * OSTINATO_MIN_REPEATS > n:
                continue
            base = pitches[i:i + cell]
            repeats = 1
            while pitches[i + repeats * cell: i + (repeats + 1) * cell] == base:
                repeats += 1
            if repeats >= OSTINATO_MIN_REPEATS:
                span = stream[i:i + repeats * cell]
                out.append(PatternObservation(
                    pattern="ostinato", department="universal", note_ids=_ids(span),
                    start_ms=span[0].start_ms, end_ms=span[-1].end_ms,
                    confidence=min(0.92, 0.6 + 0.08 * repeats),
                    evidence={"cell_length": cell, "repeats": repeats, "cell": base},
                    falsifier="any repetition differing in pitch from the first cell",
                    stem=stem,
                ))
                i += repeats * cell
                matched = True
                break
        if not matched:
            i += 1
    return out


def detect_pedal(stream: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """One pitch persisting (re-struck or held) across a long span - a
    drone/pedal point. Not the same as rearticulation: a pedal spans
    BEATS, a rearticulation is a sub-beat stutter."""
    out: List[PatternObservation] = []
    n = len(stream)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and int(stream[j + 1].pitch) == int(stream[i].pitch):
            j += 1
        run = stream[i:j + 1]
        if len(run) >= PEDAL_MIN_REPEATS and beat_ms > 0:
            beats = (run[-1].end_ms - run[0].start_ms) / beat_ms
            if beats >= PEDAL_MIN_BEATS:
                out.append(PatternObservation(
                    pattern="pedal_point", department="universal", note_ids=_ids(run),
                    start_ms=run[0].start_ms, end_ms=run[-1].end_ms,
                    confidence=0.75,
                    evidence={"pitch": int(run[0].pitch), "repeats": len(run),
                              "span_beats": round(beats, 2)},
                    falsifier="a pitch change inside the span, or a span under "
                              f"{PEDAL_MIN_BEATS} beats",
                    stem=stem,
                ))
        i = j + 1
    return out


def detect_rearticulation(stream: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """Same pitch re-struck several times within a fraction of a beat.

    This is the ARTIFACT detector and the most directly useful one in
    APPLY mode: it is the measured 'machine-gun' signature of Basic
    Pitch fragmenting one sustained note (52% of adjacent notes in a
    voice were same-pitch on the produced page). Real tremolo exists,
    which is exactly why this reports rather than deletes - the apply
    layer additionally requires low confidence before acting."""
    out: List[PatternObservation] = []
    if beat_ms <= 0:
        return out
    max_ioi = REARTIC_MAX_IOI_BEATS * beat_ms
    n = len(stream)
    i = 0
    while i < n:
        j = i
        while (j < n - 1
               and int(stream[j + 1].pitch) == int(stream[i].pitch)
               and (stream[j + 1].start_ms - stream[j].start_ms) <= max_ioi):
            j += 1
        run = stream[i:j + 1]
        if len(run) >= REARTIC_MIN_NOTES:
            confidences = [float(getattr(x, "confidence", 1.0)) for x in run]
            out.append(PatternObservation(
                pattern="rearticulation", department="universal", note_ids=_ids(run),
                start_ms=run[0].start_ms, end_ms=run[-1].end_ms,
                confidence=0.7,
                evidence={"pitch": int(run[0].pitch), "count": len(run),
                          "mean_note_confidence": round(sum(confidences) / len(confidences), 3),
                          "max_ioi_beats": REARTIC_MAX_IOI_BEATS},
                falsifier="a pitch change, or a gap wider than "
                          f"{REARTIC_MAX_IOI_BEATS} beat between strike onsets",
                stem=stem,
            ))
        i = j + 1
    return out


# =====================================================================
# Department: KEYBOARD / HARMONIC
# =====================================================================

def detect_alberti(stream: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """Alberti bass: the low-high-middle-high cycle (C-G-E-G), repeated.

    NOTE: the pasted proposal's version could NEVER fire - it built a
    4-element normalized list and compared it to the 3-element [0,4,7].
    This one tests the actual shape: within each 4-note cycle
    p0 < p2 < p1 and p3 == p1, the three distinct pitches form a triad,
    and the cycle repeats at least ALBERTI_MIN_CYCLES times."""
    out: List[PatternObservation] = []
    pitches = _pitches(stream)
    n = len(pitches)

    def is_cycle(k: int) -> bool:
        if k + 4 > n:
            return False
        p0, p1, p2, p3 = pitches[k:k + 4]
        if not (p0 < p2 < p1):
            return False
        if p3 != p1:
            return False
        return _is_chord_shaped((p0, p1, p2)) is not None

    i = 0
    while i < n:
        if not is_cycle(i):
            i += 1
            continue
        cycles = 1
        while is_cycle(i + cycles * 4) and pitches[i + cycles * 4:i + cycles * 4 + 4] == pitches[i:i + 4]:
            cycles += 1
        if cycles >= ALBERTI_MIN_CYCLES:
            span = stream[i:i + cycles * 4]
            out.append(PatternObservation(
                pattern="alberti_bass", department="keyboard", note_ids=_ids(span),
                start_ms=span[0].start_ms, end_ms=span[-1].end_ms,
                confidence=0.85,
                evidence={"cycles": cycles, "figure": pitches[i:i + 4]},
                falsifier="a cycle not matching low-high-mid-high over one triad",
                stem=stem,
            ))
            i += cycles * 4
        else:
            i += 1
    return out


def detect_neighbor_tone(stream: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """Neighbor figure: a note steps away and returns (C-D-C / C-B-C), or
    the double-neighbor turn (C-D-C-B-C). One of the most common melodic
    ornaments in any idiom, and cheap to verify: the return must land on
    the SAME pitch it left."""
    out: List[PatternObservation] = []
    p = _pitches(stream)
    n = len(p)
    i = 0
    while i < n - 2:
        # simple neighbor: p0 -> p1 (step) -> p0
        if p[i + 2] == p[i] and 1 <= abs(p[i + 1] - p[i]) <= STEP_MAX_SEMITONES:
            size = 3
            # double neighbor / turn: ... -> other-side step -> back again
            if (i + 4 < n and p[i + 4] == p[i]
                    and 1 <= abs(p[i + 3] - p[i]) <= STEP_MAX_SEMITONES
                    and (p[i + 3] - p[i]) * (p[i + 1] - p[i]) < 0):
                size = 5
            span = stream[i:i + size]
            out.append(PatternObservation(
                pattern="neighbor_tone", department="universal", note_ids=_ids(span),
                start_ms=span[0].start_ms, end_ms=span[-1].end_ms,
                confidence=0.75 if size == 3 else 0.85,
                evidence={"center_pitch": p[i], "size": size,
                          "kind": "turn" if size == 5 else "neighbor",
                          "neighbor_interval": p[i + 1] - p[i]},
                falsifier="the figure not returning to the pitch it left, or the "
                          "neighbor being a leap rather than a step",
                stem=stem,
            ))
            i += size - 1
        else:
            i += 1
    return out


def detect_gap_fill(stream: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """Gap-fill: a melodic LEAP followed by stepwise motion back in the
    OPPOSITE direction - the most robust cross-cultural melodic principle
    there is (a leap creates a gap; the line fills it in). Strong because
    it demands a specific relationship between two events, not just a
    local shape."""
    out: List[PatternObservation] = []
    p = _pitches(stream)
    n = len(p)
    i = 0
    while i < n - 2:
        leap = p[i + 1] - p[i]
        if abs(leap) < 4 or abs(leap) > 12:
            i += 1
            continue
        j = i + 1
        filled = 0
        while j < n - 1:
            step = p[j + 1] - p[j]
            if 1 <= abs(step) <= STEP_MAX_SEMITONES and step * leap < 0:
                filled += 1
                j += 1
            else:
                break
        if filled >= 2:
            span = stream[i:j + 1]
            out.append(PatternObservation(
                pattern="gap_fill", department="universal", note_ids=_ids(span),
                start_ms=span[0].start_ms, end_ms=span[-1].end_ms,
                confidence=min(0.9, 0.6 + 0.07 * filled),
                evidence={"leap": leap, "fill_steps": filled,
                          "direction": "down-fill" if leap > 0 else "up-fill"},
                falsifier="the steps after the leap not moving opposite to it",
                stem=stem,
            ))
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------------
# VERTICAL detector: reads a whole stem, not one melodic line. Chords are
# simultaneity, which by definition cannot live inside a monophonic
# stream - and dense harmonic material is where most of the notes are.
# ---------------------------------------------------------------------

CHORD_ONSET_WINDOW_MS = 45.0
CHORD_MIN_NOTES = 3


def detect_chords(notes: Sequence, beat_ms: float, stem: Optional[str]) -> List[PatternObservation]:
    """Notes struck together whose pitch classes form a chord. Takes the
    whole stem (not a line), groups by onset, and requires a real template
    match - the arpeggio lesson applies here too, so the null model is the
    arbiter of whether this is signal or coincidence."""
    out: List[PatternObservation] = []
    ordered = sorted(notes, key=lambda n: n.start_ms)
    group: List = []
    for note in ordered:
        if group and note.start_ms - group[0].start_ms > CHORD_ONSET_WINDOW_MS:
            if len(group) >= CHORD_MIN_NOTES:
                template = _is_chord_shaped(_pitches(group))
                if template is not None:
                    out.append(PatternObservation(
                        pattern="chord", department="harmonic", note_ids=_ids(group),
                        start_ms=group[0].start_ms,
                        end_ms=max(g.end_ms for g in group),
                        confidence=0.8,
                        evidence={"size": len(group), "chord_template": list(template),
                                  "pitch_classes": sorted(set(p % 12 for p in _pitches(group)))},
                        falsifier="a pitch class outside the matched chord template, "
                                  "or the notes not being struck together",
                        stem=stem,
                    ))
            group = []
        group.append(note)
    if len(group) >= CHORD_MIN_NOTES:
        template = _is_chord_shaped(_pitches(group))
        if template is not None:
            out.append(PatternObservation(
                pattern="chord", department="harmonic", note_ids=_ids(group),
                start_ms=group[0].start_ms, end_ms=max(g.end_ms for g in group),
                confidence=0.8,
                evidence={"size": len(group), "chord_template": list(template),
                          "pitch_classes": sorted(set(p % 12 for p in _pitches(group)))},
                falsifier="a pitch class outside the matched chord template",
                stem=stem,
            ))
    return out


# Registry: name -> detector. The study pass runs ALL applicable
# detectors and records EVERY firing (witnesses, not a classifier - the
# proposal returned only the first match, which made hierarchical
# patterns impossible by construction).
DETECTORS: Dict[str, Callable[..., List[PatternObservation]]] = {
    "scale_run": detect_scale_runs,
    "arpeggio": detect_arpeggios,
    "sequence": detect_sequences,
    "ostinato": detect_ostinato,
    "pedal_point": detect_pedal,
    "rearticulation": detect_rearticulation,
    "alberti_bass": detect_alberti,
    "neighbor_tone": detect_neighbor_tone,
    "gap_fill": detect_gap_fill,
    "chord": detect_chords,
}

# Detectors that read a WHOLE STEM (simultaneity) rather than one melodic
# line. Dispatched separately by study()/null_model().
VERTICAL_DETECTORS = ("chord",)

# WHAT ACTUALLY GRADUATED (null model, 3 songs, 2026-08-06).
# The rule from GRIMLOCK_UNIVERSITY.md §7 is applied to its own authors:
# a detector that does not beat shuffled notes does not get to speak.
#
#   scale_run   REAL  lift 8.65 (Hopeful) / 5.81 (prospering)
#   arpeggio    REAL  lift 1.81 / 9.33 (End Transmission) / 3.44
#               - only after being tightened; the first version fired MORE
#                 on shuffled pitches than on real music (lift 0.73)
#   sequence    REAL  lift 1.80 / inf
#   ostinato    REAL  (fires rarely; null 0 where it does)
#   alberti_bass  never fired on this corpus - correct, not broken: there
#                 is no Alberti bass in worship/rock. Kept, unproven.
#
# DEMOTED - graded NOISE on all three songs (lift 0.34 / 0.72 / 1.3, i.e.
# it fires about as often, or more often, on shuffled notes). Same-pitch
# adjacency arises by chance in a small pitch vocabulary, so "one pitch
# repeated over 2+ beats" is not evidence of a real pedal point.
_DEMOTED_PEDAL = "pedal_point"

# NULL-MODEL GRADES, 3 songs (Hopeful / prospering / End Transmission),
# 2026-08-06, after the streaming fix. Recorded here so the tiering is a
# MEASUREMENT, not an opinion, and so it can be re-checked on new material.
#
#   GRADUATED (unanimous REAL - safe for APPLY to act on):
#     scale_run     3/3   lift 3.79 / 12.75 / 1.50
#     arpeggio      3/3   lift 1.85 /  1.62 / 2.26
#     neighbor_tone 3/3   lift 2.59 /  2.13 / 3.24   <- biggest coverage win
#     ostinato      2/2   lift  inf /  2.00 /  n/a   (silent on the third)
#
#   PROVISIONAL (real on some material, NOISE on other - studied and
#   logged, but NOT allowed to alter the page):
#     chord         2/3   REAL 1.79 / 2.27, NOISE 0.92 (End Transmission)
#     gap_fill      2/3   REAL 4.66 / 1.59, NOISE 1.23 (Hopeful)
#     sequence      1/2   REAL 3.82,        NOISE 0.95 (prospering)
#
#   FAILED: pedal_point 0/3 - NOISE everywhere. Demoted.
GRADUATED = ("scale_run", "arpeggio", "neighbor_tone", "ostinato")
PROVISIONAL = ("chord", "gap_fill", "sequence")

# DIAGNOSTIC ONLY - not redundant, but already handled upstream. Measured
# on Hopeful: with consolidation ON it finds 0 runs, with consolidation
# OFF it finds 63. note_consolidation has already absorbed exactly the
# machine-gun fragments this detector targets, so in the pipeline path it
# is correctly silent. Retained because it is a sharp probe for whether
# consolidation is doing its job.
_DIAGNOSTIC_REARTIC = "rearticulation"

UNIVERSAL_DETECTORS = ("scale_run", "arpeggio", "sequence", "ostinato",
                       "neighbor_tone", "gap_fill")
KEYBOARD_DETECTORS = ("alberti_bass",)

# Everything, including demoted/diagnostic detectors - used by the null
# model and by tools that want to re-measure the demotion decision.
ALL_DETECTOR_NAMES = UNIVERSAL_DETECTORS + KEYBOARD_DETECTORS + VERTICAL_DETECTORS + (
    _DEMOTED_PEDAL, _DIAGNOSTIC_REARTIC)


__all__ = ["DETECTORS", "UNIVERSAL_DETECTORS", "KEYBOARD_DETECTORS",
           "VERTICAL_DETECTORS", "ALL_DETECTOR_NAMES",
           "detect_neighbor_tone", "detect_gap_fill", "detect_chords",
           "detect_scale_runs", "detect_arpeggios", "detect_sequences",
           "detect_ostinato", "detect_pedal", "detect_rearticulation",
           "detect_alberti"]
