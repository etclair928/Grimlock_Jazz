# =================================================================
# MODULE: output/piano_reduction.py
# Turns a flat harmonic note stream (the merged "junk drawer") into a
# two-staff PIANO GRAND STAFF - the two wins measured in
# GRIMLOCK_6.0_PERFORMANCE_ENGRAVING.md §15 and §13, promoted from
# experiment into production:
#
#   1. assign_hands   - register split by a Viterbi with a switch penalty
#                       (hysteresis). MEASURED §15: worst-staff playability
#                       75%->99% (You Say), 98%->100% (Hopeful), and each
#                       hand gets a physically reachable register. The
#                       hysteresis beat a fixed middle-C split on both songs.
#
#   2. assign_voices  - the <=4 rhythmic-independence voicer §13. §15 showed
#                       register split alone still SPILLED past 4 voices onto
#                       an extra staff (3% Hopeful, 8% You Say) because the
#                       exporter's greedy fallback over-produces voices. This
#                       fixes that at the source: notes that share a rhythm
#                       become chord tones of one voice; genuinely independent
#                       lines become separate voices; and the RARE >4-overlap
#                       instant is absorbed by merging the overflow note into
#                       the nearest active voice as a chord tone rather than
#                       spilling a fifth staff. Guarantees <= max_voices.
#
# LAW (unchanged): this is a VIEW transform. Nothing here mutates a Note.
# It builds a NotationScore from NotationNotes and stamps voice_index on
# them; the exporter honors that verbatim. Deterministic, inspectable, no
# global solver, no ground-truth-hungry weights - the switch penalty and
# the hand limits are physical, not tuned. Not wired into the Conductor
# by default (same discipline as §6.1): call it explicitly.
# =================================================================

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from core import StemType
from output.notation_score import NotationNote, NotationPart, NotationScore

# Physical two-hand limits (§12.3 / §13). Not tuned knobs - a hand covers a
# ninth comfortably and has five fingers.
SPAN_MAX_SEMITONES = 14
FINGERS_PER_HAND = 5

# The shape limit, from output/playability.py's hand model: between thumb and
# the rest of the hand sits one large gap, and fingers 2-3-4-5 stay close. A
# span limit alone admits a five-note cluster and rejects a real tenth, so
# this is what makes the emission cost below describe a HAND rather than a
# width. Deliberately the COMFORTABLE value rather than playability's own
# 17-semitone default: that default exists so a positive means "physically
# impossible", while this decides where a staff should split, which is a
# question about what reads well, not about what is barely possible.
INNER_GAP_MAX_SEMITONES = 5

# Register-split Viterbi search grid + hysteresis. The switch penalty is the
# cost of moving the treble/bass boundary by one semitone between beats; it is
# what stops the split churning bar to bar. Grid E3..D5 covers every sane
# grand-staff split point.
SPLIT_BOUNDARY_LO = 52          # E3
SPLIT_BOUNDARY_HI = 74          # D5
SPLIT_BOUNDARY_STEP = 2
SPLIT_SWITCH_PENALTY = 0.4

# Two notes whose onsets AND ends both fall within these windows share a
# rhythm -> one chord in one voice. Divergent rhythm -> separate voices.
# Read ONLY by _chord_events' raw-millisecond fallback, which runs when no
# tempo is available; the grid-aligned path production takes ignores them.
# (The header used to call them dead outright - true of the grid path, not of
# the function.)
CHORD_ONSET_TOL_MS = 40.0
CHORD_OFFSET_TOL_MS = 90.0

# How close two RAW onsets must be to count as one chord, as a fraction of a
# grid cell, before anything is snapped. Half a cell is the distance at which
# two notes would have landed in the same cell had the boundary fallen
# elsewhere, so this catches boundary-straddle splits without inventing a
# threshold. Swept on Ellington - see _chord_events.
CHORD_RAW_TOLERANCE_CELLS = 0.5

# Acoustic gate for the legato fill (XVIII.3 #1, RECALIBRATED 2026-08-08).
#
# The first version mirrored AnechoicMa's own region thresholds (void>0.65 =
# SILENT, resonance>0.60 = RESONANT) on the assumption that
# `resonance_probability` was a calibrated probability on [0,1]. MEASURED over
# 11,126 trailing-gap observations on two songs, it is not: it occupies roughly
# [0.24, 0.62], and 0.60 sits ABOVE ITS 99th PERCENTILE. The gate fired on
# 0.07%-1.00% of gaps - it was inert. `trailing_void` bottoms out at 0.45, so
# the void half passed 97.9% of everything and was inert too.
#
# Worse, the inert gate SHADOWED the blanket fill: the old branch `continue`d
# for every note the witness had measured (73% of them), so those gaps got no
# legato fill at all. Wiring the evidence in made sustain worse than the
# threshold it replaced, which is the opposite of what §XVIII.3 intended.
#
# Two changes, both aimed at the same law - the evidence should MODULATE the
# blanket window, never silently replace it:
#   1. Thresholds are PERCENTILES OF THE OBSERVED DISTRIBUTION, not absolute
#      constants. The scale drifts between songs (median resonance 0.392 on
#      Federal Blvd vs 0.428 on the SUNO cut), so an absolute number cannot mean
#      the same thing twice. Percentiles are self-calibrating.
#   2. The policy is THREE-WAY. Clearly ringing -> extra reach; clearly silent
#      -> a real rest, blocked; anything in between -> the ordinary fill_max
#      window that was working before the gate existed.
#
# resonance and void are genuinely informative (corr = -0.42 / -0.27, i.e. more
# ringing means less silence, as it should) - they were just being read against
# the wrong ruler.
#
# HONESTY NOTE ON WHAT THE FIX ACTUALLY BOUGHT. Measured on two songs, rest/note
# fell 0.305 -> 0.198 (Federal Blvd) and 0.209 -> 0.122 (SUNO cut) with mean
# voice jump improving TOO (4.85 -> 4.49, 6.38 -> 5.95) - both axes at once,
# which they usually trade. But disabling the gate outright and simply letting
# the flat window run scores 0.207 / 0.126 - i.e. essentially ALL of the gain
# came from removing the shadowing `continue`, and almost none from the acoustic
# evidence itself. §XIX recorded "wiring audio back in" as the best fix of the
# engraving arc; measured properly, the wiring was inert and actively suppressed
# a working blanket fill. The evidence is kept because it is consistently (4/4
# metrics, 2 songs) a hair better than blind and costs nothing - not because it
# has earned the claim once made for it. Do not cite this gate as evidence that
# acoustic witnesses improve the page until something measures better than this.
#
# The percentiles below were swept against readability on two songs and were the
# best measured pair. Note that at a 25th-percentile ring threshold three
# quarters of gaps qualify as "ringing", which is closer to a wider flat window
# than to a discriminative use of the evidence - consistent with the honesty
# note above.
ACOUSTIC_RINGING_PERCENTILE = 25.0   # at/above this percentile = still sounding
ACOUSTIC_VOID_PERCENTILE = 10.0      # at/below this percentile = a real rest
ACOUSTIC_RINGING_REACH = 2.0         # ringing gaps may close this much further
ACOUSTIC_MIN_SAMPLES = 30            # below this, percentiles are noise - stay blind

# VOICE ASSIGNMENT BY REGISTER RANK (2026-08-08). The original rule sent each
# chord-event to whichever FREE voice was last nearest in pitch. Measured on
# Federal Blvd - a lead-trumpet-plus-brass-section tune - that put the top line
# in voice 1 only 38.8% of the time and made the melody change voice on 52% of
# onsets, because a sustaining lead would push the tune into voice 2 and reclaim
# it an onset later. mean_voice_jump could not see any of this: all three voices
# scored 4.9-5.7 precisely BECAUSE none of them was the melody.
#
# A section is stratified - lead on top, parts holding position underneath - so
# a voice should own a REGISTER, and voice 1 should be the upper part exactly as
# notation convention says.
#
# MEASURED AND REJECTED (2026-08-08). That reasoning is musically sound and did
# not survive contact with the data: against the nearest-pitch rule it left
# top-line stability unchanged (0.419 -> 0.420) while making voice1_is_top WORSE
# (0.413 -> 0.299) and mean jump worse (4.49 -> 5.75). The fragmentation was
# never caused by the choice rule - see VOICE_OVERLAP_TOLERANCE_BEATS below for
# what actually caused it. Kept as a switch, defaulted OFF, because the negative
# result is worth preserving: a correct account of the MUSIC is not automatically
# a correct account of the BUG.
VOICE_BY_REGISTER_RANK = False
VOICE_ANCHOR_SLOTS = 3               # slots averaged into a voice's register anchor

# VOICE OVERLAP TOLERANCE - the actual cause of the fragmented melody.
# MEASURED on Federal Blvd's harmonic stem: 63.6% of consecutive notes OVERLAP
# the note that follows them, median overlap 104ms (0.10 beats). Basic Pitch
# ends a note where its energy decays, not where the player stopped, so a
# legato line arrives as a chain of slightly-overlapping events. A strict
# interval partition cannot put two overlapping events in one voice, so a
# melody of overlapping notes is FORCED to alternate between voices - which is
# exactly the 52%-of-onsets flapping we measured, and no amount of smarter
# voice-CHOICE can fix it because the constraint is occupancy, not preference.
#
# So a voice counts as free when its current slot ends within this tolerance of
# the new slot's onset, and the outgoing slot is CLAMPED to that onset. That is
# a notation-clock edit of exactly the kind the legato fill already makes in the
# other direction - the performance clock is untouched (§2.2).
# Swept 0.0 / 0.25 / 0.50 on two songs: 0.50 was best or tied on every axis,
# taking top-line stability 0.416 -> 0.545 (Federal Blvd) and 0.396 -> 0.579
# (SUNO cut) with zero overfull measures - the check that matters, since this
# path clamps note ends.
VOICE_OVERLAP_TOLERANCE_BEATS = 0.5

# Renumber voices so voice 1 is the one that most often holds the top note
# (see the relabel block in assign_voices). Small but consistent: voice1_is_top
# 0.430 -> 0.453 and 0.486 -> 0.517, with every other axis unchanged.
RELABEL_ENABLED = True


# ---------------------------------------------------------------------------
# 1. Register split (hysteresis Viterbi) - §15 win, validated
# ---------------------------------------------------------------------------

def assign_hands(
        notes: List[NotationNote],
        tempo_bpm: float,
        *,
        switch_penalty: float = SPLIT_SWITCH_PENALTY,
        span_max: int = SPAN_MAX_SEMITONES,
        fingers_max: int = FINGERS_PER_HAND,
        inner_gap_max: int = INNER_GAP_MAX_SEMITONES,
        b_lo: int = SPLIT_BOUNDARY_LO,
        b_hi: int = SPLIT_BOUNDARY_HI,
        step: int = SPLIT_BOUNDARY_STEP,
) -> Dict[str, List[NotationNote]]:
    """Split notes into {'treble', 'bass'} by a per-beat boundary chosen to
    keep each hand playable (emission = span/finger excess) while resisting
    change (transition = switch_penalty * |Δboundary|)."""
    if not notes:
        return {"treble": [], "bass": []}

    ms_per_beat = 60000.0 / max(tempo_bpm, 1.0)
    win_pitches: Dict[int, List[int]] = defaultdict(list)
    for n in notes:
        w0 = int(math.floor(n.start_ms / ms_per_beat))
        w1 = max(w0 + 1, int(math.ceil(n.end_ms / ms_per_beat)))
        for w in range(w0, w1):
            win_pitches[w].append(n.pitch)
    windows = sorted(win_pitches.keys())
    boundaries = list(range(b_lo, b_hi + 1, step))

    # The cost of putting the treble/bass boundary at `b` during window `w`.
    #
    # WHAT CHANGED, and why it is worth a term. This used to charge only for
    # SPAN and FINGER COUNT, which cannot tell a real tenth from a cluster of
    # the same width: C-E-G-C is one wide thumb reach and four close fingers,
    # while C-G#-E is three minor sixths and no hand can shape it, and both
    # span twelve to sixteen semitones. output/playability.py's hand model
    # carries the distinction - one large gap allowed, the rest close - and
    # this is the natural place to spend it: choosing WHERE the staff splits
    # is exactly the decision that can turn an unplayable shape into two
    # playable ones.
    #
    # Graded, not binary. Viterbi needs a cost surface it can slide down; a
    # hard "impossible" would flatten every bad boundary to the same value and
    # leave the search choosing between them at random.
    def emit(w: int, b: int) -> float:
        cost = 0.0
        for group in ([p for p in win_pitches[w] if p < b],
                      [p for p in win_pitches[w] if p >= b]):
            if not group:
                continue
            uniq = sorted(set(group))
            span = uniq[-1] - uniq[0]
            cost += max(0, span - span_max) * 1.0
            cost += max(0, len(uniq) - fingers_max) * 3.0
            # The shape term: every adjacent-finger gap except the widest one
            # (the thumb's) has to be small. Charged per semitone over, so a
            # boundary that breaks one bad shape into two good ones wins.
            gaps = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
            if len(gaps) > 1:
                for gap in sorted(gaps)[:-1]:
                    cost += max(0, gap - inner_gap_max) * 1.5
        return cost

    inf = float("inf")
    dp = {b: emit(windows[0], b) for b in boundaries}
    back: List[Dict[int, int]] = [{}]
    for wi in range(1, len(windows)):
        w = windows[wi]
        ndp: Dict[int, float] = {}
        nback: Dict[int, int] = {}
        for b in boundaries:
            ec = emit(w, b)
            best, best_prev = inf, boundaries[0]
            for pb in boundaries:
                c = dp[pb] + switch_penalty * abs(b - pb) + ec
                if c < best:
                    best, best_prev = c, pb
            ndp[b], nback[b] = best, best_prev
        dp = ndp
        back.append(nback)

    b_end = min(dp, key=dp.get)
    chosen = [0] * len(windows)
    chosen[-1] = b_end
    for wi in range(len(windows) - 1, 0, -1):
        chosen[wi - 1] = back[wi][chosen[wi]]
    win_boundary = {w: chosen[i] for i, w in enumerate(windows)}

    hands: Dict[str, List[NotationNote]] = {"treble": [], "bass": []}
    for n in notes:
        w = int(math.floor(n.start_ms / ms_per_beat))
        b = win_boundary.get(w)
        if b is None:
            b = win_boundary[min(windows, key=lambda x: abs(x - w))]
        hands["treble" if n.pitch >= b else "bass"].append(n)
    return hands


# ---------------------------------------------------------------------------
# 2. The <=4 rhythmic-independence voicer - §13, the next wall from §15
# ---------------------------------------------------------------------------

class _Slot:
    """One chord-event in a voice: notes that share a rhythm, sounding as one
    notehead-stack. Tracks the span it occupies so later events know if the
    voice is free."""
    __slots__ = ("notes", "start", "end")

    def __init__(self, note: NotationNote):
        self.notes: List[NotationNote] = [note]
        self.start = note.start_ms
        self.end = note.end_ms

    def add(self, note: NotationNote) -> None:
        self.notes.append(note)
        self.end = max(self.end, note.end_ms)

    @property
    def pitch_center(self) -> float:
        return sum(n.pitch for n in self.notes) / len(self.notes)

    @property
    def trailing_resonance(self) -> Optional[float]:
        """AnechoicMa's resonance probability for the gap after this slot -
        the MAX across its notes, since a chord is still ringing if any of
        its tones is. None when the witness never ran."""
        vals = [n.trailing_resonance for n in self.notes
                if getattr(n, "trailing_resonance", None) is not None]
        return max(vals) if vals else None

    @property
    def trailing_void(self) -> Optional[float]:
        """Silence probability for the gap after this slot - the MIN across
        its notes: the gap is only truly empty if every tone has stopped."""
        vals = [n.trailing_void for n in self.notes
                if getattr(n, "trailing_void", None) is not None]
        return min(vals) if vals else None

    @property
    def cohesion(self) -> Optional[str]:
        """The Grimlock University gesture this slot belongs to, if any
        (APPLY mode only). A slot inherits the group of its notes; mixed
        slots take the first, which is the dominant case since a chord is
        struck as one gesture."""
        for n in self.notes:
            if getattr(n, "cohesion", None):
                return n.cohesion
        return None


def _chord_events(notes: List[NotationNote], ms_per_beat: float = 0.0,
                  grid_division: int = 4, end_tolerance_cells: int = 1) -> List[_Slot]:
    """Group same-rhythm notes (onset AND end both within tolerance) into one
    slot. This is the chord branch of the §13.1 rhythmic-independence test:
    notes that move together are a chord; the rest fall through to become
    their own slots (and later, their own voices)."""
    # GRID-ALIGNED grouping (§XVI.6 leverage #1). When a beat length is known,
    # two notes are the same chord if they land on the SAME notated position
    # AND the same notated length - i.e. after snapping to the 16th grid. The
    # old rule compared raw milliseconds (40 ms onset / 90 ms offset), which is
    # a detection-grade tolerance applied to a pure layout decision: notes the
    # page draws at one position were still split into separate voices, and
    # every extra voice then has to be filled with rests wherever it is silent.
    # That is the measured cause of our 10-15% chord share vs Klangio's 31-57%
    # (§XVII.4). Snapping first is fidelity-free - the notes already render
    # there. Falls back to the raw-ms rule when no tempo is available.
    use_grid = ms_per_beat > 0.0
    step = (ms_per_beat / grid_division) if use_grid else 0.0

    def cell(value: float) -> float:
        return round(value / step) if use_grid else value

    if use_grid:
        # CHORD COLLAPSE (user request 2026-08-06): "when block chords appear -
        # notes stacked vertically and held for the same duration - group them
        # into the same voice, so chords separate from melody and independent
        # movement."
        #
        # Requiring an EXACT end-cell match tore real chords apart: a block
        # chord whose notes release slightly differently (0.9 vs 1.1 beats)
        # snapped to different end cells, became different slots, and therefore
        # landed in DIFFERENT VOICES - each then needing rests wherever the
        # other sounded. Nobody hears that release difference; it is a chord.
        #
        # So: group by onset cell, then cluster by end cell with a tolerance,
        # and let the chord hold as long as its longest member (what a player
        # actually does). Genuinely divergent durations still separate - that is
        # the §13.1 rhythmic-independence test doing its real job.
        # GROUP ON RAW PROXIMITY, THEN SNAP - not the other way round.
        #
        # Grouping by snapped cell alone splits a chord whose notes straddle a
        # cell boundary: two notes 5ms apart round in opposite directions, land
        # in different cells, become different slots, and end up in DIFFERENT
        # VOICES. MEASURED on Ellington's Reflections in D, where the writing is
        # six-note orchestral voicings: every "spread" chord in our output was
        # EXACTLY 214ms wide - one 16th cell at that tempo - which is arithmetic,
        # not piano playing. Mean chord size came out 3.01 against the
        # reference's 6.07, and parallel motion (the planing that defines the
        # piece) collapsed to 4% against 49%.
        #
        # So decide chord membership on the performance clock, where
        # simultaneity is a physical fact, and only then assign the group one
        # cell. The tolerance is half a grid cell, which is by construction the
        # distance at which two onsets would have snapped together had the
        # boundary fallen anywhere else - no new magic number.
        ordered = sorted(notes, key=lambda n: (n.start_ms, n.pitch))
        tol = step * CHORD_RAW_TOLERANCE_CELLS
        raw_groups: List[List[NotationNote]] = []
        for note in ordered:
            if raw_groups and note.start_ms - raw_groups[-1][0].start_ms <= tol:
                raw_groups[-1].append(note)
            else:
                raw_groups.append([note])

        by_onset: Dict[float, List[NotationNote]] = {}
        for group in raw_groups:
            # one cell for the whole group, taken from its earliest onset
            c = cell(group[0].start_ms)
            by_onset.setdefault(c, []).extend(group)

        slots = []
        for onset_cell in sorted(by_onset):
            group = sorted(by_onset[onset_cell], key=lambda n: n.end_ms)
            current: Optional[_Slot] = None
            last_end_cell: Optional[float] = None
            for note in group:
                ec = cell(note.end_ms)
                if (current is not None and last_end_cell is not None
                        and abs(ec - last_end_cell) <= end_tolerance_cells):
                    current.add(note)          # same chord: end within tolerance
                else:
                    current = _Slot(note)
                    slots.append(current)
                last_end_cell = ec
        slots.sort(key=lambda s: s.start)
        return slots

    # RAW-MILLISECOND FALLBACK, used only when no tempo is available (see the
    # grid path above, which is what production takes). CHORD_ONSET_TOL_MS and
    # CHORD_OFFSET_TOL_MS are live here - the module header used to call them
    # dead, which was true of the grid path only.
    slots: List[_Slot] = []
    for note in sorted(notes, key=lambda n: (n.start_ms, n.pitch)):
        placed = False
        for slot in reversed(slots):
            if note.start_ms - slot.start > CHORD_ONSET_TOL_MS:
                break                                   # sorted by onset - no earlier slot can match
            if (abs(note.start_ms - slot.start) <= CHORD_ONSET_TOL_MS
                    and abs(note.end_ms - slot.end) <= CHORD_OFFSET_TOL_MS):
                slot.add(note)
                placed = True
                break
        if not placed:
            slots.append(_Slot(note))
    slots.sort(key=lambda s: s.start)
    return slots


def _smooth_slots(slots: List[_Slot], ms_per_beat: float,
                  merge_gap_beats: float = 0.5, fill_max_beats: float = 1.0,
                  beats_per_bar: int = 0, boundary_reach_beats: float = 2.0,
                  bar_origin_ms: float = 0.0) -> List[_Slot]:
    """Fix the horizontal choppiness within one voice (§16 follow-up: the
    grand staff read well but PLAYED choppy - measured 59% of adjacent notes
    separated by a rest, 52% same-pitch rearticulations, 39% <=16th-note
    fragments). A monophonic voice is a held line, so:

      1. MERGE consecutive same-pitch slots a short gap apart - the
         machine-gun rearticulation Basic Pitch emits for one sustained note
         (this is note_consolidation's insight, applied at the page).
      2. LEGATO-FILL a *small* gap: extend a slot's end to the next slot's
         onset only when the gap is short enough to be a cut-off artifact,
         not a real rest. BOUNDED on purpose - the first attempt filled
         EVERY gap and over-sustained: rest/note fell to 0.02 but playability
         cratered (97%->33%) because every voice then rang continuously. So
         gaps up to `fill_max` close (de-chop); larger gaps stay rests
         (phrasing preserved).

    Both change only the NOTATION clock (NotationNote end_ms), never a frozen
    Note - lawful page timing, exactly the 'hold until the next note unless
    evidence says stop' intent already stated in notation_score.py."""
    if not slots:
        return slots
    merge_gap = merge_gap_beats * ms_per_beat   # same-pitch rearticulation this close = one held note
    fill_max = fill_max_beats * ms_per_beat     # close gaps up to this; leave real rests alone (0 = merge only)
    merged: List[_Slot] = []
    for s in slots:
        prev = merged[-1] if merged else None
        if (prev is not None and len(s.notes) == 1 and len(prev.notes) == 1
                and s.notes[0].pitch == prev.notes[0].pitch
                and s.start - prev.end <= merge_gap):
            prev.end = max(prev.end, s.end)              # absorb the rearticulation
        else:
            merged.append(s)
    # Calibrate the acoustic gate against THIS material's own distribution.
    # An absolute threshold cannot mean the same thing on two songs whose
    # resonance scales differ; a percentile can. Too few observations and the
    # percentiles are noise, so we stay blind and let the flat window rule.
    observed_res = [s.trailing_resonance for s in merged if s.trailing_resonance is not None]
    observed_void = [s.trailing_void for s in merged if s.trailing_void is not None]
    if len(observed_res) >= ACOUSTIC_MIN_SAMPLES:
        import numpy as _np
        ring_at = float(_np.percentile(observed_res, ACOUSTIC_RINGING_PERCENTILE))
        void_at = (float(_np.percentile(observed_void, 100.0 - ACOUSTIC_VOID_PERCENTILE))
                   if len(observed_void) >= ACOUSTIC_MIN_SAMPLES else None)
    else:
        ring_at = void_at = None

    for i in range(len(merged) - 1):
        gap = merged[i + 1].start - merged[i].end
        if gap <= 0:
            continue
        # ACOUSTIC GATE (XVIII.3 #1, recalibrated). AnechoicMa measured this gap:
        # it knows whether the stem is still ringing there (resonance) or
        # genuinely silent (void). That evidence MODULATES the blind window it
        # used to replace - a ringing gap earns extra reach, a silent one is a
        # real rest and stays open, and everything else gets the ordinary fill.
        res = merged[i].trailing_resonance
        void = merged[i].trailing_void
        limit = fill_max
        if ring_at is not None and res is not None:
            silent = void_at is not None and void is not None and void >= void_at
            if silent:
                limit = 0.0                              # the audio says: real rest
            elif res >= ring_at:
                limit = fill_max * ACOUSTIC_RINGING_REACH   # still sounding: reach further
        if gap <= limit:
            merged[i].end = merged[i + 1].start          # legato: close the artifact gaps

    # METRIC-BOUNDARY FILL (user request 2026-08-06): "if a figure starts on 3
    # and lands on the AND, there realistically isn't a rest on the and - it
    # should just be an 8th to the end of the measure."
    #
    # The legato pass above can only close a gap BETWEEN two notes; it extends a
    # note toward the next onset. But the case described has NO next note - the
    # figure ends and the voice falls silent to the barline, so there is nothing
    # to fill toward and the short note keeps a trailing rest nobody plays.
    #
    # The copyist rule: a note followed by silence is written out to its natural
    # metric boundary (end of beat / half-bar / bar) rather than written short
    # with a trailing rest. Bounded deliberately - it NEVER crosses a barline
    # (that would change the harmonic rhythm) and it only extends to the FIRST
    # boundary reached, so a genuine multi-beat rest survives as a rest.
    # Bar boundaries are measured from the RESOLVED DOWNBEAT, not from absolute
    # zero (2026-08-17 audit). Measuring from zero meant this rule's "never
    # cross a barline" guard was enforced against barlines nobody draws - the
    # page's bars start at the downbeat, and the two only coincide by accident.
    if beats_per_bar > 0:
        bar_ms = beats_per_bar * ms_per_beat
        for i, slot in enumerate(merged):
            next_start = merged[i + 1].start if i + 1 < len(merged) else float("inf")
            silence_to = min(next_start, slot.end + boundary_reach_beats * ms_per_beat)
            if silence_to <= slot.end:
                continue
            rel_end = slot.end - bar_origin_ms
            bar_index = math.floor(rel_end / bar_ms)
            # candidate boundaries inside THIS bar, nearest first
            for div in (1.0, 0.5):                        # beat, then half-bar
                step = div * ms_per_beat if div == 1.0 else bar_ms / 2.0
                if step <= 0:
                    continue
                boundary = (math.floor(rel_end / step) + 1) * step
                if boundary + bar_origin_ms > silence_to:
                    continue
                if (math.floor(boundary / bar_ms) != bar_index
                        and abs(boundary % bar_ms) > 1e-6):
                    continue                              # never cross a barline
                if boundary + bar_origin_ms > slot.end:
                    slot.end = boundary + bar_origin_ms
                    break
    return merged


def assign_voices(notes: List[NotationNote], max_voices: int = 4,
                  smooth: bool = False, ms_per_beat: float = 500.0,
                  merge_gap_beats: float = 0.5, fill_max_beats: float = 1.0,
                  beats_per_bar: int = 0, boundary_reach_beats: float = 2.0,
                  bar_origin_ms: Optional[float] = None) -> List[NotationNote]:
    """Stamp voice_index (1..max_voices) on a single staff's notes.

    Interval-partition the chord-events across at most `max_voices` voices
    (optimal for interval graphs), choosing among free voices by register
    continuity so a line stays in one voice. When more than `max_voices`
    events genuinely overlap, MERGE the overflow event into the nearest
    active voice as chord tones (§13.4) rather than spilling a new
    staff/voice - the graceful degradation for the rare dense instant the
    §15 measurement showed is only 3-8% of notes.

    `smooth` (§16 follow-up) additionally de-chops each voice: merges
    same-pitch rearticulations and legato-fills gaps (see _smooth_slots)."""
    slots = _chord_events(notes, ms_per_beat=ms_per_beat)
    if not slots:
        return notes

    # Entries are always real slots - a voice is created BY being given one, so
    # there is no "empty voice" state to represent (the old Optional[...] and
    # the `act is None` test in free_voices below were never reachable).
    voices: List[_Slot] = []                          # active slot per voice
    voice_slots: List[List[_Slot]] = []               # committed slots per voice
    # Grimlock University APPLY: gesture id -> the voice it already owns, so a
    # scale run / arpeggio / sequence stays in ONE voice rather than being
    # scattered across whichever slot happened to be free. Empty (and this
    # whole path inert) unless the University ran in APPLY mode.
    group_voice: Dict[str, int] = {}

    overlap_tol = VOICE_OVERLAP_TOLERANCE_BEATS * ms_per_beat

    def free_voices(start: float) -> List[int]:
        return [i for i, act in enumerate(voices)
                if act.end <= start + overlap_tol]

    def anchor(i: int) -> Optional[float]:
        """This voice's recent REGISTER, not just its last note. Averaged over a
        few slots so one leap doesn't relocate the whole part."""
        vs = voice_slots[i]
        if not vs:
            return None
        recent = vs[-VOICE_ANCHOR_SLOTS:]
        return sum(s.pitch_center for s in recent) / len(recent)

    for slot in sorted(slots, key=lambda s: (s.start, s.pitch_center)):
        free = free_voices(slot.start)
        group = slot.cohesion
        if free:
            def last_pitch(i: int) -> float:
                return voice_slots[i][-1].pitch_center if voice_slots[i] else slot.pitch_center
            # Same gesture -> same voice, whenever that voice is actually free.
            preferred = group_voice.get(group) if group else None
            if preferred is not None and preferred in free:
                i = preferred
            elif VOICE_BY_REGISTER_RANK:
                # REGISTER RANK. A section is stratified: the lead sits on top and
                # each part holds its position in the stack. So a slot belongs to
                # the voice whose register STRATUM it falls in - rank it against
                # the current anchors and take the voice at that rank - rather
                # than to whichever free voice was last nearest in pitch. The old
                # rule let a sustaining lead push the melody into voice 2 and
                # take it back the next onset, which is why the top line changed
                # voice on 52% of onsets.
                anchored = [(a, k) for k in range(len(voices)) if (a := anchor(k)) is not None]
                if anchored:
                    ordered = [k for _, k in sorted(anchored, key=lambda t: -t[0])]
                    rank = sum(1 for a, _ in anchored if a > slot.pitch_center)
                    target = ordered[min(rank, len(ordered) - 1)]
                    i = target if target in free else min(
                        free, key=lambda k: abs((anchor(k) if anchor(k) is not None
                                                 else slot.pitch_center) - slot.pitch_center))
                else:
                    i = free[0]
            else:
                i = min(free, key=lambda k: abs(last_pitch(k) - slot.pitch_center))
            # A voice admitted under the overlap tolerance must stay monophonic:
            # clamp the outgoing slot to this onset so the two never sound at
            # once on the page (an overlapping pair inside one voice is exactly
            # what produces an overfull measure).
            prev_slot = voices[i]
            if prev_slot is not None and prev_slot.end > slot.start:
                prev_slot.end = slot.start
            voices[i] = slot
            voice_slots[i].append(slot)
            if group:
                group_voice[group] = i
        elif len(voices) < max_voices:
            voices.append(slot)
            voice_slots.append([slot])
            if group:
                group_voice[group] = len(voices) - 1
        else:
            i = min(range(len(voices)), key=lambda k: abs(voices[k].pitch_center - slot.pitch_center))
            for n in slot.notes:
                voices[i].add(n)

    if smooth:
        voice_slots = [_smooth_slots(vs, ms_per_beat, merge_gap_beats, fill_max_beats,
                                     beats_per_bar=beats_per_bar,
                                     boundary_reach_beats=boundary_reach_beats,
                                     bar_origin_ms=bar_origin_ms or 0.0)
                       for vs in voice_slots]

    # RELABEL SO VOICE 1 CARRIES THE TUNE. Voice numbers are not arbitrary:
    # notation convention puts the upper part in voice 1 (stems up) and a reader
    # looks there for the melody. Voices are CREATED in order of need, which has
    # nothing to do with musical role, so the top line landed in voice 1 only
    # 38.8% of the time.
    #
    # Rank by TOP-NOTE SHARE, not by mean register. Sorting by mean register was
    # tried and measured WORSE (voice1_is_top 0.413 -> 0.299): background figures
    # sit higher on average than a lead does, so the highest-mean voice is not
    # the melody. What we want is the voice that is most often ON TOP at an
    # onset, which is what a reader actually follows.
    order = list(range(len(voice_slots)))
    if len(voice_slots) > 1 and RELABEL_ENABLED:
        # Rank on the slot's HIGHEST note, not its pitch_center (a mean).
        # A four-note chord averaging 60 can top out at 75 and outrank a
        # single-line melody at 70 - which is how ranking by the mean made
        # voice1_is_top WORSE (0.475 -> 0.436) instead of better. A reader
        # follows the top NOTE, so that is what has to be ranked.
        #
        # SWEEP LINE, not a scan per boundary (2026-08-17 audit). This was
        # "for every distinct onset, look at every slot of every voice" -
        # O(onsets x slots) with a max() inside, i.e. quadratic in note count
        # per staff, on a page that can carry thousands. Walking slot
        # start/end events in time order gives the same answer in O(n log n).
        # (time, kind, voice, top_pitch). kind -1 = a slot ends, +1 = one starts;
        # ends are applied before starts at the same instant, matching the
        # original's half-open `s.start <= t < s.end` test.
        events: List[Tuple[float, int, int, int]] = []
        onsets: set = set()
        for i, vs in enumerate(voice_slots):
            for s in vs:
                onsets.add(s.start)
                if s.end <= s.start:
                    # A zero-length slot never sounds (the test is half-open),
                    # but its start is still an instant to score - some OTHER
                    # voice may be sounding there. Carry it as a no-op marker
                    # so the sweep stops at it; without one the sweep only
                    # visits instants that have a real event and silently drops
                    # those boundaries.
                    events.append((s.start, 0, i, 0))
                    continue
                top = max(n.pitch for n in s.notes)
                events.append((s.start, 1, i, top))
                events.append((s.end, -1, i, top))
        # ends (-1) before markers (0) before starts (+1) at the same instant,
        # which reproduces the half-open `s.start <= t < s.end` test exactly.
        events.sort(key=lambda e: (e[0], e[1]))

        top_share = [0] * len(voice_slots)
        # voice -> the top pitches of every slot of that voice sounding right
        # now. A LIST, not a single value: _smooth_slots can leave two slots of
        # one voice overlapping, and collapsing them to one entry would retire
        # the whole voice when only one of its slots ended.
        sounding: Dict[int, List[int]] = defaultdict(list)
        idx = 0
        while idx < len(events):
            now = events[idx][0]
            while idx < len(events) and events[idx][0] == now:
                _t, kind, voice, top = events[idx]
                if kind > 0:
                    sounding[voice].append(top)
                elif kind < 0 and top in sounding[voice]:
                    sounding[voice].remove(top)
                idx += 1
            # Score this instant only if a slot actually starts at it - the
            # same set of instants the old per-boundary scan scored.
            if now in onsets:
                # Ties go to the LOWEST voice index, which is what the old
                # scan did by construction (it walked voices in order and only
                # replaced its best on a strictly higher pitch).
                best_voice, best_top = None, float("-inf")
                for voice in range(len(voice_slots)):
                    tops = sounding.get(voice)
                    if tops and max(tops) > best_top:
                        best_voice, best_top = voice, max(tops)
                if best_voice is not None:
                    top_share[best_voice] += 1
        order.sort(key=lambda i: -top_share[i])
    relabel = {orig: new for new, orig in enumerate(order)}

    out: List[NotationNote] = []
    for i, vs in enumerate(voice_slots):
        for slot in vs:
            for n in slot.notes:
                out.append(replace(n, voice_index=relabel[i] + 1, end_ms=slot.end))
    return out


# ---------------------------------------------------------------------------
# 3. Orchestration: notes -> grand-staff NotationScore
# ---------------------------------------------------------------------------

def build_grand_staff(
        notes: List[NotationNote],
        tempo_bpm: float,
        time_signature: Tuple[int, int],
        key: Optional[str] = None,
        ratio_family: Optional[str] = None,
        *,
        # USER-CHOSEN BY EAR + MIDI PLAYBACK 2026-08-07. cap=3 is the LOWEST cap
        # that clears the coherence target (mean intra-voice jump <= 5.0) on all
        # three test songs, and the user judged its playback best. cap=2 scored
        # better on rest/note but FAILED coherence on 2 of 3 songs (jump 5.87 /
        # 5.79) - it buys a clean-looking page by cramming unrelated material
        # into one voice. The two axes trade near-linearly (~0.10 rest/note per
        # ~0.7 semitones), so this is a CHOICE, not a maximum.
        max_voices: int = 3,
        switch_penalty: float = SPLIT_SWITCH_PENALTY,
        smooth: bool = True,
        merge_gap_beats: float = 0.5,
        # USER-CHOSEN BY EAR 2026-08-06 (variant "C_beat"): close gaps up to a
        # full beat. Measured 1.30x raw sounding time - well inside what ships
        # (Klangio is 2.68x on material the user finds MORE readable), and it
        # removes 69% of the sub-half-beat artifact gaps that made the page
        # stutter. The earlier 0.0 default was mine, from an invented 1.6x
        # fidelity bar that §XVI.5 already recorded as stricter than the
        # commercial benchmark.
        fill_max_beats: float = 1.0,
        family: str = "piano",
        beats_per_bar: int = 0,
        bar_origin_ms: Optional[float] = None,
) -> NotationScore:
    """Assemble a two-part (treble/bass) grand-staff NotationScore with <=
    max_voices per staff already assigned. Each part is one braced-staff
    hand; the exporter honors the stamped voice_index verbatim.

    `smooth` (default on) de-chops each voice. Measured against the naked-BP
    baseline (§17), the two halves behave differently, so they are split:
      - same-pitch MERGE (always) - removes machine-gun rearticulation and
        moves note count CLOSER to Basic Pitch (n/BP 1.66 -> 1.35): pure win.
      - legato GAP-FILL (`fill_max_beats`) - closes rests but INVENTS sustain
        beyond what BP heard (sndT/BP 1.57 -> 1.69 -> 1.95 as fill grows).
        Defaults to 1.0 (a full beat), user-chosen by ear 2026-08-06; set 0.0
        to disable it and keep only the same-pitch merge. The docstring used to
        claim the default was 0/OFF, which stopped being true when the measured
        value landed and was never updated."""
    ms_per_beat = 60000.0 / max(tempo_bpm, 1.0)
    hands = assign_hands(notes, tempo_bpm, switch_penalty=switch_penalty)
    parts: List[NotationPart] = []
    for label in ("treble", "bass"):
        hand_notes = hands.get(label, [])
        if not hand_notes:
            continue
        voiced = assign_voices(hand_notes, max_voices=max_voices,
                               smooth=smooth, ms_per_beat=ms_per_beat,
                               merge_gap_beats=merge_gap_beats, fill_max_beats=fill_max_beats,
                               beats_per_bar=beats_per_bar, bar_origin_ms=bar_origin_ms)
        voiced.sort(key=lambda n: (n.start_ms, n.pitch))
        parts.append(NotationPart(
            family=family, voice_id=label, stem=StemType.OTHER,
            notes=voiced, is_drum=False,
        ))
    return NotationScore(
        parts=parts, tempo_bpm=tempo_bpm, time_signature=time_signature,
        key=key, ratio_family=ratio_family,
    )


__all__ = [
    "assign_hands", "assign_voices", "build_grand_staff",
    "SPAN_MAX_SEMITONES", "FINGERS_PER_HAND", "SPLIT_SWITCH_PENALTY",
]
