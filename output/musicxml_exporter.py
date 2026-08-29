# =================================================================
# MODULE: output/musicxml_exporter.py
# Serializes a NotationScore to MusicXML via music21. A VIEW - it reads
# the resolved symbolic object and never touches Notes or annotations.
#
# The whole reason MusicXML and not (only) MIDI: MIDI cannot carry
# separate voices as separate staves, so the guitar's independent lines
# collapse into one polyphonic mush on the page.
#
# HARD CONSTRAINT (learned the corrupted-file way, 2026-07-20 - see
# GRIMLOCK_6.0_DESIGN_DECISIONS.md): a NotationPart is NOT guaranteed
# monophonic. When voice separation collapses to a family (or over-splits
# and gets re-merged), a part carries overlapping notes. The first
# exporter inserted them flat and let makeNotation invent voices - it
# produced SEVEN voices numbered from 0 on one staff, and MuseScore
# rejected the whole file as "corrupted" because:
#   (a) MusicXML voice numbers must start at 1, never 0, and
#   (b) MuseScore represents at most 4 voices per staff.
# So this VIEW must guarantee validity structurally, whatever it is fed:
# split every part into monophonic sub-voices itself, number them from 1,
# and cap at MAX_VOICES_PER_STAFF - spilling any overflow onto additional
# staves rather than emitting an un-representable staff. A view may render
# an upstream mess uglily; it may never emit something that won't open.
# =================================================================

from __future__ import annotations

import copy
import math

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from output.notation_score import NotationScore, NotationPart, NotationNote
from fractions import Fraction
from output.beat_hierarchy import split_for_hierarchy
from quantization.duration_witness import nearest_notatable, notatable_at_most
from output.ma_legato import calibrate_from_notes

# music21 quarterLength is in quarter notes. ms -> quarterLength needs the
# tempo: at T BPM one quarter note is 60000/T ms.
_MIN_QUARTER_LENGTH = 0.125    # a 32nd note - the finest we let a duration round to

# MuseScore (and most engravers) represent at most four voices per staff.
# More than this is not a prettiness question - it is invalid, and the
# file will not open. Overflow voices spill to a new staff of the same
# instrument instead.
MAX_VOICES_PER_STAFF = 4

# Two notes whose onsets are within this window are treated as struck
# together - a chord in one voice - rather than forced into separate
# voices. Keeps a strum on one staff instead of exploding it.
_CHORD_ONSET_WINDOW_MS = 30.0

# General MIDI program per family, mirroring the MIDI engraver's table so
# a MusicXML playback and a MIDI playback pick the same voice.
_GM_PROGRAM: Dict[str, int] = {
    "bass": 33, "vocals": 52, "guitar": 27, "piano": 0,
    "warm_sustained": 48, "mid_body": 25, "bright_lead": 56,
}


# Ratio families (from the ReverseGeoCrypt-style lattice witness) in which
# tuplets are musically plausible. On binary material a "triplet" is almost
# always a per-note rounding artifact, so the gate closes and everything snaps
# binary regardless of any single note's rounded value.
_TERNARY_RATIO_FAMILIES = frozenset({"ternary", "swing"})


# Durations a single notehead can actually express, in quarter-lengths. The
# table itself lives in quantization.duration_witness - rhythm_inference needs
# the same vocabulary when it decides how long to hold a note, and two copies
# of one fact is how the two layers drift apart (2026-08-17 audit).
_representable = nearest_notatable
_representable_at_most = notatable_at_most


# A ternary value has to beat the binary one by this margin (in quarter-lengths)
# before it wins. Ties and near-ties go binary: a triplet is the more expensive
# reading for both the engraver and the player, so it should have to earn the
# call rather than win a coin flip.
_TERNARY_PREFERENCE_MARGIN = 1e-4


def _snap_quarter_length(ql: float, allow_triplet: bool, divisor=None,
                         subdivision: int = 4):
    """Snap a quarter-length to the 16th grid, or - when this note carried a
    beat-level TRIPLET VERDICT and the piece's ratio family permits tuplets -
    to whichever of the 16th and triplet grids is NEARER. The verdict comes
    from rhythm_inference (decided once per beat), never re-guessed here from
    the rounded duration: that per-note guessing manufactured 3210 tuplets >
    4971 noteheads on the first pass.

    BUG FIX (2026-08-17 audit): `allow_triplet` used to REPLACE the binary grid
    rather than widen the vocabulary, so a plain eighth inside a tuplet-flagged
    beat snapped 0.5 -> 0.667. That put binary and ternary positions in the same
    measure, and their differences are exactly the un-notatable leftovers that
    made music21 invent ratios like 24:13. "Allow" now means allow, not force.
    """
    # THE BINARY GRID IS THE BEAT'S OWN, not a fixed sixteenth lattice.
    #
    # This was `round(ql * 4.0) / 4.0` unconditionally - every onset in every
    # song snapped to the nearest SIXTEENTH, whatever subdivision the beat had
    # actually been read as. rhythm_inference decides that subdivision per beat
    # and, until now, threw the number away on binary beats (it kept it only
    # for tuplets), so the page had no way to know a beat was eighths.
    #
    # Measured on Clocks, whose published human transcription is 100% on the
    # eighth grid and contains no sixteenth in 160 bars: our onsets landed on
    # that grid 90.6% of the time and the page still came out 42% sixteenths,
    # with 89% of those sixteenth noteheads sitting on CORRECT onsets. A
    # sixteenth lattice invites sixteenth-length fragments even when every
    # onset is right, because the gap to the next snapped onset can be a
    # sixteenth.
    #
    # A beat read as 2 parts snaps to halves of a beat; as 4, to quarters. The
    # fallback stays 4 so any caller that does not know the subdivision behaves
    # exactly as before.
    grid = max(1, int(subdivision or 4))
    binary = round(ql * grid) / float(grid)
    if divisor:
        # This beat was READ as a `divisor`-part tuplet, so its own grid is the
        # right one to land on - and EXACTLY, as a Fraction. Summing a handful
        # of binary-float sixths lands on no rational music21 recognises, which
        # is where 24:13 and 48:43 came from: not tuplets anybody detected,
        # just accumulated float error being reconciled.
        tuplet = Fraction(round(ql * divisor), divisor)
        if abs(ql - float(tuplet)) + _TERNARY_PREFERENCE_MARGIN < abs(ql - binary):
            return tuplet
        return binary
    if not allow_triplet:
        return binary
    ternary = round(ql * 3.0) / 3.0       # nearest triplet-eighth
    if abs(ql - ternary) + _TERNARY_PREFERENCE_MARGIN < abs(ql - binary):
        return ternary
    return binary


# THE CONVERSION THAT DECIDES WHICH BAR A NOTE LANDS IN.
#
# It used to divide by a constant `60000/tempo_bpm`. MEASURED (2026-08-11):
# fixing the quantizer's snap alone moved Chopin recall only +9.6%, because
# this then converted the correctly-placed milliseconds back to bar positions
# at a fixed rate and undid it. A tempo map has to be the ONLY currency, or
# every site that isn't converted silently reverts the ones that are.
# `musical_time` is threaded through from the score; when it is absent the old
# constant-rate behaviour is preserved exactly.
#
# There used to be a second conversion here, `_quarter_length`, computing a
# note's DURATION the same way. It was dead: build_music21_score derives every
# duration from the difference of two snapped OFFSETS (so each voice tiles its
# bar exactly) and overwrote _quarter_length's result unconditionally. Deleted
# 2026-08-17 rather than left looking load-bearing - a duration measured
# through this function is a duration measured through musical_time, since both
# of its endpoints are.

def _offset_quarter_length(start_ms: float, origin_ms: float, tempo_bpm: float,
                           allow_triplet: bool = False, musical_time=None,
                           divisor=None, subdivision: int = 4):
    if musical_time is not None and musical_time.usable:
        pos = musical_time.to_beats(start_ms) - musical_time.to_beats(origin_ms)
        return max(0.0, _snap_quarter_length(pos, allow_triplet, divisor,
                                            subdivision))
    ms_per_quarter = 60000.0 / max(tempo_bpm, 1.0)
    return max(0.0, _snap_quarter_length((start_ms - origin_ms) / ms_per_quarter,
                                         allow_triplet, divisor, subdivision))


# Denominator cap when converting a snapped offset to an exact Fraction. The
# finest thing the rhythm layer emits is a twelfth of a beat, so 96 leaves
# generous headroom while stopping a float artifact from becoming a 10007ths.
_HIERARCHY_MAX_DEN = 96


def _chain_tie(el, index: int, total: int) -> None:
    """Tie one link of a hierarchy-split chain to its neighbours.

    A split note is ONE sounding note wearing several noteheads, so the ties
    are what stop it being read as several notes. Any tie the note already
    carried (from tie_reconstruction) is subsumed: the chain's last link keeps
    an existing tie-forward, the first keeps an existing tie-back."""
    from music21 import tie as m21tie
    had = el.tie.type if el.tie is not None else None
    first, last = index == 0, index == total - 1
    if first:
        el.tie = m21tie.Tie("continue" if had in ("stop", "continue") else "start")
    elif last:
        el.tie = m21tie.Tie("continue" if had in ("start", "continue") else "stop")
    else:
        el.tie = m21tie.Tie("continue")


def _clamp_to_beat_frame(onset, ql):
    """Shorten `ql` so the event cannot leave the beat it started in.

    A tuplet is a statement about how ONE metric unit is divided, so it must
    end inside that unit - and since a beat never spans a barline, staying
    inside the beat makes crossing a barline impossible too. Returns the
    clamped length, which may be <= 0 when the event starts exactly on the
    frame edge; the caller drops those.

    CLAMP, NOT REJECT (corrected 2026-08-18). The previous version of this
    guard rejected any tuplet whose onset was not exactly on a slot and fell
    back to the binary grid - but the smallest binary value is 1/8 of a beat,
    so a 1/12-beat scrap was rounded UP, and lengthening notes pushed 50 of
    them over barlines where there had been 1. Shortening is always safe;
    lengthening never is.

    MEASURED ORIGIN (Chopin bar 29): the gap clamp handed _apply_tuplet a
    1/24-beat leftover at offset 1.5417, which it faithfully rendered as a
    64th triplet in a frame anchored to nothing, running 0.042 past the bar."""
    import math as _math
    frame_end = _math.floor(float(onset) + 1e-9) + 1.0
    return min(float(ql), frame_end - float(onset))


# The finest unit a legato FILL may land on. A fill is not restricted to a
# single notatable value the way an isolated duration is: split_for_hierarchy
# renders whatever it is given as the tied chain the reader expects, so
# limiting the fill to one note value throws reach away for nothing.
# _representable_at_most rounded 47.4% of fill requests DOWN, a median of 0.333
# quarters each - 1.25 to 1.0, 2.5 to 2.0, 1.75 to 1.5 - and every one of those
# is an ordinary tie.
_FILL_GRID_QL = 0.5


def _fill_value_at_most(value: float, unit: float = _FILL_GRID_QL) -> float:
    """Largest multiple of `unit` not exceeding `value`. Never lengthens."""
    if value <= 0 or unit <= 0:
        return 0.0
    return max(0.0, (int((value + 1e-9) / unit)) * unit)


def _close_unwritable_gaps(m_voice) -> int:
    """Close gaps too short to be written, by holding the previous note.

    WHY THIS EXISTS. split_for_hierarchy never lengthens a note, so when a
    duration cannot be tiled exactly it drops the residue - correctly, because
    growing a note pushes the next onset. But the residue leaves a GAP, and the
    makeRests(fillGaps=True) call downstream fills every gap with whatever
    music21 can find for it. Handed 1/12 of a beat it writes a 32nd triplet
    rest, and one leftover on Clocks came out as a tuplet of 12 - both barred
    outright, and both present in every export including the shipped one.

    A gap shorter than a sixteenth is not a rest anybody performs; it is the
    previous note still sounding. Holding that note is also the only move here
    that CANNOT push anything: the following onset is already fixed, and the
    fill is bounded by it, so nothing downstream shifts. Gaps a sixteenth or
    longer are real and are left completely alone.

    Skips tuplet members - their frame is anchored to a beat and is not this
    pass's business - and any extension whose result would not be a writable
    value, since replacing an unwritable rest with an unwritable note would
    only move the defect.
    """
    closed = 0
    items = sorted(m_voice.notesAndRests, key=lambda e: (float(e.offset),
                                                         float(e.quarterLength)))
    for cur, nxt in zip(items, items[1:]):
        end = float(cur.offset) + float(cur.quarterLength)
        gap = float(nxt.offset) - end
        if gap <= 1e-9 or gap >= _MIN_WRITTEN_VALUE - 1e-9:
            continue
        wanted = float(nxt.offset) - float(cur.offset)
        # Tuplet members are where these gaps actually live - an unfilled slot
        # inside a triplet frame is what makeRests turns into a 32nd triplet
        # rest. Their arithmetic is not ours to reason about, so the extension
        # is TRIED and reverted unless music21 resolves it to a value the page
        # is allowed to show. Skipping them entirely (the first version of this
        # pass) closed nothing at all, which is how they were found.
        before = cur.quarterLength
        try:
            cur.quarterLength = wanted
        except Exception:
            continue
        if _writable_frame_marker(cur):
            closed += 1
        else:
            cur.quarterLength = before
    return closed


def _leading_tuplet_rest(onset, divisor: int):
    """The tuplet rest that holds the frame on the beat when the first sounding
    slot is not slot 0 - a rest on "Tri" before a note on "ple".

    Returns (rest_offset, rest_element) or None when the note already starts on
    its beat. The rest carries the SAME tuplet as the notes, which is what
    keeps the container visibly anchored; an ordinary rest of the same
    clock-length would sit outside the bracket and put the reader back where
    they started, hunting for the beat.

    Generalises to any divisor: the silent span is (onset - beat start), which
    is always a whole number of tuplet slots because the onset was snapped onto
    that grid."""
    from music21 import note as m21note
    beat_start = Fraction(int(Fraction(onset))) if onset >= 0 else Fraction(0)
    silent = Fraction(onset) - beat_start
    if silent <= 0:
        return None
    rest = m21note.Rest()
    # If the silence is too short to WRITE, there is no anchoring rest to
    # insert. _apply_tuplet now returns 0.0 rather than a raw length for that
    # case (a raw length makes music21 invent a 32nd-note tuplet from it), and
    # a zero-length rest cannot be serialized at all - "Cannot convert
    # durations without types". No rest is the right answer: the frame simply
    # is not anchored here, which the tuplet audit will report honestly.
    if _apply_tuplet(rest, silent, divisor) <= 0:
        return None
    # ENFORCED ON THE WRITTEN RESULT, not on the length that went in. The
    # intent above was always "too short to write means no rest", but it was
    # checked before music21 chose a note type, and music21 still resolved
    # some frames to a 32nd triplet rest or a 12-tuplet - nine of them on
    # Clocks, in every version including the shipped one. A 32nd and a tuplet
    # of 12 are both barred outright (GRIMLOCK_6.0_DESIGN_DECISIONS, the
    # notation rules): if the frame marker cannot be written in the ordinary
    # vocabulary, the frame simply is not anchored here, which is exactly what
    # this function already says it should do.
    if not _writable_frame_marker(rest):
        return None
    return float(beat_start), rest


# The page's whole legal vocabulary for a frame-marker rest. A 32nd or finer is
# never acceptable, and a tuplet that is not a duplet, triplet or sextuplet
# means the beat grid is wrong rather than the note being unusual.
_FORBIDDEN_WRITTEN_TYPES = frozenset({"32nd", "64th", "128th", "256th"})
_ALLOWED_TUPLET_ACTUALS = frozenset({2, 3, 6})


def _writable_frame_marker(el) -> bool:
    """Is this rest something the ordinary vocabulary can actually show?"""
    try:
        if el.duration.type in _FORBIDDEN_WRITTEN_TYPES:
            return False
        for t in (el.duration.tuplets or ()):
            if int(t.numberNotesActual) not in _ALLOWED_TUPLET_ACTUALS:
                return False
    except Exception:
        return False
    return True


# Which "in the time of" a subdivision is written against: the largest power of
# two at or below it, which is the convention every edition uses - 6 in the
# time of 4, 5 in the time of 4, 7 in the time of 4, 12 in the time of 8.
# A sixteenth, in quarter-lengths. The finest notehead this project will draw.
_MIN_WRITTEN_VALUE = 0.25

# How far a drum hit may be stretched to reach the next attack. One beat: past
# that the gap is a real silence, and the published kit part writes it as a rest.
_DRUM_MAX_FILL_QL = 1.0

# The ordinary window a pitched gap may be closed by when Anechoic Ma has no
# confident opinion - half a beat, the same blind default the page used before
# the witness was consulted at all. A ringing gap earns RINGING_REACH x this.
_MA_BASE_FILL_QL = 0.5

# Telemetry for the two fills above. A pass that silently does nothing is the
# failure mode these counters exist to make impossible - the same reason
# onset_refined_count and wobble_group_count are surfaced on PipelineResult.
LAST_FILL_STATS: Dict[str, int] = {}


def _writable_or_zero(ql: float) -> float:
    """The largest value we are willing to WRITE that fits inside `ql`.

    ASSIGNING A RAW LENGTH IS NOT NEUTRAL, which is the trap this exists to
    close. Setting el.quarterLength to 0.1 does not produce "a short note" -
    music21 derives a 10:8 tuplet with a thirty-second notehead from it,
    because that is the only way 1/10 of a quarter can be written. So every
    fallback path that used to hand back the raw length was quietly MAKING the
    forbidden tuplets it was trying to avoid: 148 of them on one Chopin page,
    every one a 32nd inside a 10:8.

    Returns 0.0 when nothing writable fits, and callers drop those - a note
    too short to write is evidence the beat grid is wrong, and dropping it
    surfaces that where a 32nd-note decuplet buried it.
    """
    return notatable_at_most(float(ql))


def _normal_count(divisor: int) -> int:
    n = 1
    while n * 2 <= divisor:
        n *= 2
    return n


def _apply_tuplet(el, ql, divisor: int):
    """Write `el` as a note inside a `divisor`-tuplet of one beat, lasting no
    longer than `ql`. Returns the quarter-length actually used.

    WHY THIS RETURNS A LENGTH, AND WHY IT ONLY EVER SHRINKS (2026-08-18).
    A tuplet duration is (un-tupleted note value) x (normal/actual), so the
    un-tupleted value has to be a REAL note type - a half, a quarter, a dotted
    eighth. Handing music21 an arbitrary quarter-length and letting it derive
    one lets it round UP, and a tuplet that came back longer than asked is what
    put 64 notes past their barline after the caller had carefully clamped them
    to the beat. So the value is chosen here, from the notatable set, rounding
    DOWN - shortening is always safe, lengthening never is.

    Left to itself music21 also derives the BRACKET from the quarter-length
    alone, and 1/6 is as validly a sixteenth triplet (two 3:2 groups) as a
    sextuplet (one 6:4). It picks the triplet, so a sextuplet - 297 of the 507
    tuplets in the Chopin edition - never appeared under its own name however
    correctly we detected it. The subdivision is known by this point; stating
    it beats letting the serializer infer an equivalent-but-different reading."""
    from music21 import duration as m21duration
    normal = _normal_count(divisor)
    if normal == divisor:                 # a power of two is not a tuplet
        el.quarterLength = float(ql)
        return float(ql)
    # Largest REAL note value that still fits inside the requested length once
    # the tuplet ratio is applied.
    untupleted = notatable_at_most(float(ql) * divisor / normal)
    used = untupleted * normal / divisor
    if used <= 0 or used > float(ql) + 1e-9:
        fallback = _writable_or_zero(ql)
        el.quarterLength = fallback
        return fallback
    # NOTHING FINER THAN A SIXTEENTH GETS A BRACKET (user directive,
    # 2026-08-22). The written value is `untupleted`; if that is below a
    # sixteenth then the notehead inside the bracket is a 32nd or worse, which
    # is forbidden outright - and music21 will happily build the Tuplet anyway
    # and then fail at serialization with "Cannot convert 2048th duration to
    # MusicXML", which is how this guard was found. Falling back to the plain
    # value is right: a rhythm too fine to write is evidence the beat grid is
    # wrong, and it should surface as a coarse note rather than as a bracket
    # nobody can read.
    if untupleted < _MIN_WRITTEN_VALUE - 1e-9:
        fallback = _writable_or_zero(ql)
        el.quarterLength = fallback
        return fallback
    try:
        el.duration = m21duration.Duration(untupleted)
        el.duration.appendTuplet(m21duration.Tuplet(divisor, normal))
    except Exception:
        el.quarterLength = _writable_or_zero(used)
    return float(el.quarterLength)


# Percussion staff placement (GRIMLOCK_6.0_OPEN_PROBLEMS.md §XVI.10 item 2).
# Drums are NOT pitched: writing them as MIDI 36/38/42 on a normal staff puts a
# kick on F2 and a hi-hat on F#3, which is wrong notation - no drummer reads
# that, and it was our worst staff in the Klangio comparison (their drums:
# proper <unpitched>, rest/note 0.02; ours: pitched, 0.74). MusicXML expresses
# a drum as <unpitched> with a display-step/display-octave saying WHERE on the
# staff the notehead sits, plus a percussion clef.
#
# Positions follow the standard 5-line drum-set convention (and match what
# Klangio emitted on the same audio: kick F4, snare C5, hi-hat G5).
_DRUM_STAFF_POSITION: Dict[int, Tuple[str, int, Optional[str]]] = {
    35: ("F", 4, None), 36: ("F", 4, None),          # kick
    37: ("C", 5, "x"),                                # side stick
    38: ("C", 5, None), 40: ("C", 5, None),          # snare
    39: ("C", 5, "x"),                                # hand clap
    41: ("A", 4, None), 43: ("A", 4, None),          # low tom
    45: ("B", 4, None), 47: ("D", 5, None),          # mid toms
    48: ("E", 5, None), 50: ("F", 5, None),          # high toms
    42: ("G", 5, "x"),                                # closed hi-hat
    44: ("D", 4, "x"),                                # pedal hi-hat
    46: ("G", 5, "circle-x"),                         # open hi-hat
    49: ("A", 5, "x"), 57: ("A", 5, "x"),            # crash
    51: ("F", 5, "x"), 59: ("F", 5, "x"),            # ride
    52: ("B", 5, "x"), 55: ("B", 5, "x"),            # china / splash
    53: ("F", 5, "diamond"),                          # ride bell
}
_DRUM_DEFAULT = ("C", 5, None)


def _unpitched_for(pitch: int):
    """A music21 Unpitched placed at the conventional drum-staff position."""
    from music21 import note as m21note
    step, octave, notehead = _DRUM_STAFF_POSITION.get(int(pitch), _DRUM_DEFAULT)
    u = m21note.Unpitched()
    # music21 renamed these across versions; set whichever exists.
    for attr, value in (("displayStep", step), ("displayOctave", octave)):
        try:
            setattr(u, attr, value)
        except Exception:
            pass
    if notehead:
        try:
            u.notehead = notehead
        except Exception:
            pass
    return u


def _m21_key(key: str):
    """The pipeline's key string ("C", "Am") as a music21 Key, or None if it
    can't be parsed (a bad key means no key signature, never a crash).

    BUG FIX (2026-08-17 audit): this used to be `Key(key.replace("m", ""))`,
    which hands music21 the PARALLEL MAJOR - Key("A") is A major, three sharps,
    where A minor has none. Every minor-key export therefore carried the wrong
    key signature and accidentals on nearly every diatonic note. music21
    distinguishes mode by CASE: Key("a") is A minor. scribe_engraver's
    _key_to_key_number already got this right for MIDI, so the two exports of
    the same transcription disagreed about its key."""
    from music21 import key as m21key
    if not key:
        return None
    key = key.strip()
    is_minor = key.endswith("m")
    tonic = key[:-1] if is_minor else key
    if not tonic:
        return None
    try:
        return m21key.Key(tonic.lower() if is_minor else tonic)
    except Exception:
        return None


def _clef_for(mean_pitch: float):
    from music21 import clef
    # Guitar/vocals read an octave high in treble (8vb); low parts get bass.
    if mean_pitch >= 60:
        return clef.TrebleClef()
    if mean_pitch >= 52:
        return clef.Treble8vbClef()
    return clef.BassClef()


def _part_name(family: str, voice_tag: str, staff_index: int, staff_count: int) -> str:
    base = f"{family} {voice_tag}" if voice_tag and voice_tag != "all" else family
    # When one part had to spill onto extra staves, tag them so a reader
    # can see they are the same instrument.
    return base if staff_count == 1 else f"{base} [{staff_index + 1}]"


# Group chord events on the QUANTIZED GRID rather than by raw-millisecond
# proximity (GRIMLOCK_6.0_OPEN_PROBLEMS.md §XVI.6, leverage item #1).
#
# The old rule grouped notes struck within _CHORD_ONSET_WINDOW_MS (30 ms) and
# only THEN quantized - so two notes that the page places at the SAME notated
# position could still be split into separate voices merely because they were
# performed 40 ms apart. That is a detection-grade tolerance applied to what is
# purely a layout decision, and it is the measured cause of our chord share
# sitting at 10-15% against Klangio's 31-57% (§XVII.4): we split into voices
# what they stack into chords, and every extra voice must then be filled with
# rests for all the time it is not sounding.
#
# Grouping AFTER the snap is fidelity-free by construction: the notes already
# render at that position, so stacking them changes nothing about pitch or
# time - only how many rhythmic slots (and therefore voices, and therefore
# rests) the staff needs.
def _origin_before(bar_origin_ms, earliest_ms, beats_per_bar, tempo_bpm, musical_time):
    """Step the bar origin back whole bars until it is at or before the first
    note, so a pickup sits inside a leading bar instead of moving every barline.

    Whole BARS, never a partial step: the origin has to stay a real downbeat or
    the barlines stop meaning anything, which is the whole point of having
    resolved a downbeat at all."""
    if bar_origin_ms <= earliest_ms:
        return float(bar_origin_ms)
    beats = max(1, int(beats_per_bar))
    if musical_time is not None and getattr(musical_time, "usable", False):
        origin_b = musical_time.to_beats(bar_origin_ms)
        earliest_b = musical_time.to_beats(earliest_ms)
        bars_back = math.ceil((origin_b - earliest_b) / beats)
        return float(musical_time.to_ms(origin_b - bars_back * beats))
    ms_per_bar = (60000.0 / max(tempo_bpm, 1.0)) * beats
    bars_back = math.ceil((bar_origin_ms - earliest_ms) / ms_per_bar)
    return float(bar_origin_ms - bars_back * ms_per_bar)


def _chord_events_gridded(
        notes: List[NotationNote], tempo_bpm: float, origin_ms: float,
        ratio_family: Optional[str], musical_time=None) -> List[List[NotationNote]]:
    buckets: Dict[float, List[NotationNote]] = {}
    for nn in sorted(notes, key=lambda n: (n.start_ms, n.pitch)):
        allow = bool(nn.is_tuplet) or (ratio_family in _TERNARY_RATIO_FAMILIES)
        offset = _offset_quarter_length(nn.start_ms, origin_ms, tempo_bpm, allow,
                                        musical_time=musical_time,
                                        divisor=getattr(nn, "tuplet_divisor", None),
                                        subdivision=int(getattr(nn, "subdivision", 4) or 4))
        buckets.setdefault(round(offset, 6), []).append(nn)
    return [buckets[k] for k in sorted(buckets)]


def _chord_events(notes: List[NotationNote]) -> List[List[NotationNote]]:
    """Groups notes whose onsets fall within _CHORD_ONSET_WINDOW_MS into
    one event. Notes struck together are a chord, not separate voices -
    seven notes hit at once is ONE seven-note chord, which is both correct
    and keeps them on one staff instead of exploding into seven voices."""
    events: List[List[NotationNote]] = []
    cur: List[NotationNote] = []
    cur_onset = 0.0
    for nn in sorted(notes, key=lambda n: n.start_ms):
        if cur and nn.start_ms - cur_onset > _CHORD_ONSET_WINDOW_MS:
            events.append(cur)
            cur = []
        if not cur:
            cur_onset = nn.start_ms
        cur.append(nn)
    if cur:
        events.append(cur)
    return events


def _events_to_voices(events: List[List[NotationNote]]) -> List[List[List[NotationNote]]]:
    """Assigns chord-events to monophonic voices with REGISTER CONTINUITY:
    among the voices free at this event's onset, pick the one whose last
    event sat closest in pitch, instead of the first free slot. A melodic
    line then stays in ONE voice instead of being scattered across whatever
    slot happened to be open.

    MEASURED (2026-07-28 voice-separation experiment, Hopeful, 2337 events):
    at the SAME voice count as first-free (9), this cuts mean intra-voice
    pitch-jump from 8.2 to 2.5 semitones - coherent lines for free. It beat
    a global min-cost path-cover on both axes (the graph floors at ~177
    voices because it snaps a voice at every rest-gap). This is the cheap
    win; it does NOT reduce gappiness (~inherent to notating real polyphony)
    and does NOT change the voice count, so it is rest-neutral - unlike the
    register+sparse-merge combo that was tried and reverted for raising the
    rest ratio.

    Rendering-only: guarantees legal MusicXML, not musical truth. Genuine
    voice assignment is the upstream data-association problem (§XIV.3), which
    the experiment showed voice is under-determined for (edge entropy 0.68)."""
    voices: List[List[List[NotationNote]]] = []
    free_at: List[float] = []
    last_pitch: List[float] = []
    for ev in sorted(events, key=lambda e: min(n.start_ms for n in e)):
        start = min(n.start_ms for n in ev)
        end = max(n.end_ms for n in ev)
        pitch = sum(n.pitch for n in ev) / len(ev)
        free = [i for i, f in enumerate(free_at) if start >= f]
        if free:
            i = min(free, key=lambda k: abs(last_pitch[k] - pitch))
            voices[i].append(ev)
            free_at[i] = end
            last_pitch[i] = pitch
        else:
            voices.append([ev])
            free_at.append(end)
            last_pitch.append(pitch)
    return voices


def _voices_from_index(notes: List[NotationNote], grid=None) -> List[List[List[NotationNote]]]:
    """Honor an upstream voicer's explicit voice_index (e.g.
    piano_reduction's <=4 rhythmic-independence voicer): one voice per
    distinct voice_index, chord-grouped within each. Unlike
    _events_to_voices, this does NOT re-guess voicing - it renders the
    decision it was handed. Same return shape (voices -> events -> chord
    notes) so the rest of the exporter is unchanged."""
    by_index: Dict[int, List[NotationNote]] = defaultdict(list)
    for n in notes:
        by_index[n.voice_index].append(n)
    chunk = grid if grid is not None else _chord_events
    return [chunk(by_index[vi]) for vi in sorted(by_index)]


def _staff_groups(voices: List) -> List[List]:
    """Chunks voices into staves of at most MAX_VOICES_PER_STAFF."""
    return [voices[i:i + MAX_VOICES_PER_STAFF]
            for i in range(0, len(voices), MAX_VOICES_PER_STAFF)]


def _normalize_voice_numbers(path: str) -> None:
    """Belt-and-suspenders: rewrite every part's <voice> numbers to a
    dense 1..N per part. music21's own export numbering is not reliably
    1-based (it emitted voice 0, which MuseScore rejects), so this makes
    the invariant true in the bytes on disk regardless of what music21
    chose.

    BUG FIX (2026-08-17 audit): this used ElementTree.parse/write, which does
    not preserve the DOCTYPE - so every export silently lost music21's
    `<!DOCTYPE score-partwise PUBLIC ...>` line. MuseScore tolerates that;
    validating parsers do not, and this module's own law is that a view may
    render an upstream mess uglily but may never emit something that won't
    open. The doctype is captured before the rewrite and restored after."""
    import re
    import xml.etree.ElementTree as ET

    with open(path, "r", encoding="utf-8") as fh:
        original = fh.read()
    doctype_match = re.search(r"^<!DOCTYPE[^>]*>", original, re.MULTILINE)
    doctype = doctype_match.group(0) if doctype_match else None

    tree = ET.parse(path)
    root = tree.getroot()
    for part in root.findall(".//part"):
        seen: Dict[str, str] = {}
        for v in part.iter("voice"):
            old = v.text
            if old not in seen:
                seen[old] = str(len(seen) + 1)
            v.text = seen[old]
    tree.write(path, encoding="UTF-8", xml_declaration=True)

    if doctype is not None:
        with open(path, "r", encoding="utf-8") as fh:
            rewritten = fh.read()
        if "<!DOCTYPE" not in rewritten:
            # Slot it back between the XML declaration and the root element,
            # which is the only place a doctype is legal.
            lines = rewritten.split("\n", 1)
            if len(lines) == 2 and lines[0].lstrip().startswith("<?xml"):
                rewritten = f"{lines[0]}\n{doctype}\n{lines[1]}"
            else:
                rewritten = f"{doctype}\n{rewritten}"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(rewritten)


def build_music21_score(score: NotationScore, grid_chords: bool = True):
    """Returns a music21.stream.Score. Each NotationPart becomes one or
    more staves; each staff carries at most MAX_VOICES_PER_STAFF voices,
    numbered from 1 - always valid, whatever the input polyphony."""
    from music21 import stream, note as m21note, chord as m21chord, tempo as m21tempo
    from music21 import meter as m21meter, key as m21key, instrument as m21instrument
    from music21 import tie as m21tie


    # WHERE BAR 1 STARTS, AND WHERE THE PICKUP GOES.
    #
    # The resolved downbeat when the pipeline found one; otherwise the earliest
    # note, which is only right by luck - a score whose first note is an upbeat
    # gets every barline shifted by that upbeat's offset (2026-08-17 audit).
    #
    # THE min() THAT USED TO BE HERE THREW THE DOWNBEAT AWAY (fixed 2026-08-18).
    # It clamped the origin to the earliest note so nothing could land at a
    # negative offset - which meant that the moment ANY note preceded the
    # downbeat, the downbeat was discarded and bar 1 restarted on the anacrusis.
    # That is precisely the pickup case, so pickups were the one thing it broke:
    # measured on Hopeful, which opens on beats 4-5-6 of 6/4, measure 1 came out
    # a full 6.000 long and every barline after it sat three beats early, so the
    # big ONE of the first full bar landed on measure 1 beat 4.
    #
    # The fix keeps the downbeat and steps the origin back a WHOLE number of
    # bars until it is at or before the first note. Barlines then land on real
    # downbeats, and the anacrusis occupies the tail of a leading bar instead of
    # displacing the entire piece. (A short anacrusis MEASURE, rather than a
    # full leading bar with rests, is the further refinement.)
    earliest_ms = min(
        (n.start_ms for p in score.parts for n in p.notes),
        default=0.0,
    )
    if score.bar_origin_ms is None:
        origin_ms = earliest_ms
    else:
        origin_ms = _origin_before(score.bar_origin_ms, earliest_ms,
                                   score.time_signature[0], score.tempo_bpm,
                                   score.musical_time)

    # The gate for whether a note is written as a triplet is rhythm_inference's
    # per-beat posterior verdict (is_tuplet) - a lattice-level decision that
    # weighs the whole beat's onsets, NOT the exporter re-guessing from a
    # rounded duration (that per-note guessing gave 3210 spurious triplets).
    #
    # The coarser ratio_family gate (ReverseGeoCrypt-style) is intentionally
    # NOT ANDed in here: measured on the test library, the lattice witness
    # reports `binary` for every song including the triplet-feel ones (Hopeful,
    # Gospel, No Pasaran), so ANDing it would force every page to 0 tuplets.
    # Strengthening ternary/swing detection is the prerequisite to using it as
    # the gate; until then the beat-level verdict is the trustworthy signal. A
    # ratio_family that is CONFIDENTLY ternary/swing can only ADD permission,
    # never remove the beat-level one.
    # NOTE (2026-08-06): a "one grid per measure" variant of this was tried -
    # any measure containing a tuplet beat rendered WHOLLY on the triplet grid,
    # to stop k/3 and k/4 positions mixing. It was a clear REGRESSION and was
    # reverted: prospering nonsense ratios 4 -> 7 and bad measures 0 -> 3;
    # Hopeful exploded to 1502 triplets with nine distinct nonsense ratios and
    # 22 bad measures, because it forced genuinely binary material onto thirds.
    # The per-beat verdict below stays. The residual nonsense ratios (4 notes of
    # 2773 on prospering) come from rhythm_inference's per-beat tuplet verdicts
    # themselves; the real fix is upstream tuplet detection, not the exporter.
    def _triplet(event: List[NotationNote]) -> bool:
        return (any(n.is_tuplet for n in event)
                or score.ratio_family in _TERNARY_RATIO_FAMILIES)

    # One calibration for the whole score: the resonance scale drifts between
    # songs, so it must be read off THIS material. Drums contribute nothing
    # (no Ma report) and simply do not appear in the distribution.
    _ma_gate = calibrate_from_notes([n for pt in score.parts for n in pt.notes])
    LAST_FILL_STATS.clear()
    LAST_FILL_STATS.update(drum_filled=0, ma_seen=0, ma_blocked_silent=0,
                           ma_no_evidence=0, ma_filled=0, ma_clamped_away=0)

    def _subdivision(event: List[NotationNote]) -> int:
        """Equal parts the beat this event sits on was read as.

        THE SHIPPING PATH DID NOT ASK FOR THIS. `subdivision` was added to
        NotationNote, carried correctly from rhythm_inference, and consulted
        only by _chord_events_gridded - so every onset and every duration the
        exporter actually wrote went on the default sixteenth lattice, and the
        fix that added it moved nothing. The beat's own reading is the grid.
        """
        vals = [int(getattr(n, "subdivision", 4) or 4) for n in event]
        return max(vals) if vals else 4

    def _divisor(event: List[NotationNote]):
        """The subdivision THIS beat was read as, if any.

        Meter-independent by construction: a sextuplet is a sextuplet in 4/4
        exactly as in 6/8, and nothing here consults the time signature. The
        only reason tuplets ever looked like a compound-meter feature is that
        until now the sole tuplet unit the page could write was the triplet's
        1/3 - so on any other subdivision the note was flattened to a binary
        value and the tuplet vanished, whatever the meter said."""
        for n in event:
            d = getattr(n, "tuplet_divisor", None)
            if d:
                return int(d)
        return None

    def _make_element(event: List[NotationNote], is_drum: bool = False):
        """A Note, or a Chord when several notes were struck together.
        Drums become Unpitched objects on the percussion staff (§XVI.10 #2).

        Deliberately does NOT set a duration. The caller derives quarterLength
        from the SNAPPED onset and end (grid_end - grid_onset), which is what
        makes each voice tile its bar. This used to compute a duration here as
        well and have it overwritten one line later - dead work that also meant
        the rubato-aware `musical_time` path in _quarter_length never reached
        the page (2026-08-17 audit)."""
        if is_drum:
            if len(event) == 1:
                el = _unpitched_for(event[0].pitch)
            else:
                try:
                    from music21 import percussion as m21perc
                    el = m21perc.PercussionChord([_unpitched_for(n.pitch) for n in event])
                except Exception:
                    el = _unpitched_for(event[0].pitch)
            return el
        if len(event) == 1:
            el = m21note.Note(event[0].pitch)
        else:
            el = m21chord.Chord(sorted(n.pitch for n in event))
        el.volume.velocity = max(n.velocity for n in event)
        # Ties (tie_reconstruction, via NotationNote flags): a held note that
        # tie_reconstruction joined to the next same-pitch note. start->next,
        # stop<-prev, both = a middle note in a tie chain -> continue.
        start = any(n.tie_start for n in event)
        stop = any(n.tie_stop for n in event)
        if start and stop:
            el.tie = m21tie.Tie("continue")
        elif start:
            el.tie = m21tie.Tie("start")
        elif stop:
            el.tie = m21tie.Tie("stop")
        return el

    m_score = stream.Score()
    # A metronome mark is a SYSTEM-level object: it belongs to the score once,
    # not to every staff. Emitting it per part put an identical "quarter = N"
    # <direction> above EVERY staff (drums included), which MuseScore renders
    # as a stack of tempo markings - reported from the page, and present since
    # long before Grimlock University. Time signature and key stay per-part:
    # those genuinely are per-part <attributes> in MusicXML.
    tempo_emitted = False
    for part in score.parts:
        if not part.notes:
            continue

        # If an upstream voicer stamped voice_index (piano_reduction), render
        # that verbatim - <= MAX_VOICES_PER_STAFF by construction, so no spill.
        # Otherwise fall back to the exporter's own greedy voicing.
        # Grid-aligned chord grouping (§XVI.6 leverage #1) - see
        # _chord_events_gridded. Falls back to the raw-ms rule when disabled.
        def _group(ns):
            if grid_chords:
                return _chord_events_gridded(ns, score.tempo_bpm, origin_ms, score.ratio_family,
                                 musical_time=score.musical_time)
            return _chord_events(ns)

        if any(n.voice_index is not None for n in part.notes):
            voices = _voices_from_index(part.notes, grid=_group)
        else:
            voices = _events_to_voices(_group(part.notes))
        staves = _staff_groups(voices)
        voice_tag = part.voice_id.split("::")[-1]

        for staff_idx, staff_voices in enumerate(staves):
            m_part = stream.Part()
            m_part.partName = _part_name(part.family, voice_tag, staff_idx, len(staves))

            if part.is_drum:
                inst = m21instrument.UnpitchedPercussion()
            else:
                inst = m21instrument.Instrument()
                program = _GM_PROGRAM.get(part.family)
                if program is not None:
                    inst.midiProgram = program
            m_part.insert(0.0, inst)
            if not tempo_emitted:
                m_part.insert(0.0, m21tempo.MetronomeMark(number=round(score.tempo_bpm)))
                tempo_emitted = True
            m_part.insert(0.0, m21meter.TimeSignature(
                f"{score.time_signature[0]}/{score.time_signature[1]}"))
            if score.key:
                key_object = _m21_key(score.key)
                if key_object is not None:
                    m_part.insert(0.0, key_object)
            if part.is_drum:
                # Percussion staff: drums are unpitched, so a pitched clef
                # would place them at meaningless staff positions (§XVI.10 #2).
                from music21 import clef as m21clef
                m_part.insert(0.0, m21clef.PercussionClef())
            else:
                staff_mean = sum(n.pitch for v in staff_voices for ev in v for n in ev) / \
                    sum(len(ev) for v in staff_voices for ev in v)
                m_part.insert(0.0, _clef_for(staff_mean))

            # One music21 Voice per monophonic line, numbered from 1.
            for v_num, voice_events in enumerate(staff_voices, start=1):
                m_voice = stream.Voice(id=str(v_num))
                # Snap each event's ONSET and END to the same grid and derive the
                # duration from those two grid points (dur = grid_end - grid_onset),
                # rather than snapping the duration independently. Independent
                # snapping (plus the sub-16th min-duration clamp) put notes like
                # 5.0 + 0.125 = 5.125 off the bar grid, so a voice's content summed
                # to 6.125 in a 6.0 bar -> overfull measures -> MuseScore flags the
                # file "corrupted" (it auto-repairs on open). Grid-aligning both
                # ends guarantees each voice tiles the bar. Then clamp so a note
                # never runs into the next onset in its own (monophonic) voice.
                placed = []
                for event in voice_events:
                    triplet = _triplet(event)
                    divisor = _divisor(event)
                    subdiv = _subdivision(event)
                    onset = _offset_quarter_length(
                        min(n.start_ms for n in event), origin_ms, score.tempo_bpm, triplet,
                        musical_time=score.musical_time, divisor=divisor,
                        subdivision=subdiv)
                    end = _offset_quarter_length(
                        max(n.end_ms for n in event), origin_ms, score.tempo_bpm, triplet,
                        musical_time=score.musical_time, divisor=divisor,
                        subdivision=subdiv)
                    # The floor is the beat's OWN subdivision, not a sixteenth.
                    # On a beat read as two, nothing finer than an eighth may be
                    # written - which is what stops a note being engraved as a
                    # 16th and the remainder of its eighth becoming a rest.
                    min_grid = (Fraction(1, divisor) if divisor
                                else (1.0 / 3.0) if triplet
                                else max(0.25, 1.0 / max(1, subdiv)))
                    el = _make_element(event, is_drum=part.is_drum)
                    # Snap the DURATION onto the same grid as the onset, rather
                    # than just subtracting two snapped points. A difference of
                    # two grid points is not itself guaranteed to be a
                    # representable note value - when a triplet-snapped end met
                    # a binary-snapped onset the leftover was un-notatable, and
                    # music21 expressed it as nonsense tuplets (24:13, 12:11,
                    # 24:23 were all present in one prospering export). MuseScore
                    # then boxes those measures in red because it cannot
                    # reconcile them either. Staying on one grid keeps every
                    # duration notatable; a slightly-wrong readable value always
                    # beats an exact un-notatable one on the page.
                    _ql = _representable(max(min_grid, end - onset), triplet,
                                         divisor=divisor)
                    if divisor:
                        # A tuplet occupies ONE beat and may not spill out of
                        # it - see rhythm_inference's anchoring rule. Enforced
                        # here as well because the legato fill and the voice
                        # clamp can both lengthen a note after the fact. A beat
                        # never spans a barline, so this also makes a
                        # barline-crossing bracket impossible.
                        _ql = _clamp_to_beat_frame(onset, _ql)
                        if _ql <= 0:
                            continue
                        _ql = _apply_tuplet(el, _ql, divisor)
                        if _ql <= 0:
                            continue
                    else:
                        el.quarterLength = _ql
                    # `event` is carried through: the Ma gate below needs THIS
                    # event's own acoustic evidence, and reading the first loop's
                    # leftover variable silently gave every note the last one's.
                    placed.append([onset, el, triplet, divisor, event])
                placed.sort(key=lambda oe: oe[0])
                # How far into the voice real content already reaches, so a
                # frame-marker rest is never written over an existing note.
                _voice_filled_to = float("-inf")
                for i, (onset, el, triplet, divisor, event) in enumerate(placed):
                    if i + 1 < len(placed):
                        gap = placed[i + 1][0] - onset
                        # Clamp THROUGH _representable, not straight to `gap`.
                        # BUG FIX (2026-08-17 audit): assigning the raw gap
                        # undid the snap three lines above it - a gap is the
                        # difference of two grid points and is not itself
                        # guaranteed notatable (1/3 + 1/4 = 0.5833), which is
                        # exactly the leftover music21 renders as 24:13 and
                        # MuseScore boxes in red. Round DOWN to a representable
                        # value so the note still cannot run into the next onset.
                        if gap > 0 and el.quarterLength > gap:
                            _clamped = _representable_at_most(
                                gap, triplet, divisor=divisor)
                            if divisor:
                                _clamped = _clamp_to_beat_frame(onset, _clamped)
                                if _clamped <= 0:
                                    continue
                                if _apply_tuplet(el, _clamped, divisor) <= 0:
                                    continue
                            else:
                                el.quarterLength = _clamped
                        # A DRUM HIT IS AN IMPULSE, AND ITS SILENCE IS NOT A
                        # REST. An unpitched notehead marks an ATTACK; its
                        # written length is nominal, so the convention is to
                        # write the value that reaches the next attack. Left
                        # alone, every hit was engraved at the grid floor and
                        # the remainder of its beat became a rest: on Clocks
                        # the kit came out 100% sixteenth notes and 1069
                        # sixteenth rests, where the published transcription
                        # writes it as 100% eighths and NO sixteenth rests.
                        # 1069 of our 1120 note-then-rest pairs were here.
                        #
                        # Clamped to the beat frame, so a fill can never cross
                        # a beat - and since a beat never spans a barline, it
                        # can never cross one of those either. Capped at one
                        # beat so a genuine silence between hits stays the rest
                        # it really is; the published kit part writes 430 of
                        # them and those are real.
                        elif (part.is_drum and not divisor
                              and gap > 0 and el.quarterLength < gap):
                            _fill = _representable_at_most(
                                min(gap, _DRUM_MAX_FILL_QL), triplet, divisor=None)
                            _fill = _clamp_to_beat_frame(onset, _fill)
                            if _fill > el.quarterLength:
                                el.quarterLength = _fill
                                LAST_FILL_STATS["drum_filled"] += 1
                        # ANECHOIC MA GATES EVERY OTHER REST ON THE PAGE. Basic
                        # Pitch's note ends are cut-offs, not releases, so the
                        # leftover of a beat became a rest nobody performs. Ma
                        # already measured whether that gap is still ringing or
                        # genuinely silent, on every pitched note of every run,
                        # and only the piano staff ever read it. A ringing gap
                        # is closed toward the next attack; a silent one is a
                        # real rest and is left exactly as it is.
                        elif (not divisor and gap > 0
                              and el.quarterLength < gap and not _ma_gate.blind):
                            _res = max((n.trailing_resonance for n in event
                                        if n.trailing_resonance is not None),
                                       default=None)
                            _void = max((n.trailing_void for n in event
                                         if n.trailing_void is not None),
                                        default=None)
                            LAST_FILL_STATS["ma_seen"] += 1
                            if _res is None:
                                LAST_FILL_STATS["ma_no_evidence"] += 1
                            _reach = _ma_gate.limit(_res, _void, _MA_BASE_FILL_QL,
                                                    gap=gap)
                            if _reach <= 0:
                                LAST_FILL_STATS["ma_blocked_silent"] += 1
                            else:
                                # NOT clamped to the beat frame. That helper
                                # exists for tuplets, which are a statement
                                # about how ONE beat is divided; a legato fill
                                # is not, and clamping it there killed 607 of
                                # 817 fills - every note starting late in a
                                # beat could not reach the attack it was being
                                # extended toward. split_for_hierarchy below
                                # renders a beat- or bar-crossing duration as
                                # the TIED chain the reader expects, which is
                                # the rule ("use ties if a duration crosses
                                # those boundaries") being satisfied properly
                                # rather than avoided by refusing to extend.
                                # The fill is still bounded by `gap`, so it can
                                # never run into the next attack in its voice.
                                _want = min(gap, el.quarterLength + _reach)
                                _fill = (_representable_at_most(_want, triplet, divisor=None)
                                         if triplet else
                                         _fill_value_at_most(_want))
                                if _fill > el.quarterLength:
                                    el.quarterLength = _fill
                                    LAST_FILL_STATS["ma_filled"] += 1
                                else:
                                    LAST_FILL_STATS["ma_clamped_away"] += 1
                    if el.quarterLength <= 0:
                        continue

                    # BEAT HIERARCHY (output/beat_hierarchy.py). One sounding
                    # note becomes the tied chain the page should show, so no
                    # un-tied duration ever hides a barline, a half-measure or
                    # a beat the reader is looking for. Tuplets pass through
                    # whole - they are anchored to one beat and must not be
                    # broken. A single-element result is the common case and
                    # costs one function call.
                    if divisor and onset >= _voice_filled_to - 1e-9:
                        # Hold the tuplet frame on its beat with a tuplet rest
                        # when the first sounding slot is not slot 0.
                        #
                        # ONLY INTO EMPTY SPACE. The span before this onset may
                        # already be occupied by the previous note in this
                        # voice, and inserting a rest on top of it DOUBLE-BOOKS
                        # the voice: measured on the Chopin page, voice 1 summed
                        # to 5.5 quarter-lengths in a 4.0 bar while voice 2 sat
                        # at exactly 4.0, and the overflow hanging off the end
                        # was being reported as tuplets crossing the barline.
                        # The rest is a frame marker, not extra time.
                        lead = _leading_tuplet_rest(onset, divisor)
                        if lead is not None and lead[0] >= _voice_filled_to - 1e-9:
                            m_voice.insert(lead[0], lead[1])
                            _voice_filled_to = onset

                    chain = split_for_hierarchy(
                        Fraction(onset).limit_denominator(_HIERARCHY_MAX_DEN),
                        Fraction(el.quarterLength).limit_denominator(_HIERARCHY_MAX_DEN),
                        score.time_signature[0], score.time_signature[1],
                        is_tuplet=bool(divisor))
                    if len(chain) == 1:
                        m_voice.insert(onset, el)
                        _voice_filled_to = onset + float(el.quarterLength)
                        continue
                    for seg_i, (seg_start, seg_dur) in enumerate(chain):
                        piece = copy.deepcopy(el)
                        piece.quarterLength = seg_dur
                        _chain_tie(piece, seg_i, len(chain))
                        m_voice.insert(float(seg_start), piece)
                        _voice_filled_to = float(seg_start) + float(seg_dur)
                LAST_FILL_STATS["gaps_closed"] = (
                    LAST_FILL_STATS.get("gaps_closed", 0)
                    + _close_unwritable_gaps(m_voice))
                m_voice.makeRests(fillGaps=True, inPlace=True)
                m_part.insert(0.0, m_voice)

            m_part = m_part.makeNotation(inPlace=False)
            m_score.insert(0.0, m_part)

    return m_score


def export_musicxml(score: NotationScore, path: str, grid_chords: bool = True) -> str:
    """Writes `score` to `path` as MusicXML, guaranteed voice-legal
    (voices 1..N per part, <= MAX_VOICES_PER_STAFF per staff). Returns
    the path."""
    m_score = build_music21_score(score, grid_chords=grid_chords)
    m_score.write("musicxml", fp=path)
    _normalize_voice_numbers(path)
    # The bytes are the only place left to enforce the barline - see
    # output/musicxml_repair.py and the open-defect note below.
    from output.musicxml_repair import repair_measures
    repair_measures(path)
    return path


# ---------------------------------------------------------------------
# OPEN DEFECT: elements that cross a barline, and why no in-memory guard
# can stop them (measured 2026-08-20, Hopeful/HRV/Chopin FULLRUN).
#
# THE RULE IS ABSOLUTE - "Tuplets must never cross barlines. NEVER" - and it
# is currently violated 13 times on Hopeful (0.3% of 4703 notes), 12 on HRV,
# 12 on Chopin. Before chasing it again, three measured facts, because the
# obvious fixes have all been tried and none of them work:
#
# 1. IT IS MOSTLY NOT A TUPLET PROBLEM. Only 2 of Hopeful's 13 crossing
#    elements carry a tuplet at all. The rest are ordinary notes and rests
#    with clean binary durations. `_clamp_to_beat_frame` is doing its job;
#    fixing the bracket code again will not move this number.
#
# 2. THE DOMINANT CAUSE IS A CONTAMINATED ONSET, NOT A LONG DURATION. Eight
#    of the thirteen sit at k + 1/24 - e.g. a clean 2.0-quarter note at
#    offset 4.0417 running to 6.0417 in a 6/4 bar. Nothing is wrong with the
#    length, so no duration-side guard can see it. 1/24 of a quarter is 17ms
#    at 148bpm. The fix belongs upstream, where onsets are snapped: a note
#    may not land 1/24 of a beat off a legal metric position.
#
# 3. A GUARD HERE CANNOT WORK, WHICH IS WHY THERE ISN'T ONE. music21's
#    MusicXML writer runs its OWN notation pass at serialization, and that
#    pass pads short voices with a rest of a full `barDuration` regardless of
#    the offset it starts at (offset 1.0 + a 6.0 rest in a 6/4 bar - three of
#    Hopeful's thirteen). It runs AFTER anything we do in memory: a clamp
#    inserted before `.write()` reported 0 elements touched while the written
#    file still had 13. Passing `makeNotation=False` to suppress that pass
#    raises "Cannot convert complex durations to MusicXML", and calling
#    `splitAtDurations()` first does not clear it. So the only enforcement
#    point that is provably last is the post-write XML pass in
#    `_normalize_voice_numbers` - walking <measure> with <divisions>, <backup>
#    and <forward> to trim what overruns. That is where a future fix goes.
#
# 4. THE OFF-GRID ONSET HAS THE SAME AUTHOR (measured 2026-08-21). Chopin's
#    last surviving grid violation is not ours either. Built in memory the
#    score has 2326 onsets and NONE off-grid; serialized and read back it has
#    2327, one of them at 17/24 of a beat. The writer's notation pass adds a
#    note and places it off the grid. Both remaining hard-rule violations on
#    this page therefore have one cause and one fix site - the post-write pass
#    above - and neither is a defect in what this module engraves.
# ---------------------------------------------------------------------


__all__ = ["build_music21_score", "export_musicxml"]
