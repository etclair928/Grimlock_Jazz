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

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from output.notation_score import NotationScore, NotationPart, NotationNote

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


# Durations a single notehead can actually express, in quarter-lengths.
# Snapping to a GRID is not enough: a k/3 triplet grid happily produces 5/3 and
# 7/3, which no note value represents, so music21 invents ratios like 24:13 and
# 12:7 to make the arithmetic work and MuseScore boxes the measure in red
# because it cannot reconcile them either. Snapping to this explicit SET makes
# every emitted duration notatable by construction.
_BINARY_DURATIONS = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
_TRIPLET_DURATIONS = (1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0)


def _representable(ql: float, allow_triplet: bool) -> float:
    """Nearest duration a notehead can actually be written as."""
    allowed = _BINARY_DURATIONS + (_TRIPLET_DURATIONS if allow_triplet else ())
    return min(allowed, key=lambda a: abs(a - ql))


def _snap_quarter_length(ql: float, allow_triplet: bool) -> float:
    """Snap a quarter-length to the 16th grid, or - when this note carried a
    beat-level TRIPLET VERDICT and the piece's ratio family permits tuplets -
    to the triplet grid. The verdict comes from rhythm_inference (decided once
    per beat), never re-guessed here from the rounded duration: that per-note
    guessing manufactured 3210 tuplets > 4971 noteheads on the first pass."""
    binary = round(ql * 4.0) / 4.0        # nearest 16th
    if allow_triplet:
        return round(ql * 3.0) / 3.0      # nearest triplet-eighth
    return binary


# THE TWO CONVERSIONS THAT DECIDE WHICH BAR A NOTE LANDS IN.
#
# Both used to divide by a constant `60000/tempo_bpm`. MEASURED (2026-08-11):
# fixing the quantizer's snap alone moved Chopin recall only +9.6%, because
# these two then converted the correctly-placed milliseconds back to bar
# positions at a fixed rate and undid it. A tempo map has to be the ONLY
# currency, or every site that isn't converted silently reverts the ones that
# are. `musical_time` is threaded through from the score; when it is absent the
# old constant-rate behaviour is preserved exactly.

def _quarter_length(duration_ms: float, tempo_bpm: float, allow_triplet: bool = False,
                    musical_time=None, start_ms: Optional[float] = None) -> float:
    if musical_time is not None and musical_time.usable and start_ms is not None:
        # A duration is STRUCTURAL: measure it in beats, so both ends move
        # together under rubato and a stretched quarter still reads as a quarter.
        ql = _snap_quarter_length(
            musical_time.to_beats(start_ms + duration_ms) - musical_time.to_beats(start_ms),
            allow_triplet)
    else:
        ms_per_quarter = 60000.0 / max(tempo_bpm, 1.0)
        ql = _snap_quarter_length(duration_ms / ms_per_quarter, allow_triplet)
    return max(_MIN_QUARTER_LENGTH, ql)   # never let a real note vanish to zero


def _offset_quarter_length(start_ms: float, origin_ms: float, tempo_bpm: float,
                           allow_triplet: bool = False, musical_time=None) -> float:
    if musical_time is not None and musical_time.usable:
        pos = musical_time.to_beats(start_ms) - musical_time.to_beats(origin_ms)
        return max(0.0, _snap_quarter_length(pos, allow_triplet))
    ms_per_quarter = 60000.0 / max(tempo_bpm, 1.0)
    return max(0.0, _snap_quarter_length((start_ms - origin_ms) / ms_per_quarter, allow_triplet))


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
def _chord_events_gridded(
        notes: List[NotationNote], tempo_bpm: float, origin_ms: float,
        ratio_family: Optional[str], musical_time=None) -> List[List[NotationNote]]:
    buckets: Dict[float, List[NotationNote]] = {}
    for nn in sorted(notes, key=lambda n: (n.start_ms, n.pitch)):
        allow = bool(nn.is_tuplet) or (ratio_family in _TERNARY_RATIO_FAMILIES)
        offset = _offset_quarter_length(nn.start_ms, origin_ms, tempo_bpm, allow,
                                        musical_time=musical_time)
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
    chose."""
    import xml.etree.ElementTree as ET
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


def build_music21_score(score: NotationScore, grid_chords: bool = True):
    """Returns a music21.stream.Score. Each NotationPart becomes one or
    more staves; each staff carries at most MAX_VOICES_PER_STAFF voices,
    numbered from 1 - always valid, whatever the input polyphony."""
    from music21 import stream, note as m21note, chord as m21chord, tempo as m21tempo
    from music21 import meter as m21meter, key as m21key, instrument as m21instrument
    from music21 import tie as m21tie

    origin_ms = min(
        (n.start_ms for p in score.parts for n in p.notes),
        default=0.0,
    )

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

    def _make_element(event: List[NotationNote], is_drum: bool = False):
        """A Note, or a Chord when several notes were struck together.
        Drums become Unpitched objects on the percussion staff (§XVI.10 #2)."""
        allow = _triplet(event)
        ql = max(_quarter_length(n.duration_ms, score.tempo_bpm, allow,
                                 musical_time=score.musical_time,
                                 start_ms=n.start_ms) for n in event)
        if is_drum:
            if len(event) == 1:
                el = _unpitched_for(event[0].pitch)
            else:
                try:
                    from music21 import percussion as m21perc
                    el = m21perc.PercussionChord([_unpitched_for(n.pitch) for n in event])
                except Exception:
                    el = _unpitched_for(event[0].pitch)
            el.quarterLength = ql
            return el
        if len(event) == 1:
            el = m21note.Note(event[0].pitch)
        else:
            el = m21chord.Chord(sorted(n.pitch for n in event))
        el.quarterLength = ql
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
                try:
                    m_part.insert(0.0, m21key.Key(score.key.replace("m", "").strip()))
                except Exception:
                    pass
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
                    onset = _offset_quarter_length(
                        min(n.start_ms for n in event), origin_ms, score.tempo_bpm, triplet,
                        musical_time=score.musical_time)
                    end = _offset_quarter_length(
                        max(n.end_ms for n in event), origin_ms, score.tempo_bpm, triplet,
                        musical_time=score.musical_time)
                    min_grid = (1.0 / 3.0) if triplet else 0.25
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
                    el.quarterLength = _representable(max(min_grid, end - onset), triplet)
                    placed.append([onset, el])
                placed.sort(key=lambda oe: oe[0])
                for i, (onset, el) in enumerate(placed):
                    if i + 1 < len(placed):
                        gap = placed[i + 1][0] - onset
                        if gap > 0 and el.quarterLength > gap:
                            el.quarterLength = gap
                    if el.quarterLength <= 0:
                        continue
                    m_voice.insert(onset, el)
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
    return path


__all__ = ["build_music21_score", "export_musicxml"]
