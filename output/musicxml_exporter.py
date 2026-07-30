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

from typing import Dict, List, Optional

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


def _quarter_length(duration_ms: float, tempo_bpm: float, allow_triplet: bool = False) -> float:
    ms_per_quarter = 60000.0 / max(tempo_bpm, 1.0)
    ql = _snap_quarter_length(duration_ms / ms_per_quarter, allow_triplet)
    return max(_MIN_QUARTER_LENGTH, ql)   # never let a real note vanish to zero


def _offset_quarter_length(start_ms: float, origin_ms: float, tempo_bpm: float, allow_triplet: bool = False) -> float:
    ms_per_quarter = 60000.0 / max(tempo_bpm, 1.0)
    return max(0.0, _snap_quarter_length((start_ms - origin_ms) / ms_per_quarter, allow_triplet))


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


def build_music21_score(score: NotationScore):
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
    def _triplet(event: List[NotationNote]) -> bool:
        return (any(n.is_tuplet for n in event)
                or score.ratio_family in _TERNARY_RATIO_FAMILIES)

    def _make_element(event: List[NotationNote]):
        """A Note, or a Chord when several notes were struck together."""
        allow = _triplet(event)
        ql = max(_quarter_length(n.duration_ms, score.tempo_bpm, allow) for n in event)
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
    for part in score.parts:
        if not part.notes:
            continue

        events = _chord_events(part.notes)
        voices = _events_to_voices(events)
        staves = _staff_groups(voices)
        voice_tag = part.voice_id.split("::")[-1]

        for staff_idx, staff_voices in enumerate(staves):
            m_part = stream.Part()
            m_part.partName = _part_name(part.family, voice_tag, staff_idx, len(staves))

            inst = m21instrument.Instrument()
            program = _GM_PROGRAM.get(part.family)
            if program is not None:
                inst.midiProgram = program
            m_part.insert(0.0, inst)
            m_part.insert(0.0, m21tempo.MetronomeMark(number=round(score.tempo_bpm)))
            m_part.insert(0.0, m21meter.TimeSignature(
                f"{score.time_signature[0]}/{score.time_signature[1]}"))
            if score.key:
                try:
                    m_part.insert(0.0, m21key.Key(score.key.replace("m", "").strip()))
                except Exception:
                    pass
            staff_mean = sum(n.pitch for v in staff_voices for ev in v for n in ev) / \
                sum(len(ev) for v in staff_voices for ev in v)
            m_part.insert(0.0, _clef_for(staff_mean))

            # One music21 Voice per monophonic line, numbered from 1.
            for v_num, voice_events in enumerate(staff_voices, start=1):
                m_voice = stream.Voice(id=str(v_num))
                for event in voice_events:
                    offset = _offset_quarter_length(
                        min(n.start_ms for n in event), origin_ms, score.tempo_bpm, _triplet(event))
                    m_voice.insert(offset, _make_element(event))
                m_voice.makeRests(fillGaps=True, inPlace=True)
                m_part.insert(0.0, m_voice)

            m_part = m_part.makeNotation(inPlace=False)
            m_score.insert(0.0, m_part)

    return m_score


def export_musicxml(score: NotationScore, path: str) -> str:
    """Writes `score` to `path` as MusicXML, guaranteed voice-legal
    (voices 1..N per part, <= MAX_VOICES_PER_STAFF per staff). Returns
    the path."""
    m_score = build_music21_score(score)
    m_score.write("musicxml", fp=path)
    _normalize_voice_numbers(path)
    return path


__all__ = ["build_music21_score", "export_musicxml"]
