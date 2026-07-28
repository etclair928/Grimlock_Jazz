# =================================================================
# MODULE: output/scribe_engraver.py
# Scribe Engraver (GRIMLOCK_6.0_DESIGN_DECISIONS.md §7 Output): the
# single Compositor/commit step. Reads the immutable Notes + every
# attached Annotation and emits MIDI in ONE auditable place - nothing
# upstream of this writes a MIDI file, and this module never mutates a
# Note's pitch/timing/velocity, it only reads.
#
# One pretty_midi.Instrument per resolved instrument-family label (via
# the "instrument_family" Annotation instrument_attribution/resolve.py
# writes) - not per stem, since Instrument Attribution's whole point
# was giving each voice LINE one real identity; grouping tracks by stem
# instead would throw that resolution away at the last step.
# =================================================================

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pretty_midi

from core import AnnotationStore, MusicBox, Note, StemType
from instrument_attribution.resolve import ANNOTATION_KIND
from quantization import (
    QUANTIZATION_ANNOTATION_KIND, SUSTAIN_RECOVERY_ANNOTATION_KIND,
    MICRO_NOTE_PURGE_ANNOTATION_KIND, PURGE_CANDIDATE,
    NOTATION_TIMING_ANNOTATION_KIND,
    ONSET_REFINEMENT_ANNOTATION_KIND,
    CONSOLIDATION_ANNOTATION_KIND,
)
from acoustic_witness import HARMONIC_LEGITIMACY_ANNOTATION_KIND, NOTE_SUPPORT_ANNOTATION_KIND, UNSUPPORTED

_HARMONIC_ILLEGITIMATE_VERDICTS = ("noise", "hallucination")

# Standard notation-software resolution. pretty_midi defaults to 220;
# 480 PPQN is the near-universal convention MuseScore/Finale/Sibelius/
# Dorico all expect, so the tick grid lines up on import without the
# host having to rescale.
MIDI_RESOLUTION_PPQN = 480

# Tempo-map segment merging: madmom's tracked beats carry a few ms of
# jitter each, so a naive one-tempo-event-per-beat map would spray
# hundreds of meaningless micro tempo changes into the file (and
# MuseScore renders each as a tempo marking - visual noise, the exact
# thing this map exists to reduce). Consecutive beats whose implied BPM
# stays within this tolerance of the running segment's mean merge into
# ONE segment whose tempo is computed from the segment's total duration,
# so segment BOUNDARIES stay exactly beat-aligned and interior beats
# drift only by the jitter itself (a few ms), never accumulating.
MIDI_TEMPO_MAP_MERGE_TOLERANCE_BPM = 1.5

# Audio before the first tracked beat shorter than this is treated as
# zero (beat 0 sits at tick 0, everything shifts earlier by < 50ms);
# longer intros get a real pre-beat segment so beat 0 still lands on an
# exact beat boundary in ticks.
_PRE_BEAT_MIN_OFFSET_S = 0.050


def _build_beat_tick_scales(
        beat_times_ms: Sequence[float], resolution: int,
) -> Optional[Tuple[List[Tuple[int, float]], int]]:
    """Turns the pipeline's TRACKED beat grid into a pretty_midi tick-scale
    list [(tick, seconds_per_tick), ...] that pins beat i to tick
    T0 + i*resolution. This is the structural binding raw export lacks:
    with it, a note's tick position is defined RELATIVE to where the beats
    actually fell in the performance, so a notation host draws it on the
    right beat instead of re-deriving its own grid from free-floating
    absolute times. Absolute note times are unchanged - only the tick
    lattice underneath them moves. Returns (tick_scales, beat0_tick), or
    None when the grid is unusable (too few beats / non-monotonic)."""
    beats_s = [b / 1000.0 for b in beat_times_ms]
    if len(beats_s) < 2:
        return None
    intervals = [beats_s[i + 1] - beats_s[i] for i in range(len(beats_s) - 1)]
    if min(intervals) <= 0:
        return None

    scales: List[Tuple[int, float]] = []

    # Pre-beat segment: the intro before beat 0 occupies a whole number of
    # grid beats (>=1) so beat 0 itself lands on an exact beat tick.
    #
    # CRITICAL: only build this segment when the pickup rounds to a WHOLE
    # beat. Forcing a sub-beat lead-in into one beat (the old max(1, ...))
    # crams e.g. a 175ms phase offset into a full 480-tick beat and thereby
    # fabricates an absurd intro tempo - a real case produced a 343 BPM
    # leading event that made MuseScore mis-read the whole file's tempo.
    # When the pickup is less than half a beat, anchor beat 0 at tick 0
    # instead; the short anacrusis simply falls early in bar 1, and the
    # tempo map stays honest (one rate, no phantom segment).
    first_spb = intervals[0]
    beat0_s = beats_s[0]
    n_pickup_beats = int(round(beat0_s / first_spb)) if (
        beat0_s >= _PRE_BEAT_MIN_OFFSET_S and first_spb > 0) else 0
    if n_pickup_beats >= 1:
        beat0_tick = n_pickup_beats * resolution
        scales.append((0, beat0_s / beat0_tick))
    else:
        beat0_tick = 0

    # Merge per-beat intervals into tempo segments (see tolerance above).
    segments: List[Tuple[int, int]] = []  # (first_interval_idx, last_interval_idx)
    seg_start = 0
    for i in range(1, len(intervals)):
        seg_duration = beats_s[i] - beats_s[seg_start]
        seg_bpm = 60.0 * (i - seg_start) / seg_duration
        next_bpm = 60.0 / intervals[i]
        if abs(next_bpm - seg_bpm) > MIDI_TEMPO_MAP_MERGE_TOLERANCE_BPM:
            segments.append((seg_start, i - 1))
            seg_start = i
    segments.append((seg_start, len(intervals) - 1))

    for first, last in segments:
        start_tick = beat0_tick + first * resolution
        n_beats = last - first + 1
        duration_s = beats_s[last + 1] - beats_s[first]
        scales.append((start_tick, duration_s / (n_beats * resolution)))

    return scales, beat0_tick

# key string (e.g. "C", "Bm") -> pretty_midi key_number (0-11 major C..B,
# 12-23 minor c..b - the SMF/pretty_midi convention). Built from the same
# pitch-class table key_intelligence uses.
_NOTE_TO_PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}


def _key_to_key_number(key: str) -> Optional[int]:
    """Converts a key string ("C" major, "Bm" minor) to pretty_midi's
    key_number, or None if it can't be parsed (a bad key just means no
    KeySignature meta-event, never a crash)."""
    if not key:
        return None
    is_minor = key.endswith('m')
    tonic = key[:-1] if is_minor else key
    pc = _NOTE_TO_PITCH_CLASS.get(tonic)
    if pc is None:
        return None
    return pc + (12 if is_minor else 0)

_GM_PROGRAM: Dict[str, int] = {
    "bass": 33,             # Electric Bass (finger)
    "vocals": 52,           # Choir Aahs
    "guitar": 27,           # Electric Guitar (clean)
    "piano": 0,             # Acoustic Grand Piano
    # The three "other"-stem brightness buckets. These are NEUTRAL KEYBOARD
    # patches ordered dark->bright to match the centroid tertiles resolve.py
    # sorts lines into (warm=lowest centroid ... bright=highest).
    #
    # They used to map to a brass tiering (French Horn/Trombone/Trumpet),
    # which is why non-brass songs "came out all brass": that was a genre
    # guess, not measured. We then MEASURED (2026-07-27, across the labeled
    # test library: Gospel=brass, Hopeful/YouSay/NoPasaran=piano/EP/guitar)
    # whether blind per-note DSP can actually tell brass from piano, testing
    # spectral centroid, bandwidth, sustain ratio, absolute attack time, and
    # decay slope. It CANNOT: brass's attack (~58ms) overlaps piano's (~45ms)
    # completely, its decay is only marginally slower, and its centroid sits
    # dead between the two pianos'. The ONE feature that separates cleanly is
    # brightness/darkness (guitar alone at centroid ~820). So we render only
    # what we can honestly measure - brightness - as a neutral keyboard of
    # matching brightness, and we DON'T claim a brass/string/wind family we
    # can't detect. See resolve.py's header for the full negative result;
    # genuine family ID would need reference-template matching, not blind DSP.
    "warm_sustained": 4,    # Electric Piano 1 - warm, neutral keyboard
    "mid_body": 0,          # Acoustic Grand Piano - the neutral default
    "bright_lead": 1,       # Bright Acoustic Piano - bright, neutral keyboard
}
_DEFAULT_PROGRAM = 0
_DRUM_FAMILY = "drums"


def _family_for_note(note: Note, annotations: AnnotationStore) -> str:
    family = annotations.latest_value(note.id, ANNOTATION_KIND)
    if family is not None:
        return family
    # No Instrument Attribution annotation exists for this note (e.g. a
    # stem it never ran on) - fall back to the stem name itself rather
    # than silently dropping the note from the export.
    return note.stem.value


def _timing_for_note(
        note: Note, annotations: AnnotationStore,
        use_quantized_timing: bool, use_notation_timing: bool,
) -> tuple:
    """Returns (start_ms, end_ms) for the requested timeline. THREE
    timelines exist and they are mutually exclusive in intent:

    - raw (default): Basic Pitch's own detected timing, untouched (§2's
      frozen-detection-floor law). What you listened to and liked.
    - notation (`use_notation_timing`): hard-snapped onset + clean
      symbolic duration (quantization.notation_quantizer) - the
      reading-optimized timeline, for a clean MuseScore/notation import.
      Takes precedence over groove timing when both are set, because
      it's the stricter, more opinionated of the two.
    - groove (`use_quantized_timing`): confidence-weighted snap that
      PRESERVES human microtiming inside a pocket + sustain-recovery
      tail extension - the playback-feel timeline.

    Every mode falls back to raw for any note lacking the relevant
    annotation (e.g. drums never go through these passes), never raises.

    Sustain recovery's proposed end (groove mode only) is measured from
    the note's ORIGINAL end and the later of the two wins: a real
    acoustic ringing tail shouldn't be cut short just because a grid-
    snap moved the note earlier. Notation mode deliberately ignores it -
    a clean symbolic duration is the whole point there.

    ONSET REFINEMENT applies underneath all of this, because unlike the
    three timelines it is not an interpretive choice - it corrects where
    the detector said a note began versus where its attack actually is.
    Basic Pitch fires on a threshold crossing, which lands progressively
    later the more slowly an instrument speaks, so melodic stems drift
    late while drums and bass do not. Correcting that is not a matter of
    taste, so even the raw timeline gets it: raw means "the performance
    as played," not "the detector's latency included."

    The refined onset moves the START only. The note's END is not in
    question - it rang until it rang - so an earlier onset lengthens the
    note rather than sliding it."""
    refined = annotations.latest_value(note.id, ONSET_REFINEMENT_ANNOTATION_KIND)
    base_start_ms = refined.get("start_ms", note.start_ms) if refined else note.start_ms

    if use_notation_timing:
        notation = annotations.latest_value(note.id, NOTATION_TIMING_ANNOTATION_KIND)
        if notation:
            return notation.get("start_ms", base_start_ms), notation.get("end_ms", note.end_ms)
        return base_start_ms, note.end_ms

    if not use_quantized_timing:
        return base_start_ms, note.end_ms
    start_ms, end_ms = base_start_ms, note.end_ms
    proposal = annotations.latest_value(note.id, QUANTIZATION_ANNOTATION_KIND)
    if proposal:
        start_ms = proposal.get("start_ms", start_ms)
        end_ms = proposal.get("end_ms", end_ms)
    sustain = annotations.latest_value(note.id, SUSTAIN_RECOVERY_ANNOTATION_KIND)
    if sustain:
        end_ms = max(end_ms, sustain.get("end_ms", end_ms))
    return start_ms, end_ms


def engrave(
        notes: List[Note],
        annotations: AnnotationStore,
        output_path: str,
        music_box: Optional[MusicBox] = None,
        use_quantized_timing: bool = False,
        use_notation_timing: bool = False,
        use_consolidated_timing: bool = False,
        drop_purge_candidates: bool = False,
        tempo_bpm: Optional[float] = None,
        time_signature: Optional[Tuple[int, int]] = None,
        key: Optional[str] = None,
        beat_times_ms: Optional[Sequence[float]] = None,
) -> pretty_midi.PrettyMIDI:
    """Writes `notes` to a MIDI file at `output_path`, one Instrument
    track per resolved instrument-family label. Returns the in-memory
    PrettyMIDI object as well, for callers that want to inspect it
    without re-reading the file.

    `tempo_bpm`/`time_signature`/`key`: the pipeline's resolved
    findings, stamped as meta-events at tick 0 so notation software
    (MuseScore/Finale/etc.) reads the real tempo/meter/key instead of
    pretty_midi's silent 120bpm-4/4 defaults. Absolute note timing is
    unchanged - pretty_midi converts each note's seconds to ticks
    against this tempo map at write time, so playback is identical; only
    the barline/beat grid the host DRAWS changes. None for any of them
    just omits that meta-event (the pre-findings default behavior),
    never a crash.

    `beat_times_ms`: the pipeline's TRACKED beat grid (the anchor tempo
    witness's actual beat positions). When provided (>= 2 beats), the
    file's tempo map is built FROM it via _build_beat_tick_scales - beat
    i is pinned to an exact beat tick, so notes' tick positions are
    defined relative to where the beats really fell rather than floating
    at absolute times against one flat tempo. This is what stops a
    notation host from discarding our stamped tempo and re-deriving its
    own grid: the notes and the tempo map finally tell the SAME story.
    Playback is unchanged (absolute seconds are preserved; only the tick
    lattice underneath moves). Overrides the scalar `tempo_bpm` for the
    tempo map itself; `tempo_bpm` remains the fallback when no usable
    grid is supplied.

    `use_quantized_timing` / `use_notation_timing`: which of the three
    timing timelines to export (see _timing_for_note). Both default
    False = raw timing (what you listened to and liked). `use_notation_
    timing` is the reading-optimized timeline (hard-snapped onsets +
    clean symbolic durations, quantization.notation_quantizer) and is
    the one to pick for a clean MuseScore/notation import; it takes
    precedence if both are somehow set. `use_quantized_timing` is the
    playback-feel timeline (groove-preserving snap + sustain tails).
    Correct meta-events fix WHERE barlines land; notation timing is what
    actually makes dense polyphony read cleanly instead of as endless
    tied fragments. Even notation timing has a MIDI ceiling: ties/voices/
    beaming are the host importer's decisions, not ours - controlling
    those needs a MusicXML export path (a separate future module that
    would build ON these symbolic onset/duration decisions).

    `drop_purge_candidates`: a separate opt-in from `use_quantized_timing`
    - it changes which notes exist in the export, not their timing, so
    it gets its own flag rather than being folded into that one. When
    True, notes flagged illegitimate by any of three independent
    "is this note real" witnesses are left out of the MIDI: MicroNotePurge
    (quantization.micro_note_purge, duration+confidence heuristic) marking
    "purge_candidate"; SchoenbergMirror (acoustic_witness, harmonic-
    series/ZCR/retrograde-symmetry/spectral-inversion) marking NOISE/
    HALLUCINATION; or the note-support filter (acoustic_witness.
    note_support - weak own-f0 energy AND acoustically inactive, the two
    signals a 3-song diagnostic showed actually discriminate over-
    detection/bleed) marking UNSUPPORTED. They answer the same underlying
    question from different evidence, so they share one opt-in flag
    rather than proliferating near-duplicate ones. NOTE: the diagnostic
    found MicroNotePurge's and SchoenbergMirror's signals barely
    discriminate real over-detection (bleed/doubled notes are genuinely
    tonal and confidently detected) - note_support is the one carrying
    the weight here; the other two stay because they catch the rarer
    genuine-noise cases. The Note itself is never touched upstream either
    way - this is purely an export-time filter."""
    init_tempo = tempo_bpm if (tempo_bpm is not None and tempo_bpm > 0) else 120.0
    midi = pretty_midi.PrettyMIDI(resolution=MIDI_RESOLUTION_PPQN, initial_tempo=init_tempo)

    # Meta-events at tick 0, from the pipeline's resolved findings - the
    # whole reason tempo/meter/key were computed. Without these the file
    # silently claims 120bpm/4-4/no-key and the host mis-draws every barline.
    if time_signature is not None:
        numerator, denominator = time_signature
        midi.time_signature_changes.append(pretty_midi.TimeSignature(numerator, denominator, 0.0))
    key_number = _key_to_key_number(key) if key is not None else None
    if key_number is not None:
        midi.key_signature_changes.append(pretty_midi.KeySignature(key_number, 0.0))

    tracks: Dict[str, pretty_midi.Instrument] = {}

    for note in notes:
        if drop_purge_candidates:
            verdict = annotations.latest_value(note.id, MICRO_NOTE_PURGE_ANNOTATION_KIND)
            if verdict is not None and verdict.get("verdict") == PURGE_CANDIDATE:
                continue
            harmonic_verdict = annotations.latest_value(note.id, HARMONIC_LEGITIMACY_ANNOTATION_KIND)
            if harmonic_verdict is not None and harmonic_verdict.get("verdict") in _HARMONIC_ILLEGITIMATE_VERDICTS:
                continue
            support = annotations.latest_value(note.id, NOTE_SUPPORT_ANNOTATION_KIND)
            if support is not None and support.get("verdict") == UNSUPPORTED:
                continue

        # Consolidation (opt-in): a same-pitch fragment absorbed into an
        # earlier note is not written at all; the primary it belongs to
        # carries the merged span (its end is extended below). Default off,
        # so raw output is byte-for-byte unchanged.
        consolidation = (annotations.latest_value(note.id, CONSOLIDATION_ANNOTATION_KIND)
                         if use_consolidated_timing else None)
        if consolidation is not None and consolidation.get("role") == "absorbed":
            continue

        family = _family_for_note(note, annotations)
        is_drum = family == _DRUM_FAMILY or note.stem == StemType.DRUMS

        if family not in tracks:
            program = _GM_PROGRAM.get(family, _DEFAULT_PROGRAM)
            tracks[family] = pretty_midi.Instrument(program=program, is_drum=is_drum, name=family)

        start_ms, end_ms = _timing_for_note(note, annotations, use_quantized_timing, use_notation_timing)
        if consolidation is not None and consolidation.get("role") == "primary":
            end_ms = max(end_ms, consolidation.get("end_ms", end_ms))
        tracks[family].notes.append(pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=start_ms / 1000.0,
            end=max(end_ms / 1000.0, start_ms / 1000.0 + 0.001),
        ))

    for track in tracks.values():
        track.notes.sort(key=lambda n: n.start)
        midi.instruments.append(track)

    # Beat-grid tempo map (see docstring): replace the single-tempo tick
    # scale the constructor installed with one built from the tracked
    # beats, then rebuild pretty_midi's tick<->time table so write()'s
    # seconds->tick conversion uses the real map (without the rebuild it
    # would extrapolate every note against the final segment's tempo).
    tempo_map_segments = 0
    if beat_times_ms is not None and len(beat_times_ms) >= 2:
        built = _build_beat_tick_scales(beat_times_ms, MIDI_RESOLUTION_PPQN)
        if built is not None:
            tick_scales, _beat0_tick = built
            max_end_s = max((inst_note.end for inst in midi.instruments for inst_note in inst.notes), default=0.0)
            last_tick, last_scale = tick_scales[-1]
            # The tick table must cover every note write() will convert;
            # the last beat's time is an upper bound on the final
            # segment's start, so this over-allocates slightly - cheap.
            last_beat_s = beat_times_ms[-1] / 1000.0
            extra_ticks = int(max(0.0, max_end_s - last_beat_s) / last_scale) if last_scale > 0 else 0
            max_tick = last_tick + extra_ticks + 8 * MIDI_RESOLUTION_PPQN
            midi._tick_scales = tick_scales
            midi._update_tick_to_time(max_tick)
            tempo_map_segments = len(tick_scales)

    midi.write(output_path)

    if music_box is not None:
        music_box.log_decision(
            stage_name="scribe_engraver",
            decision_type="midi_written",
            before_state={"note_count": len(notes)},
            after_state={
                "output_path": output_path,
                "track_count": len(tracks),
                "tracks": {name: len(inst.notes) for name, inst in tracks.items()},
                "use_quantized_timing": use_quantized_timing,
                "use_notation_timing": use_notation_timing,
                "use_consolidated_timing": use_consolidated_timing,
                "drop_purge_candidates": drop_purge_candidates,
                "tempo_bpm": init_tempo,
                "tempo_map_segments": tempo_map_segments,
                "time_signature": list(time_signature) if time_signature else None,
                "key": key,
            },
            reasoning=f"Wrote {len(notes)} notes across {len(tracks)} track(s) to {output_path}",
            reversible=False,
        )

    return midi


__all__ = ["engrave"]
