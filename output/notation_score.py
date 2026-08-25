# =================================================================
# MODULE: output/notation_score.py
# The NotationScore - the symbolic object that sits between the
# pipeline's evidence (frozen Notes + AnnotationStore) and the
# exporters (see GRIMLOCK_6.0_NOTATION_GRAPH.md).
#
# It owns exactly the relationships a copyist needs and MIDI cannot
# carry. This first slice owns ONE of them - VOICE membership - because
# that is the one the user named ("guitar is a mess of polyphony... a
# trouble spot") and the one whose input (VoiceLine.line_id) the
# pipeline already computes and then, until now, discarded.
#
# The law is unchanged: nothing here mutates a Note. The score is
# BUILT from notes + annotations and READ by serializers. Every future
# edge (ties, tuplets, beams, spelling, gesture groups) becomes another
# field on NotationPart/NotationScore, added one measured slice at a
# time - never a global solver, never a magic number, never a guess
# that fails silently.
# =================================================================

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core import AnnotationStore, Note, StemType
from instrument_attribution.resolve import ANNOTATION_KIND as FAMILY_ANNOTATION_KIND
from instrument_attribution.resolve import VOICE_ANNOTATION_KIND
from quantization import (
    ONSET_REFINEMENT_ANNOTATION_KIND, SUSTAIN_RECOVERY_ANNOTATION_KIND,
    NOTATION_TIMING_ANNOTATION_KIND, TIE_RECONSTRUCTION_ANNOTATION_KIND,
)
from quantization.note_consolidation import CONSOLIDATION_ANNOTATION_KIND
from acoustic_witness import ACOUSTIC_ACTIVITY_ANNOTATION_KIND
from pitch_engine.bass_octave import BASS_OCTAVE_ANNOTATION_KIND

# A voice's notes are monophonic by construction (stream_into_lines only
# joins a note to a line when it does NOT overlap that line's last note).
# So "hold each note until the next one in its own voice" is not a policy
# choice with a magic threshold - it is what a monophonic line IS. The
# only real decision is when to break the legato with a rest, and that is
# driven by SustainRecovery's posteriorgram evidence (the extended end),
# not by a constant. This is exactly why voice separation and clean
# rhythm are the SAME move, not two.
_DRUM_FAMILY = "drums"


@dataclass
class NotationNote:
    """One note as the page will see it. The physical Note is untouched;
    these are the resolved values a serializer reads."""
    pitch: int
    start_ms: float           # onset the page uses (refined when available)
    end_ms: float             # end the page uses (sustain-extended when available)
    velocity: int
    source_note_id: str       # provenance back to the frozen Note
    is_tuplet: bool = False   # rhythm_inference's beat-level triplet verdict (not re-guessed)
    # HOW MANY equal parts the beat was read as, when it is a tuplet. The
    # bool above says "not binary"; only this says "into how many", and the
    # page needs the number - a sextuplet's unit is 1/6 of a beat and a
    # quintuplet's is 1/5, which are different noteheads. Written by
    # rhythm_inference since 2026-08-17 and not READ until 2026-08-18,
    # which is why every non-triplet tuplet was silently flattened to the
    # nearest binary value on export.
    tuplet_divisor: Optional[int] = None
    # Equal parts the BEAT was read as, binary or not. 2 means eighths, and
    # the exporter must then snap this note's onset to the eighth grid rather
    # than to the sixteenth lattice it used unconditionally until now.
    subdivision: int = 4
    tie_start: bool = False   # tie_reconstruction: this note is held into the next same-pitch note
    tie_stop: bool = False    # ...and/or continues a tie from the previous one
    # Explicit engraving voice (1-based) when an upstream voicer has already
    # decided it - e.g. piano_reduction's <=4 rhythmic-independence voicer.
    # When set, the exporter honors it verbatim instead of running its own
    # greedy fallback voicing (musicxml_exporter._events_to_voices). None =
    # unchanged legacy behavior (the exporter voices the part itself).
    voice_index: Optional[int] = None
    # Grimlock University APPLY: the id of the gesture (scale run, arpeggio,
    # sequence, Alberti figure) this note belongs to. The voicer keeps one
    # gesture in ONE voice instead of scattering it - a musical line should
    # not change stems mid-phrase. None when the University never ran.
    cohesion: Optional[str] = None
    # AnechoicMa's evidence about the gap immediately AFTER this note
    # (acoustic_activity annotation). None when the witness never ran.
    # resonance high  -> the stem is still ringing: write the note out.
    # void high       -> genuine silence: the rest is real, keep it.
    # This replaces a blanket by-ear fill threshold with measured evidence
    # (GRIMLOCK_6.0_OPEN_PROBLEMS.md XVIII.3 #1).
    trailing_resonance: Optional[float] = None
    trailing_void: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass
class NotationPart:
    """One staff. A single monophonic voice of a single instrument
    family - the unit that stops the polyphony mush."""
    family: str
    voice_id: str
    stem: StemType
    notes: List[NotationNote] = field(default_factory=list)
    is_drum: bool = False

    @property
    def mean_pitch(self) -> float:
        return sum(n.pitch for n in self.notes) / len(self.notes) if self.notes else 60.0


@dataclass
class NotationScore:
    """The resolved symbolic world, as a page. One part per voice."""
    parts: List[NotationPart] = field(default_factory=list)
    tempo_bpm: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    key: Optional[str] = None
    # The lattice's rhythmic ratio family (binary / ternary / swing, from the
    # ReverseGeoCrypt-style lattice witness). The exporter uses it as the GATE
    # for whether tuplets are permitted at all - triplets on strictly-binary
    # material are almost always a per-note rounding artifact, not real.
    ratio_family: Optional[str] = None
    # THE MAP between clock time and musical position (core/musical_time.py).
    # When present the exporter converts through it instead of dividing by
    # 60000/tempo_bpm, so a note played inside a ritardando lands in the bar it
    # belongs to. Absent, the old constant-rate behaviour is preserved exactly.
    # `tempo_bpm` above is then a DISPLAY statistic, not the currency.
    musical_time: Optional[object] = None
    # WHERE BAR 1 STARTS, in performance ms - the resolved downbeat, not a
    # guess. None means "nobody knows", and consumers fall back to the earliest
    # note in the score (the old behaviour). Before this existed the exporter
    # anchored bars to the earliest note and piano_reduction anchored them to
    # absolute zero, so the two disagreed about where every barline was
    # (2026-08-17 audit).
    bar_origin_ms: Optional[float] = None

    @property
    def total_notes(self) -> int:
        return sum(len(p.notes) for p in self.parts)


def _page_pitch(note: Note, annotations: AnnotationStore) -> int:
    """The pitch the page should use.

    Basic Pitch's octave on the bass is arbitrated by CREPE
    (pitch_engine/bass_octave.py), whose verdict arrives as an annotation
    rather than a rewritten Note (DESIGN_DECISIONS §2.2). Notes with no such
    annotation - which is all of them outside the bass, and most within it -
    return their detected pitch untouched.

    This exists as one helper rather than two inline reads because
    NotationNote is constructed at two sites, and the last field added to
    both was added to one, shipped, and cost a whole run.
    """
    corrected = annotations.latest_value(note.id, BASS_OCTAVE_ANNOTATION_KIND)
    if corrected is None:
        return int(note.pitch)
    pitch = corrected.get("corrected_pitch")
    return int(pitch) if pitch is not None else int(note.pitch)


def _page_timing(note: Note, annotations: AnnotationStore, use_notation_timing: bool = True) -> Tuple[float, float]:
    """The (start, end) the page should use.

    `use_notation_timing` (default): prefer the NOTATION_TIMING annotation -
    the beat-level Bayesian rhythm inference (rhythm_inference.py), which
    places each onset on a symbolic beat fraction (0, 1/4, 1/3, 1/2, ...)
    instead of at its raw performed millisecond. This is THE fix for the
    "notation confetti": raw-ms onsets land at arbitrary 32nd offsets, so
    every gap becomes a rest and every span a tie; beat-relative symbolic
    onsets land on the grid and let tuplets exist. Falls back (per note,
    for beats rhythm_inference couldn't confidently parse) to the refined
    onset + sustain-recovery end, then to the frozen values."""
    if use_notation_timing:
        nt = annotations.latest_value(note.id, NOTATION_TIMING_ANNOTATION_KIND)
        if nt is not None:
            return float(nt.get("start_ms", note.start_ms)), float(nt.get("end_ms", note.end_ms))

    refined = annotations.latest_value(note.id, ONSET_REFINEMENT_ANNOTATION_KIND)
    start_ms = refined.get("start_ms", note.start_ms) if refined else note.start_ms

    end_ms = note.end_ms
    sustain = annotations.latest_value(note.id, SUSTAIN_RECOVERY_ANNOTATION_KIND)
    if sustain:
        end_ms = max(end_ms, sustain.get("end_ms", end_ms))

    # A refined onset moves the START only; if it landed past the end,
    # keep the note non-degenerate rather than inverting it.
    if start_ms >= end_ms:
        end_ms = start_ms + max(1.0, note.end_ms - note.start_ms)
    return start_ms, end_ms


def _consolidated_end_ms(consolidation: dict, end_ms: float,
                         annotations: AnnotationStore,
                         use_notation_timing: bool) -> float:
    """End of a consolidated run on the timeline THIS PAGE is drawing.

    The consolidation annotation stores the run's end as a raw performance
    millisecond, since that is the only timeline that exists when it runs.
    Pasting that onto a notation-snapped start gave every merged run's primary
    a duration off the notation grid - the un-notatable leftovers the exporter
    then has to round away (2026-08-17 audit). The run's last fragment already
    carries its own end on every timeline, so ask it."""
    raw_end = consolidation.get("end_ms", end_ms)
    if use_notation_timing:
        last_id = consolidation.get("last_note_id")
        if last_id:
            value = annotations.latest_value(last_id, NOTATION_TIMING_ANNOTATION_KIND)
            if value and value.get("end_ms") is not None:
                return max(end_ms, float(value["end_ms"]))
    return max(end_ms, raw_end)


def build_notation_score(
        notes: List[Note],
        annotations: AnnotationStore,
        tempo_bpm: float,
        time_signature: Tuple[int, int],
        key: Optional[str] = None,
        use_voices: bool = True,
        use_consolidation: bool = True,
        use_notation_timing: bool = True,
        ratio_family: Optional[str] = None,
        musical_time=None,
        bar_origin_ms: Optional[float] = None,
) -> NotationScore:
    """Groups notes into parts and resolves each note's page timing.

    `use_voices=True` (default): one part per (family, voice), using the
    `voice` annotation from voice_continuity. This is only as good as the
    voice separation upstream - while that over-fragments, it yields one
    part per greedy line (many tiny staves).

    `use_voices=False`: one part per FAMILY, leaving all of an
    instrument's polyphony together. The exporter then splits that
    polyphony into the fewest voice-legal staves itself. Until voice
    separation is fixed this is the more readable of the two - a handful
    of dense staves instead of hundreds of fragments.

    Notes with no `voice` annotation (e.g. drums) always fall back to a
    single per-family part - they are never dropped."""
    buckets: Dict[Tuple[str, str, StemType], List[NotationNote]] = defaultdict(list)
    drum_flag: Dict[Tuple[str, str, StemType], bool] = {}

    for note in notes:
        # Consolidation (opt-in): a same-pitch fragment absorbed into a held
        # note never becomes its own notehead on the page (that IS the
        # confetti); the primary carries the merged end. Same evidence the
        # engraver's use_consolidated_timing reads - §XII.3's "notation
        # confetti" root, removed before a single duration is quantized.
        consolidation = (annotations.latest_value(note.id, CONSOLIDATION_ANNOTATION_KIND)
                         if use_consolidation else None)
        if consolidation is not None and consolidation.get("role") == "absorbed":
            continue

        family = annotations.latest_value(note.id, FAMILY_ANNOTATION_KIND)
        is_drum = note.stem == StemType.DRUMS or family == _DRUM_FAMILY
        if family is None:
            family = _DRUM_FAMILY if is_drum else "unknown"

        voice_id = annotations.latest_value(note.id, VOICE_ANNOTATION_KIND) if use_voices else None
        if voice_id is None:
            # Drums, un-voiced notes, and family-only mode collapse to one
            # part per family - a single bucket the exporter then voices
            # legally. Correct for percussion, and honest ("no voice
            # evidence used here") for the rest.
            voice_id = f"{family}::all"

        key_tuple = (family, voice_id, note.stem)
        start_ms, end_ms = _page_timing(note, annotations, use_notation_timing=use_notation_timing)
        if consolidation is not None and consolidation.get("role") == "primary":
            end_ms = _consolidated_end_ms(consolidation, end_ms, annotations,
                                          use_notation_timing)
        # The beat-level tuplet verdict, carried from rhythm_inference - never
        # re-derived from the rounded duration on the page.
        nt = annotations.latest_value(note.id, NOTATION_TIMING_ANNOTATION_KIND) if use_notation_timing else None
        is_tuplet = bool(nt.get("is_tuplet", False)) if nt else False
        tuplet_divisor = nt.get("tuplet_divisor") if nt else None
        subdivision = int(nt.get("subdivision") or 4) if nt else 4
        buckets[key_tuple].append(NotationNote(
            pitch=_page_pitch(note, annotations), start_ms=start_ms, end_ms=end_ms,
            velocity=note.velocity, source_note_id=note.id, is_tuplet=is_tuplet,
            tuplet_divisor=tuplet_divisor,
            subdivision=subdivision,
        ))
        drum_flag[key_tuple] = is_drum

    # Ties (tie_reconstruction): a note held into the next same-pitch note
    # renders as tied notes rather than two separate ones. Only set a tie when
    # BOTH ends SURVIVE into the score - when consolidation is on it has
    # already ABSORBED these same-pitch fragments into one sustained note (a
    # cleaner result than a tie), so the target is gone and no tie is drawn.
    # This is why the signal is a correct no-op on the consolidated page and a
    # real edge only in the faithful (non-consolidated) view.
    by_source: Dict[str, NotationNote] = {
        nn.source_note_id: nn for nns in buckets.values() for nn in nns
    }
    for nn in list(by_source.values()):
        tie = annotations.latest_value(nn.source_note_id, TIE_RECONSTRUCTION_ANNOTATION_KIND)
        if tie is None:
            continue
        target = by_source.get(tie.get("tied_to_note_id"))
        if target is not None:          # both ends survived -> draw the tie
            nn.tie_start = True
            target.tie_stop = True

    parts: List[NotationPart] = []
    for (family, voice_id, stem), nnotes in buckets.items():
        nnotes.sort(key=lambda n: n.start_ms)
        parts.append(NotationPart(
            family=family, voice_id=voice_id, stem=stem,
            notes=nnotes, is_drum=drum_flag[(family, voice_id, stem)],
        ))

    # Stable, readable part order: non-drums by descending register (so a
    # score reads treble-to-bass top-to-bottom), drums last.
    parts.sort(key=lambda p: (p.is_drum, -p.mean_pitch))
    return NotationScore(
        parts=parts, tempo_bpm=tempo_bpm,
        time_signature=time_signature, key=key, ratio_family=ratio_family,
        musical_time=musical_time, bar_origin_ms=bar_origin_ms,
    )


# ========================================================================
# Per-stem staff routing (user directive 2026-07-31): each stem gets its
# own staff; a GRAND STAFF is used only when the stem earns it - it is a
# keyboard/plucked timbre (guitar/piano/harp) or the merged harmonic
# "other" junk drawer, OR it is verifiably polyphonic. A melodic/vocal
# stem stays on ONE staff, and if it has real polyphonic independence it
# is sorted into Voice 1/2/... BEFORE ever reaching for a grand staff.
# Bass and drums each get their own single staff. Timbre (the family
# annotation) and polyphony are the deciders.
# ========================================================================

# Timbres that warrant a grand staff outright (polyphonic keyboard/plucked).
_GRAND_STAFF_FAMILIES = {"guitar", "piano", "harp", "keyboard", "keys"}
# Stems that are grand-staff candidates by identity (the merged harmonic
# junk drawer, and the un-merged keyboard/guitar heads).
_GRAND_STAFF_STEMS = (StemType.OTHER, StemType.GUITAR, StemType.PIANO)
# A voice-worthy amount of simultaneity on a non-keyboard melodic/vocal
# stem: 3+ notes sounding at once is real independence, not a passing
# double-stop, so sort it with voices rather than leaving it as chords.
_POLYPHONY_VOICE_THRESHOLD = 3


def _max_simultaneity(nns: List[NotationNote]) -> int:
    events: List[Tuple[float, int]] = []
    for n in nns:
        events.append((n.start_ms, 1))
        events.append((n.end_ms, -1))
    events.sort()
    cur = mx = 0
    for _, delta in events:
        cur += delta
        mx = max(mx, cur)
    return mx


def build_routed_score(
        notes: List[Note],
        annotations: AnnotationStore,
        tempo_bpm: float,
        time_signature: Tuple[int, int],
        key: Optional[str] = None,
        use_consolidation: bool = True,
        use_notation_timing: bool = True,
        ratio_family: Optional[str] = None,
        # MEASURED ON THE WHOLE LIBRARY (2026-08-09, tools/voice_cap_sweep.py):
        # 9 songs, cap 2 vs cap 3, UNANIMOUS - not one exception.
        #     cap 2:  rest 0.342  topstab 0.518  v1top 0.532  jump 4.85
        #     cap 3:  rest 0.432  topstab 0.446  v1top 0.427  jump 4.08
        # Rest/note -21%, top-line stability +16%, voice-1-is-the-tune +25%, and
        # note counts move <0.5% - cap 2 does not DISCARD music, it REPRESENTS
        # the same music with fewer independent rhythmic layers (§XX.7: Klangio
        # carries 1.19-1.85x our notes with 6-9x fewer rests, so clutter was
        # never a note-count problem). Klangio's max_voices is 2 on every song.
        #
        # Cap 2 loses ONE axis, mean_voice_jump (4.08 -> 4.85). That axis alone
        # is what rejected cap 2 the first time, before top_line_stability
        # existed - it read "passing" while the melody changed voice on 52% of
        # onsets. It may not veto a layout change on its own again (§XX.6).
        #
        # The honest cost: a genuine 3-voice contrapuntal passage cannot be
        # represented. On this repertoire that trade is measurably worth it.
        max_voices: int = 2,
        honor_university: bool = False,
        # OctaveStack (acoustic_witness.octave_stack). Off by default like
        # every other output-changing switch here, and for the same reason:
        # raw output stays byte-identical until someone asks for the change.
        # When on, the interior of a 3+-octave stack struck within one
        # gesture is left off the page. Worth it only where the input has
        # stem bleed: +0.0035 F1 on a six-stem run of a solo piano record,
        # -0.0033 on the same piece with one clean harmonic stem.
        drop_octave_stacks: bool = False,
        fill_max_beats: float = 1.0,
        musical_time=None,
        bar_origin_ms: Optional[float] = None,
) -> NotationScore:
    """Route each STEM to the staff layout it warrants (see header):
      drums/bass          -> one single staff each
      other/guitar/piano, or any grand-staff timbre -> GRAND STAFF (reduction)
      vocals/other melodic: polyphonic -> <=max_voices on one staff;
                            monophonic  -> one single staff
    Frozen-note law holds - this builds a NotationScore from notes +
    annotations and never mutates a Note."""
    from output.piano_reduction import build_grand_staff, assign_voices

    ms_per_beat = 60000.0 / max(tempo_bpm, 1.0)
    by_stem: Dict[StemType, List[NotationNote]] = defaultdict(list)
    stem_family: Dict[StemType, Counter] = defaultdict(Counter)

    # Grimlock University APPLY mode (opt-in, off by default): the page may
    # drop low-confidence re-strike echoes it identified as one sustained
    # note, and hold the surviving strike across the run. Frozen Notes are
    # untouched - this is the notation clock only, so MIDI/playback is
    # identical either way. When the University never ran, these are empty
    # and this whole path is a no-op.
    uni_drop: set = set()
    uni_extend: Dict[str, float] = {}
    uni_cohesion: Dict[str, str] = {}
    # RANGE PLAUSIBILITY, on the page as well as in the MIDI. Written since
    # 2026-08-06 and consumed by nothing until the annotation audit found it.
    # A bass note above G4 is bleed, not music, and the bound is deliberately
    # generous so it catches artifacts rather than policing unusual playing.
    if drop_octave_stacks:
        from instrument_attribution.range_check import (
            RANGE_ANNOTATION_KIND, IMPLAUSIBLE as _RANGE_BAD)
        for n in notes:
            verdict = annotations.latest_value(n.id, RANGE_ANNOTATION_KIND)
            if verdict is not None and verdict.get("verdict") == _RANGE_BAD:
                uni_drop.add(n.id)

    if drop_octave_stacks:
        # Folded into the same drop set the loop below already consults, so
        # there is exactly one place a note can vanish from the page.
        from acoustic_witness.octave_stack import octave_suppression_ids
        uni_drop |= octave_suppression_ids(annotations, [n.id for n in notes])
    if honor_university:
        try:
            from university.apply import page_suppression_map
            from university.observation_types import VOICE_COHESION_ANNOTATION_KIND
            uni_drop, uni_extend = page_suppression_map(
                annotations, [n.id for n in notes])
            for n in notes:
                value = annotations.latest_value(n.id, VOICE_COHESION_ANNOTATION_KIND)
                if value and value.get("group"):
                    uni_cohesion[n.id] = str(value["group"])
        except Exception:
            uni_drop, uni_extend, uni_cohesion = set(), {}, {}

    for note in notes:
        if note.id in uni_drop:
            continue
        consolidation = (annotations.latest_value(note.id, CONSOLIDATION_ANNOTATION_KIND)
                         if use_consolidation else None)
        if consolidation is not None and consolidation.get("role") == "absorbed":
            continue
        family = annotations.latest_value(note.id, FAMILY_ANNOTATION_KIND)
        is_drum = note.stem == StemType.DRUMS or family == _DRUM_FAMILY
        if family is None:
            family = _DRUM_FAMILY if is_drum else "unknown"
        start_ms, end_ms = _page_timing(note, annotations, use_notation_timing=use_notation_timing)
        if consolidation is not None and consolidation.get("role") == "primary":
            end_ms = _consolidated_end_ms(consolidation, end_ms, annotations,
                                          use_notation_timing)
        if note.id in uni_extend:
            end_ms = max(end_ms, uni_extend[note.id])
        nt = annotations.latest_value(note.id, NOTATION_TIMING_ANNOTATION_KIND) if use_notation_timing else None
        is_tuplet = bool(nt.get("is_tuplet", False)) if nt else False
        tuplet_divisor = nt.get("tuplet_divisor") if nt else None
        # Equal parts the beat was read as. 4 when unknown, which is the
        # sixteenth lattice the exporter used unconditionally before this
        # existed - so an intermediate written without it behaves as it did.
        subdivision = int(nt.get("subdivision") or 4) if nt else 4
        acoustic = annotations.latest_value(note.id, ACOUSTIC_ACTIVITY_ANNOTATION_KIND)
        by_stem[note.stem].append(NotationNote(
            pitch=_page_pitch(note, annotations), start_ms=start_ms, end_ms=end_ms,
            velocity=note.velocity, source_note_id=note.id, is_tuplet=is_tuplet,
            tuplet_divisor=tuplet_divisor,
            subdivision=subdivision,
            cohesion=uni_cohesion.get(note.id),
            trailing_resonance=(acoustic or {}).get("trailing_resonance"),
            trailing_void=(acoustic or {}).get("trailing_void"),
        ))
        stem_family[note.stem][family] += 1

    # Ties across everything that survived (same rule as build_notation_score).
    by_source: Dict[str, NotationNote] = {
        nn.source_note_id: nn for nns in by_stem.values() for nn in nns
    }
    for nn in list(by_source.values()):
        tie = annotations.latest_value(nn.source_note_id, TIE_RECONSTRUCTION_ANNOTATION_KIND)
        if tie is None:
            continue
        target = by_source.get(tie.get("tied_to_note_id"))
        if target is not None:
            nn.tie_start = True
            target.tie_stop = True

    parts: List[NotationPart] = []
    for stem, nns in by_stem.items():
        nns.sort(key=lambda n: (n.start_ms, n.pitch))
        dominant = stem_family[stem].most_common(1)[0][0] if stem_family[stem] else "unknown"
        is_drum = stem == StemType.DRUMS or dominant == _DRUM_FAMILY
        label = dominant if dominant not in ("unknown", _DRUM_FAMILY) else stem.value

        if is_drum:
            parts.append(NotationPart(family=dominant, voice_id=f"{dominant}::all",
                                      stem=stem, notes=nns, is_drum=True))
            continue

        # Bass is its own single staff, a bass LINE - never a grand staff and
        # never 4-part. Basic Pitch emits overtone/artifact simultaneity on
        # bass that would otherwise trip the polyphony test, so cap it hard at
        # 2 voices (a real occasional double-stop, nothing more).
        if stem == StemType.BASS:
            voiced = assign_voices(nns, max_voices=2, smooth=True, ms_per_beat=ms_per_beat,
                                   fill_max_beats=fill_max_beats,
                                   beats_per_bar=time_signature[0],
                                   bar_origin_ms=bar_origin_ms)
            parts.append(NotationPart(family=dominant, voice_id="bass::line",
                                      stem=stem, notes=voiced, is_drum=False))
            continue

        wants_grand = stem in _GRAND_STAFF_STEMS or dominant in _GRAND_STAFF_FAMILIES
        if wants_grand:
            # Label the grand staff by a real timbre when we have one, else by
            # the STEM ("other"/"guitar"/"piano") - never by an internal
            # brightness-bucket family (mid_body/bright_lead/warm_sustained),
            # which must not leak onto the page.
            gs_label = dominant if dominant in _GRAND_STAFF_FAMILIES else stem.value
            gs = build_grand_staff(nns, tempo_bpm, time_signature, key,
                                   ratio_family=ratio_family, max_voices=max_voices,
                                   family=gs_label, fill_max_beats=fill_max_beats,
                                   beats_per_bar=time_signature[0],
                                   bar_origin_ms=bar_origin_ms)
            parts.extend(gs.parts)
        elif _max_simultaneity(nns) >= _POLYPHONY_VOICE_THRESHOLD:
            voiced = assign_voices(nns, max_voices=max_voices, smooth=True,
                                   ms_per_beat=ms_per_beat, fill_max_beats=fill_max_beats,
                                   beats_per_bar=time_signature[0],
                                   bar_origin_ms=bar_origin_ms)
            parts.append(NotationPart(family=dominant, voice_id=f"{label}::poly",
                                      stem=stem, notes=voiced, is_drum=False))
        else:
            parts.append(NotationPart(family=dominant, voice_id=f"{label}::all",
                                      stem=stem, notes=nns, is_drum=False))

    parts.sort(key=lambda p: (p.is_drum, -p.mean_pitch))
    return NotationScore(parts=parts, tempo_bpm=tempo_bpm,
                         time_signature=time_signature, key=key, ratio_family=ratio_family,
                         musical_time=musical_time, bar_origin_ms=bar_origin_ms)


__all__ = [
    "NotationNote",
    "NotationPart",
    "NotationScore",
    "build_notation_score",
    "build_routed_score",
]
