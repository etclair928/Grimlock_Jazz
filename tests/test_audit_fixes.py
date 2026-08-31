# =================================================================
# MODULE: tests/test_audit_fixes.py
# Regression tests for the 2026-08-17 audit.
#
# Every defect the audit found was a pure function over plain data -
# reproducible in milliseconds with no audio, no models, and no separation.
# That is precisely why they survived: the project measures its DECISIONS
# carefully (and records those measurements in the module headers) but had
# nothing pinning the arithmetic underneath them, so a correct measurement
# could sit on top of a wrong conversion indefinitely.
#
# Each test below names the defect it pins and asserts the property, not the
# implementation - a future rewrite of any of these modules should keep these
# passing or be deliberately changing behaviour.
#
#   python -m pytest tests/ -q          (or: python tests/test_audit_fixes.py)
# =================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Note, Provenance, StemType, TempoMeter, MusicalTime  # noqa: E402
from key_intelligence.key_detector import KeyResult, _key_pitch_classes, key_fit  # noqa: E402
from quantization.duration_witness import (  # noqa: E402
    nearest_notatable, notatable_at_most, notatable_values,
)
from quantization.lattice_judge import build_lattice  # noqa: E402
from quantization.notation_quantizer import _subdivisions_per_beat  # noqa: E402
from quantization.rhythm_inference import infer_voice_rhythm  # noqa: E402
from output.scribe_engraver import _build_beat_tick_scales, MIDI_RESOLUTION_PPQN  # noqa: E402

BEAT_MS = 500.0


def _note(pitch: int, start_ms: float, end_ms: float, velocity: int = 90,
          confidence: float = 0.9, stem: StemType = StemType.OTHER) -> Note:
    return Note(pitch=pitch, start_ms=start_ms, end_ms=end_ms, velocity=velocity,
                confidence=confidence, stem=stem, source=Provenance.BASIC_PITCH)


# --- #01  minor keys are not their parallel major -------------------------

def test_minor_key_pitch_classes_match_the_relative_major():
    """A minor key's 7 pitch classes ARE its relative major's - that is what
    "relative" means. The bug transposed them down 3 semitones, landing on the
    parallel major, so A minor came out as A major."""
    a_minor = _key_pitch_classes("Am")
    c_major = _key_pitch_classes("C")
    assert a_minor == c_major, f"Am {sorted(a_minor)} != C {sorted(c_major)}"
    # A B C D E F G
    assert sorted(a_minor) == [0, 2, 4, 5, 7, 9, 11]
    assert _key_pitch_classes("Dm") == _key_pitch_classes("F")
    assert _key_pitch_classes("F#m") == _key_pitch_classes("A")


def test_key_fit_calls_the_tonic_triad_in_key():
    """The concrete symptom: in A minor, C natural was "out of key" and C#
    was "in key" - the two notes that most define the mode, both inverted."""
    a_minor = KeyResult(key="Am", confidence=0.9)
    assert key_fit(60, a_minor)[0] is True, "C natural must be in A minor"
    assert key_fit(61, a_minor)[0] is False, "C# must not be in A minor"
    assert key_fit(65, a_minor)[0] is True, "F natural must be in A minor"
    assert key_fit(66, a_minor)[0] is False, "F# must not be in A minor"


# --- #02  minor keys engrave with their own key signature -----------------

def test_musicxml_minor_key_is_minor():
    """`Key(key.replace("m",""))` handed music21 the PARALLEL MAJOR: Am -> 3
    sharps where A minor has none. music21 distinguishes mode by CASE."""
    from output.musicxml_exporter import _m21_key
    assert _m21_key("Am").sharps == 0
    assert _m21_key("F#m").sharps == 3
    assert _m21_key("C").sharps == 0
    assert _m21_key("Bb").sharps == -2
    assert _m21_key("") is None
    assert _m21_key("nonsense") is None


def test_midi_and_musicxml_agree_about_mode():
    """scribe_engraver got minor right while musicxml_exporter got it wrong,
    so the two exports of one transcription disagreed about its key."""
    from output.musicxml_exporter import _m21_key
    from output.scribe_engraver import _key_to_key_number
    for key in ("C", "Am", "F#m", "Eb", "Bbm"):
        midi_is_minor = _key_to_key_number(key) >= 12
        xml_is_minor = _m21_key(key).mode == "minor"
        assert midi_is_minor == xml_is_minor, f"{key}: MIDI and MusicXML disagree"


# --- #03  notation durations are not capped at one beat -------------------

def test_a_whole_note_is_written_as_a_whole_note():
    """The notation end came from the next onset in the beat or the beat line,
    never from the note itself - so one beat was the longest value this
    function could emit and a whole note became a quarter plus three beats of
    manufactured rest."""
    note = _note(60, 0.0, 4 * BEAT_MS)
    timing = infer_voice_rhythm([note], BEAT_MS, 0.0)[note.id]
    held = (timing.notation_end_ms - timing.notation_start_ms) / BEAT_MS
    assert held == 4.0, f"whole note written as {held} beats"


def test_a_sustained_note_stops_at_the_next_onset():
    """Lengthening must never overlap the next onset in the same voice, or the
    voice stops tiling its bar and the measure goes overfull."""
    held, next_up = _note(60, 0.0, 10 * BEAT_MS), _note(62, 1.5 * BEAT_MS, 2 * BEAT_MS)
    result = infer_voice_rhythm([held, next_up], BEAT_MS, 0.0)
    timing = result[held.id]
    assert timing.notation_end_ms <= next_up.start_ms + 1e-6
    assert (timing.notation_end_ms - timing.notation_start_ms) / BEAT_MS == 1.5


def test_a_short_note_still_fills_its_beat():
    """The floor is unchanged: a note that dies early inside its own beat is
    still written to the beat line, which is what a copyist writes."""
    note = _note(60, 0.0, 120.0)
    timing = infer_voice_rhythm([note], BEAT_MS, 0.0)[note.id]
    assert (timing.notation_end_ms - timing.notation_start_ms) == BEAT_MS


def test_every_inferred_duration_is_notatable():
    """Whatever the inference decides, the page has to be able to draw it."""
    notes = [_note(60, 0.0, 4 * BEAT_MS), _note(62, 4 * BEAT_MS, 4.5 * BEAT_MS),
             _note(64, 6 * BEAT_MS, 9 * BEAT_MS), _note(65, 12 * BEAT_MS, 12.3 * BEAT_MS)]
    for timing in infer_voice_rhythm(notes, BEAT_MS, 0.0).values():
        beats = (timing.notation_end_ms - timing.notation_start_ms) / BEAT_MS
        assert any(abs(beats - v) < 1e-6 for v in notatable_values(True)), \
            f"{beats} beats is not a notatable duration"


# --- #04  the quantizer grid has a phase ----------------------------------

def test_grid_lines_land_on_the_tracked_beats():
    """The lattice's grid was `k * period_ms` from ABSOLUTE ZERO - a tempo with
    no phase. On a track whose first beat is at 300ms every grid line sat 300ms
    from every real beat."""
    beats = tuple(300.0 + i * BEAT_MS for i in range(40))
    tm = TempoMeter(tempo_bpm=120.0, confidence=0.8, time_signature_numerator=4,
                    time_signature_denominator=4, beat_times_ms=beats)
    lattice = build_lattice(tm)
    assert lattice.get_beat_time(0) == 300.0
    # subdivisions_per_beat grid lines later we should be on the next beat
    assert abs(lattice.get_beat_time(lattice.subdivisions_per_beat) - 800.0) < 1e-6


def test_a_note_on_the_beat_is_not_snapped_off_it():
    """The symptom that proves the phase bug: a note landing exactly on a
    tracked beat got moved AWAY from it."""
    beats = tuple(300.0 + i * BEAT_MS for i in range(40))
    tm = TempoMeter(tempo_bpm=120.0, confidence=0.8, time_signature_numerator=4,
                    time_signature_denominator=4, beat_times_ms=beats)
    lattice = build_lattice(tm, musical_time=MusicalTime.from_beats(beats))
    on_the_beat = _note(60, 800.0, 1000.0, confidence=0.2)
    snapped, _end, _why = lattice.quantize_note(on_the_beat)
    assert abs(snapped - 800.0) < 1e-6, f"on-beat note moved to {snapped}"


def test_a_late_note_is_pulled_toward_the_beat_not_away():
    beats = tuple(300.0 + i * BEAT_MS for i in range(40))
    tm = TempoMeter(tempo_bpm=120.0, confidence=0.8, time_signature_numerator=4,
                    time_signature_denominator=4, beat_times_ms=beats)
    lattice = build_lattice(tm, musical_time=MusicalTime.from_beats(beats))
    late = _note(60, 845.0, 1000.0, confidence=0.2)
    snapped, _end, _why = lattice.quantize_note(late)
    assert 800.0 <= snapped < 845.0, f"late note moved to {snapped}, away from beat 800"


# --- #05  the resolved meter reaches the quantizer ------------------------

def test_compound_meter_reaches_the_subdivision_tests():
    """resolve_tempo defaults time_signature to 4/4 and the Conductor cannot
    pass one (meter is resolved later), so the TempoMeter carried 4/4 all run
    and both compound-meter branches were unreachable in detected mode."""
    from dataclasses import replace
    tm = TempoMeter(tempo_bpm=100.0, confidence=0.8, time_signature_numerator=4,
                    time_signature_denominator=4,
                    beat_times_ms=tuple(i * 600.0 for i in range(40)))
    compound = replace(tm, time_signature_numerator=6, time_signature_denominator=8)
    assert build_lattice(compound).subdivisions_per_beat == 3
    assert _subdivisions_per_beat(compound) == 3
    assert compound.measure_duration_ms == 6 * compound.beat_duration_ms


def test_conductor_propagates_the_resolved_meter():
    """The wiring itself: whatever resolve_meter returns must be what the
    TempoMeter downstream stages read."""
    import inspect
    from orchestration import conductor
    source = inspect.getsource(conductor.transcribe_file)
    assert "time_signature_numerator=meter_resolution.numerator" in source, \
        "the resolved meter is no longer propagated into the TempoMeter"


# --- #06  the MIDI beat grid is beat-aligned ------------------------------

def test_beat_zero_always_lands_on_a_beat_tick():
    """Beat 0 must sit on a whole multiple of the resolution or every beat in
    the file inherits the remainder. The old code anchored beat 0 at tick 0
    without shifting anything, which displaced the whole grid by the pickup."""
    for beat0_ms in (5.0, 20.0, 60.0, 175.0, 249.0, 300.0, 450.0, 600.0, 1100.0):
        beats = [beat0_ms + i * BEAT_MS for i in range(20)]
        scales, beat0_tick, pad_s = _build_beat_tick_scales(beats, MIDI_RESOLUTION_PPQN)
        seconds_per_tick = scales[0][1]
        landed = (beat0_ms / 1000.0 + pad_s) / seconds_per_tick
        assert abs(landed - beat0_tick) < 0.5, (
            f"pickup {beat0_ms}ms: beat 0 lands at tick {landed:.1f}, "
            f"not the beat tick {beat0_tick}")


def test_the_pickup_shift_stays_under_one_beat():
    """The shift is the cost of correct barring; it must stay small enough to
    be a lead-in rather than a re-timing of the piece."""
    for beat0_ms in (5.0, 60.0, 175.0, 249.0, 300.0, 600.0):
        beats = [beat0_ms + i * BEAT_MS for i in range(20)]
        _scales, _tick, pad_s = _build_beat_tick_scales(beats, MIDI_RESOLUTION_PPQN)
        assert abs(pad_s) < BEAT_MS / 1000.0, f"pickup {beat0_ms}ms shifted {pad_s}s"


def test_no_absurd_leading_tempo():
    """The original bug this area's guard was added for: cramming a 175ms
    pickup into one beat fabricated a 343 BPM leading event that made
    MuseScore mis-read the file's tempo. Must not come back."""
    beats = [175.0 + i * BEAT_MS for i in range(20)]
    scales, _tick, _pad = _build_beat_tick_scales(beats, MIDI_RESOLUTION_PPQN)
    leading_bpm = 60.0 / (scales[0][1] * MIDI_RESOLUTION_PPQN)
    assert 30.0 < leading_bpm < 250.0, f"leading tempo event is {leading_bpm:.0f} BPM"


# --- #07 / #08  every emitted duration is notatable -----------------------

def test_allow_triplet_widens_the_vocabulary_it_does_not_replace_it():
    """`allow_triplet` used to return round(ql*3)/3 unconditionally, so a plain
    eighth inside a tuplet-flagged beat became a triplet eighth - which is how
    binary and ternary positions ended up in one measure."""
    import output.musicxml_exporter as mx
    from output.musicxml_exporter import _snap_quarter_length
    # Tested UNPOLISHED. This pins the primitive - that allow_triplet widens
    # the vocabulary instead of replacing it - which is a different question
    # from Polish's policy of refusing anything finer than an eighth. Polish
    # sits on top and is asserted separately below.
    before = mx.MAX_SUBDIVISION
    try:
        mx.MAX_SUBDIVISION = None
        assert _snap_quarter_length(0.5, True) == 0.5
        assert _snap_quarter_length(1.0, True) == 1.0
        assert abs(_snap_quarter_length(0.34, True) - 1.0 / 3.0) < 1e-6
        assert _snap_quarter_length(0.5, False) == 0.5
    finally:
        mx.MAX_SUBDIVISION = before


def test_polish_suppresses_the_triplet_the_primitive_would_allow():
    """The other half of the pair above: with Polish on, the same ternary
    value must NOT come back as a triplet, because the divisor path would
    otherwise route straight around the eighth ceiling."""
    import output.musicxml_exporter as mx
    from output.musicxml_exporter import _snap_quarter_length
    before = mx.MAX_SUBDIVISION
    try:
        mx.MAX_SUBDIVISION = 2
        got = _snap_quarter_length(0.34, True)
        assert abs(got - 1.0 / 3.0) > 1e-6, "a triplet survived the cap"
        assert abs(got * 2 - round(got * 2)) < 1e-9, "must land on the eighth grid"
    finally:
        mx.MAX_SUBDIVISION = before


def test_the_gap_clamp_produces_a_notatable_duration():
    """The clamp assigned a RAW gap - the difference of two snapped offsets,
    e.g. 1/3 + 1/4 = 0.5833 - straight onto quarterLength, undoing the snap
    three lines above it. That is the 24:13 nonsense-ratio source."""
    gap = 1.0 / 3.0 + 0.25
    clamped = notatable_at_most(gap, True)
    assert clamped <= gap + 1e-9, "the clamp must never round UP past the ceiling"
    assert any(abs(clamped - v) < 1e-9 for v in notatable_values(True))


def test_notatable_helpers_agree_with_each_other():
    """UPDATED when notatable_at_most stopped violating its own ceiling.

    This used to assert the result was always a member of the vocabulary,
    which is exactly what let the bug through: when nothing in the table fit
    under the ceiling the function returned the SMALLEST member - a value too
    LONG - and this test passed, because that value is indeed in the
    vocabulary. The property worth asserting is the ceiling itself; "or 0.0,
    meaning nothing fits" is the honest completion of the contract."""
    for value in (0.1, 0.26, 0.4, 0.51, 0.9, 1.4, 2.9, 5.0, 100.0):
        assert nearest_notatable(value, True) in notatable_values(True)
        at_most = notatable_at_most(value, False)
        assert at_most <= value + 1e-9, "the ceiling must never be exceeded"
        assert at_most == 0.0 or at_most in notatable_values(False)


def test_a_ceiling_below_every_notatable_value_yields_nothing():
    """THE k/24 BUG, pinned. A note with 1/12 of a beat of room used to be
    written 1/8 long - the smallest binary value - overrunning by exactly
    1/8 - 1/12 = 1/24 and displacing every onset after it onto a k/24
    position. There is no binary value that small, and saying so is the only
    correct answer."""
    assert notatable_at_most(1.0 / 12.0, False) == 0.0
    assert notatable_at_most(0.001, False) == 0.0


# --- #09  the MusicXML DOCTYPE survives the voice renumbering -------------

def test_export_keeps_the_doctype(tmp_path=None):
    """_normalize_voice_numbers round-tripped the file through ElementTree,
    which does not preserve the doctype - so every export silently lost it."""
    import tempfile
    from output.notation_score import NotationNote, NotationPart, NotationScore
    from output.musicxml_exporter import export_musicxml

    notes = [NotationNote(pitch=60 + i, start_ms=i * BEAT_MS,
                          end_ms=i * BEAT_MS + 400.0, velocity=80,
                          source_note_id=f"n{i}") for i in range(8)]
    score = NotationScore(
        parts=[NotationPart(family="piano", voice_id="p::all",
                            stem=StemType.OTHER, notes=notes)],
        tempo_bpm=120.0, time_signature=(4, 4), key="Am")

    directory = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    path = os.path.join(directory, "score.musicxml")
    export_musicxml(score, path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert "<!DOCTYPE score-partwise" in text, "the doctype was stripped"
    assert text.index("<!DOCTYPE") < text.index("<score-partwise"), \
        "the doctype must precede the root element"
    # and the key signature is the minor one (#02, end to end)
    assert "<fifths>0</fifths>" in text, "Am must engrave with no sharps"


def test_voice_numbers_are_dense_and_one_based():
    """The invariant _normalize_voice_numbers exists for - MuseScore rejects
    voice 0 outright."""
    import re
    import tempfile
    from output.notation_score import NotationNote, NotationPart, NotationScore
    from output.musicxml_exporter import export_musicxml

    notes = []
    for i in range(12):
        notes.append(NotationNote(pitch=60 + (i % 5) * 3, start_ms=(i // 3) * BEAT_MS,
                                  end_ms=(i // 3) * BEAT_MS + 900.0, velocity=80,
                                  source_note_id=f"n{i}"))
    score = NotationScore(
        parts=[NotationPart(family="piano", voice_id="p::all",
                            stem=StemType.OTHER, notes=notes)],
        tempo_bpm=120.0, time_signature=(4, 4), key="C")
    path = os.path.join(tempfile.mkdtemp(), "voices.musicxml")
    export_musicxml(score, path)
    with open(path, encoding="utf-8") as fh:
        voices = {int(v) for v in re.findall(r"<voice>(\d+)</voice>", fh.read())}
    assert voices, "no voices emitted"
    assert min(voices) == 1, f"voice numbers start at {min(voices)}, must be 1"
    assert voices == set(range(1, max(voices) + 1)), f"voice numbers are sparse: {sorted(voices)}"


# --- #10  one map between clock time and musical position -----------------

def test_musical_time_is_the_only_interpolator():
    """notation_quantizer carried its own copy of MusicalTime.to_beats/to_ms.
    Two implementations of one map is how the +9.6% Chopin fix got reverted one
    layer down the first time."""
    import quantization.notation_quantizer as nq
    assert not hasattr(nq, "_beat_position"), "the duplicate interpolator is back"
    assert not hasattr(nq, "_beat_to_ms"), "the duplicate interpolator is back"


def test_musical_time_round_trips():
    beats = tuple(300.0 + i * BEAT_MS for i in range(20))
    mt = MusicalTime.from_beats(beats)
    for t in (0.0, 300.0, 512.0, 1234.5, 9000.0):
        assert abs(mt.to_ms(mt.to_beats(t)) - t) < 1e-6, f"round trip failed at {t}"


def test_musical_time_is_exported_from_core():
    """core/__init__'s own law: import from `core`, not `core.submodule`."""
    import core
    assert "MusicalTime" in core.__all__


# --- #11  the downbeat phase is computed AND consumed ---------------------

def test_meter_estimator_returns_its_phase():
    from rhythm_engine import estimate_time_signature_with_phase
    import inspect
    signature = inspect.signature(estimate_time_signature_with_phase)
    assert "beat_times_ms" in signature.parameters


def test_conductor_consumes_the_downbeat_phase():
    import inspect
    from orchestration import conductor
    source = inspect.getsource(conductor.transcribe_file)
    assert "estimate_time_signature_with_phase" in source
    assert "downbeat_times_ms=resolved_downbeats" in source, \
        "the resolved downbeat no longer reaches the TempoMeter"
    assert "bar_origin_ms=bar_origin_ms" in source, \
        "the notation layers no longer receive the bar origin"


def test_bar_origin_defaults_are_safe():
    """No downbeat evidence must leave behaviour exactly as it was."""
    from output.notation_score import NotationScore
    assert NotationScore().bar_origin_ms is None


# --- #15  consolidation ends stay on one timeline -------------------------

def test_consolidation_records_its_last_fragment():
    """The merged end is a RAW millisecond. Rendering a different timeline
    needs to ask the run's last fragment for ITS end on that timeline rather
    than pasting a raw value onto a snapped start."""
    from core import AnnotationStore
    from quantization.note_consolidation import (
        consolidate_fragments, CONSOLIDATION_ANNOTATION_KIND,
    )
    fragments = [_note(60, 0.0, 100.0), _note(60, 100.0, 200.0), _note(60, 200.0, 340.0)]
    annotations = AnnotationStore()
    runs, absorbed = consolidate_fragments(fragments, annotations)
    assert (runs, absorbed) == (1, 2)
    primary = annotations.latest_value(fragments[0].id, CONSOLIDATION_ANNOTATION_KIND)
    assert primary["role"] == "primary"
    assert primary["end_ms"] == 340.0
    assert primary["last_note_id"] == fragments[2].id


def test_consolidation_is_gated_on_a_real_attack():
    """A detected attack at the boundary, ACROSS A REAL GAP, means the player
    struck the pitch again - the run must stop rather than swallow it.

    This is the Ellington protection: planing textures re-struck by design lost
    31% of their notes and 43% of their 4+ note chords before the gate existed.
    The gap here is 40ms, inside the 15-60ms band where a fast repeat lives.
    """
    from core import AnnotationStore
    from quantization.note_consolidation import consolidate_fragments
    fragments = [_note(60, 0.0, 100.0), _note(60, 140.0, 240.0)]
    annotations = AnnotationStore()
    runs, absorbed = consolidate_fragments(
        fragments, annotations, onsets_ms=[140.0])
    assert (runs, absorbed) == (0, 0), "an attack across a real gap must block the merge"


def test_touching_notes_merge_even_with_an_attack_at_the_boundary():
    """AMENDED CONTRACT (2026-08-23). This case used to assert the opposite,
    with a gap of exactly ZERO - encoding the assumption that any onset at a
    boundary proves a re-articulation.

    Measurement contradicts it. Of the merges the gate blocks across the
    corpus, 96-99% are notes touching within 15ms: 642 of Chopin's 666, 471 of
    Burden's 478. Fifteen milliseconds is a sixty-fourth note at 240bpm - no
    player re-articulates that fast, so the onset found there is the note's own
    attack seen again, not a second one. Blocking those left runs absorbing
    24-64% of eligible fragments where pre-gate runs absorbed 100%.
    """
    from core import AnnotationStore
    from quantization.note_consolidation import (
        consolidate_fragments, MIN_REARTICULATION_GAP_MS,
    )
    assert MIN_REARTICULATION_GAP_MS < 31.0, (
        "must stay below a 32nd note at 240bpm, or it starts eating real repeats")
    fragments = [_note(60, 0.0, 100.0), _note(60, 100.0, 200.0)]
    annotations = AnnotationStore()
    runs, absorbed = consolidate_fragments(
        fragments, annotations, onsets_ms=[100.0])
    assert (runs, absorbed) == (1, 1), "touching notes are one sounding event"


if __name__ == "__main__":
    import traceback
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
