# =================================================================
# MODULE: orchestration/conductor.py
# Orchestration Conductor (GRIMLOCK_6.0_DESIGN_DECISIONS.md §7): 6.0 is
# correct, deterministic, LINEAR - one straight pass, no feedback/auto-
# re-analysis loops, no parallel DAG (explicitly deferred to 6.1+, §8).
# When witnesses disagree (tempo, meter), SURFACE it via Epistemic and
# record what happened - never loop back and re-run anything.
#
# Pipeline: decode -> separate -> pitch (Basic Pitch per stem, CREPE as
# bass's own extra read) -> drums (Rhythm Engine, not Basic Pitch) ->
# Instrument Attribution (per pitched stem) -> FOUR independent tempo
# witnesses (librosa/madmom beat-tracking, note-onset-pattern from
# Basic Pitch's own notes, ReverseGeoCrypt's event-lattice search) ->
# Epistemic resolves tempo/meter -> PulseField (multi-hypothesis) +
# GrooveField (bass-vs-kick relational phase) -> per-note Quantization
# (lattice_judge + duration_witness, written as Annotations, never
# mutating a Note) -> Scribe Engraver commits the MIDI (raw timing by
# default; quantized timing opt-in). Exactly one place each thing happens.
# =================================================================

from __future__ import annotations

import gc
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from audio_engine import AudioEngine, AudioTrack
from core import Annotation, AnnotationStore, MusicBox, MusicalFindingsMap, Note, Provenance, Separation, StemType
from separation_engine import separate
from separation_engine import merge_harmonic_stems as _merge_harmonic_stems
from pitch_engine import (
    transcribe_basic_pitch, transcribe_basic_pitch_with_posteriorgram,
    transcribe_crepe_bass, BASIC_PITCH_SAMPLE_RATE,
)
from instrument_attribution import resolve_instrument_identity
from key_intelligence import analyze_key, key_fit, KEY_FIT_ANNOTATION_KIND
from rhythm_engine import (
    run_librosa_tempo, run_madmom_tempo, run_note_onset_tempo_witness, run_lattice_witness,
    run_pulse_field, estimate_groove, compute_phase_deltas, detect_drums,
    build_phase_locked_grid, estimate_time_signature, sample_beat_accents,
    fft_meter_candidate, resolve_denominator, detect_onset_candidates,
)
from epistemic import resolve_tempo, resolve_meter
from quantization import (
    build_lattice, infer_duration, QUANTIZATION_ANNOTATION_KIND,
    propose_sustain_extensions, SUSTAIN_RECOVERY_ANNOTATION_KIND,
    evaluate_legitimacy, MICRO_NOTE_PURGE_ANNOTATION_KIND, PURGE_CANDIDATE,
    find_tie_candidates, TIE_RECONSTRUCTION_ANNOTATION_KIND,
    build_trouble_map, notation_quantize_note, NOTATION_TIMING_ANNOTATION_KIND,
    infer_voice_rhythm,
)
from acoustic_witness import (
    analyze_stem, AnechoicReport, ACOUSTIC_ACTIVITY_ANNOTATION_KIND,
    audit_note, HarmonicVerdict, HARMONIC_LEGITIMACY_ANNOTATION_KIND,
    evaluate_stem_support, NOTE_SUPPORT_ANNOTATION_KIND, NOTE_SUPPORT_SAMPLE_RATE, UNSUPPORTED,
)
from output import engrave
from model_registry import unload_demucs

# htdemucs_6s is the doc's preferred model (§4: splits guitar/piano out
# of "other" at the audio layer). Its checkpoint once failed torch.hub's
# hash check - that turned out to be a corrupted cached download, not an
# upstream integrity problem: a fresh download verifies cleanly against
# the pinned SHA256 (34c22ccb...), the model loads with all 6 sources,
# and a real 60s separation produced genuine guitar/piano stems with the
# "other" residual dropping to a fraction of its 4-stem energy. The hash
# check stays fully enforced - nothing was bypassed.
DEFAULT_SEPARATION_MODEL = "htdemucs_6s"
DURATION_ANNOTATION_KIND = "duration_hypothesis"

_PITCHED_STEMS = (StemType.BASS, StemType.VOCALS, StemType.OTHER, StemType.GUITAR, StemType.PIANO)


@dataclass
class PipelineResult:
    audio_path: str
    output_midi_path: str
    separation_model: str
    separation_seed: Optional[int] = 0
    note_counts_by_stem: Dict[str, int] = field(default_factory=dict)
    crepe_bass_note_count: int = 0
    tempo_bpm: float = 0.0
    tempo_confidence: float = 0.0
    time_signature: tuple = (4, 4)
    tempo_contention: Optional[dict] = None
    meter_contention: Optional[dict] = None
    total_notes_exported: int = 0
    elapsed_seconds: float = 0.0
    findings: Optional[MusicalFindingsMap] = None


def transcribe_file(
        audio_path: Union[str, Path],
        output_midi_path: Union[str, Path],
        separation_model: str = DEFAULT_SEPARATION_MODEL,
        device: str = "cpu",
        use_quantized_timing: bool = False,
        use_notation_timing: bool = False,
        drop_purge_candidates: bool = False,
        merge_harmonic_stems: bool = True,
        separation_seed: Optional[int] = 0,
) -> PipelineResult:
    start_time = time.time()
    audio_path = str(audio_path)
    output_midi_path = str(output_midi_path)

    engine = AudioEngine()
    music_box = MusicBox()
    findings = MusicalFindingsMap()
    annotations = AnnotationStore()

    music_box.log_decision(
        stage_name="conductor", decision_type="session_start",
        before_state={}, after_state={"audio_path": audio_path, "separation_model": separation_model,
                                      "separation_seed": separation_seed},
        reasoning=f"Starting 6.0 pipeline for {audio_path}", reversible=False,
    )

    master_track = engine.decode(audio_path)

    # The separation seed is passed EXPLICITLY and recorded, not left implicit.
    # It is fixed today (determinism, §III), but a run should carry the seed
    # that produced it: the day stochasticity is deliberately re-enabled for
    # robustness testing - many seeds, same audio - every archived run needs to
    # say which draw it was. A reproducible result that doesn't record what made
    # it reproducible is only accidentally reproducible.
    separation: Separation = separate(
        engine, master_track, model_name=separation_model, device=device, seed=separation_seed,
    )
    music_box.log_decision(
        stage_name="separation_engine", decision_type="separated",
        before_state={},
        after_state={"model": separation.model_used, "stems": [s.value for s in separation.stems],
                     "separation_seed": separation_seed},
        reasoning=f"Separated into {len(separation.stems)} stems via {separation.model_used} "
                  f"(seed={separation_seed})", reversible=False,
    )

    # De-duplicate the harmonic stems (§XII.1). htdemucs_6s's guitar/piano
    # heads reassign the SAME content between guitar/piano/other over time,
    # so transcribing them independently counts that content more than once -
    # measured at 53% duplication on Hopeful (4645 notes across the three,
    # 2183 when summed and transcribed once). Merging restores the 4-stem
    # "other" semantics while keeping the reliable drums/bass/vocals heads
    # separate. This is DE-DUPLICATION, not filtering: no note is judged or
    # deleted, the same audio is simply transcribed once instead of thrice.
    if merge_harmonic_stems:
        stems_before = [s.value for s in separation.stems]
        separation = _merge_harmonic_stems(separation)
        music_box.log_decision(
            stage_name="separation_engine", decision_type="harmonic_stems_merged",
            before_state={"stems": stems_before},
            after_state={"stems": [s.value for s in separation.stems], "model": separation.model_used},
            reasoning="Merged guitar+piano+other into one 'other' stem - the 6-stem split "
                      "reassigns the same content between them over time, which inflates the "
                      "note count when each is transcribed separately (§XII.1).",
            reversible=False,
        )

    # Memory is a policy, not a place (§7): Demucs is typically the
    # single largest resident model, and this run never needs it again -
    # unload it now rather than carrying it through Pitch/Rhythm Engine.
    unload_demucs(separation_model, device)
    gc.collect()

    # Key Intelligence (ported): song-level key detection from the full
    # mix - a chord/harmony fact needs every instrument's contribution,
    # not one isolated stem's. Independent of everything else (no
    # tempo/pitch dependency), so it can run this early.
    key_result = analyze_key(engine, master_track)
    findings.key = key_result.key
    findings.key_confidence = key_result.confidence
    music_box.log_decision(
        stage_name="key_intelligence", decision_type="key_detected",
        before_state={}, after_state={"key": key_result.key, "confidence": key_result.confidence},
        reasoning=f"Key resolved to {key_result.key} (confidence {key_result.confidence:.2f})",
        reversible=False,
    )

    all_notes: List[Note] = []
    pitched_notes: List[Note] = []
    bass_notes: List[Note] = []
    note_counts: Dict[str, int] = {}
    crepe_bass_count = 0

    drum_notes: List[Note] = []
    if separation.has_stem(StemType.DRUMS):
        drum_notes = detect_drums(engine, separation.get_stem(StemType.DRUMS), annotations)
        all_notes.extend(drum_notes)
        note_counts[StemType.DRUMS.value] = len(drum_notes)
        music_box.log_decision(
            stage_name="rhythm_engine", decision_type="drums_detected",
            before_state={}, after_state={"hit_count": len(drum_notes)},
            reasoning=f"Detected {len(drum_notes)} drum hits", reversible=False,
        )

    sustain_extended_count = 0
    stem_tracks_by_type: Dict[StemType, AudioTrack] = {}
    anechoic_reports: Dict[StemType, AnechoicReport] = {}
    for stem in _PITCHED_STEMS:
        if not separation.has_stem(stem):
            continue
        stem_track = separation.get_stem(stem)
        stem_tracks_by_type[stem] = stem_track
        stem_notes, posteriorgram = transcribe_basic_pitch_with_posteriorgram(engine, stem_track, stem)

        # AnechoicMa (ported): frame-level silence/resonance/activity
        # evidence for this stem's own audio - queried per note below.
        anechoic_reports[stem] = analyze_stem(engine, stem_track, stem)

        # SustainRecovery (Ritornello Pass 2, ported): extend a note's END
        # when it's still acoustically ringing past Basic Pitch's own
        # nominal end - never touches pitch/start, never mutates the Note.
        # Written as an Annotation; Scribe Engraver decides whether to
        # read it (opt-in via use_quantized_timing, same law as
        # quantization's snapped timing).
        stem_samples = engine.view(stem_track, BASIC_PITCH_SAMPLE_RATE).samples
        extensions = propose_sustain_extensions(stem_notes, stem_samples, BASIC_PITCH_SAMPLE_RATE, posteriorgram)
        for ext in extensions:
            annotations.add(Annotation(
                note_id=ext.note_id, kind=SUSTAIN_RECOVERY_ANNOTATION_KIND,
                value={"end_ms": ext.extended_end_ms, "reason": ext.reason},
                source=Provenance.RITORNELLO, confidence=ext.confidence,
            ))
        sustain_extended_count += len(extensions)

        if stem == StemType.BASS:
            bass_notes = stem_notes
            crepe_notes = transcribe_crepe_bass(engine, stem_track)
            crepe_bass_count = len(crepe_notes)
            music_box.log_decision(
                stage_name="pitch_engine", decision_type="crepe_bass_witness",
                before_state={}, after_state={"crepe_note_count": crepe_bass_count,
                                               "basic_pitch_note_count": len(stem_notes)},
                reasoning="CREPE's bass read is its own witness (§3) - Basic Pitch's notes are what's "
                          "exported, so bass isn't double-counted; CREPE's read is logged, not merged.",
                reversible=False,
            )

        lines = resolve_instrument_identity(engine, stem_track, stem, stem_notes, annotations)
        all_notes.extend(stem_notes)
        pitched_notes.extend(stem_notes)
        note_counts[stem.value] = len(stem_notes)

        families_found = sorted({
            annotations.latest_value(line.notes[0].id, "instrument_family")
            for line in lines if line.notes
        } - {None})
        if families_found:
            findings.instrument_families[stem] = families_found

    music_box.log_decision(
        stage_name="quantization", decision_type="sustain_recovery",
        before_state={}, after_state={"notes_extended": sustain_extended_count},
        reasoning=f"SustainRecovery proposed an extension for {sustain_extended_count} of "
                  f"{len(pitched_notes)} pitched notes (posteriorgram + narrowband-energy witnesses)",
        reversible=False,
    )

    # Rhythm Engine: FOUR independent tempo witnesses from the full
    # original track + the pitched notes just detected. Each audio-only
    # witness is already octave-corrected internally (tempo_witness.py).
    librosa_witness = run_librosa_tempo(engine, master_track)
    madmom_witness = run_madmom_tempo(engine, master_track)
    note_onset_witness = run_note_onset_tempo_witness(pitched_notes)
    lattice_witness, lattice = run_lattice_witness(engine, master_track)

    tempo_witnesses = [librosa_witness, madmom_witness, note_onset_witness]
    if lattice_witness is not None:
        tempo_witnesses.append(lattice_witness)
    tempo_resolution = resolve_tempo(tempo_witnesses)

    # ratio_family (binary/ternary/swing/...) is real per-track evidence
    # the lattice search already computed - it decides simple vs compound
    # for a 6/9/12-beat bar, not the numerator alone (see meter.py).
    ratio_family = lattice.ratio_family if lattice is not None else None

    # ONE canonical beat grid anchored to the RESOLVED tempo, not each
    # witness's own independently-tracked grid - each witness's own
    # tempo error accumulates into phase drift fast enough to corrupt
    # exactly the longer bar lengths (6/4 in particular) a meter test
    # cares about most (see meter.py's module docstring for the math).
    onset_candidates = detect_onset_candidates(engine, master_track)
    phase_locked_grid = build_phase_locked_grid(
        tempo_resolution.tempo_meter.tempo_bpm,
        onset_candidates.combined_ms,
        master_track.duration_seconds * 1000.0,
    )

    meter_candidates = []
    if phase_locked_grid:
        meter_candidates.append(
            estimate_time_signature(engine, master_track, phase_locked_grid, ratio_family=ratio_family)
        )
        # Independent second method against the SAME grid: spectral
        # periodicity of the beat-accent sequence itself, rather than
        # only the downbeat-salience heuristic - genuine corroboration,
        # not three copies of the same test on different data.
        beat_accents = sample_beat_accents(engine, master_track, phase_locked_grid)
        fft_candidate = fft_meter_candidate(beat_accents)
        if fft_candidate is not None:
            fft_numerator, fft_power = fft_candidate
            meter_candidates.append(
                (fft_numerator, resolve_denominator(fft_numerator, ratio_family), fft_power)
            )

    if not meter_candidates:
        meter_candidates.append((4, 4, 0.3))
    meter_resolution = resolve_meter(meter_candidates)

    swing_ratio, groove_confidence = estimate_groove(engine, master_track, tempo_resolution.tempo_meter.beat_times_ms)

    # PulseField: multi-hypothesis pulse tracking seeded from the
    # resolved tempo, reinforced by real onsets and (if available) the
    # ReverseGeoCrypt lattice reading.
    pulse_result = run_pulse_field(
        engine, master_track,
        initial_tempo_bpm=tempo_resolution.tempo_meter.tempo_bpm,
        initial_confidence=tempo_resolution.tempo_meter.confidence,
        lattice_period_ms=lattice.subdivision_ms if lattice is not None else None,
        lattice_confidence=lattice.confidence if lattice is not None else 0.0,
    )

    # GrooveField: cross-instrument bass-vs-kick phase delta - the
    # "Dilla pocket" relational feel, distinct from PulseField's
    # single-track multi-hypothesis tracking.
    kick_notes = [n for n in drum_notes if annotations.latest_value(n.id, "drum_type") == "kick"]
    phase_delta_result = compute_phase_deltas(bass_notes, kick_notes)

    findings.tempo_meter = tempo_resolution.tempo_meter
    findings.tempo_contention = tempo_resolution.contention
    findings.swing_ratio = swing_ratio
    findings.groove_confidence = groove_confidence

    music_box.log_decision(
        stage_name="epistemic", decision_type="tempo_resolved",
        before_state={"witness_count": len(tempo_witnesses),
                      "raw_bpms": [w.tempo_bpm for w in tempo_witnesses]},
        after_state={"resolved_bpm": tempo_resolution.tempo_meter.tempo_bpm,
                     "contention": tempo_resolution.contention is not None},
        reasoning=f"Tempo resolved to {tempo_resolution.tempo_meter.tempo_bpm:.1f}bpm "
                  f"across {len(tempo_witnesses)} witnesses", reversible=False,
    )
    music_box.log_decision(
        stage_name="epistemic", decision_type="meter_resolved",
        before_state={"candidates": meter_candidates},
        after_state={"numerator": meter_resolution.numerator, "denominator": meter_resolution.denominator,
                     "contention": meter_resolution.contention is not None},
        reasoning=f"Meter resolved to {meter_resolution.numerator}/{meter_resolution.denominator}",
        reversible=False,
    )
    music_box.log_decision(
        stage_name="rhythm_engine", decision_type="groove_resolved",
        before_state={}, after_state={
            "pulse_best_bpm": pulse_result.best_tempo_bpm if pulse_result else None,
            "pulse_coherence": pulse_result.coherence if pulse_result else None,
            "phase_delta_groove_type": phase_delta_result.groove_type.value,
            "phase_delta_avg_ms": phase_delta_result.average_ms,
        },
        reasoning=f"Groove: {phase_delta_result.groove_type.value} "
                  f"(bass/kick avg delta {phase_delta_result.average_ms:.1f}ms, "
                  f"{phase_delta_result.sample_count} pairs)",
        reversible=False,
    )

    # Quantization: per-note snap proposal + duration hypothesis, written
    # as Annotations - Note.start_ms/end_ms are never touched (§2's
    # frozen-detection-floor law holds by construction here).
    lattice_judge = build_lattice(
        tempo_resolution.tempo_meter,
        swing_ratio=swing_ratio,
        groove_confidence=groove_confidence,
        lattice_confidence=lattice.confidence if lattice is not None else 0.0,
        phase_delta_ms=phase_delta_result.average_ms,
    )
    beat_ms = tempo_resolution.tempo_meter.beat_duration_ms

    # Beat-level Bayesian rhythm inference for the notation timeline,
    # run PER STEM (~one voice each, thanks to 6-stem separation) before
    # the per-note loop. This is the sequence-level upgrade over per-note
    # snapping: it recovers triplets/tuplet groups and notates swung
    # eighths straight (using the measured swing_ratio as the style
    # prior) - context a per-note snapper can't see. Notes whose beat
    # didn't parse confidently fall back to notation_quantize_note below.
    grid_origin_ms = (tempo_resolution.tempo_meter.beat_times_ms[0]
                      if tempo_resolution.tempo_meter.beat_times_ms else 0.0)
    inferred_rhythm: Dict[str, "object"] = {}
    _notes_by_stem_for_rhythm: Dict[StemType, List[Note]] = defaultdict(list)
    for note in pitched_notes:
        _notes_by_stem_for_rhythm[note.stem].append(note)
    for stem_notes in _notes_by_stem_for_rhythm.values():
        inferred_rhythm.update(
            infer_voice_rhythm(stem_notes, beat_ms, grid_origin_ms, swing_ratio=swing_ratio)
        )
    rhythm_inferred_count = 0

    # Note-support pre-pass (over-detection / bleed / hallucination
    # filter): built from a 3-song diagnostic that measured which signals
    # actually separate spurious notes. Per stem, flags a note UNSUPPORTED
    # only if BOTH its own fundamental is near-silent in its own stem's
    # audio AND AnechoicMa says the window is acoustically inactive - the
    # two signals that discriminated (harmonic legitimacy and Basic Pitch
    # confidence did NOT). The AND is the precision guard. Written as an
    # Annotation; Scribe Engraver drops UNSUPPORTED notes only under
    # drop_purge_candidates (§2.2).
    support_by_note_id: Dict[str, "object"] = {}
    for stem, report in anechoic_reports.items():
        stem_track = stem_tracks_by_type.get(stem)
        if stem_track is None:
            continue
        stem_notes = _notes_by_stem_for_rhythm.get(stem, [])
        if not stem_notes:
            continue
        support_samples = engine.view(stem_track, NOTE_SUPPORT_SAMPLE_RATE).samples
        active_by_id = {n.id: report.query(n.start_ms, n.end_ms).active_material_probability for n in stem_notes}
        support_by_note_id.update(
            evaluate_stem_support(stem_notes, support_samples, NOTE_SUPPORT_SAMPLE_RATE, active_by_id)
        )
    unsupported_count = 0

    purge_candidate_count = 0
    harmonic_illegitimate_count = 0
    for note in pitched_notes:
        quantized_start, quantized_end, reason = lattice_judge.quantize_note(note)
        annotations.add(Annotation(
            note_id=note.id, kind=QUANTIZATION_ANNOTATION_KIND,
            value={"start_ms": quantized_start, "end_ms": quantized_end, "reason": reason},
            source=Provenance.TEMPORAL_LATTICE,
            confidence=lattice_judge.quantize_strength,
        ))

        duration_testimony = infer_duration(note, beat_ms)
        primary = duration_testimony.primary
        annotations.add(Annotation(
            note_id=note.id, kind=DURATION_ANNOTATION_KIND,
            value=primary.symbolic_value, source=Provenance.TEMPORAL_LATTICE,
            confidence=primary.probability, contested=duration_testimony.has_contention,
        ))

        # Notation timing (the reading-optimized THIRD timeline), exported
        # only when use_notation_timing is set. Prefer the beat-level
        # Bayesian rhythm inference (sequence-aware: recovers tuplets,
        # notates swing straight); fall back to per-note hard-snap +
        # clean symbolic duration for beats it couldn't confidently parse.
        # Either way it's an Annotation - the Note is never touched (§2.2).
        beat_timing = inferred_rhythm.get(note.id)
        if beat_timing is not None:
            annotations.add(Annotation(
                note_id=note.id, kind=NOTATION_TIMING_ANNOTATION_KIND,
                value={"start_ms": beat_timing.notation_start_ms, "end_ms": beat_timing.notation_end_ms,
                       "reason": beat_timing.reason},
                source=Provenance.TEMPORAL_LATTICE,
            ))
            rhythm_inferred_count += 1
        else:
            notation = notation_quantize_note(note, tempo_resolution.tempo_meter, duration_testimony)
            if notation is not None:
                annotations.add(Annotation(
                    note_id=note.id, kind=NOTATION_TIMING_ANNOTATION_KIND,
                    value={"start_ms": notation.notation_start_ms, "end_ms": notation.notation_end_ms,
                           "symbolic_duration": notation.symbolic_duration, "reason": notation.reason},
                    source=Provenance.TEMPORAL_LATTICE,
                ))

        # AnechoicMa (ported): frame-level silence/resonance/activity
        # evidence for this note's own time window - a real, acoustically-
        # grounded complement to purely symbolic evidence (duration,
        # confidence, centroid). Recorded as evidence; nothing acts on it
        # yet ("don't build ahead of need" - no consumer needs to yet).
        report = anechoic_reports.get(note.stem)
        if report is not None:
            state = report.query(note.start_ms, note.end_ms)
            annotations.add(Annotation(
                note_id=note.id, kind=ACOUSTIC_ACTIVITY_ANNOTATION_KIND,
                value={
                    "rhythmic_void_probability": state.rhythmic_void_probability,
                    "resonance_probability": state.resonance_probability,
                    "active_material_probability": state.active_material_probability,
                    "confidence_penalty": state.get_confidence_penalty(),
                },
                source=Provenance.ANECHOIC_MA,
            ))

        # SchoenbergMirror (ported): harmonic-legitimacy audit against
        # this note's own stem audio - TONAL/PERCUSSION/UNCERTAIN/NOISE/
        # HALLUCINATION, corroborated by two independent literal-
        # transformation witnesses (retrograde symmetry, spectral
        # inversion). Written as a verdict Annotation, never mutating
        # the Note (§2.2) - Scribe Engraver may optionally drop NOISE/
        # HALLUCINATION verdicts under drop_purge_candidates.
        harmonic_match_ratio = None
        stem_track = stem_tracks_by_type.get(note.stem)
        if stem_track is not None:
            audit = audit_note(engine, stem_track, note, note.stem)
            annotations.add(Annotation(
                note_id=note.id, kind=HARMONIC_LEGITIMACY_ANNOTATION_KIND,
                value={"verdict": audit.verdict.value, "reason": audit.reason},
                source=Provenance.SCHOENBERG_MIRROR,
                confidence=audit.confidence, contested=audit.contested,
            ))
            if audit.verdict in (HarmonicVerdict.NOISE, HarmonicVerdict.HALLUCINATION):
                harmonic_illegitimate_count += 1
            if audit.partial_count > 0:
                harmonic_match_ratio = audit.harmonic_match_confidence

        # Key Intelligence (ported): does this note's pitch fit the
        # resolved song key? Written as a key_fit Annotation - never
        # applied to the note's confidence directly (unlike Symphony's
        # apply_key_context, which mutated it), matching this session's
        # annotation-not-mutation law.
        is_in_key, weight = key_fit(note.pitch, key_result)
        annotations.add(Annotation(
            note_id=note.id, kind=KEY_FIT_ANNOTATION_KIND,
            value={"key": key_result.key, "is_in_key": is_in_key, "weight": weight},
            source=Provenance.KEY_INTELLIGENCE, confidence=key_result.confidence,
        ))

        # MicroNotePurge (Ritornello Pass 3, ported): judges whether a very
        # short, low-confidence note is a detection artifact - written as
        # a verdict Annotation, never deleting the Note itself (§2.2). Now
        # fed SchoenbergMirror's real harmonic-match evidence when a
        # harmonic series was actually detected for this note, closing the
        # gap micro_note_purge.py's own docstring flagged; falls back to
        # the stricter no-evidence floor when Schoenberg found no series.
        verdict = evaluate_legitimacy(note, harmonic_match_ratio)
        annotations.add(Annotation(
            note_id=note.id, kind=MICRO_NOTE_PURGE_ANNOTATION_KIND,
            value={"verdict": verdict.verdict, "reason": verdict.reason},
            source=Provenance.RITORNELLO,
        ))
        if verdict.verdict == PURGE_CANDIDATE:
            purge_candidate_count += 1

        # Note-support verdict (from the pre-pass above): the data-driven
        # over-detection filter. Written as an Annotation Scribe Engraver
        # optionally acts on; never touches the Note.
        support = support_by_note_id.get(note.id)
        if support is not None:
            annotations.add(Annotation(
                note_id=note.id, kind=NOTE_SUPPORT_ANNOTATION_KIND,
                value={"verdict": support.verdict, "reason": support.reason},
                source=Provenance.ANECHOIC_MA,
            ))
            if support.verdict == UNSUPPORTED:
                unsupported_count += 1

    music_box.log_decision(
        stage_name="quantization", decision_type="micro_note_purge",
        before_state={}, after_state={"purge_candidates": purge_candidate_count},
        reasoning=f"MicroNotePurge flagged {purge_candidate_count} of {len(pitched_notes)} "
                  f"pitched notes as purge candidates (verdict only - notes are not deleted)",
        reversible=False,
    )
    music_box.log_decision(
        stage_name="quantization", decision_type="rhythm_inference",
        before_state={}, after_state={"beat_inferred": rhythm_inferred_count,
                                       "per_note_fallback": len(pitched_notes) - rhythm_inferred_count},
        reasoning=f"Beat-level Bayesian rhythm inference resolved {rhythm_inferred_count} of "
                  f"{len(pitched_notes)} pitched notes' notation timing; the rest fell back to "
                  f"per-note snapping (notation timeline only - raw/groove unaffected)",
        reversible=False,
    )
    music_box.log_decision(
        stage_name="acoustic_witness", decision_type="note_support",
        before_state={}, after_state={"unsupported": unsupported_count},
        reasoning=f"Note-support filter flagged {unsupported_count} of {len(pitched_notes)} pitched "
                  f"notes UNSUPPORTED (weak own-f0 energy AND acoustically inactive - the two "
                  f"signals a 3-song diagnostic showed actually discriminate over-detection; "
                  f"verdict only, dropped at export only under drop_purge_candidates)",
        reversible=False,
    )
    music_box.log_decision(
        stage_name="acoustic_witness", decision_type="schoenberg_mirror",
        before_state={}, after_state={"illegitimate_verdicts": harmonic_illegitimate_count},
        reasoning=f"SchoenbergMirror flagged {harmonic_illegitimate_count} of {len(pitched_notes)} "
                  f"pitched notes as NOISE/HALLUCINATION (verdict only - notes are not deleted)",
        reversible=False,
    )

    # TieReconstruction (Ritornello Pass 4, ported): same-pitch notes
    # whose gap straddles a beat line, scoped per stem so unrelated
    # instruments' notes never look adjacent - written as a tie_candidate
    # Annotation, never merging the two Notes (§2.2).
    notes_by_stem: Dict[StemType, List[Note]] = defaultdict(list)
    for note in pitched_notes:
        notes_by_stem[note.stem].append(note)

    tie_candidate_count = 0
    for stem_notes in notes_by_stem.values():
        for tie in find_tie_candidates(stem_notes, tempo_resolution.tempo_meter.beat_times_ms):
            annotations.add(Annotation(
                note_id=tie.note_id, kind=TIE_RECONSTRUCTION_ANNOTATION_KIND,
                value={"tied_to_note_id": tie.tied_to_note_id, "boundary_ms": tie.boundary_ms, "reason": tie.reason},
                source=Provenance.RITORNELLO,
            ))
            tie_candidate_count += 1

    music_box.log_decision(
        stage_name="quantization", decision_type="tie_reconstruction",
        before_state={}, after_state={"tie_candidates": tie_candidate_count},
        reasoning=f"TieReconstruction flagged {tie_candidate_count} same-pitch note pairs "
                  f"straddling a beat line as tie candidates (verdict only - notes are not merged)",
        reversible=False,
    )

    # TroubleMap (Ritornello Pass 7, ported): pure diagnosis of the whole
    # ensemble at once (unlike TieReconstruction, which stays within one
    # voice) - flags measures that are low-confidence and/or fragmented.
    # Read-only reporting on MusicalFindingsMap; nothing acts on it.
    findings.trouble_measures = build_trouble_map(pitched_notes, tempo_resolution.tempo_meter)
    music_box.log_decision(
        stage_name="quantization", decision_type="trouble_map",
        before_state={}, after_state={"flagged_measures": len(findings.trouble_measures)},
        reasoning=f"TroubleMap flagged {len(findings.trouble_measures)} measure(s) as "
                  f"low-confidence and/or fragmented (diagnosis only)",
        reversible=False,
    )

    # beat_times_ms is the anchor witness's TRACKED beat grid - the plug
    # that used to be left disconnected: the Rhythm Engine computed real
    # beat positions and the engraver only ever received one scalar
    # tempo, so exported notes floated at absolute times against a flat
    # grid and notation hosts re-derived their own. Passing the grid
    # binds the file's tempo map to where the beats actually fell.
    midi = engrave(all_notes, annotations, output_midi_path, music_box=music_box,
                   use_quantized_timing=use_quantized_timing, use_notation_timing=use_notation_timing,
                   drop_purge_candidates=drop_purge_candidates,
                   tempo_bpm=tempo_resolution.tempo_meter.tempo_bpm,
                   time_signature=(meter_resolution.numerator, meter_resolution.denominator),
                   key=findings.key,
                   beat_times_ms=tempo_resolution.tempo_meter.beat_times_ms)

    elapsed = time.time() - start_time
    music_box.log_decision(
        stage_name="conductor", decision_type="session_end",
        before_state={}, after_state={"total_notes": len(all_notes), "elapsed_seconds": elapsed},
        reasoning=f"Pipeline complete in {elapsed:.1f}s", reversible=False,
    )

    return PipelineResult(
        audio_path=audio_path,
        output_midi_path=output_midi_path,
        separation_model=separation.model_used,
        separation_seed=separation_seed,
        note_counts_by_stem=note_counts,
        crepe_bass_note_count=crepe_bass_count,
        tempo_bpm=tempo_resolution.tempo_meter.tempo_bpm,
        tempo_confidence=tempo_resolution.tempo_meter.confidence,
        time_signature=(meter_resolution.numerator, meter_resolution.denominator),
        tempo_contention=tempo_resolution.contention,
        meter_contention=meter_resolution.contention,
        total_notes_exported=len(all_notes),
        elapsed_seconds=elapsed,
        findings=findings,
    )


__all__ = ["transcribe_file", "PipelineResult", "DEFAULT_SEPARATION_MODEL"]
