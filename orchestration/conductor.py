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
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple, Union

from audio_engine import AudioEngine, AudioTrack
from core import Annotation, AnnotationStore, MusicBox, MusicalFindingsMap, Note, Provenance, Separation, StemType
from core import TempoMeter
from separation_engine import separate
from separation_engine import merge_harmonic_stems as _merge_harmonic_stems
from separation_engine import load_cached_separation as _load_cached_separation
from separation_engine import (detect_solo_instrument, SOLO_PIANO, SOLO_GUITAR,
                               ENSEMBLE)
from pitch_engine import (
    transcribe_basic_pitch, transcribe_basic_pitch_with_posteriorgram,
    transcribe_crepe_bass, BASIC_PITCH_SAMPLE_RATE, continuous_f0, make_f0_sampler,
)
from instrument_attribution import resolve_instrument_identity, check_range, RANGE_ANNOTATION_KIND
from key_intelligence import (analyze_key, analyze_key_from_notes, analyze_key_stability,
                              key_fit, KEY_FIT_ANNOTATION_KIND, KeyResult)
from rhythm_engine import (
    estimate_time_signature_with_phase, classify_drift,
    run_librosa_tempo, run_madmom_tempo, run_note_onset_tempo_witness, run_lattice_witness,
    run_pulse_field, estimate_groove, compute_phase_deltas, detect_drums,
    build_phase_locked_grid, estimate_time_signature, sample_beat_accents,
    fft_meter_candidate, resolve_denominator, detect_onset_candidates,
    run_madmom_downbeat,
)
from core.musical_time import MusicalTime
from epistemic import resolve_tempo, resolve_meter, arbitrate_tempo_octave, TempoResolution, MeterResolution
from quantization import (
    build_lattice, infer_duration, QUANTIZATION_ANNOTATION_KIND,
    propose_sustain_extensions, SUSTAIN_RECOVERY_ANNOTATION_KIND,
    refine_note_onsets, ONSET_REFINEMENT_ANNOTATION_KIND,
    find_wobble_groups, PITCH_WOBBLE_ANNOTATION_KIND,
    evaluate_legitimacy, MICRO_NOTE_PURGE_ANNOTATION_KIND, PURGE_CANDIDATE,
    find_tie_candidates, TIE_RECONSTRUCTION_ANNOTATION_KIND,
    build_trouble_map, notation_quantize_note, NOTATION_TIMING_ANNOTATION_KIND,
    infer_voice_rhythm,
    consolidate_fragments, CONSOLIDATION_ANNOTATION_KIND,
)
from acoustic_witness import (
    analyze_stem, AnechoicReport, ACOUSTIC_ACTIVITY_ANNOTATION_KIND,
    audit_note, HarmonicVerdict, HARMONIC_LEGITIMACY_ANNOTATION_KIND,
    evaluate_stem_support, NOTE_SUPPORT_ANNOTATION_KIND, NOTE_SUPPORT_SAMPLE_RATE, UNSUPPORTED,
    write_octave_annotations,
)
from check import run_check
from university import UniversityMode
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
# METER VOTE: how confident madmom's downbeat tracker must be before its
# reading is allowed into the vote at all.
#
# It is the only meter witness reporting on an absolute-ish scale - the other
# two report a chance-corrected salience and a relative spectral power - and
# resolve_meter SUMS them, so whichever scale runs hottest decides. MEASURED
# across the library (tools/meter_harness.py), it answers 6 on four songs of
# five whatever the truth is, and its mode-share term is 1.00 everywhere, so
# that half of its confidence carries no information about correctness.
#
# The spacing-CV half DOES: where the witness is right it is confident (No
# Pasaran 0.920, HRV 0.877, Hopeful 0.533) and on the one song it is badly
# wrong it is not (Chopin 0.384, against a published edition reading 4/4).
# A floor lets two agreeing calibrated witnesses carry a beat the trained one
# cannot see, without touching the songs it was added to fix. Same shape as
# OCTAVE_ARBITER_MIN_CONFIDENCE, and for the same reason.
#
# HONEST ABOUT THE NUMBER: 0.50 is a round value in the single gap between
# 0.384 and 0.533 in a four-song sample. The RULE is principled; the threshold
# is calibrated on very little and should be revisited as the library grows.
# Scored under §XVII.3's metrical-equivalence rule (6/4 is two 3/4 bars), this
# takes meter from 3/4 songs correct to 4/4.
METER_DOWNBEAT_MIN_CONFIDENCE = 0.50

DEFAULT_SEPARATION_MODEL = "htdemucs_6s"
DURATION_ANNOTATION_KIND = "duration_hypothesis"

_PITCHED_STEMS = (StemType.BASS, StemType.VOCALS, StemType.OTHER, StemType.GUITAR,
                  StemType.PIANO, StemType.BRASS)


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
    # Both are annotation-only passes: they change what the engraver WRITES,
    # never how many notes exist. Surfaced here so a run says out loud how
    # much they actually did - a pass that silently does nothing is the
    # failure mode these counters exist to make impossible.
    onset_refined_count: int = 0
    wobble_group_count: int = 0
    elapsed_seconds: float = 0.0
    findings: Optional[MusicalFindingsMap] = None


def transcribe_file(
        audio_path: Union[str, Path],
        output_midi_path: Union[str, Path],
        separation_model: str = DEFAULT_SEPARATION_MODEL,
        device: str = "cpu",
        use_quantized_timing: bool = False,
        use_notation_timing: bool = False,
        use_consolidated_timing: bool = False,
        output_musicxml_path: Optional[Union[str, Path]] = None,
        drop_purge_candidates: bool = False,
        merge_harmonic_stems: bool = True,
        separation_seed: Optional[int] = 0,
        guided_tempo_bpm: Optional[float] = None,
        guided_time_signature: Optional[Tuple[int, int]] = None,
        guided_key: Optional[str] = None,
        # GUIDED SEPARATION (§2.7, same law as guided tempo/meter/key): the
        # user's knowledge of what is playing is a hard lock, not a vote.
        #   None           - auto: the solo probe decides
        #   "solo_piano"   - skip Demucs, the master audio IS the piano stem
        #   "solo_guitar"  - skip Demucs, the master audio IS the guitar stem
        #   "ensemble"     - separate, never skip, whatever the probe thinks
        # The vocabulary is deliberately the SAME words the probe returns, so
        # a user overriding it is speaking the machine's own language rather
        # than a parallel set of flags.
        guided_separation: Optional[str] = None,
        # BASIC PITCH'S OWN DIALS, which the conductor has never passed - every
        # stem has run on library defaults since the beginning. Swept against
        # the Chopin answer key (tools/, 2026-08-23), 30 configurations:
        #
        #   baseline  onset 0.50 frame 0.30 minlen 58   3204 notes  F1 0.609
        #   tuned     onset 0.60 frame 0.45 minlen 107  2045 notes  F1 0.679
        #
        # +0.070 F1, precision 0.517 -> 0.710, and the short-note share falls
        # from 32% to 25%. The surface is FLAT across frame 0.45-0.50 and onset
        # 0.60-0.65 - the top seven configurations sit within 0.010 - so these
        # are a robust region rather than a fitted maximum.
        #
        # BUT IT DOES NOT GENERALISE. Re-swept on Burden of Sentiment against
        # Klangio, every configuration lands within 0.004 of baseline: note
        # count falls 44%, precision rises, recall falls, and they cancel.
        # Chopin is solo piano under heavy pedal, which is exactly the case a
        # stricter frame threshold should help. So these stay None (library
        # defaults) and are set per-material by the caller - see the solo
        # fast path, which already knows when it is looking at a piano.
        detection_onset_threshold: Optional[float] = None,
        detection_frame_threshold: Optional[float] = None,
        detection_min_note_ms: Optional[float] = None,
        routed_layout: bool = True,
        save_intermediate_path: Optional[Union[str, Path]] = None,
        university_mode: Union[str, "UniversityMode"] = "off",
        stem_cache_dir: Optional[Union[str, Path]] = None,
        # The solo fast path (see the call site). On by default because the
        # gate measured zero false "solo" verdicts across every recording in
        # the corpus, and because the failure it prevents is worse than the
        # one it risks - but it is a switch, so a caller who wants Demucs on
        # a solo record can still have it.
        skip_separation_when_solo: bool = True,
        # A MusicBox to log into, instead of the private one built below.
        # Exists so a front end can hand in `MusicBox(log_path=..., buffer_size=1)`
        # and TAIL the ledger while the run proceeds: every decision lands on
        # disk the moment it is made, so live progress is the engine's own
        # audit trail rather than a progress bar invented alongside it.
        music_box: Optional[MusicBox] = None,
        university_corpus_path: Optional[Union[str, Path]] = None,
) -> PipelineResult:
    start_time = time.time()
    audio_path = str(audio_path)
    output_midi_path = str(output_midi_path)

    engine = AudioEngine()
    music_box = music_box if music_box is not None else MusicBox()
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
    # Cached stems (opt-in): skip Demucs when this song has already been
    # separated by tools/build_stem_cache.py with the SAME model and seed.
    # Demucs is the largest model in a run, and re-separating audio we have
    # already separated buys nothing - it also makes repeat runs bit-identical
    # in their separation, removing a source of variation when comparing
    # notation changes. Falls back to real separation if the cache is absent or
    # unreadable, so this can never silently produce a different pipeline.
    separation: Optional[Separation] = None
    if stem_cache_dir is not None:
        separation = _load_cached_separation(stem_cache_dir)
        if separation is not None:
            music_box.log_decision(
                stage_name="separation_engine", decision_type="separation_cache_hit",
                before_state={"cache_dir": str(stem_cache_dir)},
                after_state={"stems": [s.value for s in separation.stems],
                             "model": separation.model_used},
                reasoning="Loaded pre-separated stems from the cache instead of running "
                          "Demucs. Same model+seed as a live run, so the stems are "
                          "identical; skips the largest model in the pipeline.",
                reversible=False,
            )
    # SOLO FAST PATH (separation_engine/solo_detector.py). Demucs is the
    # largest model in a run, and on a solo recording it is not just wasted
    # but harmful: separating the solo piano Nocturne into six stems produced
    # 1546 "drum", 463 "bass" and 200 "vocal" notes on a recording with no
    # drummer, bassist or singer, and passing the same audio through as ONE
    # harmonic stem moved F1 from 0.507 to 0.603. So the cheap physical test
    # runs first, and a confident solo verdict skips separation entirely.
    #
    # ONLY a confident solo verdict. The asymmetry is the whole design: a
    # wrong "solo" silently discards real instruments and nothing downstream
    # could notice, while a wrong "ensemble" costs only time.
    # GUIDED SEPARATION is a hard lock and, like guided tempo, it SKIPS the
    # detector rather than out-voting it - there is no point paying for a
    # verdict that cannot change the outcome.
    if separation is None and guided_separation in (SOLO_PIANO, SOLO_GUITAR):
        stem = StemType.PIANO if guided_separation == SOLO_PIANO else StemType.GUITAR
        separation = Separation(
            stems={stem: master_track},
            model_used=f"guided({guided_separation})",
            confidence=1.0, separation_time_seconds=0.0,
        )
        music_box.log_decision(
            stage_name="separation_engine", decision_type="separation_guided",
            before_state={"model": separation_model},
            after_state={"stem": stem.value},
            reasoning=(f"Separation HARD-LOCKED by the user to {guided_separation} "
                       f"(guided mode, §2.7). Demucs skipped and the solo probe "
                       f"NOT run - a verdict that cannot change the outcome is "
                       f"not worth computing. The master audio is the "
                       f"{stem.value} stem."),
            reversible=False,
        )
    elif separation is None and guided_separation == ENSEMBLE:
        music_box.log_decision(
            stage_name="separation_engine", decision_type="separation_guided",
            before_state={}, after_state={"forced": "separate"},
            reasoning=("Separation HARD-LOCKED by the user to ensemble (guided "
                       "mode, §2.7): Demucs runs and the solo fast path is not "
                       "consulted, however the probe would have read the audio."),
            reversible=False,
        )

    if separation is None and guided_separation is None and skip_separation_when_solo:
        solo = detect_solo_instrument(engine, master_track)
        music_box.log_decision(
            stage_name="separation_engine", decision_type="solo_probe",
            before_state={}, after_state={
                "verdict": solo.verdict, "confidence": round(solo.confidence, 3),
                "instrument_confidence": round(solo.instrument_confidence, 3),
                "kit_hf_fraction": round(solo.kit_hf_fraction, 4),
                "crescendo_fraction": round(solo.crescendo_fraction, 4),
                "low_energy_fraction": round(solo.low_energy_fraction, 4)},
            reasoning=solo.reason, reversible=False,
        )
        if solo.is_solo:
            stem = (StemType.PIANO if solo.verdict == SOLO_PIANO
                    else StemType.GUITAR)
            separation = Separation(
                stems={stem: master_track},
                model_used=f"solo_fast_path({solo.verdict})",
                confidence=solo.confidence, separation_time_seconds=0.0,
            )
            music_box.log_decision(
                stage_name="separation_engine", decision_type="separation_skipped",
                before_state={"model": separation_model},
                after_state={"stem": stem.value},
                reasoning=(f"Skipped Demucs: {solo.reason}. The master audio IS "
                           f"the {stem.value} stem, so no drums/bass/vocals stem "
                           f"exists for the detector to hallucinate into."),
                reversible=False,
            )

    if separation is None:
        separation = separate(
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
    if guided_key is not None:
        # §2.7 "guided means guided": the user's key is a hard lock at
        # confidence 1.0. analyze_key is skipped entirely, not blended -
        # a guided value the witnesses could still outvote is not guided.
        key_result = KeyResult(key=guided_key, confidence=1.0)
        findings.key = key_result.key
        findings.key_confidence = key_result.confidence
        music_box.log_decision(
            stage_name="key_intelligence", decision_type="key_guided",
            before_state={}, after_state={"key": key_result.key, "guided": True},
            reasoning=f"Key HARD-LOCKED to user-supplied {key_result.key} (guided mode; "
                      f"detection skipped, §2.7).", reversible=False,
        )
    else:
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
    onset_refined_count = 0
    stem_tracks_by_type: Dict[StemType, AudioTrack] = {}
    anechoic_reports: Dict[StemType, AnechoicReport] = {}
    stem_onsets_by_type: Dict[StemType, List[float]] = {}
    for stem in _PITCHED_STEMS:
        if not separation.has_stem(stem):
            continue
        stem_track = separation.get_stem(stem)
        stem_tracks_by_type[stem] = stem_track
        _detect_kwargs = {}
        if detection_onset_threshold is not None:
            _detect_kwargs["onset_threshold"] = detection_onset_threshold
        if detection_frame_threshold is not None:
            _detect_kwargs["frame_threshold"] = detection_frame_threshold
        if detection_min_note_ms is not None:
            _detect_kwargs["min_note_duration_ms"] = detection_min_note_ms
        stem_notes, posteriorgram = transcribe_basic_pitch_with_posteriorgram(
            engine, stem_track, stem, **_detect_kwargs)

        # AnechoicMa (ported): frame-level silence/resonance/activity
        # evidence for this stem's own audio - queried per note below.
        anechoic_reports[stem] = analyze_stem(engine, stem_track, stem)

        # SustainRecovery (Ritornello Pass 2, ported): extend a note's END
        # when it's still acoustically ringing past Basic Pitch's own
        # nominal end - never touches pitch/start, never mutates the Note.
        # Written as an Annotation; Scribe Engraver decides whether to
        # read it (opt-in via use_quantized_timing, same law as
        # quantization's snapped timing).
        # OnsetRefinement: correct WHERE a note began, using this stem's own
        # transients. Basic Pitch fires on an onset-probability threshold
        # crossing, which lands later the more slowly an instrument speaks -
        # so vocals/guitar/piano drift late while drums (a real transient
        # detector) and bass (two witnesses, fast attack) land right. The
        # dual-witness machinery for this already existed in rhythm_engine
        # and was wired to nothing. Annotation-only; Note.start_ms stands.
        # OWN-STEM ONSETS for the consolidation attack-gate (see
        # note_consolidation._has_attack). Consulting the MASTER MIX's onsets
        # regressed dense material badly - every drum hit voted on whether a
        # held vocal note had been restruck, so the gate opened constantly and
        # You Say God Says came back at 33.5% sub-32nd notes, the exact defect
        # consolidation exists to prevent. A piano re-articulation has to be
        # evidenced by a PIANO attack.
        stem_onsets_by_type[stem] = sorted(
            detect_onset_candidates(engine, stem_track).combined_ms)

        onset_refinements = refine_note_onsets(engine, stem_track, stem_notes)
        for ref in onset_refinements:
            annotations.add(Annotation(
                note_id=ref.note_id, kind=ONSET_REFINEMENT_ANNOTATION_KIND,
                value={"start_ms": ref.refined_start_ms,
                       "shift_ms": ref.shift_ms,
                       "disagreement": ref.disagreement},
                source=Provenance.RITORNELLO, confidence=ref.confidence,
            ))
        onset_refined_count += len(onset_refinements)
        if onset_refinements:
            shifts = [r.shift_ms for r in onset_refinements]
            music_box.log_decision(
                stage_name="quantization", decision_type="onsets_refined",
                before_state={"stem": stem.value, "notes": len(stem_notes)},
                after_state={"refined": len(onset_refinements),
                             "median_shift_ms": round(float(median(shifts)), 1),
                             "earlier": sum(1 for s in shifts if s < 0),
                             "later": sum(1 for s in shifts if s > 0)},
                reasoning="Resolved Basic Pitch's onsets against this stem's own "
                          "librosa-backtracked + madmom transients. Negative shift = "
                          "the attack was earlier than the threshold crossing.",
                reversible=True,
            )

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

    # Key, re-read from OUR OWN NOTES (§XVII.2). The audio path above reads
    # chroma off the FULL MIX, where drums/percussion smear the pitch-class
    # profile; the transcribed notes are 90-99% diatonic - a much cleaner
    # signal we already compute. Two witnesses, one place decides: the
    # notes-based reading wins when it is at least as confident, and both are
    # logged. Guided key stays a hard lock (§2.7) and is never arbitrated.
    if guided_key is None and pitched_notes:
        notes_key = analyze_key_from_notes(pitched_notes)
        audio_key = key_result
        if notes_key.confidence >= audio_key.confidence and notes_key.key != audio_key.key:
            key_result = notes_key
            findings.key = notes_key.key
            findings.key_confidence = notes_key.confidence
        # STABILITY (key_intelligence/key_stability.py). A key label is worth
        # what the span supports, and until now nothing measured that. Chopin
        # returned 0.84 for D# minor on a 180-second EXCERPT of a B major
        # piece - and that reading is correct for the excerpt, because the
        # published edition reads D# minor over the same span. What was wrong
        # was the certainty. The reported key is never overridden here; only
        # its confidence is damped, and the windows are logged so a modulation
        # can be pointed at instead of averaged away.
        stability = analyze_key_stability(pitched_notes)
        findings.key_confidence = stability.confidence
        music_box.log_decision(
            stage_name="key_intelligence", decision_type="key_stability",
            before_state={"key": stability.key,
                          "raw_confidence": round(stability.raw_confidence, 3)},
            after_state={"confidence": round(stability.confidence, 3),
                         "agreement": round(stability.agreement, 3),
                         "modulates": stability.modulates,
                         "windows": [k for _s, _e, k in stability.windows],
                         "cadence_key": stability.cadence_key,
                         "cadence_agrees": stability.cadence_agrees},
            reasoning=stability.notes,
            reversible=False,
        )
        music_box.log_decision(
            stage_name="key_intelligence", decision_type="key_from_notes",
            before_state={"audio_key": audio_key.key,
                          "audio_confidence": round(audio_key.confidence, 3)},
            after_state={"notes_key": notes_key.key,
                         "notes_confidence": round(notes_key.confidence, 3),
                         "resolved": key_result.key},
            reasoning="Key re-read from the transcribed notes rather than mix chroma "
                      "(§XVII.2: mix chroma is smeared by percussion). The more "
                      "confident witness wins; both are recorded.",
            reversible=True,
        )

    # Pitch-range plausibility (§XVII.6): flag notes outside their own stem's
    # physical range - a bass note above G4, a "vocal" below C2. Measured
    # against Klangio on identical audio, which held a tight, realistic bass
    # range while ours ran to midi 77. Annotation-only: nothing is dropped or
    # re-pitched, the verdict simply exists for the engraver or a later
    # reducer to consult.
    range_verdicts = check_range(all_notes)
    for rv in range_verdicts:
        annotations.add(Annotation(
            note_id=rv.note_id, kind=RANGE_ANNOTATION_KIND,
            value={"verdict": rv.verdict, "reason": rv.reason,
                   "stem": rv.stem, "pitch": rv.pitch},
            source=Provenance.TIMBRE_INTELLIGENCE,
        ))
    if range_verdicts:
        by_stem: Dict[str, int] = {}
        for rv in range_verdicts:
            by_stem[rv.stem] = by_stem.get(rv.stem, 0) + 1
        music_box.log_decision(
            stage_name="instrument_attribution", decision_type="range_plausibility",
            before_state={"notes": len(all_notes)},
            after_state={"implausible": len(range_verdicts), "by_stem": by_stem},
            reasoning="Notes outside their own stem's physical pitch range - octave/"
                      "overtone artifacts or cross-stem bleed. Verdict only; nothing "
                      "is dropped.",
            reversible=True,
        )

    music_box.log_decision(
        stage_name="quantization", decision_type="sustain_recovery",
        before_state={}, after_state={"notes_extended": sustain_extended_count},
        reasoning=f"SustainRecovery proposed an extension for {sustain_extended_count} of "
                  f"{len(pitched_notes)} pitched notes (posteriorgram + narrowband-energy witnesses)",
        reversible=False,
    )

    # Onsets feed both the guided grid (phase-lock at the user's tempo) and
    # the detected-meter path below; computed once, and cheap - this is
    # onset detection, NOT tempo estimation.
    onset_candidates = detect_onset_candidates(engine, master_track)
    duration_ms = master_track.duration_seconds * 1000.0

    # Which tracked beat carries beat 1. Stays 0 unless the meter estimator
    # finds significant evidence otherwise (guided mode never estimates it).
    onset_downbeat_phase = 0

    if guided_tempo_bpm is not None:
        # §2.7 "guided means guided": guided mode does NOT run the tempo
        # witnesses at all. The user supplied the tempo, so the pipeline
        # spends zero cycles trying to re-derive it - this is "do not
        # compute," not "compute then override." (madmom, librosa, the
        # note-onset witness, and the ReverseGeoCrypt lattice search are all
        # skipped.) The one input the user did NOT give is PHASE - where
        # beat 1 actually falls - so that alone is solved, from onsets. That
        # is a small, well-conditioned problem, and it is not tempo
        # detection: without it there is no way to place a barline.
        tempo_witnesses: List = []
        lattice = None
        ratio_family = None
        downbeat_witness = None
        ts_num, ts_den = guided_time_signature if guided_time_signature is not None else (4, 4)
        guided_grid = build_phase_locked_grid(guided_tempo_bpm, onset_candidates.combined_ms, duration_ms)
        guided_downbeats = tuple(guided_grid[i] for i in range(0, len(guided_grid), ts_num)) \
            if guided_grid else ()
        guided_tm = TempoMeter(
            tempo_bpm=guided_tempo_bpm, confidence=1.0,
            time_signature_numerator=ts_num, time_signature_denominator=ts_den,
            beat_times_ms=tuple(guided_grid), downbeat_times_ms=guided_downbeats,
            source=Provenance.GUIDED, is_guided=True,
        )
        tempo_resolution = TempoResolution(tempo_meter=guided_tm, contention=None)
        music_box.log_decision(
            stage_name="rhythm_engine", decision_type="tempo_guided",
            before_state={"witnesses_run": 0},
            after_state={"tempo_bpm": guided_tempo_bpm, "beats": len(guided_grid),
                         "phase_ms": round(guided_grid[0], 1) if guided_grid else 0.0},
            reasoning=f"Tempo HARD-LOCKED to user-supplied {guided_tempo_bpm}bpm (guided mode, "
                      f"§2.7). Tempo witnesses SKIPPED ENTIRELY - not computed, not arbitrated; "
                      f"only beat PHASE was solved against onsets.",
            reversible=False,
        )
    else:
        # Rhythm Engine: FOUR independent tempo witnesses from the full
        # original track + the pitched notes just detected. Each audio-only
        # witness is already octave-corrected internally (tempo_witness.py).
        librosa_witness = run_librosa_tempo(engine, master_track)
        madmom_witness = run_madmom_tempo(engine, master_track)
        note_onset_witness = run_note_onset_tempo_witness(pitched_notes)

        # NOTE on Anchor (rhythm_engine.lattice_witness): kick attacks are the
        # natural anchor set, and the lattice's _anchor_alignment machinery is
        # ready for them - but feeding them here was MEASURED to be a net
        # regression and deliberately NOT done. Anchoring made the lattice a
        # more confident witness, which let it survive the referee's #270
        # exclusion filter and drag the weighted tempo UP toward the note-onset
        # witness's known ~19%-high reading (#271): Hopeful 148->153.8,
        # No Pasaran 132->136.4, both wrong, on 2026-07-20. Anchors belong in a
        # place that consumes the lattice DIRECTLY, not the tempo vote. Dormant.
        lattice_witness, lattice = run_lattice_witness(engine, master_track)

        tempo_witnesses = [librosa_witness, madmom_witness, note_onset_witness]
        if lattice_witness is not None:
            tempo_witnesses.append(lattice_witness)

        # ratio_family (binary/ternary/swing/...) is real per-track evidence
        # the lattice search already computed - it decides simple vs compound
        # for a 6/9/12-beat bar, not the numerator alone (see meter.py).
        ratio_family = lattice.ratio_family if lattice is not None else None

        tempo_resolution = resolve_tempo(tempo_witnesses)

        # Distributed-downbeat witness (open problem #7): madmom's trained
        # RNN+DBN bar tracker. Computed here (once) because it serves BOTH
        # the meter vote below AND the #2 tempo-octave arbiter next.
        downbeat_witness = run_madmom_downbeat(engine, master_track)

        # Tempo-octave arbiter (open problem #2): the per-witness octave
        # correction over-fires its compound-tactus test on a triplet-SHUFFLE
        # feel and halves a normal pulse (No Pasaran quarter=132 -> 66). A
        # CONFIDENT downbeat witness reading a clean 2x the resolved tempo is
        # the evidence that halving was wrong; it does NOT touch a genuinely
        # slow triplet-feel song (Gospel 67), whose witness is only 0.30
        # confident. Guided tempo is a hard lock (§2.7) - never arbitrated.
        if downbeat_witness is not None and len(downbeat_witness.beat_times_ms) >= 2:
            bt = downbeat_witness.beat_times_ms
            witness_iois = [bt[i + 1] - bt[i] for i in range(len(bt) - 1)]
            witness_bpm = 60000.0 / median(witness_iois) if witness_iois else 0.0
            before_bpm = tempo_resolution.tempo_meter.tempo_bpm
            tempo_resolution = arbitrate_tempo_octave(
                tempo_resolution, witness_bpm, bt,
                downbeat_witness.confidence, downbeat_witness.downbeat_times_ms,
            )
            after_bpm = tempo_resolution.tempo_meter.tempo_bpm
            if after_bpm != before_bpm:
                music_box.log_decision(
                    stage_name="epistemic", decision_type="tempo_octave_arbitrated",
                    before_state={"resolved_bpm": round(before_bpm, 1)},
                    after_state={"corrected_bpm": round(after_bpm, 1),
                                 "witness_bpm": round(witness_bpm, 1),
                                 "witness_confidence": round(downbeat_witness.confidence, 2)},
                    reasoning="#2: a confident downbeat witness read ~2x the resolved tempo - "
                              "the compound-tactus correction had octave-halved it. Adopted the "
                              "witness's octave and tracked grid.",
                    reversible=False,
                )

    if guided_time_signature is not None:
        # Meter is the user's, at confidence 1.0 - no estimation, no vote.
        g_num, g_den = guided_time_signature
        meter_candidates = [(g_num, g_den, 1.0)]  # the "candidate" was the lock (for logging below)
        meter_resolution = MeterResolution(numerator=g_num, denominator=g_den,
                                           confidence=1.0, contention=None)
        music_box.log_decision(
            stage_name="rhythm_engine", decision_type="meter_guided",
            before_state={}, after_state={"time_signature": f"{g_num}/{g_den}"},
            reasoning=f"Meter HARD-LOCKED to user-supplied {g_num}/{g_den} (guided mode, §2.7).",
            reversible=False,
        )
    else:
        # THE GRID THE METER TEST IS SAMPLED ON (fixed 2026-08-17, Chopin).
        #
        # meter.py's fix 1 replaced "each witness's own grid" with ONE grid
        # anchored to the resolved tempo, because a witness's own TEMPO ERROR
        # accumulates into phase drift. That reasoning is right about an
        # EXTRAPOLATED grid and wrong about a TRACKED one: the anchor witness's
        # beat_times_ms are tracked per beat by madmom's DBN, so they do not
        # accumulate anything - whereas rebuilding an isochronous grid from a
        # single scalar tempo throws away every bit of rubato the tracker found.
        #
        # MEASURED on Rubinstein's Op.62/1 (beat CV 0.168, local tempo 63-102):
        #   * the isochronous grid sits a median 194ms - and up to 394ms - from
        #     the real beats, with 48% of them more than 200ms away;
        #   * sampled on it, NO meter candidate is significant at all and 4/4
        #     collapses to score 0.0217, so the estimator returns its
        #     (4, 4, 0.3) fallback and gets outvoted;
        #   * sampled on the TRACKED beats the same estimator returns 4/4 at
        #     score 0.1592, significant, phase 0 - which is what the published
        #     edition says.
        # The pipeline resolved 6/4 on a piece in 4/4 purely because the accent
        # test was reading a grid this project already documents as wrong
        # (core/musical_time.py's header measures the same drift).
        #
        # So: sample on the tracked beats when we have them, and fall back to
        # the isochronous reconstruction only when we do not.
        tracked_beats = list(tempo_resolution.tempo_meter.beat_times_ms or ())
        if len(tracked_beats) >= 8:
            meter_grid = tracked_beats
            meter_grid_kind = "tracked beats"
        else:
            meter_grid = list(build_phase_locked_grid(
                tempo_resolution.tempo_meter.tempo_bpm,
                onset_candidates.combined_ms,
                duration_ms,
            ))
            meter_grid_kind = "isochronous phase-locked grid (no tracked beats)"

        # Still needed by the guided-grid path and as a phase reference.
        phase_locked_grid = meter_grid

        meter_candidates = []
        if phase_locked_grid:
            # ...and its DOWNBEAT PHASE, which this estimator has always
            # computed and always discarded (2026-08-17 audit). Without it
            # nothing in the pipeline knows which beat is beat 1, so the
            # notation layers each picked their own barline origin.
            ts_num, ts_den, ts_conf, onset_downbeat_phase = estimate_time_signature_with_phase(
                engine, master_track, phase_locked_grid, ratio_family=ratio_family)
            meter_candidates.append((ts_num, ts_den, ts_conf))
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

        # Distributed-downbeat witness (#7): madmom's trained RNN+DBN bar
        # tracker. Both onset-based estimates above read the downbeat from
        # accent LOUDNESS, which we measured is a weak/flat cue (harmonic
        # change 1.04-1.23x chance, kick-on-1 1.10-1.38x) - blind to a
        # downbeat felt through harmony/bass. The trained model fuses those
        # cues implicitly; it matched the ear on the songs the onset test
        # missed (No Pasaran 4/4 not 65-halved, Hopeful 6/4 not 4/4). Voted,
        # not obeyed: resolve_meter sums it against the two above by its own
        # stability-derived confidence.
        # Reuse the witness already computed above for the #2 arbiter; only
        # run it here if it wasn't (i.e. tempo was guided so that block was skipped).
        if downbeat_witness is None:
            downbeat_witness = run_madmom_downbeat(engine, master_track)
        if (downbeat_witness is not None
                and downbeat_witness.confidence >= METER_DOWNBEAT_MIN_CONFIDENCE):
            meter_candidates.append((
                downbeat_witness.beats_per_bar,
                resolve_denominator(downbeat_witness.beats_per_bar, ratio_family),
                downbeat_witness.confidence,
            ))
            music_box.log_decision(
                stage_name="rhythm_engine", decision_type="downbeat_witness",
                before_state={}, after_state={
                    "beats_per_bar": downbeat_witness.beats_per_bar,
                    "downbeats": len(downbeat_witness.downbeat_times_ms),
                    "confidence": round(downbeat_witness.confidence, 2)},
                reasoning="madmom RNN+DBN downbeat tracker (open problem #7): a trained "
                          "witness for the metrical top level, voted into resolve_meter "
                          "against the onset-loudness estimates it outperforms on harmony-"
                          "felt downbeats.",
                reversible=False,
            )
        elif downbeat_witness is not None:
            music_box.log_decision(
                stage_name="rhythm_engine", decision_type="downbeat_witness_excluded",
                before_state={"confidence": round(downbeat_witness.confidence, 3),
                              "floor": METER_DOWNBEAT_MIN_CONFIDENCE},
                after_state={"beats_per_bar": downbeat_witness.beats_per_bar,
                             "voted": False},
                reasoning="The downbeat tracker is not confident enough to outvote the "
                          "onset-accent and spectral-periodicity witnesses, which report "
                          "on calibrated scales. Its reading is recorded and discarded.",
                reversible=True,
            )

        if not meter_candidates:
            meter_candidates.append((4, 4, 0.3))
        meter_resolution = resolve_meter(meter_candidates)
        music_box.log_decision(
            stage_name="rhythm_engine", decision_type="meter_grid",
            before_state={"grid": meter_grid_kind, "beats": len(meter_grid)},
            after_state={"candidates": meter_candidates,
                         "resolved": f"{meter_resolution.numerator}/{meter_resolution.denominator}"},
            reasoning="Which beat grid the downbeat-accent test was sampled on. An "
                      "isochronous reconstruction of a rubato performance sits ~200ms "
                      "off the real beats and flattens every candidate's salience, so "
                      "the tracked beats are used whenever they exist.",
            reversible=True,
        )

    # THE RESOLVED METER GOES BACK INTO THE TEMPO METER, and everything
    # downstream reads it from there (2026-08-17 audit).
    #
    # resolve_tempo takes `time_signature` as a parameter and the Conductor
    # cannot supply it - meter is resolved forty lines later than tempo. So the
    # TempoMeter carried resolve_tempo's DEFAULT 4/4 for the whole rest of the
    # run, and meter_resolution was used only to stamp the MIDI/MusicXML
    # header. Everything that consumes tempo_meter therefore believed 4/4:
    #   * build_lattice's compound-meter test (6/8, 9/8, 12/8 -> 3 subdivisions
    #     per beat instead of 4) could never fire;
    #   * notation_quantizer._subdivisions_per_beat, same test, same result;
    #   * build_trouble_map bucketed notes into 4-beat bars regardless.
    # Two whole branches written specifically for compound meter were
    # unreachable in detected mode. Guided mode was always correct - it builds
    # its TempoMeter with the user's meter directly - which is exactly why this
    # stayed invisible. One fact, one home (§9).
    #
    # The DOWNBEAT PHASE rides along with it. Three separate places computed a
    # downbeat and none of them reached the page: estimate_time_signature threw
    # its phase away, madmom's tracked downbeats stopped at the octave arbiter,
    # and detect_pickup was never called at all. So the notation layers each
    # invented a barline origin - the MusicXML exporter used the earliest note
    # in the score, piano_reduction used absolute zero - and neither is beat 1.
    # Preference order: madmom's trained tracker (it reads harmony-felt
    # downbeats the accent test is blind to), then the accent phase, then beat 0.
    resolved_beats = tempo_resolution.tempo_meter.beat_times_ms
    resolved_downbeats = tempo_resolution.tempo_meter.downbeat_times_ms
    if not resolved_downbeats and resolved_beats:
        phase = onset_downbeat_phase if onset_downbeat_phase < len(resolved_beats) else 0
        resolved_downbeats = tuple(
            resolved_beats[i] for i in range(phase, len(resolved_beats), meter_resolution.numerator)
        )
    tempo_resolution = TempoResolution(
        tempo_meter=replace(
            tempo_resolution.tempo_meter,
            time_signature_numerator=meter_resolution.numerator,
            time_signature_denominator=meter_resolution.denominator,
            downbeat_times_ms=resolved_downbeats,
        ),
        contention=tempo_resolution.contention,
    )
    music_box.log_decision(
        stage_name="rhythm_engine", decision_type="downbeat_phase",
        before_state={"beats": len(resolved_beats)},
        after_state={"downbeats": len(resolved_downbeats),
                     "phase_beat_index": onset_downbeat_phase,
                     "bar_origin_ms": round(resolved_downbeats[0], 1) if resolved_downbeats else None},
        reasoning="Which tracked beat is beat 1. Previously computed in three "
                  "places and consumed in none, so every barline downstream was "
                  "anchored to an arbitrary origin.",
        reversible=True,
    )

    # Tempo DRIFT (evidence only). MusicalTime's whole thesis is that a
    # performer is not a metronome; classify_drift is the module that says HOW
    # the tempo moves - accelerando, ritardando, a step change, a shift into
    # double time - and it was written, exported, and called from nowhere
    # (2026-08-17 audit). Logged as a finding, not acted on: nothing consumes a
    # drift verdict yet, and the map itself already follows the tracked beats.
    # Conservative by construction (burden of proof is on "the tempo moved").
    drift = classify_drift(engine, master_track, tempo_resolution.tempo_meter.tempo_bpm)
    music_box.log_decision(
        stage_name="rhythm_engine", decision_type="tempo_drift",
        before_state={"reference_bpm": round(tempo_resolution.tempo_meter.tempo_bpm, 1)},
        after_state={"kind": drift.kind.value, "span_bpm": round(drift.span_bpm, 1),
                     "confidence": round(drift.confidence, 2), "detail": drift.detail},
        reasoning=f"Tempo reads {drift.kind.value} over the track "
                  f"({drift.span_bpm:.1f} BPM span, confidence {drift.confidence:.2f}). "
                  f"Evidence only - the notation clock already follows the tracked "
                  f"beats via MusicalTime.",
        reversible=True,
    )

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

    if guided_tempo_bpm is None:
        # (guided mode logs its own tempo_guided decision above; there are
        # no witnesses to report here)
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

    # THE MAP between clock time and musical position (core/musical_time.py).
    # Built ONCE from the tracked beats and handed to EVERY stage that converts
    # between the two, so there is one currency instead of each site dividing
    # by 60000/bpm on its own. Measured: converting only the quantizer moved
    # Chopin recall +9.6% because the exporter's own conversions silently
    # reverted it - a partial conversion is worth nothing.
    #
    # Built HERE, before quantization, rather than after it (2026-08-17 audit):
    # it used to be constructed 250 lines further down, which is why the
    # lattice judge and the rhythm inference - the two stages that most need it
    # - were the two that never got it.
    musical_time = MusicalTime.from_beats(
        tempo_resolution.tempo_meter.beat_times_ms,
        fallback_beat_ms=tempo_resolution.tempo_meter.beat_duration_ms)
    music_box.log_decision(
        stage_name="quantization", decision_type="musical_time",
        before_state={"tempo_bpm": tempo_resolution.tempo_meter.tempo_bpm},
        after_state={"tracked_beats": len(musical_time.beats_ms),
                     "usable": bool(musical_time.usable),
                     "summary_bpm": round(musical_time.summary_bpm(), 2)},
        reasoning="beat-space map for notation timing", reversible=True)

    # Quantization: per-note snap proposal + duration hypothesis, written
    # as Annotations - Note.start_ms/end_ms are never touched (§2's
    # frozen-detection-floor law holds by construction here).
    lattice_judge = build_lattice(
        tempo_resolution.tempo_meter,
        swing_ratio=swing_ratio,
        groove_confidence=groove_confidence,
        lattice_confidence=lattice.confidence if lattice is not None else 0.0,
        phase_delta_ms=phase_delta_result.average_ms,
        musical_time=musical_time,
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
            infer_voice_rhythm(stem_notes, beat_ms, grid_origin_ms, swing_ratio=swing_ratio,
                               musical_time=musical_time)
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
                # tuplet_divisor MUST travel with is_tuplet. Dropping it (as this
                # did until 2026-08-21) silently disabled the entire deliberate
                # tuplet path: the exporter's _apply_tuplet and
                # _leading_tuplet_rest are both gated on `if divisor`, so with
                # the flag but no number the page emitted NO tuplets at all -
                # measured, zero group starts on every song - while the widened
                # snap still produced ternary durations. music21 then invented
                # a bracket for each one at serialization, placing it wherever
                # the arithmetic fell and reconciling leftovers as 6:5. That is
                # where the junk ratios and the off-beat groups both came from.
                # InferredNoteTiming has carried this field, documented as "the
                # page needs the NUMBER, not just the flag", the whole time.
                value={"start_ms": beat_timing.notation_start_ms, "end_ms": beat_timing.notation_end_ms,
                       "reason": beat_timing.reason, "is_tuplet": beat_timing.is_tuplet,
                       "tuplet_divisor": beat_timing.tuplet_divisor,
                       # A beat read as 2 parts is EIGHTHS. Without this the
                       # exporter snaps every onset to a sixteenth lattice.
                       "subdivision": beat_timing.subdivision},
                source=Provenance.TEMPORAL_LATTICE,
            ))
            rhythm_inferred_count += 1
        else:
            notation = notation_quantize_note(note, tempo_resolution.tempo_meter, duration_testimony,
                                              musical_time=musical_time)
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
            # TRAILING window - the gap immediately AFTER this note. This is the
            # evidence the notation layer actually needs to decide whether a gap
            # is a real rest or a cut-off artifact: if the stem is still ringing
            # there, the note should be written out; if it is genuinely silent,
            # the rest is real. Until now the legato fill was a blanket
            # "close every gap under N beats" threshold chosen BY EAR
            # (§XVII.14), while this witness was already measuring the answer
            # and being written to an annotation nobody read (§XVIII.2).
            trail = report.query(note.end_ms, note.end_ms + beat_ms)
            annotations.add(Annotation(
                note_id=note.id, kind=ACOUSTIC_ACTIVITY_ANNOTATION_KIND,
                value={
                    "rhythmic_void_probability": state.rhythmic_void_probability,
                    "resonance_probability": state.resonance_probability,
                    "active_material_probability": state.active_material_probability,
                    "confidence_penalty": state.get_confidence_penalty(),
                    # trailing-gap evidence for the notation layer
                    "trailing_resonance": trail.resonance_probability,
                    "trailing_void": trail.rhythmic_void_probability,
                    "trailing_active": trail.active_material_probability,
                    "trailing_region": trail.region_type,
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
    # OctaveStack: the interior of a 3+-octave stack struck as one gesture
    # is the pitch detector reporting harmonics of a note it already has.
    # Verdict only, like its three sibling witnesses - the notes stay.
    octave_counts = write_octave_annotations(
        pitched_notes, annotations, Provenance.OCTAVE_STACK)
    music_box.log_decision(
        stage_name="acoustic_witness", decision_type="octave_stack",
        before_state={}, after_state=dict(octave_counts),
        reasoning=(f"OctaveStack flagged {octave_counts['interior_flagged']} of "
                   f"{len(pitched_notes)} pitched notes as the INTERIOR of an "
                   f"octave stack across {octave_counts['distinct_stacks']} distinct "
                   f"stacks - a published edition never writes a pitch class at more "
                   f"than 2 octaves in one struck chord, so the outer pair is real and "
                   f"the filling reads as harmonics (verdict only - notes are not "
                   f"deleted; dropped at export only under drop_purge_candidates)"),
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

    # PitchWobbleCollapse - VOCALS ONLY, and only with CREPE's continuous f0.
    #
    # The problem is specific: Basic Pitch's note posteriorgram is an 88-bin,
    # one-bin-per-semitone grid. A singer sustaining a pitch that genuinely
    # sits BETWEEN two semitones (a blues neutral third, a scoop settling shy
    # of the target) has no home bin, so the model flickers between the two it
    # straddles - emitting a stutter of short near-pitch fragments that are
    # symbolically indistinguishable from a fast real chromatic figure.
    #
    # Gate 4 is what makes this safe, and it is why 5.6.1's version was not:
    # only a NON-quantized pitch reader can tell "one off-grid pitch" from
    # "two real notes," and CREPE's raw f0 is exactly that. Without the
    # sampler find_wobble_groups() returns nothing at all - structural
    # plausibility alone is deliberately not sufficient evidence.
    #
    # Annotation-only. The notes are not merged, and nothing downstream acts
    # on this yet: this run is to see how many groups it actually finds and
    # how tight their pitch spreads are before deciding whether to collapse.
    wobble_group_count = 0
    vocals_track = stem_tracks_by_type.get(StemType.VOCALS)
    vocals_notes = notes_by_stem.get(StemType.VOCALS, [])
    if vocals_track is not None and vocals_notes:
        times_ms, freq_hz = continuous_f0(engine, vocals_track)
        sampler = make_f0_sampler(times_ms, freq_hz)
        wobble_groups = find_wobble_groups(
            vocals_notes, tempo_resolution.tempo_meter.tempo_bpm, sampler,
        )
        for group in wobble_groups:
            for member_id in group.member_note_ids:
                annotations.add(Annotation(
                    note_id=member_id, kind=PITCH_WOBBLE_ANNOTATION_KIND,
                    value={"anchor_note_id": group.anchor_note_id,
                           "anchor_pitch": group.anchor_pitch,
                           "member_note_ids": list(group.member_note_ids),
                           "continuous_pitch_spread_semitones":
                               group.continuous_pitch_spread_semitones,
                           "reason": group.reason},
                    source=Provenance.RITORNELLO,
                ))
            wobble_group_count += 1

        voiced_frames = int((freq_hz > 0).sum())
        music_box.log_decision(
            stage_name="quantization", decision_type="pitch_wobble_collapse",
            before_state={"vocals_notes": len(vocals_notes),
                          "crepe_voiced_frames": voiced_frames},
            after_state={"wobble_groups": wobble_group_count,
                         "notes_in_groups": sum(len(g.member_note_ids) for g in wobble_groups)},
            reasoning="CREPE's continuous f0 on the vocals stem confirmed which "
                      "Basic-Pitch fragment clusters are one off-grid sustained pitch "
                      "rather than distinct notes. Annotation only - nothing is merged.",
            reversible=True,
        )

    # TroubleMap (Ritornello Pass 7, ported): pure diagnosis of the whole
    # ensemble at once (unlike TieReconstruction, which stays within one
    # voice) - flags measures that are low-confidence and/or fragmented.
    # Read-only reporting on MusicalFindingsMap; nothing acts on it.
    # (`musical_time` is built once, up with the quantizer that needs it.)
    findings.trouble_measures = build_trouble_map(pitched_notes, tempo_resolution.tempo_meter)
    music_box.log_decision(
        stage_name="quantization", decision_type="trouble_map",
        before_state={}, after_state={"flagged_measures": len(findings.trouble_measures)},
        reasoning=f"TroubleMap flagged {len(findings.trouble_measures)} measure(s) as "
                  f"low-confidence and/or fragmented (diagnosis only)",
        reversible=False,
    )

    # Check (form/self-similarity, annotation-only, top of the ladder):
    # detect repeated SECTIONS + MOTIFS on the master, label every note
    # with its section, and score how consistently each repeated section
    # is transcribed. Evidence only - notes/timing are untouched; the
    # scores flag where the same idea was written inconsistently so a
    # later reconciliation (or the engraver) can prefer the consistent
    # reading. Soft by design (real music varies).
    check_result = run_check(engine, master_track, all_notes, annotations)
    music_box.log_decision(
        stage_name="check", decision_type="form_self_similarity",
        before_state={}, after_state={
            "sections": len(check_result.sections),
            "repeated_sections": check_result.repeat_groups,
            "group_consistency": {k: round(v, 2) for k, v in check_result.group_consistency.items()},
            "motifs": len(check_result.motifs),
            "drift_flagged": check_result.drift_flagged,
        },
        reasoning=f"Check found {len(check_result.sections)} sections "
                  f"({sum(1 for v in check_result.repeat_groups.values() if v >= 2)} recurring) and "
                  f"{len(check_result.motifs)} recurring motifs; labeled {check_result.notes_labeled} "
                  f"notes by section; flagged {check_result.drift_flagged} notes in outlier repeat "
                  f"instances as reconciliation evidence (annotation-only - note rewrite deferred)",
        reversible=False,
    )

    # Consolidation (annotation-only, §2): glue Basic Pitch's fragmented
    # same-pitch runs back into single sustained notes. Runs on the pitched
    # notes only (drums are discrete hits, never sustained). Always emits its
    # annotations; the engraver RENDERS the merged spans only under
    # use_consolidated_timing, so default output is unchanged - this is an
    # opt-in timeline to A/B against raw, like notation/groove timing.
    # Attack evidence gates the merge. Same-pitch fragments abut EXACTLY (gap
    # 0.0ms, no overlap, velocity ratio ~0.98) whether Basic Pitch re-triggered
    # a held note or the player struck the chord again, so only the audio can
    # tell them apart. Without this, planing textures lose their voicings -
    # measured on Ellington's Reflections in D: 31% of notes and 43% of the
    # 4+ note chords deleted. onset_candidates is already computed upstream
    # (for the phase-locked grid), so this costs nothing new.
    _attack_ms = sorted(onset_candidates.combined_ms) if onset_candidates else ()
    runs_merged, fragments_absorbed = consolidate_fragments(
        pitched_notes, annotations, onsets_ms=_attack_ms,
        onsets_by_stem=stem_onsets_by_type)
    music_box.log_decision(
        stage_name="quantization", decision_type="note_consolidation",
        before_state={"pitched_notes": len(pitched_notes)},
        after_state={"runs_merged": runs_merged, "fragments_absorbed": fragments_absorbed,
                     "rendered": use_consolidated_timing},
        reasoning=f"Consolidated {fragments_absorbed} fragmented same-pitch notes into "
                  f"{runs_merged} sustained runs (annotation-only; "
                  f"{'rendered' if use_consolidated_timing else 'not rendered - raw timeline'})",
        reversible=True,
    )

    # Grimlock University (GRIMLOCK_UNIVERSITY.md) - OFF by default.
    # Runs AFTER consolidation on purpose: raw Basic Pitch output is ~52%
    # same-pitch rearticulation and ~39% sub-16th fragments, so a pattern
    # layer reading it upstream would be studying detector artifacts, not
    # music. Read-only w.r.t. Notes in every mode; STUDY only annotates and
    # logs (output byte-identical), APPLY additionally lets the NOTATION
    # view honor what was found (the performance clock / MIDI is untouched
    # either way, so playback fidelity is unaffected by construction).
    uni_mode = UniversityMode.parse(university_mode)
    university_report = None
    if uni_mode.runs:
        from university import (
            apply_observations, study, write_corpus, write_study_annotations,
        )
        university_report = study(
            all_notes, annotations,
            tempo_bpm=tempo_resolution.tempo_meter.tempo_bpm,
            mode=uni_mode, use_consolidation=True,
        )
        annotated = write_study_annotations(university_report, annotations)
        applied = {}
        if uni_mode.applies:
            applied = apply_observations(
                university_report, annotations, {n.id: n for n in all_notes})
        if university_corpus_path is not None:
            try:
                write_corpus(university_report, str(university_corpus_path),
                             song=Path(audio_path).stem)
            except Exception:
                pass
        music_box.log_decision(
            stage_name="grimlock_university", decision_type="pattern_study",
            before_state={"mode": uni_mode.value,
                          "notes_studied": university_report.notes_studied},
            after_state={**university_report.summary(),
                         "annotations_written": annotated, **applied},
            reasoning=(
                f"Grimlock University ran in {uni_mode.value} mode: "
                f"{len(university_report.observations)} pattern observations covering "
                f"{university_report.coverage:.1%} of studied notes. "
                + ("APPLY: the notation view honors page-suppression and voice "
                   "cohesion; Notes and MIDI are unchanged."
                   if uni_mode.applies else
                   "STUDY: evidence only - pipeline output is unchanged.")
            ),
            reversible=True,
        )

    # beat_times_ms is the anchor witness's TRACKED beat grid - the plug
    # that used to be left disconnected: the Rhythm Engine computed real
    # beat positions and the engraver only ever received one scalar
    # tempo, so exported notes floated at absolute times against a flat
    # grid and notation hosts re-derived their own. Passing the grid
    # binds the file's tempo map to where the beats actually fell.
    midi = engrave(all_notes, annotations, output_midi_path, music_box=music_box,
                   use_quantized_timing=use_quantized_timing, use_notation_timing=use_notation_timing,
                   use_consolidated_timing=use_consolidated_timing,
                   drop_purge_candidates=drop_purge_candidates,
                   tempo_bpm=tempo_resolution.tempo_meter.tempo_bpm,
                   time_signature=(meter_resolution.numerator, meter_resolution.denominator),
                   key=findings.key,
                   beat_times_ms=tempo_resolution.tempo_meter.beat_times_ms)

    # Notation export (open problem #3): serialize the symbolic score in
    # notation space directly to MusicXML - never through MIDI, which by §V's
    # quotient argument cannot carry ties/tuplets/voices/beams. Opt-in path.
    # Consolidation-aware (the confetti is gone before a duration is drawn)
    # and fed the now-correct tempo/meter/grid from #2/#7. use_voices=False:
    # one dense staff per family that the exporter voices legally, more
    # readable than the over-fragmented per-line staves until voice
    # separation improves.
    if output_musicxml_path is not None:
        from output.musicxml_exporter import export_musicxml
        # Bar 1 starts on the resolved downbeat (see the downbeat_phase block
        # above), not wherever the earliest note happens to be.
        _downbeats = tempo_resolution.tempo_meter.downbeat_times_ms
        bar_origin_ms = _downbeats[0] if _downbeats else None
        if routed_layout:
            # Per-stem staff routing (user directive 2026-07-31): each stem its
            # own staff; a GRAND STAFF only where earned (guitar/piano/harp
            # timbre or the merged 'other', or verifiably polyphonic); a
            # melodic/vocal stem sorted into Voice 1/2/... before a grand staff;
            # bass and drums each a single staff. Timbre (family annotation) +
            # polyphony decide - via Timbre Intelligence and Voice Continuity.
            from output.notation_score import build_routed_score
            notation_score = build_routed_score(
                all_notes, annotations,
                tempo_bpm=tempo_resolution.tempo_meter.tempo_bpm,
                time_signature=(meter_resolution.numerator, meter_resolution.denominator),
                key=findings.key, use_consolidation=True, ratio_family=ratio_family,
                honor_university=uni_mode.applies,
                drop_octave_stacks=drop_purge_candidates,
                musical_time=musical_time,
                bar_origin_ms=bar_origin_ms,
            )
            layout_desc = ("per-stem routed (grand staff where earned)"
                           + (" + university APPLY" if uni_mode.applies else ""))
        else:
            from output.notation_score import build_notation_score
            notation_score = build_notation_score(
                all_notes, annotations,
                tempo_bpm=tempo_resolution.tempo_meter.tempo_bpm,
                time_signature=(meter_resolution.numerator, meter_resolution.denominator),
                key=findings.key, use_voices=False, use_consolidation=True,
                ratio_family=ratio_family,
                musical_time=musical_time,
                bar_origin_ms=bar_origin_ms,
            )
            layout_desc = "one staff per family"
        export_musicxml(notation_score, str(output_musicxml_path))
        music_box.log_decision(
            stage_name="scribe_engraver", decision_type="musicxml_written",
            before_state={}, after_state={"output_path": str(output_musicxml_path),
                                          "parts": len(notation_score.parts),
                                          "notes": notation_score.total_notes,
                                          "layout": layout_desc},
            reasoning=f"#3: serialized {notation_score.total_notes} notes across "
                      f"{len(notation_score.parts)} parts to MusicXML ({layout_desc}).",
            reversible=False,
        )

    # Save the expensive intermediate (frozen notes + all annotations + the
    # resolved tempo/meter/key/ratio_family) so notation LAYOUT can be
    # re-exported in seconds without re-running separation + Basic Pitch +
    # the acoustic witnesses (the ~hour cost). Notation is a view over this;
    # iterating on staff routing / voicing should never pay for detection again.
    if save_intermediate_path is not None:
        try:
            import pickle
            # WHAT AN INTERMEDIATE HAS TO CARRY, and why this grew.
            #
            # The point of saving one is that notation can be rebuilt from it
            # in seconds instead of re-running the hour. That is only true if
            # it holds everything the ENGRAVER needs - and it did not. Without
            # the beat grid and the bar origin, a re-export builds its score on
            # a flat isochronous clock starting at the first note, which is a
            # different engraving of the same notes: measured on Chopin, the
            # pipeline's own file carried 2310 onsets and a re-export of its
            # own pkl carried 2602. Two paths, two answers, and every "0
            # off-grid onsets" measured on a re-export was measuring the wrong
            # one.
            #
            # beat_times_ms is the tracked grid MusicalTime is built from;
            # bar_origin_ms is where bar 1 starts, which is what keeps a pickup
            # from displacing every barline in the piece.
            _dbs = tempo_resolution.tempo_meter.downbeat_times_ms
            payload = {
                "all_notes": all_notes,
                "annotations": annotations,
                "tempo_bpm": tempo_resolution.tempo_meter.tempo_bpm,
                "time_signature": (meter_resolution.numerator, meter_resolution.denominator),
                "key": findings.key,
                "ratio_family": ratio_family,
                "beat_times_ms": tuple(tempo_resolution.tempo_meter.beat_times_ms or ()),
                "downbeat_times_ms": tuple(_dbs or ()),
                "bar_origin_ms": (_dbs[0] if _dbs else None),
                # WHERE THE WITNESSES DISAGREED. The referee builds a full
                # record - every witness's raw bpm, confidence, weight, fold
                # ratio, and whether it was included or thrown out - and until
                # now that record went onto PipelineResult and nowhere else,
                # so it died with the process. It is the single most useful
                # thing to show a user in guided mode: when four witnesses
                # split, the right answer is to ASK, and this says who said
                # what. Chopin is the case in point - we report 70bpm where
                # the edition implies 55.4, and the disagreement behind that
                # number was being discarded.
                "tempo_contention": tempo_resolution.contention,
                "meter_contention": meter_resolution.contention,
                "tempo_confidence": tempo_resolution.tempo_meter.confidence,
            }
            with open(str(save_intermediate_path), "wb") as fh:
                pickle.dump(payload, fh)
            music_box.log_decision(
                stage_name="conductor", decision_type="intermediate_saved",
                before_state={}, after_state={"path": str(save_intermediate_path),
                                              "notes": len(all_notes)},
                reasoning="Saved notes+annotations+resolved params so notation can be "
                          "re-exported without re-running the pipeline.",
                reversible=False,
            )
        except Exception as e:
            music_box.log_decision(
                stage_name="conductor", decision_type="intermediate_save_failed",
                before_state={}, after_state={"error": f"{type(e).__name__}: {e}"},
                reasoning="Intermediate save failed (non-fatal).", reversible=False,
            )

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
        onset_refined_count=onset_refined_count,
        wobble_group_count=wobble_group_count,
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
