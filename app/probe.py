# =================================================================
# MODULE: app/probe.py
# WHAT IS THIS RECORDING? ANSWERED IN SECONDS, BEFORE THE HOUR IS SPENT.
#
# WHY THIS EXISTS. A full run costs 30-60 minutes. The single decision with
# the largest measured effect on the result - separate, or treat this as one
# instrument - is made in the first thirty seconds of it, and on the Chopin
# recording getting that decision right moved F1 from 0.507 to 0.603. There
# is no reason a user should wait an hour to find out what the machine
# concluded in half a minute, and no reason they should be unable to
# disagree with it before paying.
#
# So this composes the cheap evidence into a preview: what is playing, at
# what tempo, in what meter, and - stated plainly - what the pipeline WILL
# do if run with these settings. Nothing here writes a file, loads Demucs or
# Basic Pitch, or takes longer than reading the audio.
#
# IT REPORTS ITS WORKING. Every number that produced the verdict is carried
# on the result, because this verdict skips an entire stage of the pipeline
# and a verdict nobody can audit is a verdict nobody should act on. The
# front end shows those numbers next to the override control.
#
# ON THE OVERRIDE. The solo gate is deliberately asymmetric: it says ENSEMBLE
# unless the evidence is clear, because a false "solo" silently discards real
# instruments and nothing downstream could notice, while a false "ensemble"
# costs only time. That asymmetry is right for a machine guessing. It is
# wrong for a person who KNOWS the recording is solo piano, and guided mode
# (DESIGN_DECISIONS §2.7) is how that knowledge enters: as a hard lock that
# skips the detector, not as another vote.
# =================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from audio_engine import AudioEngine
from core.stem_types import StemType
from separation_engine import (
    ENSEMBLE, SOLO_GUITAR, SOLO_PIANO, SoloVerdict, detect_solo_instrument,
)

# Only this much audio is read for the tempo estimate. The question is "what
# is roughly going on", and a preview that takes minutes is not a preview.
TEMPO_PREVIEW_SECONDS = 90.0


@dataclass(frozen=True)
class ProbeResult:
    """A cheap preview of a recording, plus what a run would do with it."""
    audio_path: str
    duration_seconds: float

    # what is playing
    solo: Optional[SoloVerdict] = None

    # a rough reading, from one witness rather than the full referee
    tempo_bpm: float = 0.0
    tempo_confidence: float = 0.0
    time_signature: Tuple[int, int] = (4, 4)
    meter_confidence: float = 0.0

    elapsed_seconds: float = 0.0
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def will_skip_separation(self) -> bool:
        return bool(self.solo is not None and self.solo.is_solo)

    @property
    def planned_stem(self) -> Optional[StemType]:
        """The single stem a run would use if it skips separation."""
        if not self.will_skip_separation or self.solo is None:
            return None
        return StemType.PIANO if self.solo.verdict == SOLO_PIANO else StemType.GUITAR

    def plan(self, guided_separation: Optional[str] = None) -> str:
        """One sentence: what a run WILL do, given an override or without one.

        Deliberately phrased as a prediction rather than a recommendation.
        The user is being asked to confirm a plan, not to grade a classifier.
        """
        if guided_separation in (SOLO_PIANO, SOLO_GUITAR):
            stem = "piano" if guided_separation == SOLO_PIANO else "guitar"
            return (f"Demucs will be SKIPPED and the whole recording treated as "
                    f"one {stem} stem, because you said so. The probe is not "
                    f"consulted.")
        if guided_separation == ENSEMBLE:
            return ("Demucs WILL run and separate this into stems, because you "
                    "said so. The probe is not consulted.")
        if self.solo is None:
            return "Could not probe this audio; Demucs will run."
        if self.solo.is_solo:
            stem = "piano" if self.solo.verdict == SOLO_PIANO else "guitar"
            return (f"Demucs will be SKIPPED and the whole recording treated as "
                    f"one {stem} stem. Saves roughly half an hour, and avoids "
                    f"inventing drum, bass and vocal notes that are not there.")
        return ("Demucs WILL run and separate this into stems - the probe found "
                "more than one instrument. Expect the run to take 30-60 minutes.")


def probe_audio(audio_path: str,
                tempo_preview_seconds: float = TEMPO_PREVIEW_SECONDS) -> ProbeResult:
    """Read a recording and report what a run would do with it.

    Never raises on analysis failure: a probe that cannot answer returns what
    it has, with a warning, so the front end can offer the full run anyway.
    """
    started = time.time()
    warnings_out = []

    engine = AudioEngine()
    track = engine.decode(str(audio_path))
    duration = float(getattr(track, "duration_seconds", 0.0))

    solo: Optional[SoloVerdict] = None
    try:
        solo = detect_solo_instrument(engine, track)
    except Exception as exc:                                # pragma: no cover
        warnings_out.append(f"solo probe failed ({type(exc).__name__}); "
                            f"a run will separate normally")

    tempo_bpm = tempo_confidence = 0.0
    ts: Tuple[int, int] = (4, 4)
    meter_confidence = 0.0
    try:
        tempo_bpm, tempo_confidence, ts, meter_confidence = _rough_tempo_meter(
            engine, track, tempo_preview_seconds)
    except Exception as exc:                                # pragma: no cover
        warnings_out.append(f"tempo preview failed ({type(exc).__name__}); "
                            f"the run will estimate it properly")

    return ProbeResult(
        audio_path=str(audio_path), duration_seconds=duration, solo=solo,
        tempo_bpm=tempo_bpm, tempo_confidence=tempo_confidence,
        time_signature=ts, meter_confidence=meter_confidence,
        elapsed_seconds=time.time() - started,
        warnings=tuple(warnings_out),
    )


def _rough_tempo_meter(engine, track, preview_seconds: float):
    """ONE tempo witness, not the four-witness referee.

    This is a preview and is labelled as one everywhere it surfaces. The real
    reading comes from epistemic.resolve_tempo across every witness, and the
    two are allowed to disagree - when they do, that is the run correcting a
    guess, which is the arrangement working rather than failing.

    `preview_seconds` is accepted and currently unused: the witnesses take an
    AudioEngine and an AudioTrack rather than raw samples, so slicing here
    would mean hand-building a truncated track and diverging from the code
    path the real run uses. Reading the whole file costs seconds and keeps the
    preview honest, which is the better trade - the parameter stays so a
    future fast path has somewhere to land.
    """
    from rhythm_engine import estimate_time_signature, run_librosa_tempo

    witness = run_librosa_tempo(engine, track)
    tempo_bpm = float(getattr(witness, "tempo_bpm", 0.0) or 0.0)
    tempo_conf = float(getattr(witness, "confidence", 0.0) or 0.0)

    ts, meter_conf = (4, 4), 0.0
    beats = list(getattr(witness, "beat_times_ms", ()) or ())
    if tempo_bpm > 0 and len(beats) >= 8:
        num, den, conf = estimate_time_signature(engine, track, beats)
        ts, meter_conf = (int(num), int(den)), float(conf)
    return tempo_bpm, tempo_conf, ts, meter_conf


__all__ = ["ProbeResult", "probe_audio", "TEMPO_PREVIEW_SECONDS"]
