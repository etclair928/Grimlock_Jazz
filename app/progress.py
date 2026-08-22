# =================================================================
# MODULE: app/progress.py
# HOW FAR ALONG IS THIS RUN, AND IS IT STILL ALIVE?
#
# Two different questions, and conflating them is how progress bars come to
# lie. This module answers both separately, and refuses to fake either.
#
# THE HARD PART: THE PIPELINE GOES SILENT. Its expensive stages - Demucs, and
# Basic Pitch once per stem - emit nothing at all while they run, for many
# minutes at a stretch. A bar driven by decision events therefore freezes for
# most of the wall clock, and freezes hardest at exactly the moment a user most
# wants reassurance. Measured on the Chopin runs, thirty decisions arrive
# across a run lasting between thirty and seventy minutes, and the gap between
# session_start and the first separation decision is most of it.
#
# So three signals, doing three jobs:
#
#   THE CLOCK never stops. It is driven by wall time, not by events, so it
#   moves every second whatever the pipeline is doing. This is the actual
#   "something is happening" signal and everything else is secondary to it.
#
#   THE MILESTONES are honest about what they are: a count of known decisions
#   reached, out of the decisions a complete run is known to emit. Not a time
#   estimate. A run that has reached 20 of 30 milestones may still have half
#   its wall clock ahead of it, because the milestones are not evenly spaced
#   in time - and pretending otherwise is the lie this avoids.
#
#   THE QUIET TIMER says how long since anything was heard. That is what turns
#   a frozen bar from alarming into informative: "separation_engine - working,
#   nothing for 4m 12s" is a system running normally, and the same silence with
#   no explanation is a system that looks hung.
# =================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# The decisions a complete run emits, in the order one actually emitted them
# (captured from a full guided Chopin run). Used ONLY to count how far along a
# run is; a decision that never arrives simply never ticks, and an unknown one
# is counted as progress without being on the list.
#
# Stages interleave - key_intelligence appears three times, quantization eight
# - so this is a milestone sequence, not a stage sequence. Grouping it by stage
# for display is done below and is a presentation choice, not a claim about
# execution order.
MILESTONES: Tuple[Tuple[str, str], ...] = (
    ("conductor", "session_start"),
    ("separation_engine", "solo_probe"),
    ("separation_engine", "separation_guided"),
    ("separation_engine", "separation_skipped"),
    ("separation_engine", "separated"),
    ("separation_engine", "harmonic_stems_merged"),
    ("key_intelligence", "key_detected"),
    ("quantization", "onsets_refined"),
    ("key_intelligence", "key_stability"),
    ("key_intelligence", "key_from_notes"),
    ("quantization", "sustain_recovery"),
    ("rhythm_engine", "meter_grid"),
    ("rhythm_engine", "downbeat_phase"),
    ("rhythm_engine", "tempo_drift"),
    ("epistemic", "tempo_resolved"),
    ("epistemic", "meter_resolved"),
    ("rhythm_engine", "groove_resolved"),
    ("quantization", "musical_time"),
    ("quantization", "micro_note_purge"),
    ("quantization", "rhythm_inference"),
    ("acoustic_witness", "note_support"),
    ("acoustic_witness", "octave_stack"),
    ("acoustic_witness", "schoenberg_mirror"),
    ("quantization", "tie_reconstruction"),
    ("quantization", "trouble_map"),
    ("check", "form_self_similarity"),
    ("quantization", "note_consolidation"),
    ("scribe_engraver", "midi_written"),
    ("scribe_engraver", "musicxml_written"),
    ("conductor", "intermediate_saved"),
    ("conductor", "session_end"),
)

# The stages, in the order they FIRST appear, for the checklist display.
STAGE_ORDER: Tuple[str, ...] = ()
_seen: List[str] = []
for _stage, _d in MILESTONES:
    if _stage not in _seen:
        _seen.append(_stage)
STAGE_ORDER = tuple(_seen)

# Separation is optional - a guided or probed solo run skips Demucs entirely,
# so three of its milestones can never all arrive. Counting them against a run
# that legitimately skipped them would cap progress below 100%.
OPTIONAL = frozenset({
    ("separation_engine", "solo_probe"),
    ("separation_engine", "separation_guided"),
    ("separation_engine", "separation_skipped"),
})

# Longer than this with nothing heard and the UI says so explicitly rather
# than showing a bar that looks stuck.
QUIET_AFTER_SECONDS = 20.0


@dataclass
class StageState:
    name: str
    started: Optional[float] = None
    last_seen: Optional[float] = None
    finished: bool = False
    decisions: int = 0

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.last_seen or self.started) - self.started


@dataclass
class RunProgress:
    """Live state of one run. Fed events, asked for a picture."""
    started_at: float = field(default_factory=time.time)
    last_event_at: Optional[float] = None
    current_stage: str = ""
    last_decision: str = ""
    reached: set = field(default_factory=set)
    stages: Dict[str, StageState] = field(default_factory=dict)
    finished: bool = False

    def __post_init__(self) -> None:
        for name in STAGE_ORDER:
            self.stages.setdefault(name, StageState(name=name))

    # ------------------------------------------------------------------ feed
    def observe(self, event: dict) -> None:
        kind = event.get("event_type", "")
        stage = event.get("stage", "") or ""
        payload = event.get("payload", {}) or {}
        now = time.time()
        self.last_event_at = now

        if kind == "run_end":
            self.finished = True
            for state in self.stages.values():
                if state.started is not None:
                    state.finished = True
            return
        if kind != "decision":
            return

        decision = str(payload.get("decision_type", ""))
        self.last_decision = f"{stage}: {decision}" if stage else decision
        self.reached.add((stage, decision))

        if stage:
            # A stage reappearing after others ran is normal here - stages
            # interleave - so `finished` means "we have moved past it for now",
            # and it is un-finished if it speaks again.
            if stage != self.current_stage and self.current_stage:
                previous = self.stages.get(self.current_stage)
                if previous is not None:
                    previous.finished = True
            state = self.stages.setdefault(stage, StageState(name=stage))
            if state.started is None:
                state.started = now
            state.last_seen = now
            state.finished = False
            state.decisions += 1
            self.current_stage = stage

    # ------------------------------------------------------------- questions
    @property
    def elapsed_seconds(self) -> float:
        """Wall time since the run began. Never stops, never depends on events."""
        return time.time() - self.started_at

    @property
    def quiet_seconds(self) -> float:
        return time.time() - (self.last_event_at or self.started_at)

    @property
    def is_quiet(self) -> bool:
        return not self.finished and self.quiet_seconds > QUIET_AFTER_SECONDS

    @property
    def milestones_total(self) -> int:
        return len(MILESTONES) - len(OPTIONAL)

    @property
    def milestones_reached(self) -> int:
        return len([m for m in self.reached if m in set(MILESTONES) and m not in OPTIONAL])

    @property
    def fraction(self) -> float:
        if self.finished:
            return 1.0
        return min(1.0, self.milestones_reached / max(self.milestones_total, 1))

    def headline(self) -> str:
        """One line. The clock first, because it is the honest part."""
        clock = format_elapsed(self.elapsed_seconds)
        if self.finished:
            return f"finished in {clock}"
        done, total = self.milestones_reached, self.milestones_total
        stage = self.current_stage or "starting"
        line = f"{clock} elapsed   -   {stage}   -   {done}/{total} milestones"
        if self.is_quiet:
            # Not an error. Basic Pitch and Demucs are silent for minutes at a
            # time, and saying so is what stops a still bar reading as a hang.
            line += (f"   -   working, nothing reported for "
                     f"{format_elapsed(self.quiet_seconds)} "
                     f"(the heavy models do not report progress)")
        return line


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


__all__ = ["RunProgress", "StageState", "MILESTONES", "STAGE_ORDER",
           "OPTIONAL", "QUIET_AFTER_SECONDS", "format_elapsed"]
