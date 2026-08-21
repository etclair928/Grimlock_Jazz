# =================================================================
# MODULE: app/runner.py
# RUNNING A TRANSCRIPTION FROM A UI, AND WATCHING IT HAPPEN.
#
# THREE PROBLEMS, ONE SHAPE.
#
#   1. A run blocks for 30-60 minutes. It cannot sit on the UI thread.
#   2. It loads multi-gigabyte models. A Demucs OOM must not take the window
#      with it, so it does not merely get its own thread - it gets its own
#      PROCESS.
#   3. A progress bar somebody invented is a lie with a percentage on it. The
#      engine already narrates itself; the UI should show THAT.
#
# So: the pipeline runs in a subprocess, and its telemetry arrives over a file.
#
# WINDOW PANE IS THE TELEMETRY, AND THIS IS WHAT IT WAS STAGED FOR.
# core/window_pane.py has carried this note since it was written: "STATUS:
# STAGED, NOT DEAD... its consumer is the live dashboard / GUI frontend, which
# does not exist yet... the resolution is 'build the frontend,' not 'delete
# this.'" This module is that consumer.
#
# HOW IT IS ADOPTED, AND WHY NOT INSIDE THE CONDUCTOR. The obvious wiring is
# to sprinkle `pane.stage(...)` through orchestration/conductor.py. That would
# be forty-five edits to a load-bearing function, and it would make Core's
# ambient services something the pipeline has to remember to call.
#
# The conductor already names its stage on every decision it logs, and
# `transcribe_file` now accepts a MusicBox. So the adoption is a TEE: a
# MusicBox subclass that records forensically exactly as before AND mirrors
# each decision to a Window Pane. The engine is untouched, Core stays
# uncoupled, and the live feed carries every decision the engine makes -
# including its `reasoning` string, which is the part worth watching.
#
# It also keeps the two services doing what they are for. Music_Box keeps
# everything forever, for Grimlock University and for backward traceability.
# Window_Pane keeps a bounded window for whoever is watching right now, and
# lets the old events fall off. Feeding one from the other is not a shortcut
# around that distinction - it is the same event observed at two fidelities,
# which is what they were always for.
# =================================================================

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import MusicBox, WindowPane

JAZZ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PaneMusicBox(MusicBox):
    """A MusicBox that also narrates, live, to a Window Pane.

    Forensic behaviour is unchanged - `super().log_decision` still runs first,
    so nothing about the persistent ledger depends on the pane working. If the
    pane is absent or its emit fails, the run is unaffected, which is the rule
    Window_Pane sets for itself: a telemetry layer that can crash the run it
    observes is worse than no telemetry.
    """

    def __init__(self, *args, pane: Optional[WindowPane] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pane = pane

    def log_decision(self, stage_name: str, decision_type: str,
                     before_state: Dict[str, Any], after_state: Dict[str, Any],
                     reasoning: str, reversible: bool = True) -> None:
        super().log_decision(stage_name, decision_type, before_state,
                             after_state, reasoning, reversible)
        if self._pane is None:
            return
        self._pane.emit("decision", stage_name,
                        decision_type=decision_type,
                        reasoning=reasoning,
                        after=after_state)


# ---------------------------------------------------------------------
# the config a worker subprocess is given
# ---------------------------------------------------------------------

@dataclass
class RunConfig:
    """Everything a run needs, JSON-round-trippable so it can cross a process.

    The guided fields exist because the front end is built AROUND guided mode
    rather than merely offering it: what the user knows enters the pipeline the
    same way the machine's own findings do, as a hard lock (DESIGN_DECISIONS
    §2.7) that skips the detector rather than out-voting it.
    """
    audio_path: str
    out_stem: str
    out_dir: str = os.path.join(JAZZ_ROOT, "transcriptions")

    guided_separation: Optional[str] = None      # solo_piano | solo_guitar | ensemble
    guided_tempo_bpm: Optional[float] = None
    guided_time_signature: Optional[List[int]] = None
    guided_key: Optional[str] = None

    use_notation_timing: bool = True
    use_consolidated_timing: bool = True
    drop_purge_candidates: bool = False
    university_mode: str = "off"
    stem_cache_dir: Optional[str] = None

    def paths(self) -> Dict[str, str]:
        base = os.path.join(self.out_dir, self.out_stem)
        return {"midi": base + ".mid", "musicxml": base + ".musicxml",
                "pkl": base + ".pkl"}

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(text: str) -> "RunConfig":
        return RunConfig(**json.loads(text))


@dataclass
class RunStatus:
    running: bool = False
    finished: bool = False
    cancelled: bool = False
    returncode: Optional[int] = None
    error: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class TranscriptionRunner:
    """Launches a run in its own process and streams its telemetry back.

    Deliberately poll-based rather than callback-based: tkinter is single
    threaded and wants to drive its own loop, so `poll()` is called from
    `after()` and returns whatever is new. No UI toolkit is imported here.
    """

    def __init__(self, config: RunConfig, work_dir: Optional[str] = None):
        self.config = config
        self._work = work_dir or tempfile.mkdtemp(prefix="grimlock_run_")
        self.stream_path = os.path.join(self._work, "pane.jsonl")
        self.result_path = os.path.join(self._work, "result.json")
        self.config_path = os.path.join(self._work, "config.json")
        self.log_path = os.path.join(self._work, "worker.log")

        self._proc: Optional[subprocess.Popen] = None
        self._offset = 0
        self._started = 0.0
        self.status = RunStatus()

    # ------------------------------------------------------------- lifecycle
    def start(self, python_executable: Optional[str] = None) -> None:
        with open(self.config_path, "w", encoding="utf-8") as fh:
            fh.write(self.config.to_json())
        os.makedirs(self.config.out_dir, exist_ok=True)

        cmd = [python_executable or sys.executable, "-m", "app.worker",
               "--config", self.config_path,
               "--stream", self.stream_path,
               "--result", self.result_path]
        env = dict(os.environ)
        env["PYTHONPATH"] = JAZZ_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("PYTHONUNBUFFERED", "1")

        self._log = open(self.log_path, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            cmd, cwd=JAZZ_ROOT, env=env,
            stdout=self._log, stderr=subprocess.STDOUT,
        )
        self._started = time.time()
        self.status = RunStatus(running=True)

    def cancel(self) -> None:
        """Kill the run. Partial output is NOT presented as a result - a
        half-finished transcription that looks finished is worse than none."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        self.status.running = False
        self.status.cancelled = True
        self.status.finished = True

    # ------------------------------------------------------------------ read
    def poll(self) -> List[Dict[str, Any]]:
        """New telemetry since the last call, and an updated status.

        Reads by byte offset so a partially-written final line is left for the
        next poll rather than being parsed as truncated JSON.
        """
        events: List[Dict[str, Any]] = []
        try:
            if os.path.exists(self.stream_path):
                with open(self.stream_path, "r", encoding="utf-8") as fh:
                    fh.seek(self._offset)
                    text = fh.read()
                    complete, _sep, remainder = text.rpartition("\n")
                    if complete:
                        self._offset += len(complete.encode("utf-8")) + 1
                        for line in complete.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except OSError:
            pass

        if self._proc is not None and self.status.running:
            code = self._proc.poll()
            if code is not None:
                self.status.running = False
                self.status.finished = True
                self.status.returncode = code
                self.status.elapsed_seconds = time.time() - self._started
                self._finish(code)
        return events

    def _finish(self, code: int) -> None:
        try:
            self._log.close()
        except Exception:
            pass
        if os.path.exists(self.result_path):
            try:
                with open(self.result_path, "r", encoding="utf-8") as fh:
                    self.status.result = json.load(fh)
            except Exception as exc:
                self.status.error = f"could not read the run's result: {exc}"
        if code != 0 and not self.status.error:
            self.status.error = (f"the run exited with code {code}; "
                                 f"see {self.log_path}")

    def worker_log(self, last_lines: int = 40) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-last_lines:])
        except OSError:
            return ""


__all__ = ["PaneMusicBox", "RunConfig", "RunStatus", "TranscriptionRunner",
           "JAZZ_ROOT"]
