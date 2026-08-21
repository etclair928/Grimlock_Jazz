# =================================================================
# MODULE: core/window_pane.py
# Window_Pane (GRIMLOCK_6.0_DESIGN_DECISIONS.md §Core ambient
# services): the LIVE / ephemeral sink. Sibling to Music_Box.
#
#   Music_Box  = forensic, persistent, append-only, flushes to disk,
#                feeds "Grimlock University". Records EVERYTHING for
#                backward traceability. Never forgets.
#   Window_Pane = live, ephemeral, streaming. Feeds a human (or a
#                dashboard/webhook) watching a run happen RIGHT NOW.
#                Keeps only a bounded window of recent events; older
#                ones fall off by design.
#
# Like every Core ambient service: callable from anywhere, DECIDES
# NOTHING. Pure passive observability. It never blocks the pipeline
# and never raises into it - a telemetry layer that can crash the
# run it observes is worse than no telemetry.
#
# This is a from-scratch rebuild of a recovered Grimlock 4.7 sketch,
# fixing that sketch's four documented defects (see class docstring):
# per-event OS threads, an async/sync mismatch, unbounded memory
# growth, and GIL-dependent thread-safety.
#
# STATUS: STAGED, NOT DEAD. Deliberately not wired into the Conductor
# yet - its consumer is the live dashboard / GUI frontend, which does
# not exist yet. This is the observability half of that feature, built
# and probe-verified ahead of it, and it is a known, decided part of
# the 6.0 plan - not orphaned code. A wiring audit that flags this
# should read this line and move on; the resolution is "build the
# frontend," not "delete this." Standalone until that frontend adopts it.
# =================================================================

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


# ---- Event vocabulary -------------------------------------------------------
# Shared conceptually with Music_Box's forensic records, but deliberately
# leaner: a Pane event is a "something happened right now" ping, not a full
# before/after forensic record. Kept as its own type rather than sharing a
# base class with ForensicRecord - they have different needs, and a premature
# shared type would be exactly the internal-converter smell the bible warns
# against. Unify later only if a real second caller needs it.

# Canonical event_type strings. Free-form is allowed (like Music_Box's
# decision_type), but these are the ones the recovered sketch named, kept as
# constants so call sites and any future viewer agree on spelling.
EVENT_WITNESS = "witness"
EVENT_ONSET = "onset"
EVENT_DRUM_HIT = "drum_hit"
EVENT_NOTE = "note"
EVENT_CONTRADICTION = "contradiction"
EVENT_TEMPO_VERDICT = "tempo_verdict"
EVENT_STAGE_START = "stage_start"
EVENT_STAGE_END = "stage_end"
EVENT_MEMORY_SAMPLE = "memory_sample"


@dataclass(frozen=True)
class PaneEvent:
    """One immutable live-telemetry ping.

    seq       - monotonic per-pane sequence number. A viewer that has seen
                seq=N knows exactly how many events exist; if the ring buffer
                only holds the last M, it knows it is (latest_seq - N) behind.
    t_wall    - wall-clock UNIX seconds (for humans / absolute timestamps).
    t_ms      - milliseconds since this pane started (for run-relative timing).
    stage     - which pipeline stage emitted it ("" if outside any stage).
    event_type- one of the EVENT_* constants (or a free-form string).
    payload   - JSON-safe dict of whatever the event carries.
    """
    seq: int
    t_wall: float
    t_ms: float
    stage: str
    event_type: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sanitize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort JSON-safety for arbitrary payloads - large arrays/bytes get
    a short description instead of bloating the stream or failing to serialize.
    Mirrors Music_Box._sanitize so both sinks describe big objects the same way.
    """
    clean: Dict[str, Any] = {}
    for key, value in payload.items():
        if hasattr(value, "shape"):
            clean[key] = f"<array shape={value.shape}>"
        elif isinstance(value, (bytes, bytearray)):
            clean[key] = f"<bytes length={len(value)}>"
        elif isinstance(value, (list, tuple)) and len(value) > 100:
            clean[key] = f"<sequence length={len(value)}>"
        else:
            try:
                json.dumps(value, default=str)
                clean[key] = value
            except (TypeError, ValueError):
                clean[key] = str(value)
    return clean


class WindowPane:
    """Live, ephemeral, non-blocking observability for one pipeline run.

    Design contract (the four fixes over the recovered 4.7 sketch):

    1. ONE background worker, not one-thread-per-event. Webhook streaming
       drains a single bounded `queue.Queue` from a single daemon thread.
       At 500-1000 notes/song the 4.7 "threading.Thread() per send" would
       spawn thousands of short-lived threads; this spawns exactly one.

    2. NO asyncio. Every `emit_*` is synchronous and returns immediately -
       it appends to an in-memory ring and (if a webhook is configured)
       hands off to the queue with `put_nowait`. The Conductor is fully
       synchronous; the telemetry does not impose an event loop on it.

    3. BOUNDED memory. The live ring is a `deque(maxlen=capacity)`; the
       memory-sample ring likewise. Nothing grows without bound. Explicit
       budget: ~`capacity` events resident, oldest silently evicted. This
       is the ephemeral contract, not a leak.

    4. EXPLICIT thread-safety. A single `threading.Lock` guards the ring
       and the sequence counter (shared between the caller thread and the
       memory-sampler thread). The worker handoff uses `queue.Queue`, which
       is itself thread-safe. No reliance on CPython GIL atomicity.

    It NEVER blocks the pipeline (queue-full => drop-and-count) and NEVER
    raises into it (all sink I/O is wrapped). A dead webhook or a full queue
    degrades telemetry, never the transcription.
    """

    def __init__(
            self,
            capacity: int = 2000,
            webhook_url: Optional[str] = None,
            # A JSONL sink, one event per line, for a viewer in ANOTHER
            # PROCESS. The ring is in-memory and the webhook needs an HTTP
            # listener; a desktop front end that runs the pipeline as a
            # subprocess (so a Demucs OOM cannot take the window with it) can
            # do neither, and tailing a file is the cheapest thing that works.
            # Deliberately mirrors MusicBox(log_path=...) - the difference
            # stays what it always was: Music_Box keeps EVERYTHING forever,
            # this keeps a bounded window for someone watching right now.
            stream_path: Optional[Path] = None,
            webhook_queue_size: int = 1000,
            webhook_timeout_s: float = 2.0,
            enable_memory_sampling: bool = False,
            memory_sample_interval_s: float = 1.0,
            memory_sample_capacity: int = 600,
    ):
        self._session_id = uuid.uuid4().hex[:8]
        self._t0_wall = time.time()
        self._t0_mono = time.perf_counter()

        # --- bounded live ring + its guard (concerns #3, #4) ---
        self._lock = threading.Lock()
        self._ring: Deque[PaneEvent] = deque(maxlen=capacity)
        self._seq = 0                 # total events ever emitted (guarded)
        self._evicted = 0             # events aged out of the ring (informational)

        # --- webhook worker (concern #1): one thread, bounded queue ---
        self._webhook_url = webhook_url
        self._webhook_timeout_s = webhook_timeout_s
        self._webhook_q: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._webhook_sent = 0
        self._webhook_failed = 0
        self._webhook_dropped = 0     # dropped because the queue was full
        self._stream_failed = 0       # lines a stream sink could not write
        self._stop_flag = threading.Event()

        # --- JSONL stream sink: opened once, line-buffered, appended to ---
        # Line buffering is the whole point: a reader tailing this file needs
        # each event to land as it happens, not when a 4KB buffer fills.
        self._stream = None
        if stream_path is not None:
            try:
                Path(stream_path).parent.mkdir(parents=True, exist_ok=True)
                self._stream = open(str(stream_path), "a", encoding="utf-8",
                                    buffering=1)
            except Exception:
                self._stream = None   # a sink that cannot open is not fatal
        if webhook_url:
            self._webhook_q = queue.Queue(maxsize=webhook_queue_size)
            self._worker = threading.Thread(
                target=self._drain_webhook, name=f"window-pane-webhook-{self._session_id}",
                daemon=True,
            )
            self._worker.start()

        # --- optional memory sampler (concern #3 stays honest about its own cost) ---
        self._mem_ring: Deque[PaneEvent] = deque(maxlen=memory_sample_capacity)
        self._mem_thread: Optional[threading.Thread] = None
        self._mem_interval_s = memory_sample_interval_s
        if enable_memory_sampling:
            self._start_memory_sampling()

    # ------------------------------------------------------------------ emit
    def emit(self, event_type: str, stage: str = "", **payload: Any) -> None:
        """The one true emit. Synchronous, non-blocking, never raises.

        All typed convenience emitters below funnel through here.
        """
        try:
            with self._lock:
                self._seq += 1
                seq = self._seq
                if len(self._ring) == self._ring.maxlen:
                    self._evicted += 1
                event = PaneEvent(
                    seq=seq,
                    t_wall=time.time(),
                    t_ms=(time.perf_counter() - self._t0_mono) * 1000.0,
                    stage=stage,
                    event_type=str(event_type),
                    payload=_sanitize(payload),
                )
                self._ring.append(event)
            # Hand off to the webhook worker OUTSIDE the lock - never block the
            # producer; if the queue is full the event is dropped and counted.
            if self._webhook_q is not None:
                try:
                    self._webhook_q.put_nowait(event)
                except queue.Full:
                    self._webhook_dropped += 1
            # Same rule as the webhook: outside the lock, and a failure is
            # counted rather than raised. Telemetry never takes down the run.
            if self._stream is not None:
                try:
                    self._stream.write(
                        json.dumps(event.to_dict(), default=str) + chr(10))
                except Exception:
                    self._stream_failed += 1
        except Exception:
            # A telemetry sink must never take down the run it observes.
            pass

    # Typed convenience emitters (the vocabulary the 4.7 sketch named).
    def emit_witness(self, name: str, value: Any, confidence: Optional[float] = None,
                     stage: str = "", **extra: Any) -> None:
        self.emit(EVENT_WITNESS, stage, name=name, value=value,
                  confidence=confidence, **extra)

    def emit_onset(self, time_ms: float, strength: Optional[float] = None,
                   stage: str = "", **extra: Any) -> None:
        self.emit(EVENT_ONSET, stage, time_ms=time_ms, strength=strength, **extra)

    def emit_drum_hit(self, time_ms: float, drum_type: str,
                      stage: str = "", **extra: Any) -> None:
        self.emit(EVENT_DRUM_HIT, stage, time_ms=time_ms, drum_type=drum_type, **extra)

    def emit_note(self, pitch: int, start_ms: float, end_ms: float,
                  stage: str = "", **extra: Any) -> None:
        self.emit(EVENT_NOTE, stage, pitch=pitch, start_ms=start_ms,
                  end_ms=end_ms, **extra)

    def emit_contradiction(self, subject: str, detail: str,
                           stage: str = "", **extra: Any) -> None:
        self.emit(EVENT_CONTRADICTION, stage, subject=subject, detail=detail, **extra)

    def emit_tempo_verdict(self, bpm: float, confidence: Optional[float] = None,
                           stage: str = "", **extra: Any) -> None:
        self.emit(EVENT_TEMPO_VERDICT, stage, bpm=bpm, confidence=confidence, **extra)

    # --------------------------------------------------------------- staging
    def stage(self, name: str, **meta: Any) -> "_StageTimer":
        """`with pane.stage("pitch_detection"): ...` - emits stage_start on
        enter and stage_end (with elapsed_ms) on exit. Synchronous; times with
        perf_counter. Emits stage_end even if the body raises, then re-raises.
        """
        return _StageTimer(self, name, meta)

    # -------------------------------------------------------------- read-side
    def snapshot(self, last: Optional[int] = None) -> List[Dict[str, Any]]:
        """A JSON-safe copy of the current ring (or its last N events), in
        order. Taken under the lock so it never tears against a live emit.
        """
        with self._lock:
            events = list(self._ring)
        if last is not None:
            events = events[-last:]
        return [e.to_dict() for e in events]

    def memory_samples(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._mem_ring]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            resident = len(self._ring)
            emitted = self._seq
            evicted = self._evicted
        return {
            "session_id": self._session_id,
            "uptime_s": time.perf_counter() - self._t0_mono,
            "emitted": emitted,
            "resident": resident,
            "evicted": evicted,
            "webhook_sent": self._webhook_sent,
            "webhook_failed": self._webhook_failed,
            "webhook_dropped": self._webhook_dropped,
            "stream_failed": self._stream_failed,
        }

    # ------------------------------------------------------------- lifecycle
    def stop(self, replay_path: Optional[Path] = None) -> Optional[Path]:
        """Stop background threads and optionally dump a JSON replay of the
        resident ring. Idempotent-ish; safe to call once at run end.
        Returns the replay path if one was written.
        """
        self._stop_flag.set()
        if self._worker is not None:
            self._worker.join(timeout=max(self._webhook_timeout_s * 2, 3.0))
        if self._mem_thread is not None:
            self._mem_thread.join(timeout=max(self._mem_interval_s * 2, 3.0))
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if replay_path is not None:
            return self._write_replay(Path(replay_path))
        return None

    def _write_replay(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            events = list(self._ring)
            mem = list(self._mem_ring)
        blob = {
            "session_id": self._session_id,
            "started_wall": self._t0_wall,
            "stats": self.stats(),
            "events": [e.to_dict() for e in events],
            "memory_samples": [e.to_dict() for e in mem],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, default=str, indent=2)
        return path

    # ------------------------------------------------------- webhook worker
    def _drain_webhook(self) -> None:
        """Single daemon thread. Blocks on the queue, POSTs each event with a
        timeout, counts successes/failures. Never raises out of the thread.
        Uses stdlib urllib so Window_Pane pulls in no new dependency.
        """
        import urllib.request  # local import: only touched if a webhook is set
        assert self._webhook_q is not None
        while not (self._stop_flag.is_set() and self._webhook_q.empty()):
            try:
                event = self._webhook_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                data = json.dumps(event.to_dict(), default=str).encode("utf-8")
                req = urllib.request.Request(
                    self._webhook_url, data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=self._webhook_timeout_s).close()
                self._webhook_sent += 1
            except Exception:
                self._webhook_failed += 1
            finally:
                self._webhook_q.task_done()

    # ------------------------------------------------------- memory sampler
    def _start_memory_sampling(self) -> None:
        try:
            import psutil  # noqa: F401
        except Exception:
            # Honest degradation: no psutil => no sampling, one clear note,
            # rather than a silent half-feature or a forced dependency.
            self.emit(
                EVENT_MEMORY_SAMPLE, stage="",
                note="memory sampling requested but psutil is not installed; disabled",
                rss_mb=None,
            )
            return
        self._mem_thread = threading.Thread(
            target=self._sample_memory, name=f"window-pane-mem-{self._session_id}",
            daemon=True,
        )
        self._mem_thread.start()

    def _sample_memory(self) -> None:
        import os
        import psutil
        proc = psutil.Process(os.getpid())
        while not self._stop_flag.wait(self._mem_interval_s):
            try:
                rss_mb = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                continue
            with self._lock:
                self._seq += 1
                event = PaneEvent(
                    seq=self._seq,
                    t_wall=time.time(),
                    t_ms=(time.perf_counter() - self._t0_mono) * 1000.0,
                    stage="",
                    event_type=EVENT_MEMORY_SAMPLE,
                    payload={"rss_mb": round(rss_mb, 2)},
                )
                self._mem_ring.append(event)

    def __repr__(self) -> str:
        s = self.stats()
        return (f"WindowPane(session={s['session_id']}, emitted={s['emitted']}, "
                f"resident={s['resident']}, evicted={s['evicted']})")


class _StageTimer:
    """Context manager returned by WindowPane.stage(). Synchronous timing;
    emits stage_end (with elapsed_ms and ok=False) even when the body raises,
    then lets the exception propagate."""

    def __init__(self, pane: WindowPane, name: str, meta: Dict[str, Any]):
        self._pane = pane
        self._name = name
        self._meta = meta
        self._t_enter = 0.0

    def __enter__(self) -> "_StageTimer":
        self._t_enter = time.perf_counter()
        self._pane.emit(EVENT_STAGE_START, stage=self._name, **self._meta)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = (time.perf_counter() - self._t_enter) * 1000.0
        self._pane.emit(
            EVENT_STAGE_END, stage=self._name,
            elapsed_ms=round(elapsed_ms, 3),
            ok=(exc_type is None),
            error=(None if exc_type is None else f"{exc_type.__name__}: {exc}"),
        )
        return False  # never suppress the pipeline's own exceptions


__all__ = [
    "WindowPane",
    "PaneEvent",
    "EVENT_WITNESS",
    "EVENT_ONSET",
    "EVENT_DRUM_HIT",
    "EVENT_NOTE",
    "EVENT_CONTRADICTION",
    "EVENT_TEMPO_VERDICT",
    "EVENT_STAGE_START",
    "EVENT_STAGE_END",
    "EVENT_MEMORY_SAMPLE",
]
