# =================================================================
# MODULE: tests/test_app_runner.py
# Pins app/runner.py and Window Pane's stream sink - the live-telemetry path.
#
# WHAT IS ACTUALLY AT RISK HERE. Not the widgets. The three things that would
# fail silently and look fine: a tee'd MusicBox that quietly stops recording
# forensically; a pane sink that takes down the run it observes; and a tail
# that parses a half-written line as truncated JSON and loses events. Each of
# those is a bug you would only notice much later, on a run that cost an hour.
# =================================================================

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.runner import PaneMusicBox, RunConfig, TranscriptionRunner  # noqa: E402
from core import WindowPane                                          # noqa: E402


# --- Window Pane's stream sink -------------------------------------------

def test_events_are_readable_while_the_pane_is_still_live():
    """The whole point of the sink: a viewer in ANOTHER PROCESS reads events
    as they happen. Line buffering is what makes that true, and a 4KB buffer
    would make this test fail while everything still 'worked'."""
    path = os.path.join(tempfile.mkdtemp(), "pane.jsonl")
    pane = WindowPane(stream_path=path)
    pane.emit("decision", "rhythm_engine", decision_type="tempo", reasoning="x")
    pane.emit("decision", "quantization", decision_type="snap", reasoning="y")
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    assert len(rows) == 2
    assert rows[0]["stage"] == "rhythm_engine"
    assert rows[1]["payload"]["decision_type"] == "snap"
    pane.stop()


def test_a_broken_sink_never_takes_down_the_run():
    """Window_Pane's own law: a telemetry layer that can crash the run it
    observes is worse than no telemetry."""
    pane = WindowPane(stream_path=os.path.join(tempfile.mkdtemp(), "p.jsonl"))
    pane._stream.close()                      # simulate the sink dying mid-run
    pane.emit("decision", "x", decision_type="d", reasoning="r")
    assert pane.stats()["stream_failed"] >= 1
    assert pane.stats()["emitted"] >= 1       # the ring still took it
    pane.stop()


def test_an_unopenable_sink_is_not_fatal():
    pane = WindowPane(stream_path=os.path.join(tempfile.mkdtemp(), "no", "dir", "x", "p.jsonl"))
    pane.emit("decision", "x", decision_type="d", reasoning="r")
    assert pane.stats()["emitted"] >= 1
    pane.stop()


# --- the tee --------------------------------------------------------------

def test_the_tee_still_records_forensically():
    """A live feed must not cost the permanent ledger. Music_Box keeps
    everything forever; the pane keeps a bounded window. Both, not either."""
    path = os.path.join(tempfile.mkdtemp(), "pane.jsonl")
    pane = WindowPane(stream_path=path)
    box = PaneMusicBox(buffer_size=100, pane=pane)
    box.log_decision("rhythm_engine", "tempo", {}, {"bpm": 70}, "because")
    assert len(box.get_session_logs()) == 1
    with open(path, encoding="utf-8") as fh:
        assert len([l for l in fh if l.strip()]) == 1
    pane.stop()


def test_the_tee_works_without_a_pane_at_all():
    box = PaneMusicBox(buffer_size=100, pane=None)
    box.log_decision("x", "y", {}, {}, "z")
    assert len(box.get_session_logs()) == 1


def test_the_tee_carries_the_reasoning_string():
    """The reasoning is the part worth watching - a feed of decision TYPES
    would be a progress bar with extra steps."""
    path = os.path.join(tempfile.mkdtemp(), "pane.jsonl")
    pane = WindowPane(stream_path=path)
    box = PaneMusicBox(buffer_size=100, pane=pane)
    box.log_decision("separation_engine", "separation_skipped", {},
                     {"stem": "piano"}, "Skipped Demucs: no kit, nothing swells")
    with open(path, encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    assert "Skipped Demucs" in row["payload"]["reasoning"]
    pane.stop()


# --- the tail -------------------------------------------------------------

def _runner_with_stream(text: str) -> TranscriptionRunner:
    work = tempfile.mkdtemp()
    runner = TranscriptionRunner(RunConfig(audio_path="a.wav", out_stem="s"),
                                 work_dir=work)
    with open(runner.stream_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return runner


def test_the_tail_returns_only_what_is_new():
    runner = _runner_with_stream('{"seq":1,"event_type":"a","stage":"s","payload":{}}\n')
    assert len(runner.poll()) == 1
    assert runner.poll() == []                 # nothing new the second time
    with open(runner.stream_path, "a", encoding="utf-8") as fh:
        fh.write('{"seq":2,"event_type":"b","stage":"s","payload":{}}\n')
    events = runner.poll()
    assert [e["seq"] for e in events] == [2]


def test_a_half_written_line_is_left_for_the_next_poll():
    """The producer writes while the reader reads. Parsing a partial line
    would drop the event entirely - it must be held, not guessed at."""
    runner = _runner_with_stream('{"seq":1,"event_type":"a","stage":"s","payload":{}}\n'
                                 '{"seq":2,"event_ty')
    assert [e["seq"] for e in runner.poll()] == [1]
    with open(runner.stream_path, "a", encoding="utf-8") as fh:
        fh.write('pe":"b","stage":"s","payload":{}}\n')
    assert [e["seq"] for e in runner.poll()] == [2]


def test_polling_a_stream_that_does_not_exist_yet_is_fine():
    work = tempfile.mkdtemp()
    runner = TranscriptionRunner(RunConfig(audio_path="a.wav", out_stem="s"),
                                 work_dir=work)
    assert runner.poll() == []


# --- config ---------------------------------------------------------------

def test_config_round_trips_through_json():
    """It has to cross a process boundary intact, guided fields included."""
    cfg = RunConfig(audio_path="a.wav", out_stem="s",
                    guided_separation="solo_piano", guided_tempo_bpm=70.0,
                    guided_time_signature=[6, 4], guided_key="Bbm")
    back = RunConfig.from_json(cfg.to_json())
    assert back.guided_separation == "solo_piano"
    assert back.guided_time_signature == [6, 4]
    assert back.guided_key == "Bbm"
    assert back.paths()["pkl"].endswith("s.pkl")


def test_cancel_marks_the_run_cancelled_not_finished_successfully():
    """A half-finished transcription that looks finished is worse than none."""
    work = tempfile.mkdtemp()
    runner = TranscriptionRunner(RunConfig(audio_path="a.wav", out_stem="s"),
                                 work_dir=work)
    runner.cancel()
    assert runner.status.cancelled and runner.status.finished
    assert runner.status.result == {}
