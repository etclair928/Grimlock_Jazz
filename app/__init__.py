# =================================================================
# MODULE: app/__init__.py
# The front end's library layer (GRIMLOCK_6.0_FRONTEND_DESIGN.md).
#
# Everything here is UI-FREE ON PURPOSE. These modules answer the questions a
# front end needs answered - "what is this recording", "is this transcription
# trustworthy" - as functions returning data, with no tkinter import anywhere
# near them. That keeps them testable, keeps them usable from the CLI, and
# means the GUI is a thin thing on top rather than the only way to reach any
# of it. Four of the probes this session leaned on were scratch scripts that
# printed to stdout and were rewritten several times; this is where that stops.
# =================================================================

from app.probe import ProbeResult, probe_audio
from app.diagnostics import (
    HealthReport, HealthCheck, diagnose, PASS, FAIL, INFO,
)
from app.runner import (
    PaneMusicBox, RunConfig, RunStatus, TranscriptionRunner,
)

__all__ = [
    "ProbeResult", "probe_audio",
    "HealthReport", "HealthCheck", "diagnose", "PASS", "FAIL", "INFO",
    "PaneMusicBox", "RunConfig", "RunStatus", "TranscriptionRunner",
]
