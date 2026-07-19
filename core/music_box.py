# =================================================================
# MODULE: core/music_box.py
# Music_Box (GRIMLOCK_6.0_DESIGN_DECISIONS.md §7 Core): append-only
# forensic ledger. Passive logger - it records every hypothesis/
# transformation/decision for backward traceability; it does NOT
# arbitrate (that's Epistemic's job) and does NOT decide anything.
#
# Deliberately lean compared to Symphony's 5.x version: no arena/
# memory-event/veto-gate-specific methods, since those tie to 5.x
# concepts (StageInput/Output, ValidationGate, VetoReason) that don't
# exist in 6.0's linear Conductor. decision_type is a free-form string,
# not a closed enum - every caller describing its own decision in its
# own words is the normal case, not something to validate against a
# fixed vocabulary (Symphony's earlier enum-validation attempt
# manufactured a fake "error_occurred" record for every non-enum
# string, burying real errors under noise).
# =================================================================

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ForensicRecord:
    """One immutable ledger entry. Once appended, never edited or
    removed - a stage that changes its mind logs a NEW record."""
    session_id: str
    timestamp_utc: datetime
    stage_name: str
    decision_type: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    reasoning: str
    reversible: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_utc"] = self.timestamp_utc.isoformat()
        return d


def _sanitize(state: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort JSON-safety for arbitrary before/after state dicts -
    large arrays/objects get a short description instead of failing to
    serialize or bloating the ledger."""
    sanitized: Dict[str, Any] = {}
    for key, value in state.items():
        if hasattr(value, "shape"):
            sanitized[key] = f"<array shape={value.shape}>"
        elif isinstance(value, (bytes, bytearray)):
            sanitized[key] = f"<bytes length={len(value)}>"
        elif isinstance(value, (list, tuple)) and len(value) > 100:
            sanitized[key] = f"<sequence length={len(value)}>"
        else:
            try:
                json.dumps(value, default=str)
                sanitized[key] = value
            except (TypeError, ValueError):
                sanitized[key] = str(value)
    return sanitized


class MusicBox:
    """Append-only session ledger. Never deletes or mutates a record -
    `flush()` writes the buffer to disk (if a log_path is configured)
    and clears the in-memory buffer, but `get_session_logs()` never
    silently discards anything: with no log_path, it just reads back
    the buffer directly instead of calling flush() first."""

    def __init__(self, log_path: Optional[Path] = None, buffer_size: int = 100):
        self.log_path = Path(log_path) if log_path else None
        self.buffer_size = buffer_size
        self._session_id = uuid.uuid4().hex[:8]
        self._session_start = datetime.now()
        self._buffer: List[ForensicRecord] = []
        self._flushed_count = 0

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(
            self,
            stage_name: str,
            decision_type: str,
            before_state: Dict[str, Any],
            after_state: Dict[str, Any],
            reasoning: str,
            reversible: bool = True,
    ) -> None:
        record = ForensicRecord(
            session_id=self._session_id,
            timestamp_utc=datetime.now(),
            stage_name=stage_name,
            decision_type=str(decision_type),
            before_state=_sanitize(before_state),
            after_state=_sanitize(after_state),
            reasoning=reasoning,
            reversible=reversible,
        )
        self._buffer.append(record)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Writes the buffer to disk if a log_path is configured. With
        no log_path, this is a no-op - there is nowhere to flush TO, so
        the buffer is left alone rather than silently discarded."""
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            for record in self._buffer:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")
        self._flushed_count += len(self._buffer)
        self._buffer.clear()

    def get_session_logs(self) -> List[Dict[str, Any]]:
        """Every record logged this session, in order. Reads the
        in-memory buffer directly - does NOT flush first, so calling
        this never destroys unflushed records."""
        return [r.to_dict() for r in self._buffer]

    def query_entries(
            self,
            decision_type: Optional[str] = None,
            stage_name: Optional[str] = None,
            limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        records = self.get_session_logs()
        if decision_type is not None:
            records = [r for r in records if r["decision_type"] == decision_type]
        if stage_name is not None:
            records = [r for r in records if r["stage_name"] == stage_name]
        return records[:limit] if limit else records

    def get_stats(self) -> Dict[str, Any]:
        return {
            "session_id": self._session_id,
            "buffered": len(self._buffer),
            "flushed": self._flushed_count,
            "session_duration_seconds": (datetime.now() - self._session_start).total_seconds(),
        }

    def __repr__(self) -> str:
        return f"MusicBox(session={self._session_id}, buffered={len(self._buffer)}, flushed={self._flushed_count})"


__all__ = ["MusicBox", "ForensicRecord"]
