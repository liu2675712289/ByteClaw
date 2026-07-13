"""Execution tracing for ByteClaw runs."""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, message_to_dict


VALID_TRACE_MODES = {"full", "off"}
_TRACE_MODE_ALIASES = {
    "on": "full",
    "enabled": "full",
    "true": "full",
    "disabled": "off",
    "false": "off",
    "none": "off",
}
_TRACE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def normalize_trace_mode(mode: str | None) -> str:
    """Normalize a trace mode, defaulting invalid values to ``full``."""

    normalized = (mode or "").strip().lower()
    normalized = _TRACE_MODE_ALIASES.get(normalized, normalized)
    return normalized if normalized in VALID_TRACE_MODES else "full"


class TraceRecorder:
    """Record workflow events and produce compact execution summaries."""

    def __init__(self, runtime: Any, task: str = "") -> None:
        self.workspace = Path(runtime.workspace).expanduser().resolve()
        self.mode = normalize_trace_mode(runtime.trace_mode)
        self.trace_id = _trace_id(runtime.trace_id)
        self.root = (
            self.workspace / ".byteclaw" / "traces" / self.trace_id
        )
        self.task = task
        self.node_visits: dict[str, int] = {}
        self.tool_calls = 0
        self.failed_tool_calls = 0
        self.approval_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0
        self.started_at: str | None = None
        self._timeline_head: list[dict[str, Any]] = []
        self._timeline_tail: deque[dict[str, Any]] = deque(maxlen=80)
        self._event_count = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def start(
        self,
        inputs: Mapping[str, Any],
        *,
        resumed: bool = False,
        resume_event: Any = None,
    ) -> dict[str, Any] | None:
        """Record the ``run_start`` event for this trace."""

        if not self.enabled:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        events_path = self.root / "events.jsonl"
        self._reset_statistics()
        if resumed and events_path.is_file():
            self._load_existing_events(events_path)
        else:
            events_path.write_text("", encoding="utf-8")

        self.task = self.task or str(inputs.get("task", ""))
        timestamp = _utc_now()
        if self.started_at is None:
            self.started_at = timestamp
        event = {
            "type": "run_start",
            "timestamp": timestamp,
            "trace_id": self.trace_id,
            "task": self.task,
            "resumed": resumed,
            "inputs": _jsonable(dict(inputs)),
        }
        if resume_event is not None:
            event["resume_event"] = _jsonable(resume_event)
        return self._record(event)

    def record_custom_event(self, event: Any) -> dict[str, Any] | None:
        """Record a custom event and update tool and workflow statistics."""

        if not self.enabled:
            return None
        payload = _event_mapping(event)
        self._apply_statistics(payload)
        return self._record(payload)

    def record_graph_update(self, event: Any) -> dict[str, Any] | None:
        """Record a graph update and increment visited node counters."""

        if not self.enabled:
            return None
        payload = _event_mapping(event)
        payload.setdefault("type", "graph_update")
        nodes = _graph_nodes(payload)
        if nodes:
            payload.setdefault("nodes", nodes)
        self._apply_statistics(payload)
        return self._record(payload)

    def end(
        self,
        *,
        status: str,
        latest_node: str | None,
        final_state: Any,
    ) -> dict[str, Any] | None:
        """End tracing and write ``trace.json`` and ``timeline.md``."""

        if not self.enabled:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        ended_at = _utc_now()
        if self.started_at is None:
            self.started_at = ended_at
        self._record(
            {
                "type": "run_end",
                "timestamp": ended_at,
                "status": status,
                "latest_node": latest_node,
                "final_state": _jsonable(final_state),
            }
        )

        timeline_head = list(self._timeline_head)
        timeline_tail = list(self._timeline_tail)
        timeline_omitted = max(
            0,
            self._event_count - len(timeline_head) - len(timeline_tail),
        )
        payload = {
            "trace_id": self.trace_id,
            "task": self.task,
            "status": status,
            "latest_node": latest_node,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_ms": _duration_ms(self.started_at, ended_at),
            "node_visits": dict(self.node_visits),
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "approval_count": self.approval_count,
            "checkpoint_count": self.checkpoint_count,
            "handoff_count": self.handoff_count,
            "timeline_head": timeline_head,
            "timeline_tail": timeline_tail,
            "timeline_omitted": timeline_omitted,
        }
        _write_text_atomic(
            self.root / "trace.json",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        _write_text_atomic(
            self.root / "timeline.md",
            _build_timeline_markdown(payload),
        )
        return payload

    def _record(self, event: Mapping[str, Any]) -> dict[str, Any]:
        payload = _jsonable(dict(event))
        payload.setdefault("timestamp", _utc_now())
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False))
            file.write("\n")
        self._track_timeline(payload)
        return payload

    def _track_timeline(self, event: dict[str, Any]) -> None:
        if len(self._timeline_head) < 20:
            self._timeline_head.append(event)
        else:
            self._timeline_tail.append(event)
        self._event_count += 1

    def _apply_statistics(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "tool_call":
            self.tool_calls += 1
        elif event_type == "tool_result":
            if _result_field(event, "ok") is False:
                self.failed_tool_calls += 1
            if bool(_result_field(event, "requires_approval")):
                self.approval_count += 1
        elif event_type == "handoff":
            self.handoff_count += 1
        elif event_type == "checkpoint_saved":
            self.checkpoint_count += 1

        if event_type in {"graph_update", "node_output"}:
            for node in _graph_nodes(event):
                self.node_visits[node] = self.node_visits.get(node, 0) + 1

    def _load_existing_events(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if self.started_at is None and event.get("type") == "run_start":
                self.started_at = event.get("timestamp")
            self._apply_statistics(event)
            self._track_timeline(event)

    def _reset_statistics(self) -> None:
        self.node_visits = {}
        self.tool_calls = 0
        self.failed_tool_calls = 0
        self.approval_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0
        self.started_at = None
        self._timeline_head = []
        self._timeline_tail = deque(maxlen=80)
        self._event_count = 0


def _trace_id(value: Any) -> str:
    if value is None or not str(value).strip():
        return f"trace-{uuid4().hex[:12]}"
    trace_id = str(value).strip()
    if not _TRACE_ID_PATTERN.fullmatch(trace_id) or trace_id in {".", ".."}:
        raise ValueError(
            "trace_id must contain only letters, numbers, '.', '_', or '-'"
        )
    return trace_id


def _event_mapping(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    return {"type": "custom", "value": _jsonable(event)}


def _graph_nodes(event: Mapping[str, Any]) -> list[str]:
    node = event.get("node")
    if node:
        return [str(node)]
    nodes = event.get("nodes")
    if isinstance(nodes, list):
        return [str(item) for item in nodes if item]
    ignored = {"type", "timestamp", "output", "nodes"}
    return [str(key) for key in event if key not in ignored]


def _result_field(event: Mapping[str, Any], field: str) -> Any:
    if field in event:
        return event[field]
    result = event.get("result")
    if isinstance(result, Mapping):
        return result.get(field)
    return getattr(result, field, None)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return message_to_dict(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


def _duration_ms(started_at: str, ended_at: str) -> int:
    try:
        started = datetime.fromisoformat(started_at)
        ended = datetime.fromisoformat(ended_at)
    except ValueError:
        return 0
    return max(0, round((ended - started).total_seconds() * 1000))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _build_timeline_markdown(payload: Mapping[str, Any]) -> str:
    node_visits = payload.get("node_visits", {})
    node_lines = [
        f"- `{node}`: {visits}"
        for node, visits in node_visits.items()
    ] or ["- _(none)_"]
    event_lines = [
        _timeline_line(event) for event in payload.get("timeline_head", [])
    ]
    omitted = int(payload.get("timeline_omitted", 0))
    if omitted:
        event_lines.append(f"- _... {omitted} event(s) omitted ..._")
    event_lines.extend(
        _timeline_line(event) for event in payload.get("timeline_tail", [])
    )
    if not event_lines:
        event_lines.append("- _(no events)_")

    return "\n".join(
        [
            "# ByteClaw Execution Trace",
            "",
            f"- Trace ID: `{payload.get('trace_id', '')}`",
            f"- Task: {payload.get('task', '') or '_(not recorded)_'}",
            f"- Status: {payload.get('status', 'unknown')}",
            f"- Latest node: {payload.get('latest_node') or '_(not recorded)_'}",
            f"- Started at: {payload.get('started_at', 'unknown')}",
            f"- Ended at: {payload.get('ended_at', 'unknown')}",
            f"- Duration: {payload.get('duration_ms', 0)} ms",
            "",
            "## Statistics",
            "",
            f"- Tool calls: {payload.get('tool_calls', 0)}",
            f"- Failed tool calls: {payload.get('failed_tool_calls', 0)}",
            f"- Approvals: {payload.get('approval_count', 0)}",
            f"- Checkpoints: {payload.get('checkpoint_count', 0)}",
            f"- Handoffs: {payload.get('handoff_count', 0)}",
            "",
            "## Node visits",
            "",
            *node_lines,
            "",
            "## Timeline",
            "",
            *event_lines,
            "",
        ]
    )


def _timeline_line(event: Any) -> str:
    if not isinstance(event, Mapping):
        return f"- {event}"
    timestamp = event.get("timestamp", "unknown")
    event_type = event.get("type", "event")
    details = []
    for field in ("node", "name", "status", "latest_node"):
        if event.get(field) is not None:
            details.append(f"{field}={event[field]}")
    suffix = f" — {', '.join(details)}" if details else ""
    return f"- `{timestamp}` **{event_type}**{suffix}"
