"""Layered memory assembly for ByteClaw graph nodes."""

import json
from typing import Any

from byteclaw.core.state import RuntimeState
from byteclaw.graph.state import LayeredMemory

RULES_LAYER = {
    "scope": "workspace",
    "storage": "internal",
    "rules": [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, and HISTORY_SUMMARY.md as compressed history.",
        "Do not expose memory write tools to agents; layered memory is assembled by the runtime.",
    ],
}


def _short_text(text: Any, limit: int) -> str:
    """Return text bounded by ``limit`` characters with an ellipsis if trimmed."""

    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    if limit <= 3:
        return "." * max(limit, 0)
    return f"{value[: limit - 3]}..."


def _trim_handoffs(handoffs: list[dict]) -> list[dict]:
    """Keep copies of the six most recent agent handoffs."""

    return [dict(handoff) for handoff in handoffs[-6:]]


def _read_memory_file(runtime: RuntimeState, name: str) -> dict[str, Any]:
    path = runtime.workspace / name
    if not path.is_file():
        return {"path": name, "exists": False, "content": ""}
    return {
        "path": name,
        "exists": True,
        "content": path.read_text(encoding="utf-8"),
    }


def read_notepad(runtime: RuntimeState) -> dict[str, Any]:
    """Read durable notes from ``NOTEPAD.md`` when present."""

    return _read_memory_file(runtime, "NOTEPAD.md")


def read_history_summary(runtime: RuntimeState) -> dict[str, Any]:
    """Read compressed history from ``HISTORY_SUMMARY.md`` when present."""

    return _read_memory_file(runtime, "HISTORY_SUMMARY.md")


def _session_value(state: dict, name: str, fallback: str, default: Any) -> Any:
    if name in state:
        return state[name]
    session = state.get("session_context") or state.get("session") or {}
    if not isinstance(session, dict):
        return default
    return session.get(name, session.get(fallback, default))


def _prompt_sources(sources: list[Any]) -> list[dict[str, str]]:
    trimmed: list[dict[str, str]] = []
    for source in sources:
        if isinstance(source, str):
            trimmed.append({"title": "", "url": source})
        elif isinstance(source, dict):
            trimmed.append(
                {
                    "title": str(source.get("title", "")),
                    "url": str(source.get("url", "")),
                }
            )
    return trimmed


def build_layered_memory(state: dict, *, node: str = "graph") -> LayeredMemory:
    """Assemble fixed rules, current task state, and compressed durable history."""

    runtime = state["runtime"]
    notepad = read_notepad(runtime)
    history = read_history_summary(runtime)
    history_summary = history.get("content", "") or state.get(
        "history_summary", ""
    )

    working_memory = {
        "node": node,
        "task": state.get("task", ""),
        "session_id": _session_value(state, "session_id", "id", ""),
        "session_turn": _session_value(state, "session_turn", "turn", 0),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "research_notes": _short_text(
            state.get("research_notes", ""), 1600
        ),
        "sources": _prompt_sources(state.get("sources", [])),
        "agent_handoffs": _trim_handoffs(
            state.get("agent_handoffs", [])
        ),
        "code_agent_summary": _short_text(
            state.get("code_agent_summary", ""), 1000
        ),
        "verifier_summary": _short_text(
            state.get("verifier_summary", ""), 1000
        ),
        "last_error": _short_text(state.get("last_error", ""), 1400),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
    }

    history_summary_store = {
        "history_path": "HISTORY_SUMMARY.md",
        "history_exists": history.get("exists", False),
        "history_summary": _short_text(history_summary, 2200),
        "notepad_path": "NOTEPAD.md",
        "notepad_exists": notepad.get("exists", False),
        "notepad": _short_text(notepad.get("content", ""), 1800),
        "context_summary": _short_text(
            state.get("context_summary", ""), 1600
        ),
        "compression_events": state.get("compression_events", [])[-3:],
    }

    return {
        "rules": dict(RULES_LAYER),
        "working_memory": working_memory,
        "history_summary_store": history_summary_store,
    }


def format_layered_memory_for_prompt(memory: LayeredMemory) -> str:
    """Format layered memory as readable JSON for model prompts."""

    return json.dumps(memory, ensure_ascii=False, indent=2, default=str)


def memory_event(memory: LayeredMemory, *, node: str) -> dict[str, Any]:
    """Return the normalized stream event for one layered-memory snapshot."""

    return {"type": "memory", "node": node, "memory": memory}
