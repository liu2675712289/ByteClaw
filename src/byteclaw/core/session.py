"""Workspace-scoped conversation session persistence."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


SESSION_ROOT = ".byteclaw/session"
SESSION_FILE = "session.json"
SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"
MAX_SESSION_CONTEXT = 7000
MAX_TURN_CONTENT = 4000

_RECENT_FILE_LIMIT = 30
_RECENT_TURN_LIMIT = 10
_CONTEXT_FILE_PATH_LIMIT = 120
_CONTEXT_TURN_SUMMARY_LIMIT = 250
_IGNORED_WORKSPACE_DIRECTORIES = {
    ".byteclaw",
    ".git",
    ".pytest_cache",
    "__pycache__",
}


def load_or_create_session(workspace: Path) -> dict:
    """Load a workspace session, creating and persisting one when absent."""

    path = _session_root(workspace) / SESSION_FILE
    if path.is_file():
        session = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(session, dict):
            raise ValueError("Session file must contain a JSON object")
        return session

    timestamp = _utc_now()
    session = {
        "session_id": str(uuid4()),
        "turn_index": 0,
        "recent_turns": [],
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    return save_session(workspace, session)


def append_user_turn(session: dict, content: str) -> int:
    """Append a user turn and return its monotonically increasing number."""

    turn = int(session.get("turn_index", 0)) + 1
    timestamp = _utc_now()
    session["turn_index"] = turn
    session.setdefault("recent_turns", []).append(
        {
            "turn": turn,
            "role": "user",
            "content": _limit_text(content, MAX_TURN_CONTENT),
            "timestamp": timestamp,
        }
    )
    session["updated_at"] = timestamp
    return turn


def append_assistant_turn(
    session: dict,
    *,
    turn: int,
    route: str,
    content: str,
    summary: str = "",
) -> None:
    """Append an assistant response associated with a user turn."""

    if route not in {"chat", "workflow"}:
        raise ValueError("route must be 'chat' or 'workflow'")

    session["turn_index"] = max(int(session.get("turn_index", 0)), turn)
    session.setdefault("recent_turns", []).append(
        {
            "turn": turn,
            "role": "assistant",
            "route": route,
            "content": _limit_text(content, MAX_TURN_CONTENT),
            "summary": _limit_text(summary, MAX_TURN_CONTENT),
        }
    )
    session["updated_at"] = _utc_now()


def save_session(workspace: Path, session: dict) -> dict:
    """Persist session JSON and its human-readable Markdown summary."""

    root = _session_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_now()
    session.setdefault("session_id", str(uuid4()))
    session.setdefault("turn_index", 0)
    session.setdefault("recent_turns", [])
    session.setdefault("created_at", timestamp)
    session["updated_at"] = timestamp

    _write_text_atomic(
        root / SESSION_FILE,
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text_atomic(
        root / SESSION_SUMMARY_FILE,
        _build_session_summary(session),
    )
    return session


def build_session_context(workspace: Path, session: dict | None = None) -> str:
    """Build bounded session context for intent and lightweight chat nodes."""

    workspace = Path(workspace).expanduser().resolve()
    current_session = (
        session if session is not None else load_or_create_session(workspace)
    )
    files = _recent_workspace_files(workspace)
    turns = current_session.get("recent_turns", [])[-_RECENT_TURN_LIMIT:]

    lines = [
        f"Session ID: {current_session.get('session_id', '')}",
        f"Turn index: {current_session.get('turn_index', 0)}",
        "",
        "Workspace files (30 most recent):",
    ]
    if files:
        lines.extend(
            f"- {_limit_text(path, _CONTEXT_FILE_PATH_LIMIT)}"
            for path in files
        )
    else:
        lines.append("- (none)")

    lines.extend(["", "Recent conversation (10 most recent turns):"])
    if turns:
        for item in turns:
            role = str(item.get("role", "unknown"))
            route = item.get("route")
            role_label = f"{role}/{route}" if route else role
            summary = item.get("summary") or item.get("content", "")
            lines.append(
                f"- Turn {item.get('turn', '?')} [{role_label}]: "
                f"{_limit_text(_compact_text(summary), _CONTEXT_TURN_SUMMARY_LIMIT)}"
            )
    else:
        lines.append("- (none)")

    return "\n".join(lines)[:MAX_SESSION_CONTEXT]


def _session_root(workspace: Path) -> Path:
    return Path(workspace).expanduser().resolve() / SESSION_ROOT


def _recent_workspace_files(workspace: Path) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for current, directories, filenames in os.walk(workspace):
        directories[:] = [
            name
            for name in directories
            if name not in _IGNORED_WORKSPACE_DIRECTORIES
        ]
        current_path = Path(current)
        for filename in filenames:
            path = current_path / filename
            try:
                modified_at = path.stat().st_mtime_ns
                relative = path.relative_to(workspace).as_posix()
            except (OSError, ValueError):
                continue
            candidates.append((modified_at, relative))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [path for _, path in candidates[:_RECENT_FILE_LIMIT]]


def _build_session_summary(session: dict) -> str:
    lines = [
        "# ByteClaw Session Summary",
        "",
        f"- Session ID: {session.get('session_id', '')}",
        f"- Turn index: {session.get('turn_index', 0)}",
        f"- Updated at: {session.get('updated_at', '')}",
        "",
        "## Recent conversation",
    ]
    turns = session.get("recent_turns", [])[-_RECENT_TURN_LIMIT:]
    if not turns:
        lines.extend(["", "No conversation turns recorded."])
    for item in turns:
        role = str(item.get("role", "unknown")).title()
        route = item.get("route")
        route_label = f" ({route})" if route else ""
        summary = item.get("summary") or item.get("content", "")
        lines.extend(
            [
                "",
                f"### Turn {item.get('turn', '?')} - {role}{route_label}",
                "",
                str(summary),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _compact_text(value: object) -> str:
    return " ".join(str(value).split())


def _limit_text(value: object, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
