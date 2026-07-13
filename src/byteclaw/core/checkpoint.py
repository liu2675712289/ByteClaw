"""Persist and restore ByteClaw workflow checkpoints."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)


VALID_CHECKPOINT_MODES = {"light", "strict", "off"}
_CHECKPOINT_VERSION = 1
_SNAPSHOT_REPO_NAME = "snapshot-repo"
_RESUME_STATE_FIELDS = {
    "acceptance_criteria",
    "agent_handoffs",
    "attempts",
    "code_agent_summary",
    "compression_events",
    "context_next_node",
    "context_should_compress",
    "context_summary",
    "context_token_count",
    "context_token_limit",
    "final_answer",
    "history_summary",
    "last_actor_summary",
    "last_error",
    "max_attempts",
    "messages",
    "passed",
    "plan_summary",
    "research_notes",
    "sources",
    "task",
    "todos",
    "verification_checks",
    "verification_commands",
    "verification_results",
    "verifier_summary",
}


def normalize_checkpoint_mode(mode: str | None) -> str:
    """Normalize a checkpoint mode, defaulting invalid values to ``light``."""

    normalized = (mode or "").strip().lower()
    return normalized if normalized in VALID_CHECKPOINT_MODES else "light"


def workspace_manifest(workspace: Path) -> list[str]:
    """Return a stable manifest of workspace files outside checkpoint storage."""

    workspace = workspace.expanduser().resolve()
    manifest: list[str] = []
    for current, directories, filenames in os.walk(workspace):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if not _excluded_from_snapshot(
                (current_path / name).relative_to(workspace)
            )
        ]
        for filename in filenames:
            path = current_path / filename
            relative = path.relative_to(workspace)
            if not _excluded_from_snapshot(relative):
                manifest.append(relative.as_posix())
    return sorted(manifest)


def snapshot_workspace_git(
    workspace: Path, checkpoint_root: Path | None = None
) -> str | None:
    """Commit a workspace copy to an isolated Git repository and return its SHA."""

    workspace = workspace.expanduser().resolve()
    root = (
        checkpoint_root.expanduser().resolve()
        if checkpoint_root is not None
        else workspace / ".byteclaw" / "checkpoints"
    )
    root.mkdir(parents=True, exist_ok=True)
    repository = root / _SNAPSHOT_REPO_NAME
    repository.mkdir(parents=True, exist_ok=True)

    try:
        if not (repository / ".git").is_dir():
            _run_git(repository, "init", "--quiet")
        _run_git(repository, "config", "user.name", "ByteClaw Checkpoint")
        _run_git(
            repository,
            "config",
            "user.email",
            "checkpoint@byteclaw.local",
        )
        _copy_workspace_to_repository(workspace, repository)
        _run_git(repository, "add", "--all")
        _run_git(
            repository,
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            f"ByteClaw checkpoint {_utc_now()}",
        )
        return _run_git(repository, "rev-parse", "HEAD").stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def restore_workspace_git(
    workspace: Path,
    git_commit: str,
    checkpoint_root: Path | None = None,
) -> None:
    """Restore workspace files from an isolated checkpoint Git commit."""

    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", git_commit):
        raise ValueError("Invalid checkpoint Git commit")

    workspace = workspace.expanduser().resolve()
    root = (
        checkpoint_root.expanduser().resolve()
        if checkpoint_root is not None
        else workspace / ".byteclaw" / "checkpoints"
    )
    repository = root / _SNAPSHOT_REPO_NAME
    if not (repository / ".git").is_dir():
        raise FileNotFoundError(f"Checkpoint Git repository not found: {repository}")

    try:
        _run_git(repository, "reset", "--hard", "--quiet", git_commit)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Unable to restore checkpoint Git commit {git_commit}"
        ) from exc
    _copy_repository_to_workspace(repository, workspace)


def checkpoint_saved_event(
    payload: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    """Build the event emitted after a checkpoint is saved."""

    return {
        "type": "checkpoint_saved",
        "checkpoint_path": str(checkpoint_path),
        "mode": payload.get("mode"),
        "status": payload.get("status"),
        "latest_node": payload.get("latest_node"),
        "git_commit": payload.get("git_commit"),
    }


def resume_command(workspace: Path) -> str:
    """Return the CLI command used to resume a workspace."""

    workspace_text = str(workspace.expanduser().resolve())
    quoted_workspace = subprocess.list2cmdline([workspace_text])
    return f"byteclaw --resume {quoted_workspace}"


def build_recovery_markdown(payload: Mapping[str, Any]) -> str:
    """Build the human-readable recovery guide for a checkpoint payload."""

    workspace = Path(str(payload.get("workspace", ".")))
    manifest = payload.get("workspace_manifest", [])
    file_lines = []
    for item in manifest if isinstance(manifest, list) else []:
        if isinstance(item, Mapping):
            file_lines.append(f"- `{item.get('path', '')}`")
        else:
            file_lines.append(f"- `{item}`")
    if not file_lines:
        file_lines.append("- _(empty workspace)_")

    task = str(payload.get("task", "")) or "_(not recorded)_"
    git_commit = str(payload.get("git_commit") or "_(unavailable)_")
    latest_node = str(payload.get("latest_node") or "_(not recorded)_")
    return "\n".join(
        [
            "# ByteClaw Recovery",
            "",
            "## Checkpoint",
            "",
            f"- Task: {task}",
            f"- Status: {payload.get('status', 'unknown')}",
            f"- Mode: {payload.get('mode', 'light')}",
            f"- Latest node: {latest_node}",
            f"- Saved at: {payload.get('saved_at', 'unknown')}",
            f"- Git commit: `{git_commit}`",
            "",
            "## Workspace files",
            "",
            *file_lines,
            "",
            "## Resume",
            "",
            "```text",
            resume_command(workspace),
            "```",
            "",
        ]
    )


class CheckpointManager:
    """Save and restore workflow state for one workspace."""

    def __init__(self, runtime: Any, task: str = "") -> None:
        self.workspace = Path(runtime.workspace).expanduser().resolve()
        self.mode = normalize_checkpoint_mode(runtime.checkpoint_mode)
        self.root = self.workspace / ".byteclaw" / "checkpoints"
        self.task = task

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def save(
        self,
        state: Mapping[str, Any],
        *,
        status: str = "running",
        latest_node: str | None = None,
        event: Any = None,
    ) -> dict[str, Any] | None:
        """Persist state, workspace snapshot, metadata, and recovery guidance."""

        if not self.enabled:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        serializable_state = _jsonable(dict(state))
        if self.mode == "strict":
            events_path = self.root / "events.jsonl"
            events_path.touch(exist_ok=True)
            if event is not None:
                _append_jsonl(events_path, _jsonable(event))
            _write_json(self.root / "state.json", serializable_state)

        manifest = workspace_manifest(self.workspace)
        git_commit = snapshot_workspace_git(self.workspace, self.root)
        task = self.task or str(state.get("task", ""))
        payload = {
            "version": _CHECKPOINT_VERSION,
            "workspace": str(self.workspace),
            "task": task,
            "status": status,
            "mode": self.mode,
            "checkpoint_mode": self.mode,
            "latest_node": latest_node,
            "saved_at": _utc_now(),
            "git_commit": git_commit,
            "workspace_manifest": manifest,
            "state_summary": _jsonable(_state_summary(state)),
        }
        checkpoint_path = self.root / "checkpoint.json"
        _write_json(checkpoint_path, payload)
        (self.root / "RECOVERY.md").write_text(
            build_recovery_markdown(payload), encoding="utf-8"
        )
        return checkpoint_saved_event(payload, checkpoint_path)

    @classmethod
    def load_resume_inputs(
        cls,
        runtime: Any,
        task: str | None = None,
        max_attempts: int = 3,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load checkpoint state, restore files, and rebuild workflow inputs."""

        manager = cls(runtime, task or "")
        checkpoint_path = manager.root / "checkpoint.json"
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        git_commit = payload.get("git_commit")
        if git_commit:
            restore_workspace_git(manager.workspace, git_commit, manager.root)

        state_path = manager.root / "state.json"
        if payload.get("mode") == "strict" and state_path.is_file():
            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            saved_state = payload.get("state_summary", {})
        inputs = dict(saved_state) if isinstance(saved_state, Mapping) else {}
        inputs.pop("runtime", None)
        inputs["task"] = (
            task
            if task is not None
            else payload.get("task", inputs.get("task", ""))
        )
        inputs["runtime"] = runtime
        inputs["messages"] = _restore_messages(inputs.get("messages", []))
        inputs["attempts"] = int(inputs.get("attempts", 0))
        inputs["max_attempts"] = max_attempts

        resume_event = {
            "type": "checkpoint_resumed",
            "checkpoint_path": str(checkpoint_path),
            "status": payload.get("status"),
            "latest_node": payload.get("latest_node"),
            "git_commit": git_commit,
        }
        return inputs, resume_event


def _state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key in _RESUME_STATE_FIELDS
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return message_to_dict(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


def _restore_messages(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    if not value:
        return []
    if not all(
        isinstance(item, dict) and "type" in item and "data" in item
        for item in value
    ):
        return value
    try:
        return messages_from_dict(value)
    except (KeyError, TypeError, ValueError):
        return value


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False))
        file.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _excluded_from_snapshot(relative: Path) -> bool:
    if ".git" in relative.parts:
        return True
    return (
        len(relative.parts) >= 2
        and relative.parts[0] == ".byteclaw"
        and relative.parts[1] == "checkpoints"
    )


def _run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _copy_workspace_to_repository(workspace: Path, repository: Path) -> None:
    for child in repository.iterdir():
        if child.name != ".git":
            _remove_path(child)

    def ignore(current: str, names: list[str]) -> set[str]:
        relative = Path(current).resolve().relative_to(workspace)
        ignored = {".git"} if ".git" in names else set()
        if relative == Path(".byteclaw") and "checkpoints" in names:
            ignored.add("checkpoints")
        return ignored

    shutil.copytree(
        workspace,
        repository,
        dirs_exist_ok=True,
        ignore=ignore,
        symlinks=True,
    )


def _copy_repository_to_workspace(repository: Path, workspace: Path) -> None:
    for child in workspace.iterdir():
        if child.name == ".git":
            continue
        if child.name == ".byteclaw":
            for nested in child.iterdir():
                if nested.name != "checkpoints":
                    _remove_path(nested)
            continue
        _remove_path(child)

    shutil.copytree(
        repository,
        workspace,
        dirs_exist_ok=True,
        ignore=lambda _current, names: {".git"} if ".git" in names else set(),
        symlinks=True,
    )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
