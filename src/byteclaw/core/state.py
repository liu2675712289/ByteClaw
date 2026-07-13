"""Runtime state shared by ByteClaw tools."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ApprovalHandler = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """State for one agent run.

    The workspace is normalized and created once so every tool uses the same
    trusted root directory.
    """

    workspace: Path
    checkpoint_mode: str = "light"
    trace_mode: str = "full"
    trace_id: str | None = None
    approval_mode: str = "inline"
    approval_handler: ApprovalHandler | None = None
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        workspace = self.workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if not workspace.is_dir():
            raise NotADirectoryError(f"Workspace is not a directory: {workspace}")
        object.__setattr__(self, "workspace", workspace)
        if self.resume_from is not None:
            resume_from = self.resume_from.expanduser().resolve()
            object.__setattr__(self, "resume_from", resume_from)


def create_runtime(
    workspace: str | Path,
    *,
    approval_mode: str = "inline",
    approval_handler: ApprovalHandler | None = None,
    checkpoint_mode: str = "light",
    resume_from: str | Path | None = None,
    trace_mode: str = "on",
) -> RuntimeState:
    """Create a runtime configured for a new or resumed workspace."""

    resume_path = Path(resume_from) if resume_from is not None else None
    runtime_workspace = resume_path if resume_path is not None else Path(workspace)
    return RuntimeState(
        workspace=runtime_workspace,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        resume_from=resume_path,
        trace_mode=trace_mode,
    )
