"""Runtime state shared by ByteClaw tools."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """State for one agent run.

    The workspace is normalized and created once so every tool uses the same
    trusted root directory.
    """

    workspace: Path

    def __post_init__(self) -> None:
        workspace = self.workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if not workspace.is_dir():
            raise NotADirectoryError(f"Workspace is not a directory: {workspace}")
        object.__setattr__(self, "workspace", workspace)

