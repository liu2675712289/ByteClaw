"""Workspace-scoped file tools."""

from byteclaw.core.paths import resolve_workspace_path
from byteclaw.core.state import RuntimeState


class FileReadTool:
    """Read a range of lines from a UTF-8 text file."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, file_path: str, offset: int = 0, limit: int = 200) -> str:
        """Read up to ``limit`` lines, starting at zero-based ``offset``."""

        if offset < 0:
            raise ValueError("offset must be at least 0")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        path = resolve_workspace_path(
            self.state.workspace, file_path, must_exist=True
        )
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        return "".join(lines[offset : offset + limit])


class FileWriteTool:
    """Create or overwrite a UTF-8 text file inside the workspace."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, file_path: str, content: str) -> str:
        """Write ``content`` to ``file_path``, creating parent folders."""

        path = resolve_workspace_path(self.state.workspace, file_path)
        if path.exists() and not path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path.relative_to(self.state.workspace)}"


class FileEditTool:
    """Replace one unique text fragment in a workspace file."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, file_path: str, old_text: str, new_text: str) -> str:
        """Replace ``old_text`` only when it occurs exactly once."""

        if not old_text:
            raise ValueError("old_text must not be empty")

        path = resolve_workspace_path(
            self.state.workspace, file_path, must_exist=True
        )
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

        content = path.read_text(encoding="utf-8")
        matches = content.count(old_text)
        if matches == 0:
            raise ValueError("old_text was not found")
        if matches > 1:
            raise ValueError(f"old_text matched {matches} times; expected exactly once")

        path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path.relative_to(self.state.workspace)}"

