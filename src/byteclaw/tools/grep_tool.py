"""Regular-expression search tool."""

import re
from pathlib import Path

from byteclaw.core.paths import WorkspacePathError, resolve_workspace_path
from byteclaw.core.state import RuntimeState


class GrepTool:
    """Search UTF-8 text files within the workspace."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = 100,
        ignore_case: bool = False,
    ) -> str:
        """Return matching lines as ``path:line_number:text`` records."""

        if head_limit < 1:
            raise ValueError("head_limit must be at least 1")

        flags = re.IGNORECASE if ignore_case else 0
        expression = re.compile(pattern, flags)
        root = resolve_workspace_path(self.state.workspace, path, must_exist=True)
        candidates = [root] if root.is_file() else sorted(root.rglob(glob or "*"))
        matches: list[str] = []

        for candidate in candidates:
            if len(matches) >= head_limit:
                break
            if not candidate.is_file():
                continue
            if glob and root.is_file() and not candidate.match(glob):
                continue
            try:
                safe_candidate = resolve_workspace_path(
                    self.state.workspace, candidate, must_exist=True
                )
            except WorkspacePathError:
                continue
            try:
                lines = safe_candidate.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue

            relative = safe_candidate.relative_to(self.state.workspace)
            for line_number, line in enumerate(lines, start=1):
                if expression.search(line):
                    matches.append(f"{relative}:{line_number}:{line}")
                    if len(matches) >= head_limit:
                        break

        return "\n".join(matches)

