"""Workspace path validation helpers."""

from pathlib import Path


class WorkspacePathError(ValueError):
    """Raised when a requested path escapes the active workspace."""


def resolve_workspace_path(
    workspace: Path,
    requested_path: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a path and ensure its real location is inside ``workspace``."""

    root = workspace.expanduser().resolve()
    requested = Path(requested_path).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    candidate = candidate.resolve(strict=False)

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkspacePathError(
            f"Path escapes workspace: {requested_path!s}"
        ) from exc

    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {requested_path!s}")
    return candidate

