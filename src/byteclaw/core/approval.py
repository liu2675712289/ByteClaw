"""Command risk classification and approval primitives."""

import re
from dataclasses import dataclass


RISK_PATTERNS = [
    (
        r"(?:^|&&|\|\||;)\s*(?:python\s+-m\s+)?pip\s+install\b",
        "Python package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uv\s+add\b",
        "Project dependency change with uv add",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uv\s+sync\b",
        "Dependency synchronization with uv sync",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uv\s+pip\s+install\b",
        "Python package installation with uv pip",
    ),
    (
        r"(?:^|&&|\|\||;)\s*npm\s+install\b",
        "Node package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*pnpm\s+install\b",
        "Node package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*yarn\s+(?:install\b|add\b)",
        "Node package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*(?:curl|wget)\b",
        "Network download command",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uvicorn\b",
        "Long-running development server",
    ),
    (
        r"(?:^|&&|\|\||;)\s*python\s+-m\s+http\.server\b",
        "Long-running development server",
    ),
]


def classify_command_risk(command: str) -> str | None:
    """Return the first matching risk reason, or ``None`` when safe."""

    for pattern, reason in RISK_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return reason
    return None


@dataclass(frozen=True)
class ApprovalRequest:
    """A request for a human decision before running a risky command."""

    id: str
    command: str
    risk_reason: str
    tool_name: str = "BashTool"


@dataclass(frozen=True)
class ApprovalDecision:
    """The decision returned by an approval handler."""

    approved: bool
    reason: str = ""


VALID_APPROVAL_MODES = {"inline", "auto", "deny"}


def normalize_approval_mode(mode: str | None) -> str:
    """Normalize a mode, falling back to ``inline`` for invalid values."""

    normalized = (mode or "").strip().lower()
    return normalized if normalized in VALID_APPROVAL_MODES else "inline"
