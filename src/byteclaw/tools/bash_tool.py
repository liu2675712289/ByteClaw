"""Workspace-scoped subprocess tool."""

import subprocess
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

import psutil

from byteclaw.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    classify_command_risk,
    normalize_approval_mode,
)
from byteclaw.core.state import RuntimeState


ApprovalHandler = Callable[[ApprovalRequest], ApprovalDecision]


@dataclass(frozen=True)
class BashResult:
    """Structured outcome of a shell command or approval refusal."""

    command: str
    ok: bool
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    requires_approval: bool = False
    risk_reason: str | None = None

    def __str__(self) -> str:
        exit_code = self.exit_code if self.exit_code is not None else "not run"
        parts = [f"Exit code: {exit_code}"]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout.rstrip()}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr.rstrip()}")
        return "\n".join(parts)

    def __contains__(self, value: str) -> bool:
        """Keep compatibility with callers that treated results as strings."""

        return value in str(self)


class BashTool:
    """Run a shell command with the workspace as its working directory."""

    def __init__(
        self,
        state: RuntimeState,
        approval_mode: str | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self.state = state
        configured_mode = (
            state.approval_mode if approval_mode is None else approval_mode
        )
        self.approval_mode = normalize_approval_mode(configured_mode)
        self.approval_handler = (
            state.approval_handler
            if approval_handler is None
            else approval_handler
        )

    def __call__(
        self, command: str, timeout_seconds: float = 30
    ) -> BashResult:
        """Execute ``command`` through the approval-aware runner."""

        return self.run_bash(command, timeout_seconds=timeout_seconds)

    def run_bash(
        self, command: str, timeout_seconds: float = 30
    ) -> BashResult:
        """Execute ``command`` after applying configured approval policy."""

        if not command.strip():
            raise ValueError("command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        risk_reason = classify_command_risk(command)
        if risk_reason:
            refusal_reason = self._approval_refusal(command, risk_reason)
            if refusal_reason is not None:
                return BashResult(
                    command=command,
                    ok=False,
                    exit_code=None,
                    stderr=refusal_reason,
                    requires_approval=True,
                    risk_reason=risk_reason,
                )

        process = subprocess.Popen(
            command,
            cwd=self.state.workspace,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                parent = psutil.Process(process.pid)
                descendants = parent.children(recursive=True)
                for child in descendants:
                    child.kill()
                parent.kill()
                psutil.wait_procs([*descendants, parent], timeout=1)
            except psutil.Error:
                process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            raise TimeoutError(
                f"Command timed out after {timeout_seconds} seconds"
            ) from exc

        return BashResult(
            command=command,
            ok=process.returncode == 0,
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            requires_approval=risk_reason is not None,
            risk_reason=risk_reason,
        )

    def _approval_refusal(
        self, command: str, risk_reason: str
    ) -> str | None:
        if self.approval_mode == "auto":
            return None
        if self.approval_mode == "deny":
            return f"Command denied by approval policy: {risk_reason}"
        if self.approval_handler is None:
            return (
                "Approval required but no approval handler is configured: "
                f"{risk_reason}"
            )

        request = ApprovalRequest(
            id=f"approval-{uuid4().hex[:8]}",
            command=command,
            risk_reason=risk_reason,
        )
        decision = self.approval_handler(request)
        if decision.approved:
            return None
        return decision.reason or f"Command was not approved: {risk_reason}"
