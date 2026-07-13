"""Workspace-scoped subprocess tool."""

import subprocess

import psutil

from byteclaw.core.state import RuntimeState


class BashTool:
    """Run a shell command with the workspace as its working directory."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, command: str, timeout_seconds: float = 30) -> str:
        """Execute ``command`` and return its exit code and captured output."""

        if not command.strip():
            raise ValueError("command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

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

        parts = [f"Exit code: {process.returncode}"]
        if stdout:
            parts.append(f"stdout:\n{stdout.rstrip()}")
        if stderr:
            parts.append(f"stderr:\n{stderr.rstrip()}")
        return "\n".join(parts)
