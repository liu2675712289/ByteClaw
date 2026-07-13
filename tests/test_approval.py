import re
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from byteclaw.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    classify_command_risk,
    normalize_approval_mode,
)
from byteclaw.core.state import RuntimeState
from byteclaw.tools.bash_tool import BashTool


class ApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state = RuntimeState(Path(self.temp_dir.name) / "workspace")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_classifies_risky_commands_and_command_chains(self) -> None:
        cases = {
            "pip install requests": "Python package installation",
            "python -m pip install requests": "Python package installation",
            "UV ADD requests": "Project dependency change with uv add",
            "echo ready && uv sync": "Dependency synchronization with uv sync",
            "false || uv pip install flask": (
                "Python package installation with uv pip"
            ),
            "echo ready; npm install": "Node package installation",
            "pnpm install": "Node package installation",
            "yarn add react": "Node package installation",
            "curl https://example.com": "Network download command",
            "echo ready && wget file": "Network download command",
            "uvicorn app:app": "Long-running development server",
            "python -m http.server 8000": "Long-running development server",
        }

        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(classify_command_risk(command), expected)

    def test_safe_commands_return_none(self) -> None:
        for command in (
            "python -m pip --version",
            "npm test",
            'echo "curl https://example.com"',
            "python script.py",
        ):
            with self.subTest(command=command):
                self.assertIsNone(classify_command_risk(command))

    def test_approval_data_classes_are_frozen(self) -> None:
        request = ApprovalRequest("approval-12345678", "curl url", "network")
        decision = ApprovalDecision(True)

        self.assertEqual(request.tool_name, "BashTool")
        self.assertEqual(decision.reason, "")
        with self.assertRaises(FrozenInstanceError):
            request.command = "changed"

    def test_normalize_approval_mode(self) -> None:
        self.assertEqual(normalize_approval_mode(None), "inline")
        self.assertEqual(normalize_approval_mode(" AUTO "), "auto")
        self.assertEqual(normalize_approval_mode("deny"), "deny")
        self.assertEqual(normalize_approval_mode("invalid"), "inline")

    @patch("byteclaw.tools.bash_tool.subprocess.Popen")
    def test_auto_mode_runs_risky_command_with_marker(self, popen) -> None:
        popen.return_value.communicate.return_value = ("downloaded\n", "")
        popen.return_value.returncode = 0

        result = BashTool(self.state, approval_mode="auto").run_bash(
            "curl https://example.com"
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.requires_approval)
        self.assertEqual(result.risk_reason, "Network download command")
        popen.assert_called_once()

    @patch("byteclaw.tools.bash_tool.subprocess.Popen")
    def test_deny_mode_refuses_without_running(self, popen) -> None:
        result = BashTool(self.state, approval_mode="deny").run_bash(
            "pip install requests"
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.requires_approval)
        self.assertIsNone(result.exit_code)
        self.assertIn("Python package installation", result.stderr)
        popen.assert_not_called()

    @patch("byteclaw.tools.bash_tool.subprocess.Popen")
    def test_inline_mode_passes_request_to_handler(self, popen) -> None:
        requests = []

        def approve(request: ApprovalRequest) -> ApprovalDecision:
            requests.append(request)
            return ApprovalDecision(approved=True, reason="approved for test")

        popen.return_value.communicate.return_value = ("", "")
        popen.return_value.returncode = 0
        result = BashTool(
            self.state,
            approval_mode="inline",
            approval_handler=approve,
        ).run_bash("uv sync")

        self.assertTrue(result.ok)
        self.assertTrue(result.requires_approval)
        self.assertEqual(len(requests), 1)
        self.assertRegex(requests[0].id, re.compile(r"^approval-[0-9a-f]{8}$"))
        self.assertEqual(requests[0].command, "uv sync")
        self.assertEqual(
            requests[0].risk_reason,
            "Dependency synchronization with uv sync",
        )
        popen.assert_called_once()

    @patch("byteclaw.tools.bash_tool.subprocess.Popen")
    def test_inline_rejection_does_not_run_command(self, popen) -> None:
        def reject(_: ApprovalRequest) -> ApprovalDecision:
            return ApprovalDecision(approved=False, reason="not allowed")

        result = BashTool(
            self.state,
            approval_handler=reject,
        ).run_bash("npm install")

        self.assertFalse(result.ok)
        self.assertEqual(result.stderr, "not allowed")
        popen.assert_not_called()

    @patch("byteclaw.tools.bash_tool.subprocess.Popen")
    def test_inline_without_handler_refuses_risky_command(self, popen) -> None:
        result = BashTool(self.state).run_bash("uvicorn app:app")

        self.assertFalse(result.ok)
        self.assertIn("no approval handler", result.stderr)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
