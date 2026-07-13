import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from byteclaw.cli.app import app, run


class CliTests(unittest.TestCase):
    def test_installed_entrypoint_dispatches_tui(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                ["byteclaw", "tui", "--workspace", "demo"],
            ),
            patch("byteclaw.cli.tui.app.tui_cli") as tui_cli,
        ):
            run()

            tui_cli.assert_called_once_with()
            self.assertEqual(
                sys.argv,
                ["byteclaw tui", "--workspace", "demo"],
            )

    def test_task_command_renders_outputs_and_passes_runtime_options(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "new-workspace"
            events = iter(
                [
                    {
                        "type": "graph_event",
                        "event": {
                            "planner": {"plan_summary": "Write example.txt"}
                        },
                    },
                    {
                        "type": "graph_event",
                        "event": {
                            "actor": {
                                "last_actor_summary": "Wrote example.txt"
                            }
                        },
                    },
                    {
                        "type": "graph_event",
                        "event": {"verifier": {"passed": True}},
                    },
                    {
                        "type": "graph_event",
                        "event": {"final": {"final_answer": "completed"}},
                    },
                ]
            )
            with patch(
                "byteclaw.cli.app.stream_agent_events", return_value=events
            ) as stream:
                result = runner.invoke(
                    app,
                    [
                        "write a file",
                        "--workspace",
                        str(workspace),
                        "--max-attempts",
                        "5",
                        "--approval-mode",
                        "auto",
                        "--checkpoint-mode",
                        "strict",
                        "--trace-mode",
                        "off",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            stream.assert_called_once_with(
                "write a file",
                workspace=workspace,
                max_attempts=5,
                approval_mode="auto",
                checkpoint_mode="strict",
                trace_mode="off",
                resume_workspace=None,
            )
            self.assertIn("📋 Planner", result.output)
            self.assertIn("🔧 Actor", result.output)
            self.assertIn("✅ Verifier", result.output)
            self.assertIn("📝 Final", result.output)
            self.assertIn("completed", result.output)

    def test_max_attempts_defaults_to_three(self) -> None:
        runner = CliRunner()

        with patch(
            "byteclaw.cli.app.stream_agent_events", return_value=iter([])
        ) as stream:
            result = runner.invoke(app, ["write a file"])

        self.assertEqual(result.exit_code, 0, result.output)
        stream.assert_called_once_with(
            "write a file",
            workspace=Path("workspace"),
            max_attempts=3,
            approval_mode="inline",
            checkpoint_mode="light",
            trace_mode="on",
            resume_workspace=None,
        )

    def test_resume_uses_resume_workspace_without_a_task(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            resume_workspace = Path(temp_dir) / "saved-workspace"
            with patch(
                "byteclaw.cli.app.stream_agent_events", return_value=iter([])
            ) as stream:
                result = runner.invoke(
                    app,
                    ["--resume", str(resume_workspace)],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        stream.assert_called_once_with(
            "",
            workspace=resume_workspace,
            max_attempts=3,
            approval_mode="inline",
            checkpoint_mode="light",
            trace_mode="on",
            resume_workspace=resume_workspace,
        )

    def test_runtime_modes_reject_unknown_values(self) -> None:
        runner = CliRunner()

        with patch(
            "byteclaw.cli.app.stream_agent_events", return_value=iter([])
        ) as stream:
            result = runner.invoke(
                app,
                ["write a file", "--approval-mode", "prompt"],
            )

        self.assertEqual(result.exit_code, 2)
        stream.assert_not_called()

    def test_failed_verifier_uses_failure_icon(self) -> None:
        runner = CliRunner()
        events = iter(
            [
                {
                    "type": "graph_event",
                    "event": {
                        "verifier": {
                            "passed": False,
                            "last_error": "Tests failed",
                        }
                    },
                }
            ]
        )

        with patch(
            "byteclaw.cli.app.stream_agent_events", return_value=events
        ):
            result = runner.invoke(app, ["write a file"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("❌ Verifier", result.output)


if __name__ == "__main__":
    unittest.main()
