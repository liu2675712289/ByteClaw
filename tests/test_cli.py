import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from byteclaw.cli.app import app


class CliTests(unittest.TestCase):
    def test_task_command_renders_node_outputs_and_passes_max_attempts(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "new-workspace"
            events = iter(
                [
                    {
                        "type": "node_output",
                        "node": "planner",
                        "output": {"plan_summary": "Write example.txt"},
                    },
                    {
                        "type": "node_output",
                        "node": "actor",
                        "output": {"last_actor_summary": "Wrote example.txt"},
                    },
                    {
                        "type": "node_output",
                        "node": "verifier",
                        "output": {"passed": True},
                    },
                    {
                        "type": "node_output",
                        "node": "final",
                        "output": {"final_answer": "completed"},
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
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            stream.assert_called_once_with(
                "write a file", workspace=workspace, max_attempts=5
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
            "write a file", workspace=Path("workspace"), max_attempts=3
        )

    def test_failed_verifier_uses_failure_icon(self) -> None:
        runner = CliRunner()
        events = iter(
            [
                {
                    "type": "node_output",
                    "node": "verifier",
                    "output": {"passed": False, "last_error": "Tests failed"},
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
