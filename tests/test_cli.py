import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from byteclaw.cli.app import app


class CliTests(unittest.TestCase):
    def test_task_command_renders_agent_events(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "new-workspace"
            events = iter(
                [
                    {
                        "type": "tool_call",
                        "name": "file_write",
                        "args": {"file_path": "example.txt", "content": "hello"},
                    },
                    {
                        "type": "tool_result",
                        "name": "file_write",
                        "result": "Wrote example.txt",
                    },
                    {"type": "final_answer", "content": "completed"},
                ]
            )
            with patch(
                "byteclaw.cli.app.stream_agent_events", return_value=events
            ) as stream:
                result = runner.invoke(
                    app, ["write a file", "--workspace", str(workspace)]
                )

            self.assertEqual(result.exit_code, 0, result.output)
            stream.assert_called_once_with("write a file", workspace=workspace)
            self.assertIn("file_write", result.output)
            self.assertIn("example.txt", result.output)
            self.assertIn("Wrote example.txt", result.output)
            self.assertIn("completed", result.output)


if __name__ == "__main__":
    unittest.main()
