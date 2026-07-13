import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from byteclaw.cli.app import app


class CliTests(unittest.TestCase):
    def test_task_command_creates_workspace(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "new-workspace"
            with patch("byteclaw.cli.app.run_task", return_value="completed"):
                result = runner.invoke(
                    app, ["write a file", "--workspace", str(workspace)]
                )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(workspace.is_dir())
            self.assertIn("completed", result.output)


if __name__ == "__main__":
    unittest.main()
