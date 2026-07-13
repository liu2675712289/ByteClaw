import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from byteclaw.core.paths import WorkspacePathError
from byteclaw.core.state import RuntimeState
from byteclaw.tools.bash_tool import BashTool
from byteclaw.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool
from byteclaw.tools.grep_tool import GrepTool
from byteclaw.tools.registry import build_tools


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state = RuntimeState(Path(self.temp_dir.name) / "workspace")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_workspace_is_created_and_normalized(self) -> None:
        self.assertTrue(self.state.workspace.is_dir())
        self.assertTrue(self.state.workspace.is_absolute())

    def test_read_write_and_line_range(self) -> None:
        FileWriteTool(self.state)("nested/example.txt", "one\ntwo\nthree\n")
        result = FileReadTool(self.state)("nested/example.txt", offset=1, limit=1)
        self.assertEqual(result, "two\n")

    def test_file_tools_reject_path_escape(self) -> None:
        with self.assertRaises(WorkspacePathError):
            FileWriteTool(self.state)("../outside.txt", "no")

    def test_edit_requires_unique_match(self) -> None:
        path = self.state.workspace / "example.txt"
        path.write_text("same same", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "matched 2 times"):
            FileEditTool(self.state)("example.txt", "same", "new")

        path.write_text("hello world", encoding="utf-8")
        FileEditTool(self.state)("example.txt", "world", "ByteClaw")
        self.assertEqual(path.read_text(encoding="utf-8"), "hello ByteClaw")

    def test_grep_regex_glob_case_and_limit(self) -> None:
        FileWriteTool(self.state)("one.py", "Alpha\nbeta\nALPHA\n")
        FileWriteTool(self.state)("two.txt", "alpha\n")
        result = GrepTool(self.state)(
            "^alpha$", glob="*.py", head_limit=1, ignore_case=True
        )
        self.assertEqual(result, "one.py:1:Alpha")

    def test_bash_runs_in_workspace(self) -> None:
        result = BashTool(self.state)(
            f'"{sys.executable}" -c "from pathlib import Path; print(Path.cwd())"'
        )
        self.assertIn(str(self.state.workspace), result)
        self.assertIn("Exit code: 0", result)

    def test_bash_timeout(self) -> None:
        command = f'"{sys.executable}" -c "import time; time.sleep(2)"'
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            BashTool(self.state)(command, timeout_seconds=0.05)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_registry_returns_structured_tools(self) -> None:
        tools = build_tools(self.state)
        self.assertEqual(
            [tool.name for tool in tools],
            ["file_read", "file_write", "file_edit", "grep", "bash"],
        )
        self.assertTrue(all(tool.__class__.__name__ == "StructuredTool" for tool in tools))


if __name__ == "__main__":
    unittest.main()
