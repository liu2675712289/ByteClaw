import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from byteclaw.core.checkpoint import (
    CheckpointManager,
    build_recovery_markdown,
    normalize_checkpoint_mode,
    resume_command,
    workspace_manifest,
)
from byteclaw.core.state import RuntimeState


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name) / "workspace"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_modes_resume_command_and_recovery_markdown(self) -> None:
        self.assertEqual(normalize_checkpoint_mode(None), "light")
        self.assertEqual(normalize_checkpoint_mode(" STRICT "), "strict")
        self.assertEqual(normalize_checkpoint_mode("off"), "off")
        self.assertEqual(normalize_checkpoint_mode("invalid"), "light")

        command = resume_command(Path("workspace with spaces"))
        self.assertTrue(command.startswith("byteclaw --resume "))
        self.assertIn('"', command)

        markdown = build_recovery_markdown(
            {
                "workspace": str(self.workspace),
                "task": "repair tests",
                "status": "running",
                "mode": "light",
                "latest_node": "planner",
                "saved_at": "2026-07-13T00:00:00+00:00",
                "git_commit": "abc1234",
                "workspace_manifest": ["src/app.py"],
            }
        )
        self.assertIn("repair tests", markdown)
        self.assertIn("src/app.py", markdown)
        self.assertIn("abc1234", markdown)
        self.assertIn("byteclaw --resume", markdown)

    def test_off_mode_does_not_write_checkpoint_files(self) -> None:
        runtime = RuntimeState(self.workspace, checkpoint_mode="off")
        manager = CheckpointManager(runtime, task="do nothing")

        result = manager.save({"task": "do nothing"})

        self.assertFalse(manager.enabled)
        self.assertIsNone(result)
        self.assertFalse(manager.root.exists())

    @unittest.skipUnless(shutil.which("git"), "Git is required for snapshots")
    def test_light_mode_saves_metadata_guide_and_git_snapshot(self) -> None:
        runtime = RuntimeState(self.workspace, checkpoint_mode="light")
        source = runtime.workspace / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_text("print('saved')\n", encoding="utf-8")
        manager = CheckpointManager(runtime, task="build app")

        event = manager.save(
            {"task": "build app", "attempts": 1, "messages": []},
            latest_node="planner",
        )

        self.assertEqual(event["type"], "checkpoint_saved")
        self.assertEqual(event["latest_node"], "planner")
        self.assertRegex(event["git_commit"], re.compile(r"^[0-9a-f]{40,64}$"))
        self.assertTrue((manager.root / "checkpoint.json").is_file())
        self.assertTrue((manager.root / "RECOVERY.md").is_file())
        self.assertTrue((manager.root / "snapshot-repo" / ".git").is_dir())
        self.assertFalse((manager.root / "state.json").exists())
        self.assertFalse((manager.root / "events.jsonl").exists())

        payload = json.loads(
            (manager.root / "checkpoint.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["state_summary"]["attempts"], 1)
        self.assertEqual(payload["workspace_manifest"], ["src/app.py"])
        self.assertEqual(workspace_manifest(runtime.workspace), ["src/app.py"])

    @unittest.skipUnless(shutil.which("git"), "Git is required for snapshots")
    def test_strict_mode_appends_events_and_restores_inputs_and_files(self) -> None:
        runtime = RuntimeState(self.workspace, checkpoint_mode="strict")
        source = runtime.workspace / "answer.txt"
        source.write_text("saved\n", encoding="utf-8")
        manager = CheckpointManager(runtime, task="original task")
        state = {
            "task": "original task",
            "runtime": runtime,
            "messages": [HumanMessage(content="continue")],
            "attempts": 2,
            "todos": [{"id": "1", "status": "in_progress"}],
        }

        manager.save(state, latest_node="verifier", event={"step": 1})
        manager.save(state, latest_node="planner", event={"step": 2})

        events = [
            json.loads(line)
            for line in (manager.root / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(events, [{"step": 1}, {"step": 2}])
        full_state = json.loads(
            (manager.root / "state.json").read_text(encoding="utf-8")
        )
        self.assertTrue((manager.root / "events.jsonl").is_file())
        self.assertEqual(full_state["attempts"], 2)
        self.assertEqual(full_state["runtime"]["checkpoint_mode"], "strict")

        source.write_text("changed\n", encoding="utf-8")
        added = runtime.workspace / "added-after-checkpoint.txt"
        added.write_text("remove me\n", encoding="utf-8")
        inputs, resume_event = CheckpointManager.load_resume_inputs(
            runtime,
            max_attempts=5,
        )

        self.assertEqual(source.read_text(encoding="utf-8"), "saved\n")
        self.assertFalse(added.exists())
        self.assertEqual(inputs["task"], "original task")
        self.assertIs(inputs["runtime"], runtime)
        self.assertEqual(inputs["attempts"], 2)
        self.assertEqual(inputs["max_attempts"], 5)
        self.assertIsInstance(inputs["messages"][0], HumanMessage)
        self.assertEqual(inputs["messages"][0].content, "continue")
        self.assertEqual(resume_event["type"], "checkpoint_resumed")
        self.assertEqual(resume_event["latest_node"], "planner")
        self.assertTrue(manager.root.is_dir())


if __name__ == "__main__":
    unittest.main()
