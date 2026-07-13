import json
import os
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from byteclaw.core.session import (
    MAX_SESSION_CONTEXT,
    MAX_TURN_CONTENT,
    SESSION_FILE,
    SESSION_ROOT,
    SESSION_SUMMARY_FILE,
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
)


class SessionTests(unittest.TestCase):
    def test_load_or_create_session_persists_required_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            session = load_or_create_session(workspace)
            root = workspace / SESSION_ROOT
            stored = json.loads(
                (root / SESSION_FILE).read_text(encoding="utf-8")
            )
            loaded = load_or_create_session(workspace)
            summary_exists = (root / SESSION_SUMMARY_FILE).is_file()

        UUID(session["session_id"])
        self.assertEqual(session["turn_index"], 0)
        self.assertEqual(session["recent_turns"], [])
        self.assertIn("created_at", session)
        self.assertIn("updated_at", session)
        self.assertEqual(stored, session)
        self.assertEqual(loaded, session)
        self.assertTrue(summary_exists)

    def test_append_turns_truncates_content_and_save_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            session = load_or_create_session(workspace)
            turn = append_user_turn(
                session, "u" * (MAX_TURN_CONTENT + 100)
            )
            append_assistant_turn(
                session,
                turn=turn,
                route="workflow",
                content="a" * (MAX_TURN_CONTENT + 100),
                summary="Implemented the requested change",
            )
            saved = save_session(workspace, session)
            root = workspace / SESSION_ROOT
            stored = json.loads(
                (root / SESSION_FILE).read_text(encoding="utf-8")
            )
            summary = (root / SESSION_SUMMARY_FILE).read_text(
                encoding="utf-8"
            )

        self.assertEqual(turn, 1)
        self.assertEqual(saved["turn_index"], 1)
        self.assertEqual(len(saved["recent_turns"][0]["content"]), 4000)
        self.assertEqual(len(saved["recent_turns"][1]["content"]), 4000)
        self.assertEqual(saved["recent_turns"][1]["turn"], turn)
        self.assertEqual(saved["recent_turns"][1]["route"], "workflow")
        self.assertEqual(stored, saved)
        self.assertIn("Turn 1 - User", summary)
        self.assertIn("Turn 1 - Assistant (workflow)", summary)
        self.assertIn("Implemented the requested change", summary)

    def test_append_assistant_turn_rejects_unknown_route(self) -> None:
        session = {"turn_index": 1, "recent_turns": []}

        with self.assertRaisesRegex(ValueError, "route"):
            append_assistant_turn(
                session,
                turn=1,
                route="unknown",
                content="response",
            )

        self.assertEqual(session["recent_turns"], [])

    def test_build_session_context_uses_recent_files_and_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            base_time = 1_700_000_000_000_000_000
            for index in range(35):
                path = workspace / f"file-{index:02}.txt"
                path.write_text(str(index), encoding="utf-8")
                timestamp = base_time + index * 1_000_000_000
                os.utime(path, ns=(timestamp, timestamp))

            session = {
                "session_id": "session-context",
                "turn_index": 12,
                "recent_turns": [
                    {
                        "turn": index,
                        "role": "assistant",
                        "route": "chat",
                        "content": "x" * 5000,
                        "summary": f"<summary-{index}>",
                    }
                    for index in range(12)
                ],
            }
            context = build_session_context(workspace, session)

        file_lines = [
            line
            for line in context.splitlines()
            if line.startswith("- file-")
        ]
        self.assertLessEqual(len(context), MAX_SESSION_CONTEXT)
        self.assertIn("Session ID: session-context", context)
        self.assertIn("Turn index: 12", context)
        self.assertEqual(len(file_lines), 30)
        self.assertIn("- file-34.txt", file_lines)
        self.assertIn("- file-05.txt", file_lines)
        self.assertNotIn("- file-04.txt", file_lines)
        self.assertIn("Turn 11 [assistant/chat]: <summary-11>", context)
        self.assertIn("Turn 2 [assistant/chat]: <summary-2>", context)
        self.assertNotIn("Turn 1 [assistant/chat]:", context)


if __name__ == "__main__":
    unittest.main()
