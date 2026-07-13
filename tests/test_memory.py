import json
import tempfile
import unittest
from pathlib import Path

from byteclaw.core.state import RuntimeState
from byteclaw.graph.memory import (
    RULES_LAYER,
    _short_text,
    _trim_handoffs,
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
    read_history_summary,
    read_notepad,
)


class LayeredMemoryTests(unittest.TestCase):
    def test_short_text_and_handoff_limits(self) -> None:
        self.assertEqual(_short_text("short", 10), "short")
        self.assertEqual(_short_text("abcdefgh", 6), "abc...")

        handoffs = [{"instruction": str(index)} for index in range(8)]
        trimmed = _trim_handoffs(handoffs)
        self.assertEqual(
            [item["instruction"] for item in trimmed],
            ["2", "3", "4", "5", "6", "7"],
        )
        self.assertIsNot(trimmed[0], handoffs[2])

    def test_memory_event_wraps_snapshot(self) -> None:
        memory = {
            "rules": {},
            "working_memory": {"node": "planner"},
            "history_summary_store": {},
        }

        self.assertEqual(
            memory_event(memory, node="planner"),
            {"type": "memory", "node": "planner", "memory": memory},
        )

    def test_build_layered_memory_reads_and_trims_all_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "NOTEPAD.md").write_text(
                "N" * 1900, encoding="utf-8"
            )
            (workspace / "HISTORY_SUMMARY.md").write_text(
                "H" * 2300, encoding="utf-8"
            )
            runtime = RuntimeState(workspace)
            state = {
                "runtime": runtime,
                "task": "Implement memory",
                "session_context": {"session_id": "session-1", "turn": 4},
                "research_notes": "R" * 1700,
                "sources": [
                    {
                        "title": "Docs",
                        "url": "https://example.com/docs",
                        "content": "drop this",
                    }
                ],
                "agent_handoffs": [
                    {"instruction": str(index)} for index in range(8)
                ],
                "code_agent_summary": "C" * 1100,
                "verifier_summary": "V" * 1100,
                "last_error": "E" * 1500,
                "context_summary": "S" * 1700,
                "compression_events": [
                    {"node": str(index)} for index in range(5)
                ],
            }

            memory = build_layered_memory(state, node="codeAgent")

        self.assertEqual(memory["rules"], RULES_LAYER)
        working = memory["working_memory"]
        self.assertEqual(working["node"], "codeAgent")
        self.assertEqual(working["session_id"], "session-1")
        self.assertEqual(working["session_turn"], 4)
        self.assertEqual(
            working["sources"],
            [{"title": "Docs", "url": "https://example.com/docs"}],
        )
        self.assertEqual(len(working["agent_handoffs"]), 6)
        self.assertEqual(len(working["research_notes"]), 1600)
        self.assertTrue(working["research_notes"].endswith("..."))

        history = memory["history_summary_store"]
        self.assertTrue(history["history_exists"])
        self.assertTrue(history["notepad_exists"])
        self.assertEqual(len(history["history_summary"]), 2200)
        self.assertEqual(len(history["notepad"]), 1800)
        self.assertEqual(
            [item["node"] for item in history["compression_events"]],
            ["2", "3", "4"],
        )
        self.assertEqual(json.loads(format_layered_memory_for_prompt(memory)), memory)

    def test_missing_memory_files_return_empty_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RuntimeState(Path(temp_dir))

            self.assertEqual(
                read_notepad(runtime),
                {"path": "NOTEPAD.md", "exists": False, "content": ""},
            )
            self.assertEqual(
                read_history_summary(runtime),
                {
                    "path": "HISTORY_SUMMARY.md",
                    "exists": False,
                    "content": "",
                },
            )


if __name__ == "__main__":
    unittest.main()
