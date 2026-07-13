import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from byteclaw.core.state import RuntimeState
from byteclaw.core.trace import TraceRecorder, normalize_trace_mode


class TraceRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name) / "workspace"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_mode_normalization_and_generated_trace_id(self) -> None:
        self.assertEqual(normalize_trace_mode(None), "full")
        self.assertEqual(normalize_trace_mode(" FULL "), "full")
        self.assertEqual(normalize_trace_mode("on"), "full")
        self.assertEqual(normalize_trace_mode("disabled"), "off")
        self.assertEqual(normalize_trace_mode("invalid"), "full")

        recorder = TraceRecorder(RuntimeState(self.workspace))
        self.assertRegex(
            recorder.trace_id,
            re.compile(r"^trace-[0-9a-f]{12}$"),
        )

    def test_invalid_trace_id_is_rejected(self) -> None:
        runtime = RuntimeState(self.workspace, trace_id="../outside")

        with self.assertRaisesRegex(ValueError, "trace_id"):
            TraceRecorder(runtime)

    def test_off_mode_does_not_create_trace_files(self) -> None:
        runtime = RuntimeState(
            self.workspace,
            trace_mode="off",
            trace_id="trace-off",
        )
        recorder = TraceRecorder(runtime, task="disabled")

        self.assertIsNone(recorder.start({"task": "disabled"}))
        self.assertIsNone(
            recorder.record_custom_event({"type": "tool_call"})
        )
        self.assertIsNone(
            recorder.record_graph_update({"node": "planner"})
        )
        self.assertIsNone(
            recorder.end(
                status="completed",
                latest_node="final",
                final_state={},
            )
        )
        self.assertFalse(recorder.root.exists())

    def test_records_events_statistics_and_human_timeline(self) -> None:
        runtime = RuntimeState(
            self.workspace,
            trace_mode="full",
            trace_id="trace-statistics",
        )
        recorder = TraceRecorder(runtime, task="build feature")
        recorder.start(
            {"task": "build feature", "runtime": runtime},
            resumed=True,
            resume_event={"type": "checkpoint_resumed"},
        )
        recorder.record_custom_event(
            {"type": "tool_call", "name": "bash"}
        )
        recorder.record_custom_event(
            {
                "type": "tool_result",
                "name": "bash",
                "result": {"ok": False, "requires_approval": True},
            }
        )
        recorder.record_custom_event({"type": "handoff", "to": "codeAgent"})
        recorder.record_custom_event(
            {"type": "checkpoint_saved", "status": "running"}
        )
        recorder.record_graph_update({"node": "planner", "output": {}})
        recorder.record_graph_update({"node": "planner", "output": {}})
        recorder.record_graph_update({"verifier": {"passed": True}})

        summary = recorder.end(
            status="completed",
            latest_node="final",
            final_state={"passed": True},
        )

        self.assertEqual(summary["trace_id"], "trace-statistics")
        self.assertEqual(summary["task"], "build feature")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["latest_node"], "final")
        self.assertGreaterEqual(summary["duration_ms"], 0)
        self.assertEqual(summary["node_visits"], {"planner": 2, "verifier": 1})
        self.assertEqual(summary["tool_calls"], 1)
        self.assertEqual(summary["failed_tool_calls"], 1)
        self.assertEqual(summary["approval_count"], 1)
        self.assertEqual(summary["checkpoint_count"], 1)
        self.assertEqual(summary["handoff_count"], 1)
        self.assertEqual(len(summary["timeline_head"]), 9)
        self.assertEqual(summary["timeline_tail"], [])
        self.assertEqual(summary["timeline_omitted"], 0)

        trace_payload = json.loads(
            (recorder.root / "trace.json").read_text(encoding="utf-8")
        )
        self.assertEqual(trace_payload, summary)
        events = [
            json.loads(line)
            for line in (recorder.root / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(events[0]["type"], "run_start")
        self.assertTrue(events[0]["resumed"])
        self.assertEqual(events[-1]["type"], "run_end")
        timeline = (recorder.root / "timeline.md").read_text(encoding="utf-8")
        self.assertIn("ByteClaw Execution Trace", timeline)
        self.assertIn("Tool calls: 1", timeline)
        self.assertIn("`planner`: 2", timeline)

    def test_trace_summary_keeps_bounded_head_and_tail(self) -> None:
        runtime = RuntimeState(
            self.workspace,
            trace_id="trace-bounded",
        )
        recorder = TraceRecorder(runtime, task="many events")
        recorder.start({"task": "many events"})
        for index in range(105):
            recorder.record_custom_event(
                {"type": "tool_call", "name": "test", "index": index}
            )

        summary = recorder.end(
            status="completed",
            latest_node="final",
            final_state={},
        )

        self.assertEqual(summary["tool_calls"], 105)
        self.assertEqual(len(summary["timeline_head"]), 20)
        self.assertEqual(len(summary["timeline_tail"]), 80)
        self.assertEqual(summary["timeline_omitted"], 7)
        self.assertEqual(summary["timeline_head"][0]["type"], "run_start")
        self.assertEqual(summary["timeline_tail"][-1]["type"], "run_end")

    def test_resumed_trace_rebuilds_previous_statistics(self) -> None:
        runtime = RuntimeState(
            self.workspace,
            trace_id="trace-resumed",
        )
        first = TraceRecorder(runtime, task="resume task")
        first.start({"task": "resume task"})
        first.record_custom_event({"type": "tool_call", "name": "bash"})
        first.record_graph_update({"node": "planner", "output": {}})
        first_summary = first.end(
            status="interrupted",
            latest_node="planner",
            final_state={},
        )

        resumed = TraceRecorder(runtime, task="resume task")
        resumed.start(
            {"task": "resume task"},
            resumed=True,
            resume_event={"type": "checkpoint_resumed"},
        )
        resumed.record_custom_event(
            {
                "type": "tool_result",
                "ok": False,
                "requires_approval": True,
            }
        )
        resumed.record_custom_event({"type": "handoff"})
        summary = resumed.end(
            status="completed",
            latest_node="final",
            final_state={"passed": True},
        )

        self.assertEqual(summary["started_at"], first_summary["started_at"])
        self.assertEqual(summary["tool_calls"], 1)
        self.assertEqual(summary["failed_tool_calls"], 1)
        self.assertEqual(summary["approval_count"], 1)
        self.assertEqual(summary["handoff_count"], 1)
        self.assertEqual(summary["node_visits"], {"planner": 1})
        self.assertEqual(len(summary["timeline_head"]), 8)


if __name__ == "__main__":
    unittest.main()
