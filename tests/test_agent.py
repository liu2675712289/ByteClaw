import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from byteclaw.core.agent import stream_agent_events


class FakeWorkflow:
    def __init__(self, *, interrupt: bool = False) -> None:
        self.calls: list[tuple[dict, list[str]]] = []
        self.interrupt = interrupt

    def stream(self, inputs: dict, *, stream_mode: list[str]):
        self.calls.append((inputs, stream_mode))
        yield "updates", {"planner": {"plan_summary": "Create a page"}}
        if self.interrupt:
            raise KeyboardInterrupt
        yield "custom", {"type": "tool_call", "name": "file_write"}
        yield "custom", {
            "type": "tool_result",
            "name": "file_write",
            "result": {"ok": True},
        }
        yield "updates", {"verifier": {"passed": True, "attempts": 1}}
        yield "updates", {"final": {"final_answer": "Status: PASSED"}}


class FakeCheckpointManager:
    instances = []
    resume_result = None
    resume_calls = []

    def __init__(self, runtime, task: str = "") -> None:
        self.runtime = runtime
        self.task = task
        self.mode = runtime.checkpoint_mode
        self.saves = []
        self.__class__.instances.append(self)

    def save(
        self,
        state,
        *,
        status="running",
        latest_node=None,
        event=None,
    ):
        self.saves.append(
            {
                "state": dict(state),
                "status": status,
                "latest_node": latest_node,
                "event": event,
            }
        )
        return {
            "type": "checkpoint_saved",
            "status": status,
            "latest_node": latest_node,
        }

    @classmethod
    def load_resume_inputs(cls, runtime, task=None, max_attempts=3):
        cls.resume_calls.append(
            {
                "runtime": runtime,
                "task": task,
                "max_attempts": max_attempts,
            }
        )
        return cls.resume_result


class FakeTraceRecorder:
    instances = []

    def __init__(self, runtime, task: str = "") -> None:
        self.runtime = runtime
        self.task = task
        self.starts = []
        self.custom_events = []
        self.graph_updates = []
        self.ends = []
        self.__class__.instances.append(self)

    def start(self, inputs, *, resumed=False, resume_event=None):
        self.starts.append(
            {
                "inputs": inputs,
                "resumed": resumed,
                "resume_event": resume_event,
            }
        )

    def record_custom_event(self, event):
        self.custom_events.append(event)

    def record_graph_update(self, event):
        self.graph_updates.append(event)

    def end(self, *, status, latest_node, final_state):
        self.ends.append(
            {
                "status": status,
                "latest_node": latest_node,
                "final_state": dict(final_state),
            }
        )


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeCheckpointManager.instances = []
        FakeCheckpointManager.resume_result = None
        FakeCheckpointManager.resume_calls = []
        FakeTraceRecorder.instances = []

    def test_stream_records_events_and_saves_lifecycle_checkpoints(self) -> None:
        workflow = FakeWorkflow()
        approval_handler = object()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            with (
                patch(
                    "byteclaw.core.agent.build_workflow",
                    return_value=workflow,
                ) as build,
                patch(
                    "byteclaw.core.agent.CheckpointManager",
                    FakeCheckpointManager,
                ),
                patch(
                    "byteclaw.core.agent.TraceRecorder",
                    FakeTraceRecorder,
                ),
            ):
                events = list(
                    stream_agent_events(
                        "create a page",
                        workspace=workspace,
                        max_attempts=5,
                        approval_mode="auto",
                        approval_handler=approval_handler,
                        checkpoint_mode="light",
                        trace_mode="on",
                    )
                )

        build.assert_called_once_with()
        inputs, stream_modes = workflow.calls[0]
        runtime = inputs["runtime"]
        self.assertEqual(inputs["task"], "create a page")
        self.assertEqual(inputs["max_attempts"], 5)
        self.assertEqual(stream_modes, ["updates", "custom"])
        self.assertEqual(runtime.approval_mode, "auto")
        self.assertIs(runtime.approval_handler, approval_handler)
        self.assertEqual(runtime.checkpoint_mode, "light")
        self.assertEqual(runtime.trace_mode, "on")
        self.assertEqual(
            events,
            [
                {
                    "type": "graph_event",
                    "event": {
                        "planner": {"plan_summary": "Create a page"}
                    },
                },
                {
                    "type": "custom_event",
                    "event": {"type": "tool_call", "name": "file_write"},
                },
                {
                    "type": "custom_event",
                    "event": {
                        "type": "tool_result",
                        "name": "file_write",
                        "result": {"ok": True},
                    },
                },
                {
                    "type": "graph_event",
                    "event": {"verifier": {"passed": True, "attempts": 1}},
                },
                {
                    "type": "graph_event",
                    "event": {
                        "final": {"final_answer": "Status: PASSED"}
                    },
                },
            ],
        )

        manager = FakeCheckpointManager.instances[0]
        self.assertEqual(
            [save["status"] for save in manager.saves],
            ["started", "running", "running", "running", "running", "finished"],
        )
        self.assertEqual(
            [save["latest_node"] for save in manager.saves],
            ["start", "planner", "planner", "verifier", "final", "final"],
        )
        final_state = manager.saves[-1]["state"]
        self.assertEqual(final_state["plan_summary"], "Create a page")
        self.assertTrue(final_state["passed"])
        self.assertEqual(final_state["attempts"], 1)
        self.assertEqual(final_state["final_answer"], "Status: PASSED")

        trace = FakeTraceRecorder.instances[0]
        self.assertFalse(trace.starts[0]["resumed"])
        self.assertEqual(len(trace.graph_updates), 3)
        self.assertEqual(
            [event["type"] for event in trace.custom_events].count(
                "checkpoint_saved"
            ),
            6,
        )
        self.assertEqual(trace.ends[0]["status"], "finished")
        self.assertEqual(trace.ends[0]["latest_node"], "final")

    def test_resume_loads_checkpoint_inputs_and_marks_trace_resumed(self) -> None:
        workflow = FakeWorkflow()
        resume_event = {"type": "checkpoint_resumed", "latest_node": "planner"}

        with tempfile.TemporaryDirectory() as temp_dir:
            resume_workspace = Path(temp_dir) / "resume-workspace"
            FakeCheckpointManager.resume_result = (
                {
                    "task": "saved task",
                    "messages": [],
                    "attempts": 2,
                    "max_attempts": 4,
                },
                resume_event,
            )
            with (
                patch(
                    "byteclaw.core.agent.build_workflow",
                    return_value=workflow,
                ),
                patch(
                    "byteclaw.core.agent.CheckpointManager",
                    FakeCheckpointManager,
                ),
                patch(
                    "byteclaw.core.agent.TraceRecorder",
                    FakeTraceRecorder,
                ),
            ):
                list(
                    stream_agent_events(
                        "",
                        workspace="unused",
                        max_attempts=4,
                        resume_workspace=resume_workspace,
                    )
                )

        inputs, _ = workflow.calls[0]
        runtime = FakeCheckpointManager.resume_calls[0]["runtime"]
        self.assertEqual(runtime.workspace, resume_workspace.resolve())
        self.assertEqual(runtime.resume_from, resume_workspace.resolve())
        self.assertIs(inputs["runtime"], runtime)
        self.assertEqual(inputs["attempts"], 2)
        self.assertEqual(FakeCheckpointManager.resume_calls[0]["max_attempts"], 4)
        trace = FakeTraceRecorder.instances[0]
        self.assertTrue(trace.starts[0]["resumed"])
        self.assertEqual(trace.starts[0]["resume_event"], resume_event)

    def test_keyboard_interrupt_saves_interrupted_state(self) -> None:
        workflow = FakeWorkflow(interrupt=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "byteclaw.core.agent.build_workflow",
                    return_value=workflow,
                ),
                patch(
                    "byteclaw.core.agent.CheckpointManager",
                    FakeCheckpointManager,
                ),
                patch(
                    "byteclaw.core.agent.TraceRecorder",
                    FakeTraceRecorder,
                ),
            ):
                events = list(
                    stream_agent_events(
                        "interrupt me",
                        workspace=Path(temp_dir) / "workspace",
                    )
                )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "graph_event")
        manager = FakeCheckpointManager.instances[0]
        self.assertEqual(
            [save["status"] for save in manager.saves],
            ["started", "running", "interrupted"],
        )
        trace = FakeTraceRecorder.instances[0]
        self.assertEqual(trace.ends[0]["status"], "interrupted")
        self.assertEqual(trace.ends[0]["latest_node"], "planner")


if __name__ == "__main__":
    unittest.main()
