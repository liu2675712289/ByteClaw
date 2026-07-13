import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from byteclaw.core.agent import stream_agent_events


class FakeWorkflow:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, list[str]]] = []

    def stream(self, inputs: dict, *, stream_mode: list[str]):
        self.calls.append((inputs, stream_mode))
        yield "updates", {"planner": {"plan_summary": "Create a page"}}
        yield "custom", {"type": "tool_call", "name": "file_write"}
        yield "updates", {"verifier": {"passed": True, "attempts": 1}}
        yield "updates", {"final": {"final_answer": "Status: PASSED"}}


class AgentTests(unittest.TestCase):
    def test_stream_agent_events_normalizes_workflow_events(self) -> None:
        workflow = FakeWorkflow()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            with patch(
                "byteclaw.core.agent.build_workflow", return_value=workflow
            ) as build:
                events = list(
                    stream_agent_events(
                        "create a page",
                        workspace=workspace,
                        max_attempts=5,
                    )
                )

            inputs, stream_modes = workflow.calls[0]
            self.assertTrue(inputs["runtime"].workspace.is_dir())

        build.assert_called_once_with()
        self.assertEqual(inputs["task"], "create a page")
        self.assertEqual(inputs["max_attempts"], 5)
        self.assertEqual(stream_modes, ["updates", "custom"])
        self.assertEqual(
            events,
            [
                {
                    "type": "node_output",
                    "node": "planner",
                    "output": {"plan_summary": "Create a page"},
                },
                {
                    "type": "node_output",
                    "node": "planner",
                    "output": {"type": "tool_call", "name": "file_write"},
                },
                {
                    "type": "node_output",
                    "node": "verifier",
                    "output": {"passed": True, "attempts": 1},
                },
                {
                    "type": "node_output",
                    "node": "final",
                    "output": {"final_answer": "Status: PASSED"},
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
