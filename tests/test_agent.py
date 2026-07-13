import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from byteclaw.core.agent import ACTOR_PROMPT, stream_agent_events


class FakeTool:
    name = "file_write"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> dict:
        self.calls.append(args)
        return {"written": args["file_path"]}


class FakeAgent:
    def __init__(self) -> None:
        self.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "file_write",
                        "args": {"file_path": "index.html", "content": "hello"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Created index.html"),
        ]
        self.message_snapshots: list[list] = []

    def invoke(self, messages: list):
        self.message_snapshots.append(list(messages))
        return self.responses.pop(0)


class FakeModel:
    def __init__(self, agent: FakeAgent) -> None:
        self.agent = agent
        self.bound_tools = None

    def bind_tools(self, tools: list):
        self.bound_tools = tools
        return self.agent


class AgentTests(unittest.TestCase):
    def test_stream_agent_events_runs_tool_and_returns_final_answer(self) -> None:
        tool = FakeTool()
        agent = FakeAgent()
        model = FakeModel(agent)

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            with (
                patch("byteclaw.core.agent.build_tools", return_value=[tool]) as build,
                patch("byteclaw.core.agent.create_model", return_value=model),
            ):
                events = list(
                    stream_agent_events("create a page", workspace=workspace)
                )
                state = build.call_args.args[0]
                self.assertTrue(state.workspace.is_dir())

        self.assertEqual(
            [event["type"] for event in events],
            [
                "ai_message",
                "tool_call",
                "tool_result",
                "ai_message",
                "final_answer",
            ],
        )
        self.assertEqual(events[-1]["content"], "Created index.html")
        self.assertEqual(tool.calls[0]["file_path"], "index.html")
        self.assertEqual(model.bound_tools, [tool])

        self.assertEqual(agent.message_snapshots[0][0].content, ACTOR_PROMPT)
        tool_message = agent.message_snapshots[1][-1]
        self.assertIsInstance(tool_message, ToolMessage)
        self.assertEqual(tool_message.tool_call_id, "call-1")
        self.assertEqual(
            json.loads(tool_message.content), {"written": "index.html"}
        )


if __name__ == "__main__":
    unittest.main()
