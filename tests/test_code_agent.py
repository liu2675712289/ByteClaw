import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from byteclaw.agents.code_agent import CODE_AGENT_PROMPT, run_code_agent
from byteclaw.core.state import RuntimeState
from byteclaw.graph.nodes import TodoUpdateTool


class FakeTool:
    name = "file_write"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> str:
        self.calls.append(args)
        return "Wrote example.py"


class FakeAgent:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)

    def invoke(self, messages: list) -> AIMessage:
        return self.responses.pop(0)


class FakeModel:
    def __init__(self, agent: FakeAgent) -> None:
        self.agent = agent
        self.bound_tools = None

    def bind_tools(self, tools: list) -> FakeAgent:
        self.bound_tools = tools
        return self.agent


class CodeAgentTests(unittest.TestCase):
    def test_run_code_agent_executes_tools_and_persists_todos(self) -> None:
        file_tool = FakeTool()
        agent = FakeAgent(
            [
                AIMessage(
                    content="Starting implementation",
                    tool_calls=[
                        {
                            "name": "TodoUpdateTool",
                            "args": {
                                "id": "1",
                                "status": "in_progress",
                                "note": "Started",
                            },
                            "id": "todo-start",
                            "type": "tool_call",
                        },
                        {
                            "name": "file_write",
                            "args": {
                                "file_path": "example.py",
                                "content": "print('done')\n",
                            },
                            "id": "write-1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(
                    content="Implementation complete",
                    tool_calls=[
                        {
                            "name": "TodoUpdateTool",
                            "args": {
                                "id": "1",
                                "status": "completed",
                                "note": "Implemented and checked",
                            },
                            "id": "todo-finish",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Created example.py and checked its output."),
            ]
        )
        model = FakeModel(agent)
        events: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RuntimeState(Path(temp_dir))
            state = {
                "task": "Create an example program",
                "runtime": runtime,
                "todos": [
                    {
                        "id": "1",
                        "content": "Create example.py",
                        "status": "pending",
                        "note": "",
                    }
                ],
                "session_context": {"attempt": 1},
                "research_notes": ["Use an official Python example"],
                "sources": ["https://docs.python.org/3/"],
            }
            with (
                patch(
                    "byteclaw.agents.code_agent.build_tools",
                    return_value=[file_tool],
                ) as build,
                patch(
                    "byteclaw.agents.code_agent.create_model",
                    return_value=model,
                ),
            ):
                update = run_code_agent(
                    state,
                    "Implement the first todo",
                    writer=events.append,
                )

        build.assert_called_once_with(runtime)
        self.assertEqual(model.bound_tools, [file_tool, TodoUpdateTool])
        self.assertTrue(update["ok"])
        self.assertEqual(
            update["summary"], "Created example.py and checked its output."
        )
        self.assertEqual(update["todos"][0]["status"], "completed")
        self.assertEqual(state["todos"], update["todos"])
        self.assertEqual(file_tool.calls[0]["file_path"], "example.py")
        self.assertEqual(
            [event["type"] for event in update["tool_events"]],
            [
                "tool_call",
                "tool_result",
                "tool_call",
                "tool_result",
                "tool_call",
                "tool_result",
            ],
        )
        self.assertEqual(events, update["tool_events"])

        messages = update["messages"]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertEqual(messages[0].content, CODE_AGENT_PROMPT)
        self.assertIsInstance(messages[1], HumanMessage)
        request = json.loads(messages[1].content)
        self.assertEqual(request["task"], "Create an example program")
        self.assertEqual(request["instruction"], "Implement the first todo")
        self.assertEqual(request["session_context"], {"attempt": 1})
        self.assertEqual(
            request["memory"]["working_memory"]["node"], "codeAgent"
        )
        self.assertEqual(request["memory"], state["memory_snapshot"])
        self.assertEqual(
            request["memory"]["history_summary_store"]["history_exists"],
            False,
        )
        self.assertEqual(
            request["research_notes"], ["Use an official Python example"]
        )
        self.assertEqual(request["source_urls"], ["https://docs.python.org/3/"])
        self.assertTrue(any(isinstance(message, ToolMessage) for message in messages))


if __name__ == "__main__":
    unittest.main()
