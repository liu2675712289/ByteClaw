import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from byteclaw.agents.search_agent import (
    SEARCH_AGENT_PROMPT,
    run_search_agent,
)


class FakeAgent:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.message_snapshots: list[list] = []

    def invoke(self, messages: list) -> AIMessage:
        self.message_snapshots.append(list(messages))
        return self.responses.pop(0)


class FakeModel:
    def __init__(self, agent: FakeAgent) -> None:
        self.agent = agent
        self.bound_tools = None

    def bind_tools(self, tools: list) -> FakeAgent:
        self.bound_tools = tools
        return self.agent


class FakeSearchTool:
    name = "WebSearchTool"

    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> dict:
        self.calls.append(args)
        return self.result


class SearchAgentTests(unittest.TestCase):
    def test_run_search_agent_collects_results_and_streams_events(self) -> None:
        agent = FakeAgent(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "WebSearchTool",
                            "args": {"query": "official Python documentation"},
                            "id": "search-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Python.org provides the official documentation."),
            ]
        )
        model = FakeModel(agent)
        events: list[dict] = []
        search_result = {
            "ok": True,
            "query": "official Python documentation",
            "answer": "Python.org hosts Python's documentation.",
            "results": [
                {
                    "title": "Python Docs",
                    "url": "https://docs.python.org/3/",
                    "content": "Official Python documentation.",
                    "score": 0.99,
                }
            ],
        }
        search_tool = FakeSearchTool(search_result)

        with (
            patch("byteclaw.agents.search_agent.create_model", return_value=model),
            patch("byteclaw.agents.search_agent.WebSearchTool", search_tool),
        ):
            update = run_search_agent(
                {
                    "task": "Build a Python application",
                    "research_notes": ["Prefer primary sources"],
                },
                "Find the official language documentation",
                writer=events.append,
            )

        self.assertTrue(update["ok"])
        self.assertEqual(
            update["summary"],
            "Python.org provides the official documentation.",
        )
        self.assertEqual(
            update["queries"], ["official Python documentation"]
        )
        self.assertEqual(update["sources"], ["https://docs.python.org/3/"])
        self.assertEqual(
            [event["type"] for event in update["tool_events"]],
            ["tool_call", "search_results"],
        )
        self.assertEqual(events, update["tool_events"])
        self.assertEqual(model.bound_tools, [search_tool])
        self.assertEqual(
            search_tool.calls, [{"query": "official Python documentation"}]
        )

        messages = update["messages"]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertEqual(messages[0].content, SEARCH_AGENT_PROMPT)
        self.assertIsInstance(messages[1], HumanMessage)
        request = json.loads(messages[1].content)
        self.assertEqual(request["task"], "Build a Python application")
        self.assertEqual(
            request["instruction"], "Find the official language documentation"
        )
        self.assertEqual(request["research_notes"], ["Prefer primary sources"])
        self.assertIsInstance(messages[-2], ToolMessage)
        self.assertEqual(messages[-2].tool_call_id, "search-1")


if __name__ == "__main__":
    unittest.main()
