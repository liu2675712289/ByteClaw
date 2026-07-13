import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from byteclaw.core.state import RuntimeState
from byteclaw.graph.nodes import (
    CallCodeAgentTool,
    CallSearchAgentTool,
    TodoUpdateTool,
    TodoWriteTool,
    _call_code_agent_tool,
    _call_search_agent_tool,
    actor_node,
    planner_node,
    verifier_node,
    verifier_route,
)


class FakeTool:
    def __init__(self, name: str, result: object) -> None:
        self.name = name
        self.result = result
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> object:
        self.calls.append(args)
        return self.result


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

    def invoke(self, messages: list) -> AIMessage:
        return self.agent.invoke(messages)


class GraphNodeTests(unittest.TestCase):
    def test_planner_creates_structured_plan(self) -> None:
        plan_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "TodoWriteTool",
                    "args": {
                        "plan_summary": "Build and test the page",
                        "todos": [
                            {
                                "id": "1",
                                "content": "Create index.html",
                                "status": "pending",
                                "note": "",
                            }
                        ],
                        "acceptance_criteria": ["Page loads"],
                        "verification_commands": [
                            "cd /workspace && python -m unittest"
                        ],
                    },
                    "id": "plan-1",
                    "type": "tool_call",
                },
                {
                    "name": "CallSearchAgentTool",
                    "args": {"instruction": "Find official guidance"},
                    "id": "search-1",
                    "type": "tool_call",
                },
                {
                    "name": "CallCodeAgentTool",
                    "args": {"instruction": "Implement the plan"},
                    "id": "code-1",
                    "type": "tool_call",
                },
            ],
        )
        agent = FakeAgent(
            [plan_response, AIMessage(content="Implementation delegated.")]
        )
        model = FakeModel(agent)
        code_message = AIMessage(content="Implemented index.html")

        with (
            patch("byteclaw.graph.nodes.create_model", return_value=model),
            patch(
                "byteclaw.graph.nodes.run_search_agent",
                return_value={
                    "ok": True,
                    "summary": "Official guidance",
                    "sources": ["https://example.com/docs"],
                },
            ),
            patch(
                "byteclaw.graph.nodes.run_code_agent",
                return_value={
                    "ok": True,
                    "summary": "Implemented index.html",
                    "todos": [
                        {
                            "id": "1",
                            "content": "Create index.html",
                            "status": "completed",
                            "note": "Done",
                        }
                    ],
                    "messages": [code_message],
                },
            ),
        ):
            update = planner_node({"task": "Build a page"})

        self.assertEqual(update["plan_summary"], "Build and test the page")
        self.assertEqual(update["todos"][0]["status"], "completed")
        self.assertEqual(
            update["verification_commands"], ["python -m unittest"]
        )
        self.assertEqual(
            model.bound_tools,
            [TodoWriteTool, CallSearchAgentTool, CallCodeAgentTool],
        )
        self.assertEqual(update["research_notes"], "Official guidance")
        self.assertEqual(
            update["sources"], [{"url": "https://example.com/docs"}]
        )
        self.assertEqual(update["code_agent_summary"], "Implemented index.html")
        self.assertEqual(update["messages"], [code_message])
        self.assertEqual(
            [item["to_agent"] for item in update["agent_handoffs"]],
            ["searchAgent", "codeAgent"],
        )

    def test_planner_revises_failed_plan(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "TodoWriteTool",
                    "args": {
                    "plan_summary": "Revised plan",
                    "todos": [
                        {
                            "id": "1",
                            "content": "Fix test",
                            "status": "pending",
                            "note": "",
                        }
                    ],
                    "acceptance_criteria": ["Tests pass"],
                    "verification_commands": ["python -m unittest"],
                    },
                    "id": "plan-1",
                    "type": "tool_call",
                },
                {
                    "name": "CallCodeAgentTool",
                    "args": {"instruction": "Fix only the failing test"},
                    "id": "code-1",
                    "type": "tool_call",
                },
            ],
        )
        agent = FakeAgent([response, AIMessage(content="Fix delegated.")])

        with (
            patch(
                "byteclaw.graph.nodes.create_model",
                return_value=FakeModel(agent),
            ),
            patch(
                "byteclaw.graph.nodes.run_code_agent",
                return_value={
                    "ok": True,
                    "summary": "Fixed the failing test",
                    "todos": [
                        {
                            "id": "1",
                            "content": "Fix test",
                            "status": "completed",
                            "note": "Done",
                        }
                    ],
                    "messages": [],
                },
            ) as code_agent,
        ):
            update = planner_node(
                {
                    "task": "Fix project",
                    "todos": [
                        {
                            "id": "1",
                            "content": "Old step",
                            "status": "blocked",
                            "note": "failed",
                        }
                    ],
                    "last_error": "Tests failed",
                }
            )

        prompt = agent.message_snapshots[0][-1].content
        self.assertIn("Tests failed", prompt)
        self.assertEqual(update["plan_summary"], "Revised plan")
        code_agent.assert_called_once()
        self.assertEqual(
            code_agent.call_args.args[1], "Fix only the failing test"
        )

    def test_specialist_tool_helpers_update_handoff_state(self) -> None:
        state = {
            "research_notes": "Existing note",
            "sources": [{"url": "https://example.com/existing"}],
            "todos": [],
            "messages": [],
        }
        code_message = AIMessage(content="Code work")
        events: list[dict] = []

        with (
            patch(
                "byteclaw.graph.nodes.run_search_agent",
                return_value={
                    "summary": "New research",
                    "sources": [
                        "https://example.com/existing",
                        "https://example.com/new",
                    ],
                },
            ),
            patch(
                "byteclaw.graph.nodes.run_code_agent",
                return_value={
                    "summary": "Implemented",
                    "todos": [
                        {
                            "id": "1",
                            "content": "Implement",
                            "status": "completed",
                            "note": "Done",
                        }
                    ],
                    "messages": [code_message],
                },
            ),
        ):
            _call_search_agent_tool(state, events.append, "Research")
            _call_code_agent_tool(state, events.append, "Implement")

        self.assertEqual(state["research_notes"], "Existing note\n\nNew research")
        self.assertEqual(
            state["sources"],
            [
                {"url": "https://example.com/existing"},
                {"url": "https://example.com/new"},
            ],
        )
        self.assertEqual(state["todos"][0]["status"], "completed")
        self.assertEqual(state["code_agent_summary"], "Implemented")
        self.assertEqual(state["messages"], [code_message])
        self.assertEqual(
            [event["to"] for event in events if event["type"] == "handoff"],
            ["searchAgent", "codeAgent"],
        )

    def test_actor_streams_events_and_updates_todo(self) -> None:
        file_tool = FakeTool("file_write", "Wrote index.html")
        agent = FakeAgent(
            [
                AIMessage(
                    content="Working",
                    tool_calls=[
                        {
                            "name": "TodoUpdateTool",
                            "args": {
                                "id": "1",
                                "status": "in_progress",
                                "note": "Started",
                            },
                            "id": "todo-1",
                            "type": "tool_call",
                        },
                        {
                            "name": "file_write",
                            "args": {
                                "file_path": "index.html",
                                "content": "hello",
                            },
                            "id": "file-1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="Created index.html"),
            ]
        )
        model = FakeModel(agent)
        events: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            state = {
                "task": "Build a page",
                "runtime": RuntimeState(Path(temp_dir)),
                "plan_summary": "Create the page",
                "todos": [
                    {
                        "id": "1",
                        "content": "Create index.html",
                        "status": "pending",
                        "note": "",
                    }
                ],
                "acceptance_criteria": ["Page loads"],
            }
            with (
                patch(
                    "byteclaw.graph.nodes.build_tools", return_value=[file_tool]
                ),
                patch("byteclaw.graph.nodes.create_model", return_value=model),
                patch(
                    "byteclaw.graph.nodes.get_stream_writer",
                    return_value=events.append,
                ),
            ):
                update = actor_node(state)

        self.assertEqual(update["last_actor_summary"], "Created index.html")
        self.assertEqual(update["todos"][0]["status"], "in_progress")
        self.assertTrue(any(isinstance(item, ToolMessage) for item in update["messages"]))
        self.assertEqual(file_tool.calls[0]["file_path"], "index.html")
        self.assertIs(model.bound_tools[-1], TodoUpdateTool)
        self.assertEqual(
            [event["type"] for event in events],
            [
                "ai_message",
                "tool_call",
                "tool_result",
                "tool_call",
                "tool_result",
                "ai_message",
                "final_answer",
            ],
        )

    def test_actor_reports_when_tool_step_limit_is_reached(self) -> None:
        file_tool = FakeTool("file_read", "contents")
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "file_read",
                        "args": {"file_path": "example.txt"},
                        "id": f"read-{index}",
                        "type": "tool_call",
                    }
                ],
            )
            for index in range(2)
        ]
        events: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            state = {
                "task": "Inspect a file",
                "runtime": RuntimeState(Path(temp_dir)),
            }
            with (
                patch(
                    "byteclaw.graph.nodes.build_tools", return_value=[file_tool]
                ),
                patch(
                    "byteclaw.graph.nodes.create_model",
                    return_value=FakeModel(FakeAgent(responses)),
                ),
                patch("byteclaw.graph.nodes.MAX_ACTOR_STEPS", 2),
                patch(
                    "byteclaw.graph.nodes.get_stream_writer",
                    return_value=events.append,
                ),
            ):
                update = actor_node(state)

        self.assertIn("tool-step limit", update["last_actor_summary"])
        self.assertEqual(events[-1]["content"], update["last_actor_summary"])

    def test_verifier_runs_commands_and_completes_todos(self) -> None:
        verdict = {
            "passed": True,
            "reason": "All checks passed",
            "checks": [
                {"name": "output", "passed": True, "detail": "Looks correct"}
            ],
            "recommended_next_instruction": "",
        }
        agent = FakeAgent([AIMessage(content=json.dumps(verdict))])

        with tempfile.TemporaryDirectory() as temp_dir:
            command = f'"{sys.executable}" -c "print(123)"'
            state = {
                "runtime": RuntimeState(Path(temp_dir)),
                "verification_commands": [command],
                "todos": [
                    {
                        "id": "1",
                        "content": "Build",
                        "status": "in_progress",
                        "note": "",
                    }
                ],
                "attempts": 1,
            }
            with (
                patch("byteclaw.graph.nodes.build_read_only_tools", return_value=[]),
                patch(
                    "byteclaw.graph.nodes.create_model",
                    return_value=FakeModel(agent),
                ),
            ):
                update = verifier_node(state)

        self.assertTrue(update["passed"])
        self.assertEqual(update["attempts"], 2)
        self.assertEqual(update["verification_results"][0]["exit_code"], 0)
        self.assertIn("123", update["verification_results"][0]["stdout"])
        self.assertEqual(update["todos"][0]["status"], "completed")

    def test_failed_command_overrides_model_pass(self) -> None:
        verdict = {
            "passed": True,
            "reason": "Output looks correct",
            "checks": [],
            "recommended_next_instruction": "Fix the failing command",
        }
        agent = FakeAgent([AIMessage(content=json.dumps(verdict))])

        with tempfile.TemporaryDirectory() as temp_dir:
            command = f'"{sys.executable}" -c "import sys; sys.exit(7)"'
            state = {
                "runtime": RuntimeState(Path(temp_dir)),
                "verification_commands": [command],
                "todos": [
                    {
                        "id": "1",
                        "content": "Build",
                        "status": "in_progress",
                        "note": "",
                    }
                ],
            }
            with (
                patch("byteclaw.graph.nodes.build_read_only_tools", return_value=[]),
                patch(
                    "byteclaw.graph.nodes.create_model",
                    return_value=FakeModel(agent),
                ),
            ):
                update = verifier_node(state)

        self.assertFalse(update["passed"])
        self.assertEqual(update["verification_results"][0]["exit_code"], 7)
        self.assertIn("Failed commands", update["last_error"])
        self.assertEqual(update["todos"][0]["status"], "blocked")

    def test_verifier_forces_final_verdict_after_tool_step_limit(self) -> None:
        read_tool = FakeTool("file_read", "contents")
        tool_responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "file_read",
                        "args": {"file_path": "example.txt"},
                        "id": f"read-{index}",
                        "type": "tool_call",
                    }
                ],
            )
            for index in range(2)
        ]
        verdict = AIMessage(
            content=json.dumps(
                {
                    "passed": True,
                    "reason": "Inspection passed",
                    "checks": [],
                    "recommended_next_instruction": "",
                }
            )
        )
        agent = FakeAgent([*tool_responses, verdict])

        with tempfile.TemporaryDirectory() as temp_dir:
            state = {"runtime": RuntimeState(Path(temp_dir))}
            with (
                patch(
                    "byteclaw.graph.nodes.build_read_only_tools",
                    return_value=[read_tool],
                ),
                patch(
                    "byteclaw.graph.nodes.create_model",
                    return_value=FakeModel(agent),
                ),
                patch("byteclaw.graph.nodes.MAX_VERIFIER_TOOL_STEPS", 2),
            ):
                update = verifier_node(state)

        self.assertTrue(update["passed"])
        self.assertEqual(len(read_tool.calls), 2)
        self.assertIn(
            "Return the final JSON verdict now",
            agent.message_snapshots[-1][-1].content,
        )

    def test_verifier_route(self) -> None:
        self.assertEqual(verifier_route({"passed": True}), "final")
        self.assertEqual(
            verifier_route({"passed": False, "attempts": 3, "max_attempts": 3}),
            "final",
        )
        self.assertEqual(
            verifier_route({"passed": False, "attempts": 1, "max_attempts": 3}),
            "planner",
        )


if __name__ == "__main__":
    unittest.main()
