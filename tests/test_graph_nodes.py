import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import (
    AIMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from byteclaw.core.state import RuntimeState
from byteclaw.graph.nodes import (
    CallCodeAgentTool,
    CallSearchAgentTool,
    TodoUpdateTool,
    TodoWriteTool,
    _call_code_agent_tool,
    _call_search_agent_tool,
    actor_node,
    context_compressor_node,
    context_compressor_route,
    context_monitor_node,
    context_monitor_route,
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
        memory = {
            "rules": {},
            "working_memory": {"node": "planner"},
            "history_summary_store": {},
        }

        with (
            patch("byteclaw.graph.nodes.create_model", return_value=model),
            patch(
                "byteclaw.graph.nodes.build_layered_memory",
                return_value=memory,
            ),
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
        self.assertEqual(update["context_next_node"], "verifier")
        self.assertEqual(update["memory_snapshot"], memory)
        self.assertIn(
            "Layered memory:", agent.message_snapshots[0][-1].content
        )
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
        memory = {
            "rules": {},
            "working_memory": {"node": "planner"},
            "history_summary_store": {},
        }

        with (
            patch(
                "byteclaw.graph.nodes.create_model",
                return_value=FakeModel(agent),
            ),
            patch(
                "byteclaw.graph.nodes.build_layered_memory",
                return_value=memory,
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
        self.assertIn("Layered memory:", prompt)
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
        events: list[dict] = []

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
                patch(
                    "byteclaw.graph.nodes.get_stream_writer",
                    return_value=events.append,
                ),
            ):
                update = verifier_node(state)

        self.assertTrue(update["passed"])
        self.assertEqual(update["attempts"], 2)
        self.assertEqual(update["verification_results"][0]["exit_code"], 0)
        self.assertIn("123", update["verification_results"][0]["stdout"])
        self.assertEqual(update["todos"][0]["status"], "completed")
        self.assertEqual(events[0]["type"], "memory")
        self.assertEqual(events[0]["node"], "verifier")
        self.assertEqual(
            update["memory_snapshot"]["working_memory"]["node"],
            "verifier",
        )
        self.assertIn(
            "Layered memory:", agent.message_snapshots[0][-1].content
        )

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
        self.assertEqual(update["context_next_node"], "planner")

    def test_context_monitor_uses_model_token_count(self) -> None:
        class TokenModel:
            def __init__(self) -> None:
                self.messages = []

            def get_num_tokens_from_messages(self, messages):
                self.messages = messages
                return 25

        model = TokenModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            state = {
                "runtime": RuntimeState(Path(temp_dir)),
                "messages": [AIMessage(content="Current context")],
                "context_token_limit": 20,
                "context_next_node": "planner",
            }
            with patch("byteclaw.graph.nodes.create_model", return_value=model):
                update = context_monitor_node(state)

        self.assertEqual(update["context_token_count"], 25)
        self.assertTrue(update["context_should_compress"])
        self.assertEqual(update["context_next_node"], "planner")
        self.assertIsInstance(model.messages[-1], SystemMessage)
        self.assertIn("working_memory", model.messages[-1].content)

    def test_context_monitor_falls_back_to_character_estimate(self) -> None:
        class FailingTokenModel:
            def get_num_tokens_from_messages(self, messages):
                raise RuntimeError("tokenizer unavailable")

        state = {"messages": [AIMessage(content="x" * 20)]}
        with (
            patch("byteclaw.graph.nodes.create_model", return_value=FailingTokenModel()),
            patch("byteclaw.graph.nodes.build_layered_memory", return_value={}),
            patch(
                "byteclaw.graph.nodes.format_layered_memory_for_prompt",
                return_value="y" * 20,
            ),
        ):
            update = context_monitor_node(
                {**state, "context_token_limit": 9}
            )

        self.assertEqual(update["context_token_count"], 10)
        self.assertTrue(update["context_should_compress"])
        self.assertEqual(update["context_next_node"], "verifier")

    def test_context_monitor_route(self) -> None:
        self.assertEqual(
            context_monitor_route(
                {"passed": True, "context_should_compress": True}
            ),
            "final",
        )
        self.assertEqual(
            context_monitor_route({"context_should_compress": True}),
            "context_compressor",
        )
        self.assertEqual(
            context_monitor_route({"context_next_node": "planner"}),
            "planner",
        )
        self.assertEqual(context_monitor_route({}), "verifier")

    def test_context_compressor_route(self) -> None:
        self.assertEqual(
            context_compressor_route({"context_next_node": "planner"}),
            "planner",
        )
        self.assertEqual(context_compressor_route({}), "verifier")

    def test_context_compressor_replaces_and_persists_history(self) -> None:
        compressed_payload = {
            "summary": "Compressed task context",
            "active_goal": "Finish the implementation",
            "completed_work": ["Created the memory module"],
            "open_todos": ["Run verification"],
            "important_files": ["src/example.py"],
            "tool_findings": ["Tests currently pass"],
            "sources": [
                {
                    "title": "Official docs",
                    "url": "https://example.com/docs",
                }
            ],
            "next_steps": ["Verify the result"],
            "risks": [],
        }

        class CompressionModel:
            def __init__(self) -> None:
                self.messages = []

            def invoke(self, messages):
                self.messages = messages
                return AIMessage(content=json.dumps(compressed_payload))

            def get_num_tokens_from_messages(self, messages):
                return 7

        model = CompressionModel()
        old_message = AIMessage(content="Old transcript", id="old-1")
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            state = {
                "runtime": RuntimeState(workspace),
                "task": "Finish the feature",
                "messages": [old_message],
                "plan_summary": "P" * 1300,
                "research_notes": "R" * 1700,
                "sources": [],
                "agent_handoffs": [
                    {
                        "instruction": str(index) + "I" * 700,
                        "result": "X" * 1100,
                    }
                    for index in range(8)
                ],
                "code_agent_summary": "C" * 1100,
                "verifier_summary": "V" * 1100,
                "last_error": "E" * 1500,
                "context_token_count": 500000,
                "context_next_node": "planner",
                "compression_events": [{"node": "previous"}],
            }
            with patch("byteclaw.graph.nodes.create_model", return_value=model):
                update = context_compressor_node(state)

            history = (workspace / "HISTORY_SUMMARY.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(history, "Compressed task context")
        self.assertIsInstance(update["messages"][0], RemoveMessage)
        self.assertEqual(update["messages"][0].id, REMOVE_ALL_MESSAGES)
        self.assertEqual(
            update["messages"][1].content, "Compressed task context"
        )
        reduced = add_messages(state["messages"], update["messages"])
        self.assertEqual(len(reduced), 1)
        self.assertEqual(reduced[0].content, "Compressed task context")
        self.assertEqual(update["context_summary"], "Compressed task context")
        self.assertEqual(update["history_summary"], "Compressed task context")
        self.assertEqual(update["context_token_count"], 7)
        self.assertFalse(update["context_should_compress"])
        self.assertEqual(len(update["plan_summary"]), 1200)
        self.assertEqual(len(update["research_notes"]), 1600)
        self.assertEqual(len(update["agent_handoffs"]), 6)
        self.assertEqual(len(update["agent_handoffs"][0]["instruction"]), 600)
        self.assertEqual(len(update["agent_handoffs"][0]["result"]), 1000)
        self.assertEqual(
            update["sources"],
            [
                {
                    "title": "Official docs",
                    "url": "https://example.com/docs",
                }
            ],
        )
        self.assertEqual(len(update["compression_events"]), 2)
        event = update["compression_events"][-1]
        self.assertEqual(event["token_count_before"], 500000)
        self.assertEqual(event["token_count_after"], 7)
        self.assertEqual(event["next_node"], "planner")
        self.assertIsInstance(model.messages[0], SystemMessage)
        self.assertIn("context_compressor", model.messages[0].content)
        self.assertIn("Old transcript", model.messages[1].content)
        self.assertIn("layered_memory", model.messages[1].content)

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
