"""Focused code-implementation agent."""

import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from byteclaw.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
)
from byteclaw.graph.nodes import TodoUpdateTool
from byteclaw.providers.openai_provider import create_model
from byteclaw.tools.registry import build_tools

CODE_AGENT_PROMPT = """You are codeAgent, a focused implementation specialist.

You implement the planner's instruction inside the workspace using file and
shell tools.

Rules:
- You must update todo progress explicitly.
- Before starting a todo, call TodoUpdateTool with status "in_progress".
- After finishing that todo, call TodoUpdateTool with status "completed".
- If a todo is impossible, call TodoUpdateTool with status "blocked" and explain.
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool for non-interactive checks.
- Use NotepadAppendTool to record durable findings, decisions, important files,
  blockers, and next-step context that should survive compression.
- Use NotepadReadTool when you need to recover prior notes.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- Incorporate research notes and source URLs when the task asks for researched content.
- End with a concise summary of files changed and checks run.
"""


def build_layered_memory_snapshot(state: dict) -> Any:
    """Build and retain the current codeAgent memory snapshot."""

    memory = build_layered_memory(state, node="codeAgent")
    state["memory_snapshot"] = memory
    return memory


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _execute_tool(call: dict, tools_by_name: dict[str, Any]) -> Any:
    tool = tools_by_name.get(call["name"])
    if tool is None:
        return {"error": f"Unknown tool: {call['name']}"}
    try:
        return tool.invoke(call["args"])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _update_todos(state: dict, args: dict) -> tuple[list[dict], dict]:
    update = TodoUpdateTool.model_validate(args)
    todos = [dict(todo) for todo in state.get("todos", [])]
    result = {"error": f"Unknown todo id: {update.id}"}
    for todo in todos:
        if todo["id"] == update.id:
            todo["status"] = update.status
            todo["note"] = update.note
            result = {
                "id": update.id,
                "status": update.status,
                "note": update.note,
            }
            break
    state["todos"] = todos
    return todos, result


def _code_agent_input(state: dict, instruction: str, memory: dict) -> str:
    request = {
        "task": state.get("task", ""),
        "instruction": instruction,
        "session_context": state.get(
            "session_context", state.get("session", {})
        ),
        "todos": state.get("todos", []),
        "research_notes": state.get("research_notes", []),
        "source_urls": state.get("sources", []),
    }
    return (
        f"{json.dumps(request, ensure_ascii=False, default=str)}"
        "\n\nLayered memory:\n"
        f"{format_layered_memory_for_prompt(memory)}"
    )


def run_code_agent(
    state: dict,
    instruction: str,
    *,
    writer: Callable[[dict], Any] | None = None,
    max_loops: int = 10,
) -> dict:
    """Implement one planner instruction and persist todo progress in state."""

    tools = build_tools(state["runtime"])
    tools_by_name = {tool.name: tool for tool in tools}
    agent = create_model().bind_tools([*tools, TodoUpdateTool])
    memory = build_layered_memory_snapshot(state)
    memory_snapshot_event = memory_event(memory, node="codeAgent")
    tool_events: list[dict] = [memory_snapshot_event]
    if writer is not None:
        writer(memory_snapshot_event)
    messages = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=_code_agent_input(state, instruction, memory)),
    ]
    todos = [dict(todo) for todo in state.get("todos", [])]
    summary = ""
    last_content = ""

    for _ in range(max_loops):
        response = agent.invoke(messages)
        messages.append(response)
        last_content = _content_to_text(response.content)
        if not response.tool_calls:
            summary = last_content
            break

        for call in response.tool_calls:
            call_event = {
                "type": "tool_call",
                "name": call["name"],
                "args": call["args"],
            }
            tool_events.append(call_event)
            if writer is not None:
                writer(call_event)

            if call["name"] == TodoUpdateTool.__name__:
                todos, result = _update_todos(state, call["args"])
            else:
                result = _execute_tool(call, tools_by_name)

            result_event = {
                "type": "tool_result",
                "name": call["name"],
                "result": result,
            }
            tool_events.append(result_event)
            if writer is not None:
                writer(result_event)

            messages.append(
                ToolMessage(
                    content=json.dumps(
                        result, ensure_ascii=False, default=str
                    ),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )

    if not summary:
        summary = last_content or (
            f"codeAgent reached the {max_loops}-loop limit without a final summary."
        )

    return {
        "ok": True,
        "summary": summary,
        "todos": todos,
        "messages": messages,
        "tool_events": tool_events,
    }
