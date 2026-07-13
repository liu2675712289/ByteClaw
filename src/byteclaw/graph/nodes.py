"""Core planner, specialist-supervisor, and verifier nodes."""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel

from byteclaw.core.state import RuntimeState
from byteclaw.graph.state import (
    ByteGraphState,
    SourceItem,
    TodoItem,
    VerificationResult,
)
from byteclaw.prompts.stage2 import ACTOR_PROMPT
from byteclaw.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from byteclaw.providers.openai_provider import create_model
from byteclaw.tools.registry import build_read_only_tools, build_tools

TodoStatus = Literal["pending", "in_progress", "completed", "blocked"]
MAX_PLANNER_STEPS = 10
MAX_ACTOR_STEPS = 25
MAX_VERIFIER_TOOL_STEPS = 10

_WORKSPACE_CD_PREFIX = re.compile(
    r"^\s*cd\s+[\"']?/workspace[\"']?\s*(?:&&|;)\s*",
    flags=re.IGNORECASE,
)


class TodoInput(BaseModel):
    id: str
    content: str
    status: TodoStatus = "pending"
    note: str = ""


class TodoWriteTool(BaseModel):
    """Write a complete implementation and verification plan."""

    plan_summary: str
    todos: list[TodoInput]
    acceptance_criteria: list[str]
    verification_commands: list[str]


class TodoUpdateTool(BaseModel):
    """Update the status and note of one todo item."""

    id: str
    status: TodoStatus
    note: str = ""


class CallSearchAgentTool(BaseModel):
    """Delegate a research instruction to searchAgent."""

    instruction: str


class CallCodeAgentTool(BaseModel):
    """Delegate an implementation instruction to codeAgent."""

    instruction: str


class VerificationCheckOutput(BaseModel):
    name: str
    passed: bool
    detail: str


class VerifierOutput(BaseModel):
    passed: bool
    reason: str
    checks: list[VerificationCheckOutput]
    recommended_next_instruction: str


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _parse_json_content(content: Any) -> dict[str, Any]:
    text = _content_to_text(content).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from the model")
    return value


def _execute_tool(call: dict, tools_by_name: dict[str, Any]) -> Any:
    tool = tools_by_name.get(call["name"])
    if tool is None:
        return {"error": f"Unknown tool: {call['name']}"}
    try:
        return tool.invoke(call["args"])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _event_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda event: None


def _plan_fields(state: ByteGraphState) -> dict:
    return {
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
    }


def _workspace_relative_commands(commands: list[str]) -> list[str]:
    normalized = [
        _WORKSPACE_CD_PREFIX.sub("", command, count=1).strip()
        for command in commands
    ]
    return [command for command in normalized if command]


def run_search_agent(
    state: dict, instruction: str, *, writer: Any = None
) -> dict:
    """Load and run searchAgent without creating an import cycle."""

    from byteclaw.agents.search_agent import run_search_agent as agent_runner

    return agent_runner(state, instruction, writer=writer)


def run_code_agent(
    state: dict, instruction: str, *, writer: Any = None
) -> dict:
    """Load and run codeAgent without creating an import cycle."""

    from byteclaw.agents.code_agent import run_code_agent as agent_runner

    return agent_runner(state, instruction, writer=writer)


def _result_summary(result: dict) -> str:
    summary = result.get("summary") or result.get("error") or ""
    return _content_to_text(summary)


def _normalized_sources(items: Any) -> list[SourceItem]:
    sources: list[SourceItem] = []
    if not isinstance(items, list):
        return sources
    for item in items:
        if isinstance(item, str):
            source: SourceItem = {"url": item}
        elif isinstance(item, dict):
            source = dict(item)
        else:
            continue
        if source and source not in sources:
            sources.append(source)
    return sources


def _append_handoff(
    state: ByteGraphState,
    *,
    to_agent: str,
    instruction: str,
    result: dict,
) -> None:
    handoffs = [dict(item) for item in state.get("agent_handoffs", [])]
    handoffs.append(
        {
            "from_agent": "planner",
            "to_agent": to_agent,
            "instruction": instruction,
            "result": _result_summary(result),
        }
    )
    state["agent_handoffs"] = handoffs


def _call_search_agent_tool(
    state: ByteGraphState, writer: Any, instruction: str
) -> dict:
    writer(
        {
            "type": "handoff",
            "from": "planner",
            "to": "searchAgent",
            "instruction": instruction,
        }
    )
    result = run_search_agent(state, instruction, writer=writer)

    existing_notes = state.get("research_notes", "")
    if isinstance(existing_notes, list):
        existing_notes = "\n\n".join(str(item) for item in existing_notes)
    summary = _result_summary(result)
    if summary and summary not in existing_notes:
        state["research_notes"] = "\n\n".join(
            item for item in (existing_notes, summary) if item
        )
    else:
        state["research_notes"] = existing_notes

    state["sources"] = _normalized_sources(
        [*state.get("sources", []), *result.get("sources", [])]
    )
    _append_handoff(
        state,
        to_agent="searchAgent",
        instruction=instruction,
        result=result,
    )
    return result


def _call_code_agent_tool(
    state: ByteGraphState, writer: Any, instruction: str
) -> dict:
    writer(
        {
            "type": "handoff",
            "from": "planner",
            "to": "codeAgent",
            "instruction": instruction,
        }
    )
    result = run_code_agent(state, instruction, writer=writer)

    state["todos"] = result.get("todos", state.get("todos", []))
    state["code_agent_summary"] = _result_summary(result)
    state["messages"] = [
        *state.get("messages", []),
        *result.get("messages", []),
    ]
    _append_handoff(
        state,
        to_agent="codeAgent",
        instruction=instruction,
        result=result,
    )
    return result


def planner_node(state: ByteGraphState) -> dict:
    """Plan work and supervise searchAgent and codeAgent through tools."""

    if state.get("todos"):
        request = {
            "task": state.get("task", ""),
            "current_plan": _plan_fields(state),
            "last_error": state.get("last_error", ""),
            "instruction": "Revise the plan to address the verification failure.",
        }
    else:
        request = {
            "task": state.get("task", ""),
            "instruction": "Create the initial implementation plan.",
        }

    tools = [TodoWriteTool, CallSearchAgentTool, CallCodeAgentTool]
    agent = create_model().bind_tools(tools)
    writer = _event_writer()
    working_state: ByteGraphState = dict(state)
    initial_message_count = len(state.get("messages", []))
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(
            content=json.dumps(request, ensure_ascii=False, default=str)
        ),
    ]
    completed = False

    for _ in range(MAX_PLANNER_STEPS):
        response = agent.invoke(messages)
        messages.append(response)
        writer({"type": "ai_message", "content": response.content})
        if not response.tool_calls:
            writer(
                {
                    "type": "final_answer",
                    "content": _content_to_text(response.content),
                }
            )
            completed = True
            break

        for call in response.tool_calls:
            name = call["name"]
            writer({"type": "tool_call", "name": name, "args": call["args"]})
            try:
                if name == TodoWriteTool.__name__:
                    result = TodoWriteTool.model_validate(
                        call["args"]
                    ).model_dump()
                    result["verification_commands"] = (
                        _workspace_relative_commands(
                            result["verification_commands"]
                        )
                    )
                    working_state.update(result)
                elif name == CallSearchAgentTool.__name__:
                    instruction = CallSearchAgentTool.model_validate(
                        call["args"]
                    ).instruction
                    result = _call_search_agent_tool(
                        working_state, writer, instruction
                    )
                elif name == CallCodeAgentTool.__name__:
                    instruction = CallCodeAgentTool.model_validate(
                        call["args"]
                    ).instruction
                    result = _call_code_agent_tool(
                        working_state, writer, instruction
                    )
                else:
                    result = {"error": f"Unknown tool: {name}"}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}

            messages.append(
                ToolMessage(
                    content=json.dumps(
                        result, ensure_ascii=False, default=str
                    ),
                    tool_call_id=call["id"],
                    name=name,
                )
            )
            writer({"type": "tool_result", "name": name, "result": result})

    if not completed:
        writer(
            {
                "type": "final_answer",
                "content": (
                    f"Planner reached the {MAX_PLANNER_STEPS}-step limit."
                ),
            }
        )

    update = _plan_fields(working_state)
    for field in (
        "research_notes",
        "sources",
        "agent_handoffs",
        "code_agent_summary",
    ):
        if field in working_state:
            update[field] = working_state[field]
    new_messages = working_state.get("messages", [])[initial_message_count:]
    if new_messages:
        update["messages"] = new_messages
    return update


def _update_todo(todos: list[TodoItem], args: dict) -> tuple[list[TodoItem], Any]:
    update = TodoUpdateTool.model_validate(args)
    updated = [dict(todo) for todo in todos]
    for todo in updated:
        if todo["id"] == update.id:
            todo["status"] = update.status
            todo["note"] = update.note
            return updated, {
                "id": update.id,
                "status": update.status,
                "note": update.note,
            }
    return updated, {"error": f"Unknown todo id: {update.id}"}


def actor_node(state: ByteGraphState) -> dict:
    """Execute the current plan and stream custom ReAct progress events."""

    runtime = state["runtime"]
    tools = build_tools(runtime)
    tools_by_name = {tool.name: tool for tool in tools}
    agent = create_model().bind_tools([*tools, TodoUpdateTool])
    writer = _event_writer()
    todos = [dict(todo) for todo in state.get("todos", [])]
    plan_context = {
        "task": state.get("task", ""),
        "plan_summary": state.get("plan_summary", ""),
        "todos": todos,
        "acceptance_criteria": state.get("acceptance_criteria", []),
    }
    messages = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(
            content=json.dumps(plan_context, ensure_ascii=False, default=str)
        ),
    ]
    new_messages = []
    last_actor_summary = ""

    completed = False
    for _ in range(MAX_ACTOR_STEPS):
        response = agent.invoke(messages)
        messages.append(response)
        new_messages.append(response)
        last_actor_summary = _content_to_text(response.content)
        writer({"type": "ai_message", "content": response.content})

        if not response.tool_calls:
            completed = True
            break

        for call in response.tool_calls:
            name = call["name"]
            writer({"type": "tool_call", "name": name, "args": call["args"]})
            if name == TodoUpdateTool.__name__:
                todos, result = _update_todo(todos, call["args"])
            else:
                result = _execute_tool(call, tools_by_name)
            tool_message = ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=call["id"],
                name=name,
            )
            messages.append(tool_message)
            new_messages.append(tool_message)
            writer({"type": "tool_result", "name": name, "result": result})

    if not completed:
        incomplete_todos = sum(
            todo["status"] != "completed" for todo in todos
        )
        last_actor_summary = (
            f"Actor reached the {MAX_ACTOR_STEPS}-step tool-step limit before "
            f"returning a final response; {incomplete_todos} todo(s) remain "
            "incomplete."
        )

    writer({"type": "final_answer", "content": last_actor_summary})
    return {
        "messages": new_messages,
        "last_actor_summary": last_actor_summary,
        "todos": todos,
    }


def _run_verification_command(
    command: str, workspace: Path, timeout_seconds: float = 60
) -> VerificationResult:
    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "ok": False,
            "exit_code": None,
            "stdout": exc.stdout or "",
            "stderr": f"Command timed out after {timeout_seconds} seconds",
        }
    return {
        "command": command,
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _verified_todos(
    todos: list[TodoItem], passed: bool, failure_note: str
) -> list[TodoItem]:
    updated = [dict(todo) for todo in todos]
    if passed:
        for todo in updated:
            todo["status"] = "completed"
        return updated

    blocked = False
    for todo in updated:
        if todo["status"] == "in_progress":
            todo["status"] = "blocked"
            todo["note"] = failure_note
            blocked = True
    if not blocked:
        for todo in updated:
            if todo["status"] != "completed":
                todo["status"] = "blocked"
                todo["note"] = failure_note
                break
    return updated


def verifier_node(state: ByteGraphState) -> dict:
    """Run verification commands and independently assess acceptance criteria."""

    runtime: RuntimeState = state["runtime"]
    verification_results = [
        _run_verification_command(command, runtime.workspace)
        for command in state.get("verification_commands", [])
    ]
    verifier_context = {
        "task": state.get("task", ""),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "verification_results": verification_results,
        "code_agent_summary": state.get("code_agent_summary", ""),
    }

    tools = build_read_only_tools(runtime)
    tools_by_name = {tool.name: tool for tool in tools}
    model = create_model()
    agent = model.bind_tools(tools)
    messages = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(
            content=json.dumps(verifier_context, ensure_ascii=False, default=str)
        ),
    ]
    final_response = None
    for _ in range(MAX_VERIFIER_TOOL_STEPS):
        response = agent.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            final_response = response
            break
        for call in response.tool_calls:
            result = _execute_tool(call, tools_by_name)
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
    if final_response is None:
        messages.append(
            HumanMessage(
                content=(
                    "Tool inspection is complete. Do not request more tools. "
                    "Return the final JSON verdict now."
                )
            )
        )
        final_response = model.invoke(messages)

    try:
        verdict = VerifierOutput.model_validate(
            _parse_json_content(final_response.content)
        )
    except ValueError as exc:
        verdict = VerifierOutput(
            passed=False,
            reason=(
                "Verifier returned an invalid final verdict: "
                f"{type(exc).__name__}"
            ),
            checks=[],
            recommended_next_instruction=(
                "Re-run verification and return the required JSON verdict."
            ),
        )
    commands_passed = all(result["ok"] for result in verification_results)
    passed = verdict.passed and commands_passed

    failure_parts: list[str] = []
    if not verdict.passed:
        failure_parts.append(verdict.reason)
    failed_commands = [
        result["command"] for result in verification_results if not result["ok"]
    ]
    if failed_commands:
        failure_parts.append(f"Failed commands: {', '.join(failed_commands)}")
    if not passed and verdict.recommended_next_instruction:
        failure_parts.append(
            f"Next instruction: {verdict.recommended_next_instruction}"
        )
    last_error = "\n".join(failure_parts)

    update = {
        "passed": passed,
        "attempts": state.get("attempts", 0) + 1,
        "verification_results": verification_results,
        "verification_checks": [check.model_dump() for check in verdict.checks],
        "todos": _verified_todos(state.get("todos", []), passed, last_error),
    }
    if not passed:
        update["last_error"] = last_error
    return update


def verifier_route(state: ByteGraphState) -> str:
    """Route successful or exhausted workflows to finalization."""

    if state.get("passed"):
        return "final"
    if state.get("attempts", 0) >= state.get("max_attempts", 3):
        return "final"
    return "planner"


def final_node(state: ByteGraphState) -> dict:
    """Format the terminal workflow state as a final answer."""

    status = "PASSED" if state.get("passed") else "FAILED"
    lines = [
        f"Status: {status}",
        f"Attempts: {state.get('attempts', 0)}/{state.get('max_attempts', 0)}",
    ]
    summary = state.get("code_agent_summary") or state.get(
        "last_actor_summary", ""
    )
    if summary:
        lines.extend(["", "Code agent summary:", summary])
    if not state.get("passed") and state.get("last_error"):
        lines.extend(["", "Last error:", state["last_error"]])
    return {"final_answer": "\n".join(lines)}
