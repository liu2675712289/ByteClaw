"""Core planner, actor, and verifier nodes for the ByteClaw graph."""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel

from byteclaw.core.state import RuntimeState
from byteclaw.graph.state import ByteGraphState, TodoItem, VerificationResult
from byteclaw.prompts.stage2 import ACTOR_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from byteclaw.providers.openai_provider import create_model
from byteclaw.tools.registry import build_read_only_tools, build_tools

TodoStatus = Literal["pending", "in_progress", "completed", "blocked"]


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


def _tool_payload(response: Any, tool_type: type[BaseModel]) -> dict[str, Any]:
    for call in response.tool_calls:
        if call["name"] == tool_type.__name__:
            return call["args"]
    return _parse_json_content(response.content)


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


def planner_node(state: ByteGraphState) -> dict:
    """Create an initial plan or revise it after failed verification."""

    if state.get("todos") and not state.get("last_error"):
        return _plan_fields(state)

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

    model = create_model().bind_tools([TodoWriteTool])
    response = model.invoke(
        [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=json.dumps(request, ensure_ascii=False, default=str)),
        ]
    )
    plan = TodoWriteTool.model_validate(
        _tool_payload(response, TodoWriteTool)
    ).model_dump()
    return plan


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

    for _ in range(10):
        response = agent.invoke(messages)
        messages.append(response)
        new_messages.append(response)
        last_actor_summary = _content_to_text(response.content)
        writer({"type": "ai_message", "content": response.content})

        if not response.tool_calls:
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
        "last_actor_summary": state.get("last_actor_summary", ""),
    }

    tools = build_read_only_tools(runtime)
    tools_by_name = {tool.name: tool for tool in tools}
    agent = create_model().bind_tools(tools)
    messages = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(
            content=json.dumps(verifier_context, ensure_ascii=False, default=str)
        ),
    ]
    final_response = None
    for _ in range(10):
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
        raise RuntimeError("Verifier exceeded the 10-step tool-call limit")

    verdict = VerifierOutput.model_validate(
        _parse_json_content(final_response.content)
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
    if state.get("last_actor_summary"):
        lines.extend(["", "Actor summary:", state["last_actor_summary"]])
    if not state.get("passed") and state.get("last_error"):
        lines.extend(["", "Last error:", state["last_error"]])
    return {"final_answer": "\n".join(lines)}

