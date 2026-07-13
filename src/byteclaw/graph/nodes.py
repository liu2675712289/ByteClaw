"""Core intent, planner, specialist-supervisor, and verifier nodes."""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.config import get_stream_writer
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field

from byteclaw.core.state import RuntimeState
from byteclaw.graph.memory import (
    _short_text,
    _trim_handoffs,
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
)
from byteclaw.graph.state import (
    ByteGraphState,
    SourceItem,
    TodoItem,
    VerificationResult,
)
from byteclaw.prompts.stage2 import ACTOR_PROMPT
from byteclaw.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from byteclaw.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from byteclaw.providers.openai_provider import create_model
from byteclaw.tools.registry import build_read_only_tools, build_tools

TodoStatus = Literal["pending", "in_progress", "completed", "blocked"]
MAX_PLANNER_STEPS = 10
MAX_ACTOR_STEPS = 25
MAX_VERIFIER_TOOL_STEPS = 10

INTENT_ROUTER_PROMPT = """You are the intent router for ByteClaw.

Classify the user's latest input into exactly one route:
- chat: greetings, thanks, identity/help questions, ordinary conceptual Q&A,
  or conversational messages that do not need workspace access.
- workflow: any request that needs creating/editing/reading files, running commands,
  installing packages, searching the web, checking the current project, verifying a
  result, or producing a concrete deliverable.

When session context is provided, use it only to understand whether the latest
input is a continuation of prior coding work. A short follow-up like "继续",
"修一下", or "运行测试" should be workflow if it refers to prior workspace work.

Return only JSON with this shape:
{"route":"chat"|"workflow","reason":"brief reason","confidence":0.0}

If uncertain, choose workflow.
"""

CHAT_RESPONDER_PROMPT = """You are ByteClaw's lightweight chat node.

Answer the user directly and concisely. Do not claim that you read files,
searched the web, ran commands, edited files, or inspected the workspace.
If the user asks for work requiring tools or project context, say that it
should be handled by the workflow route.

If session context is provided, you may use the recent conversation summary to
answer conversational follow-ups, but do not invent workspace facts.
"""

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


class IntentRouterOutput(BaseModel):
    route: Literal["chat", "workflow"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


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


def intent_router_node(state: ByteGraphState) -> dict:
    """Classify the latest input as lightweight chat or workflow work."""

    request = {
        "user_input": state.get("task", ""),
        "session_context": state.get(
            "session_context", state.get("session", {})
        ),
    }
    response = create_model().invoke(
        [
            SystemMessage(content=INTENT_ROUTER_PROMPT),
            HumanMessage(
                content=json.dumps(request, ensure_ascii=False, default=str)
            ),
        ]
    )
    try:
        result = IntentRouterOutput.model_validate(
            _parse_json_content(response.content)
        )
    except ValueError:
        return {
            "intent_route": "workflow",
            "intent_reason": "Invalid intent router response",
            "intent_confidence": 0.0,
        }

    route = result.route if result.confidence >= 0.55 else "workflow"
    return {
        "intent_route": route,
        "intent_reason": result.reason,
        "intent_confidence": result.confidence,
    }


def chat_responder_node(state: ByteGraphState) -> dict:
    """Answer a conversational request without binding or calling tools."""

    request = {
        "user_input": state.get("task", ""),
        "session_context": state.get(
            "session_context", state.get("session", {})
        ),
    }
    response = create_model().invoke(
        [
            SystemMessage(content=CHAT_RESPONDER_PROMPT),
            HumanMessage(
                content=json.dumps(request, ensure_ascii=False, default=str)
            ),
        ]
    )
    content = _content_to_text(response.content)
    return {"chat_response": content, "final_answer": content}


def intent_route_fn(state: ByteGraphState) -> str:
    return (
        "chat_responder"
        if state.get("intent_route") == "chat"
        else "planner"
    )


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


def _planner_input(state: ByteGraphState, memory: dict) -> str:
    if state.get("todos"):
        request = {
            "task": state.get("task", ""),
            "session_context": state.get("session_context", ""),
            "current_plan": _plan_fields(state),
            "last_error": state.get("last_error", ""),
            "instruction": "Revise the plan to address the verification failure.",
        }
    else:
        request = {
            "task": state.get("task", ""),
            "session_context": state.get("session_context", ""),
            "instruction": "Create the initial implementation plan.",
        }
    return (
        f"{json.dumps(request, ensure_ascii=False, default=str)}"
        "\n\nLayered memory:\n"
        f"{format_layered_memory_for_prompt(memory)}"
    )


def planner_node(state: ByteGraphState) -> dict:
    """Plan work and supervise searchAgent and codeAgent through tools."""

    tools = [TodoWriteTool, CallSearchAgentTool, CallCodeAgentTool]
    agent = create_model().bind_tools(tools)
    writer = _event_writer()
    working_state: ByteGraphState = dict(state)
    initial_message_count = len(state.get("messages", []))
    memory = build_layered_memory(working_state, node="planner")
    working_state["memory_snapshot"] = memory
    writer(memory_event(memory, node="planner"))
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=_planner_input(working_state, memory)),
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
        "memory_snapshot",
    ):
        if field in working_state:
            update[field] = working_state[field]
    new_messages = working_state.get("messages", [])[initial_message_count:]
    if new_messages:
        update["messages"] = new_messages
    update["context_next_node"] = "verifier"
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


def _verifier_input(
    state: ByteGraphState,
    verification_results: list[VerificationResult],
    memory: dict,
) -> str:
    verifier_context = {
        "task": state.get("task", ""),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "verification_results": verification_results,
        "code_agent_summary": state.get("code_agent_summary", ""),
    }
    return (
        f"{json.dumps(verifier_context, ensure_ascii=False, default=str)}"
        "\n\nLayered memory:\n"
        f"{format_layered_memory_for_prompt(memory)}"
    )


def verifier_node(state: ByteGraphState) -> dict:
    """Run verification commands and independently assess acceptance criteria."""

    runtime: RuntimeState = state["runtime"]
    verification_results = [
        _run_verification_command(command, runtime.workspace)
        for command in state.get("verification_commands", [])
    ]
    memory = build_layered_memory(state, node="verifier")
    writer = _event_writer()
    writer(memory_event(memory, node="verifier"))

    tools = build_read_only_tools(runtime)
    tools_by_name = {tool.name: tool for tool in tools}
    model = create_model()
    agent = model.bind_tools(tools)
    messages = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(
            content=_verifier_input(state, verification_results, memory)
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
        "memory_snapshot": memory,
    }
    if not passed:
        update["last_error"] = last_error
        update["context_next_node"] = "planner"
    return update


def context_monitor_node(state: ByteGraphState) -> dict:
    """Estimate prompt size and decide whether context needs compression."""

    memory = build_layered_memory(state, node="context_monitor")
    memory_payload = SystemMessage(
        content=format_layered_memory_for_prompt(memory)
    )
    messages = [*state.get("messages", []), memory_payload]
    model = create_model()
    try:
        token_count = model.get_num_tokens_from_messages(messages)
    except Exception:
        text = "\n".join(
            _content_to_text(getattr(message, "content", message))
            for message in messages
        )
        token_count = len(text) // 4

    context_token_limit = state.get("context_token_limit", 400000)
    return {
        "context_token_count": token_count,
        "context_should_compress": token_count > context_token_limit,
        "context_next_node": state.get("context_next_node", "verifier"),
    }


def context_monitor_route(state: ByteGraphState) -> str:
    """Route through compression when the assembled context is too large."""

    if state.get("passed"):
        return "final"
    if state.get("context_should_compress"):
        return "context_compressor"
    return state.get("context_next_node", "verifier")


def _compression_handoffs(state: ByteGraphState) -> list[dict]:
    handoffs = _trim_handoffs(state.get("agent_handoffs", []))
    for handoff in handoffs:
        if "instruction" in handoff:
            handoff["instruction"] = _short_text(
                handoff["instruction"], 600
            )
        if "result" in handoff:
            handoff["result"] = _short_text(handoff["result"], 1000)
    return handoffs


def _compression_sources(sources: Any) -> list[dict[str, str]]:
    normalized = _normalized_sources(sources if isinstance(sources, list) else [])
    return [
        {
            "title": _short_text(source.get("title", ""), 300),
            "url": _short_text(source.get("url", ""), 1000),
        }
        for source in normalized
    ]


def context_compressor_node(state: ByteGraphState) -> dict:
    """Compress graph messages and persist a durable history summary."""

    memory = build_layered_memory(state, node="context_compressor")
    serialized_messages = [
        message.model_dump(mode="json")
        if hasattr(message, "model_dump")
        else message
        for message in state.get("messages", [])
    ]
    compression_context = {
        "messages": serialized_messages,
        "layered_memory": memory,
    }
    model = create_model()
    response = model.invoke(
        [
            SystemMessage(content=CONTEXT_COMPRESSION_PROMPT),
            HumanMessage(
                content=json.dumps(
                    compression_context,
                    ensure_ascii=False,
                    default=str,
                )
            ),
        ]
    )
    try:
        compressed = _parse_json_content(response.content)
    except ValueError:
        compressed = {"summary": _content_to_text(response.content)}
    summary = _short_text(compressed.get("summary", ""), 6000)

    compressed_message = AIMessage(content=summary)
    try:
        token_count = model.get_num_tokens_from_messages([compressed_message])
    except Exception:
        token_count = len(summary) // 4

    history_path = state["runtime"].workspace / "HISTORY_SUMMARY.md"
    history_path.write_text(summary, encoding="utf-8")

    compression_events = [
        dict(event) for event in state.get("compression_events", [])
    ]
    compression_events.append(
        {
            "node": "context_compressor",
            "token_count_before": state.get("context_token_count", 0),
            "token_count_after": token_count,
            "summary_chars": len(summary),
            "next_node": state.get("context_next_node", "verifier"),
        }
    )

    sources = compressed.get("sources") or state.get("sources", [])
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            compressed_message,
        ],
        "context_summary": summary,
        "context_token_count": token_count,
        "context_should_compress": False,
        "plan_summary": _short_text(state.get("plan_summary", ""), 1200),
        "research_notes": _short_text(
            state.get("research_notes", ""), 1600
        ),
        "sources": _compression_sources(sources),
        "agent_handoffs": _compression_handoffs(state),
        "code_agent_summary": _short_text(
            state.get("code_agent_summary", ""), 1000
        ),
        "verifier_summary": _short_text(
            state.get("verifier_summary", ""), 1000
        ),
        "last_error": _short_text(state.get("last_error", ""), 1400),
        "history_summary": summary,
        "compression_events": compression_events,
    }


def context_compressor_route(state: ByteGraphState) -> str:
    """Resume at the node selected before context compression."""

    return state.get("context_next_node", "verifier")


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
