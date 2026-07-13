"""ReAct agent loop with streamable events."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from byteclaw.core.state import RuntimeState
from byteclaw.providers.openai_provider import create_model
from byteclaw.tools.registry import build_tools

ACTOR_PROMPT = """You are the actor node in ByteClaw's ReAct workflow.

You implement the user's task using tools. Work inside the workspace only.

Rules:
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool to run commands and test results.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- End with a concise summary of files changed and commands run.
"""


def _execute_tool(call: dict, tools_by_name: dict[str, Any]) -> Any:
    """Execute one model tool call and return an error value on failure."""

    name = call["name"]
    tool = tools_by_name.get(name)
    if tool is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return tool.invoke(call["args"])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def stream_agent_events(
    task: str,
    *,
    workspace: str | Path,
    max_loops: int = 10,
) -> Iterator[dict]:
    """Run the ByteClaw ReAct loop and yield progress events."""

    state = RuntimeState(Path(workspace))
    messages = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(content=task),
    ]
    tools = build_tools(state)
    tools_by_name = {tool.name: tool for tool in tools}
    agent = create_model().bind_tools(tools)
    last_ai_content: Any = ""

    for _ in range(max_loops):
        response = agent.invoke(messages)
        messages.append(response)
        last_ai_content = response.content
        yield {"type": "ai_message", "content": last_ai_content}

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            name = call["name"]
            yield {"type": "tool_call", "name": name, "args": call["args"]}
            result = _execute_tool(call, tools_by_name)
            tool_message = ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=call["id"],
                name=name,
            )
            messages.append(tool_message)
            yield {"type": "tool_result", "name": name, "result": result}

    yield {"type": "final_answer", "content": last_ai_content}

