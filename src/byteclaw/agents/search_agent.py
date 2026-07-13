"""Focused web-research agent."""

import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from byteclaw.providers.openai_provider import create_model
from byteclaw.tools.web_search_tool import WebSearchTool

SEARCH_AGENT_PROMPT = """You are searchAgent, a focused research specialist.

Your only external capability is WebSearchTool. Search for reliable information
needed by the planner and codeAgent.

Rules:
- Use WebSearchTool for factual research.
- Prefer official or encyclopedia-style sources when available.
- Return a concise research summary and list the useful source URLs.
- Do not write files or produce application code.
"""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _append_unique(items: list[str], value: Any) -> None:
    if isinstance(value, str) and value and value not in items:
        items.append(value)


def run_search_agent(
    state: dict,
    instruction: str,
    *,
    writer: Callable[[dict], Any] | None = None,
    max_loops: int = 4,
) -> dict:
    """Research an instruction with Tavily and return normalized evidence."""

    request = {
        "task": state.get("task", ""),
        "instruction": instruction,
        "research_notes": state.get("research_notes", []),
    }
    messages = [
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(
            content=json.dumps(request, ensure_ascii=False, default=str)
        ),
    ]
    agent = create_model().bind_tools([WebSearchTool])
    queries: list[str] = []
    sources: list[str] = []
    answers: list[str] = []
    tool_events: list[dict] = []
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

            query = call["args"].get("query", "")
            _append_unique(queries, query)
            if call["name"] == WebSearchTool.name:
                result = WebSearchTool.invoke(call["args"])
            else:
                result = {
                    "ok": False,
                    "query": query,
                    "error": f"unknown search tool: {call['name']}",
                }

            if result.get("ok"):
                _append_unique(answers, result.get("answer"))
                for item in result.get("results", []):
                    if isinstance(item, dict):
                        _append_unique(sources, item.get("url"))

            result_event = {
                "type": "search_results",
                "query": query,
                **result,
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
        summary = "\n\n".join(answers) or last_content

    return {
        "ok": True,
        "summary": summary,
        "queries": queries,
        "sources": sources,
        "messages": messages,
        "tool_events": tool_events,
    }
