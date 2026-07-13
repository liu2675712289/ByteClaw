"""Tavily-backed web search tool."""

import os
from typing import Any

from langchain_core.tools import tool

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - covered by the package dependency
    TavilyClient = None


@tool
def WebSearchTool(query: str) -> dict[str, Any]:
    """Search the web with Tavily and return normalized source results."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"ok": False, "error": "missing TAVILY_API_KEY"}
    if TavilyClient is None:
        return {
            "ok": False,
            "query": query,
            "error": "tavily-python is not installed",
        }

    try:
        response = TavilyClient(api_key=api_key).search(
            query=query,
            include_answer=True,
        )
    except Exception as exc:
        return {
            "ok": False,
            "query": query,
            "error": f"{type(exc).__name__}: {exc}",
        }

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score"),
        }
        for item in response.get("results", [])
        if isinstance(item, dict)
    ]
    return {
        "ok": True,
        "query": query,
        "answer": response.get("answer", ""),
        "results": results,
    }
