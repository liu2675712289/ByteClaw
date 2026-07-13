import os
import unittest
from unittest.mock import patch

from byteclaw.tools.web_search_tool import WebSearchTool


class FakeTavilyClient:
    api_key = None
    search_kwargs = None

    def __init__(self, *, api_key: str) -> None:
        type(self).api_key = api_key

    def search(self, **kwargs):
        type(self).search_kwargs = kwargs
        return {
            "answer": "ByteClaw is a coding agent.",
            "results": [
                {
                    "title": "ByteClaw",
                    "url": "https://example.com/byteclaw",
                    "content": "A workspace-scoped coding agent.",
                    "score": 0.9,
                    "raw_content": "ignored",
                }
            ],
        }


class WebSearchToolTests(unittest.TestCase):
    def test_missing_api_key_returns_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = WebSearchTool.invoke({"query": "ByteClaw"})

        self.assertEqual(
            result, {"ok": False, "error": "missing TAVILY_API_KEY"}
        )

    def test_search_returns_normalized_tavily_results(self) -> None:
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}),
            patch(
                "byteclaw.tools.web_search_tool.TavilyClient",
                FakeTavilyClient,
            ),
        ):
            result = WebSearchTool.invoke({"query": "ByteClaw"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["query"], "ByteClaw")
        self.assertEqual(result["answer"], "ByteClaw is a coding agent.")
        self.assertEqual(
            result["results"],
            [
                {
                    "title": "ByteClaw",
                    "url": "https://example.com/byteclaw",
                    "content": "A workspace-scoped coding agent.",
                    "score": 0.9,
                }
            ],
        )
        self.assertEqual(FakeTavilyClient.api_key, "test-key")
        self.assertEqual(
            FakeTavilyClient.search_kwargs,
            {"query": "ByteClaw", "include_answer": True},
        )


if __name__ == "__main__":
    unittest.main()
