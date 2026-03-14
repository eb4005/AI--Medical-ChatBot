"""
utils/web_search.py
Real-time web search via the Tavily API.
Returns a clean, formatted string that can be appended to an LLM prompt.
"""

import logging
from typing import Optional

from tavily import TavilyClient

logger = logging.getLogger(__name__)


def web_search(
    query: str,
    api_key: str,
    max_results: int = 5,
) -> str:
    """
    Execute a real-time web search using the Tavily API.

    Args:
        query:       The search query string.
        api_key:     Tavily API key.
        max_results: Maximum number of results to include (default 5).

    Returns:
        A formatted multi-line string containing result titles, URLs,
        and short content snippets — ready to be injected into an LLM prompt.

    Raises:
        RuntimeError: If the search request fails.
    """
    try:
        if not api_key:
            raise ValueError(
                "Tavily API key is missing. Please enter it in the sidebar."
            )
        if not query.strip():
            raise ValueError("Search query must not be empty.")

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,        # Tavily's own summarised answer
            include_raw_content=False,
        )

        # ── Format output ──────────────────────────────────────────────────────
        lines = ["### 🌐 Real-Time Web Search Results\n"]

        # Tavily sometimes returns a top-level answer
        top_answer: Optional[str] = response.get("answer")
        if top_answer:
            lines.append(f"**Summary:** {top_answer}\n")

        results = response.get("results", [])
        for i, result in enumerate(results, start=1):
            title   = result.get("title", "No title")
            url     = result.get("url", "")
            content = result.get("content", "")[:300]   # first 300 chars
            lines.append(f"**[{i}] {title}**")
            lines.append(f"URL: {url}")
            lines.append(f"Snippet: {content}…\n")

        formatted_output = "\n".join(lines)
        logger.info("Tavily search returned %d result(s).", len(results))
        return formatted_output

    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        raise RuntimeError(f"Web search failed: {exc}") from exc
