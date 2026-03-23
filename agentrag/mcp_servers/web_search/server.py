"""MCP Server: real-time web search and page fetching."""

from __future__ import annotations

import os
import re
from typing import Literal

from fastmcp import FastMCP

mcp = FastMCP("web-search", instructions="Search the web and fetch page content.")


def _truncate_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        return text[:max_length] + "... [truncated]"
    return text


def _format_search_results(results: list[dict], max_content_length: int = 200) -> str:
    if not results:
        return "No results found."

    output = []
    for i, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url") or result.get("href") or "No URL"
        content = result.get("content") or result.get("body") or ""
        content = _truncate_text(content, max_content_length)
        output.append(f"[{i}] {title}\n    {url}\n    {content}")

    return "\n\n".join(output)


async def _fetch_page_with_httpx(url: str, max_length: int) -> str:
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(url, headers={"User-Agent": "AgentRAG/0.1"})
        resp.raise_for_status()

    text = resp.text
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _truncate_text(text, max_length)

    return f"Content from {url}:\n{text}"


@mcp.tool()
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string.
        num_results: Number of results to return.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return "No results found."

        output = []
        for i, r in enumerate(results):
            output.append(f"[{i+1}] {r['title']}\n    {r['href']}\n    {r['body'][:200]}")

        return "\n\n".join(output)
    except ImportError:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Error searching: {e}"


@mcp.tool()
async def tavily_search(
    query: str,
    num_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
) -> str:
    """Search the web using Tavily.

    Args:
        query: Search query string.
        num_results: Number of results to return.
        search_depth: Tavily search depth, either "basic" or "advanced".
    """
    api_key = os.getenv("AGENTRAG_TAVILY_API_KEY")
    if not api_key:
        return "Error: AGENTRAG_TAVILY_API_KEY is not set. Use web_search instead."

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=num_results,
            search_depth=search_depth,
        )
        results = response.get("results", [])
        return _format_search_results(results)
    except ImportError:
        return "Error: tavily not installed. Run: pip install tavily-python"
    except Exception as e:
        return f"Error searching with Tavily: {e}"


@mcp.tool()
async def fetch_page(url: str, max_length: int = 3000) -> str:
    """Fetch and extract text content from a web page.

    Args:
        url: URL to fetch.
        max_length: Maximum characters to return.
    """
    crawl4ai_error: Exception | None = None

    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            markdown = result.markdown or ""
            markdown = str(markdown).strip()

        if not markdown:
            raise ValueError("Crawl4AI returned empty content")

        markdown = _truncate_text(markdown, max_length)
        return f"Content from {url}:\n{markdown}"
    except Exception as e:
        crawl4ai_error = e

    try:
        return await _fetch_page_with_httpx(url, max_length)
    except Exception as fallback_error:
        return f"Error fetching {url}: Crawl4AI failed ({crawl4ai_error}); fallback failed ({fallback_error})"


if __name__ == "__main__":
    mcp.run()
