"""MCP Server: Web Search — real-time web search and page fetching."""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("web-search", instructions="Search the web and fetch page content.")


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
async def fetch_page(url: str, max_length: int = 3000) -> str:
    """Fetch and extract text content from a web page.

    Args:
        url: URL to fetch.
        max_length: Maximum characters to return.
    """
    try:
        import httpx

        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": "AgentRAG/0.1"})
            resp.raise_for_status()

        # Simple HTML to text
        text = resp.text
        # Strip HTML tags (basic)
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"

        return f"Content from {url}:\n{text}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


if __name__ == "__main__":
    mcp.run()
