"""MCP Server: Memory — persistent conversation memory and knowledge."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("memory", instructions="Store and retrieve persistent memories and knowledge.")

MEMORY_DIR = Path("./data/memories")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


@mcp.tool()
async def save_memory(content: str, tags: list[str] | None = None, category: str = "general") -> str:
    """Save a piece of information to persistent memory."""
    mem_id = f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    memory = {
        "content": content,
        "tags": tags or [],
        "category": category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        from agentrag.storage.chroma import embed_and_upsert

        embed_and_upsert(
            "memories",
            [mem_id],
            [content],
            [{"category": category, "tags": json.dumps(tags or []), "timestamp": memory["timestamp"]}],
        )

        mem_file = MEMORY_DIR / f"{mem_id}.json"
        mem_file.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")

        return f"Memory saved (id: {mem_id})"
    except Exception as e:
        return f"Error saving memory: {e}"


@mcp.tool()
async def recall(query: str, top_k: int = 5) -> str:
    """Search memories by semantic similarity."""
    try:
        from agentrag.storage.chroma import get_collection_count, query_collection

        if get_collection_count("memories") == 0:
            return "No memories stored yet."

        results = query_collection("memories", query, top_k)
        if not results:
            return "No relevant memories found."

        output = []
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            ts = meta.get("timestamp", "?")[:10]
            cat = meta.get("category", "?")
            output.append(f"[{i+1}] ({cat}, {ts}) {doc[:300]}")

        return "\n\n".join(output) if output else "No relevant memories found."
    except Exception as e:
        return f"Error recalling: {e}"


@mcp.tool()
async def list_memories(category: str | None = None) -> str:
    """List all stored memories, optionally filtered by category."""
    try:
        from agentrag.storage.chroma import get_collection_count, get_collection_data

        if get_collection_count("memories") == 0:
            return "No memories stored."

        where = {"category": category} if category else None
        results = get_collection_data("memories", where)

        output = [f"Total: {len(results['ids'])} memories"]
        for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            output.append(f"  [{doc_id}] {meta.get('category', '?')} — {doc[:80]}")

        return "\n".join(output)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
