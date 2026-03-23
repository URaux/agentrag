"""MCP Server: Document RAG — index and search local documents."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("docs-rag", instructions="Search and retrieve from indexed documents.")

SUPPORTED = {".txt", ".md", ".pdf", ".py", ".json", ".html", ".csv", ".toml", ".yaml", ".yml", ".rst", ".xml"}


def _smart_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    def split_keep_separator(value: str, separator: str) -> list[str]:
        parts = []
        start = 0
        while True:
            index = value.find(separator, start)
            if index == -1:
                tail = value[start:]
                if tail:
                    parts.append(tail)
                return parts
            parts.append(value[start : index + len(separator)])
            start = index + len(separator)

    def split_recursive(value: str, level: int) -> list[str]:
        if len(value) <= chunk_size:
            return [value]

        if level == 0:
            parts = split_keep_separator(value, "\n\n")
            if len(parts) > 1:
                return [piece for part in parts for piece in split_recursive(part, 1)]
            return split_recursive(value, 1)

        if level == 1:
            parts = split_keep_separator(value, "\n")
            if len(parts) > 1:
                return [piece for part in parts for piece in split_recursive(part, 2)]
            return split_recursive(value, 2)

        if level == 2:
            parts = [part for part in re.split(r"(?<=[.!?] )", value) if part]
            if len(parts) > 1:
                return [piece for part in parts for piece in split_recursive(part, 3)]

        return [value[i : i + chunk_size] for i in range(0, len(value), chunk_size)]

    if not text or chunk_size <= 0:
        return []

    if chunk_size == 1:
        overlap = 0
    else:
        overlap = max(0, min(overlap, chunk_size - 1))

    base_chunks = []
    current = ""
    for piece in split_recursive(text, 0):
        if not piece:
            continue
        if not current:
            current = piece
        elif len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            base_chunks.append(current)
            current = piece
    if current:
        base_chunks.append(current)

    chunks = []
    for chunk in base_chunks:
        if not chunks or overlap == 0:
            if chunk.strip():
                chunks.append(chunk)
            continue

        remaining = chunk
        prefix = chunks[-1][-overlap:]
        while remaining:
            available = max(chunk_size - len(prefix), 1)
            combined = prefix + remaining[:available]
            if combined.strip():
                chunks.append(combined)
            remaining = remaining[available:]
            prefix = chunks[-1][-overlap:]

    return chunks


@mcp.tool()
async def index_document(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> str:
    """Index a document file for RAG retrieval."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    if path.suffix.lower() not in SUPPORTED:
        return f"Error: Unsupported file type: {path.suffix}"

    content = path.read_text(encoding="utf-8", errors="replace")
    chunks = _smart_chunk(content, chunk_size, chunk_overlap)

    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
    try:
        from agentrag.storage.chroma import embed_and_upsert

        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]
        embed_and_upsert("documents", ids, chunks, metadatas)
        return f"Indexed {len(chunks)} chunks from {path.name} (id: {doc_id})"
    except Exception as e:
        return f"Error indexing: {e}"


@mcp.tool()
async def search_documents(query: str, top_k: int = 5) -> str:
    """Search indexed documents using semantic similarity."""
    try:
        from agentrag.storage.chroma import get_collection_count, query_collection

        if get_collection_count("documents") == 0:
            return "No documents indexed yet. Use index_document first."

        results = query_collection("documents", query, top_k)
        if not results:
            return "No results found."

        output_parts = []
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            source = meta.get("source", "unknown")
            score = 1 - dist
            output_parts.append(f"[{i+1}] (score: {score:.3f}) {source}\n{doc[:300]}")

        return "\n\n---\n\n".join(output_parts) if output_parts else "No results found."
    except Exception as e:
        return f"Error searching: {e}"


@mcp.tool()
async def list_indexed() -> str:
    """List all indexed documents."""
    try:
        from agentrag.storage.chroma import get_collection_count, get_collection_data

        count = get_collection_count("documents")
        if count == 0:
            return "No documents indexed."

        data = get_collection_data("documents")
        sources = sorted(set(m.get("source", "unknown") for m in data["metadatas"]))
        return f"Indexed {count} chunks from {len(sources)} files:\n" + "\n".join(f"  - {s}" for s in sources)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
