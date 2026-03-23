"""MCP Server: Code Index — index and search code repositories."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("code-index", instructions="Index and search code repositories.")

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    ".yaml", ".yml", ".toml", ".json", ".md", ".txt",
}


@mcp.tool()
async def index_repo(repo_path: str, max_files: int = 200) -> str:
    """Index a code repository for semantic search."""
    root = Path(repo_path)
    if not root.is_dir():
        return f"Error: Not a directory: {repo_path}"

    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox"}
    files_indexed = 0
    total_chunks = 0

    try:
        from agentrag.storage.chroma import embed_and_upsert

        for path in root.rglob("*"):
            if files_indexed >= max_files:
                break
            if any(skip in path.parts for skip in skip_dirs):
                continue
            if not path.is_file() or path.suffix.lower() not in CODE_EXTENSIONS:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if len(content) < 10:
                continue

            rel_path = str(path.relative_to(root))
            chunk_size = 600
            chunks, ids, metadatas = [], [], []
            for i, start in enumerate(range(0, len(content), chunk_size - 100)):
                chunk = content[start : start + chunk_size]
                if chunk.strip():
                    doc_id = hashlib.md5(rel_path.encode()).hexdigest()[:10]
                    chunks.append(f"# File: {rel_path}\n{chunk}")
                    ids.append(f"code_{doc_id}_{i}")
                    metadatas.append({"source": rel_path, "repo": repo_path, "chunk_index": i, "language": path.suffix})

            if chunks:
                embed_and_upsert("code", ids, chunks, metadatas)
                files_indexed += 1
                total_chunks += len(chunks)

        return f"Indexed {files_indexed} files ({total_chunks} chunks) from {root.name}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def search_code(query: str, top_k: int = 5, language: str | None = None) -> str:
    """Search indexed code using semantic similarity."""
    try:
        from agentrag.storage.chroma import get_collection_count, query_collection

        if get_collection_count("code") == 0:
            return "No code indexed. Use index_repo first."

        where = {"language": language} if language else None
        results = query_collection("code", query, top_k, where)
        if not results:
            return "No results found."

        output = []
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            score = 1 - dist
            source = meta.get("source", "?")
            output.append(f"[{i+1}] (score: {score:.3f}) {source}\n```\n{doc[:400]}\n```")

        return "\n\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
