"""AgentRAG REST API — FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    from agentrag.agent.runner import create_agent

    try:
        _agent = await create_agent()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Agent init failed (will retry on first request): {e}")
        _agent = None
    yield
    _agent = None


app = FastAPI(
    title="AgentRAG API",
    description="Multi-Source Agentic RAG with MCP",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str
    model: str | None = None
    verbose: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []


class IndexRequest(BaseModel):
    path: str
    type: str = "doc"  # "doc" or "code"


class SearchRequest(BaseModel):
    query: str
    source: str = "all"  # "docs", "code", "memory", "all"
    top_k: int = 5


@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    """Ask a question — agent retrieves from multiple sources and answers."""
    from agentrag.agent.runner import create_agent, run_query

    agent = _agent
    if req.model:
        agent = await create_agent(model_name=req.model)

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    result = await run_query(agent, req.query)
    return QueryResponse(answer=result["answer"], sources=result.get("sources", []))


@app.post("/index")
async def index(req: IndexRequest):
    """Index a file or directory."""
    from pathlib import Path

    target = Path(req.path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    if req.type == "code":
        from agentrag.mcp_servers.code_index.server import index_repo

        result = await index_repo(str(target))
    else:
        from agentrag.mcp_servers.docs_rag.server import index_document

        if target.is_file():
            result = await index_document(str(target))
        else:
            results = []
            for f in target.rglob("*"):
                if f.is_file() and f.suffix in {".txt", ".md", ".pdf", ".json", ".html"}:
                    r = await index_document(str(f))
                    results.append(r)
            result = f"Indexed {len(results)} files"

    return {"result": result}


@app.post("/search")
async def search(req: SearchRequest):
    """Direct search without agent reasoning."""
    results = {}

    if req.source in ("docs", "all"):
        from agentrag.mcp_servers.docs_rag.server import search_documents

        results["docs"] = await search_documents(req.query, req.top_k)

    if req.source in ("code", "all"):
        from agentrag.mcp_servers.code_index.server import search_code

        results["code"] = await search_code(req.query, req.top_k)

    if req.source in ("memory", "all"):
        from agentrag.mcp_servers.memory.server import recall

        results["memory"] = await recall(req.query, req.top_k)

    return results


@app.get("/status")
async def status():
    """System status."""
    from agentrag.config import settings

    info = {"model": settings.model, "embedding": settings.embedding_model}

    from agentrag.storage.chroma import get_collection_count

    for name in ["documents", "code", "memories"]:
        info[name] = get_collection_count(name)

    return info


@app.get("/health")
async def health():
    return {"status": "ok"}
