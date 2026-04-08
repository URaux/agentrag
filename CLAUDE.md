# CLAUDE.md — AgentRAG Project

## Shell & Platform
Windows machine. Use PowerShell/cmd syntax. Path separator `\` or `/`.
PYTHONIOENCODING=utf-8 needed for Rich CLI output.

## Project Overview
Multi-Source Agentic RAG with MCP — resume project for AI Agent Engineer roles.
GitHub: https://github.com/URaux/agentrag

## Architecture
- **Agent**: LangGraph ReAct with MCPAgentRuntime (agentrag/agent/runner.py)
- **MCP Servers** (4 independent subprocesses via stdio):
  - docs-rag: document indexing & semantic search
  - web-search: DuckDuckGo + Tavily + Crawl4AI
  - code-index: code repo semantic search
  - memory: persistent memory store
- **Storage**: Qdrant local (file named chroma.py for historical reasons)
- **LLM**: DeepSeek (api.deepseek.com), model=deepseek-chat
- **Embeddings**: text-embedding-3-large via api.xiaocaseai.cloud
- **Interfaces**: CLI (Click), REST API (FastAPI), Web UI (Streamlit)

## Key Files
- agentrag/agent/runner.py — MCPAgentRuntime, MCP client lifecycle
- agentrag/agent/graph.py — LangGraph ReAct graph, max_tool_rounds=3
- agentrag/storage/chroma.py — Qdrant wrapper, sync + async variants
- agentrag/config.py — Pydantic settings, AGENTRAG_ env prefix
- agentrag/mcp_servers/*/server.py — FastMCP servers
- agentrag/api/app.py — FastAPI
- agentrag/ui/app.py — Streamlit (NotebookLM style)
- agentrag/cli/main.py — Click CLI
- .env — API keys (NOT in git)

## Running
```bash
# CLI
agentrag ask "question"
agentrag index ./file.md --type doc
agentrag serve  # API on :8000

# Web UI
streamlit run agentrag/ui/app.py --server.port 8501 --server.headless true

# API
uvicorn agentrag.api.app:app --port 8000
```

## Git
- All commits must be authored by: URaux <1303113210@qq.com>
- Local git config already set. Do NOT use Co-Authored-By with Claude/OpenClaw.
- .env is in .gitignore. Never commit API keys.
- gh CLI at ~/bin/gh.exe

## Known Issues
- DeepSeek tends to loop on tool calls → max_tool_rounds=3 limiter
- xiaocaseai proxy does NOT support function calling for chat models
- Crawl4AI needs `playwright install` for browser-based fetching (not yet run)

## Next Steps (discussed but not yet done)
- Mixed MCP/direct architecture (web-search via MCP, memory/code as direct imports)
- RAG evaluation script (LLM-as-judge scoring)
- Agent self-correction on bad retrieval
- README update to reflect all new features
