"""Agent runner — creates and runs the LangGraph agent with MCP tools."""

from __future__ import annotations

from agentrag.agent.graph import build_agent_graph, run_agent
from agentrag.agent.prompts import SYSTEM_PROMPT
from agentrag.config import settings


async def create_agent(model_name: str | None = None):
    """Create the agent graph with MCP tools loaded.

    Args:
        model_name: Override the default model.

    Returns:
        Compiled LangGraph agent.
    """
    model = model_name or settings.model

    # Initialize LLM
    if "claude" in model or "anthropic" in model:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, api_key=settings.anthropic_api_key)
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=settings.api_key,
            base_url=settings.api_base,
        )

    # Load MCP tools via langchain-mcp-adapters
    tools = await _load_mcp_tools()

    return build_agent_graph(llm, tools)


async def _load_mcp_tools():
    """Load tools from MCP servers.

    Uses direct function imports as LangChain tools for simplicity.
    In production, these would connect via MCP stdio/SSE transport.
    """
    from langchain_core.tools import StructuredTool

    # Import MCP server functions directly (in-process for simplicity)
    from agentrag.mcp_servers.docs_rag.server import index_document, list_indexed, search_documents
    from agentrag.mcp_servers.web_search.server import fetch_page, web_search
    from agentrag.mcp_servers.code_index.server import index_repo, search_code
    from agentrag.mcp_servers.memory.server import list_memories, recall, save_memory

    tools = [
        StructuredTool.from_function(coroutine=search_documents, name="search_documents", description="Search indexed documents using semantic similarity."),
        StructuredTool.from_function(coroutine=index_document, name="index_document", description="Index a document file for RAG retrieval."),
        StructuredTool.from_function(coroutine=list_indexed, name="list_indexed", description="List all indexed documents."),
        StructuredTool.from_function(coroutine=web_search, name="web_search", description="Search the web using DuckDuckGo."),
        StructuredTool.from_function(coroutine=fetch_page, name="fetch_page", description="Fetch and extract text from a web page."),
        StructuredTool.from_function(coroutine=search_code, name="search_code", description="Search indexed code using semantic similarity."),
        StructuredTool.from_function(coroutine=index_repo, name="index_repo", description="Index a code repository for semantic search."),
        StructuredTool.from_function(coroutine=save_memory, name="save_memory", description="Save information to persistent memory."),
        StructuredTool.from_function(coroutine=recall, name="recall", description="Search memories by semantic similarity."),
        StructuredTool.from_function(coroutine=list_memories, name="list_memories", description="List all stored memories."),
    ]

    return tools


async def run_query(agent, query: str) -> dict:
    """Run a query through the agent.

    Returns dict with 'answer' and 'sources'.
    """
    return await run_agent(agent, query, system_prompt=SYSTEM_PROMPT)
