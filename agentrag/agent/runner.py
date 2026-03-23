"""Agent runner that builds a LangGraph agent backed by MCP clients."""

from __future__ import annotations

import json
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from fastmcp import Client
from pydantic import BaseModel, Field, create_model

from agentrag.agent.graph import build_agent_graph, run_agent
from agentrag.agent.prompts import SYSTEM_PROMPT
from agentrag.config import settings


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for one MCP server process."""

    name: str
    module: str
    tools: tuple[str, ...]


MCP_SERVERS: tuple[MCPServerConfig, ...] = (
    MCPServerConfig(
        name="docs_rag",
        module="agentrag.mcp_servers.docs_rag.server",
        tools=("index_document", "search_documents", "list_indexed"),
    ),
    MCPServerConfig(
        name="web_search",
        module="agentrag.mcp_servers.web_search.server",
        tools=("web_search", "fetch_page"),
    ),
    MCPServerConfig(
        name="code_index",
        module="agentrag.mcp_servers.code_index.server",
        tools=("index_repo", "search_code"),
    ),
    MCPServerConfig(
        name="memory",
        module="agentrag.mcp_servers.memory.server",
        tools=("save_memory", "recall", "list_memories"),
    ),
)


class MCPAgentRuntime:
    """Owns MCP client processes and the compiled LangGraph agent."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.model
        self.graph = None
        self._exit_stack: AsyncExitStack | None = None
        self._clients: dict[str, Client] = {}
        self._tool_clients: dict[str, Client] = {}

    async def start(self) -> MCPAgentRuntime:
        """Start MCP clients, discover tools, and compile the graph."""
        if self.graph is not None:
            return self

        self._exit_stack = AsyncExitStack()
        try:
            llm = _build_llm(self.model_name)
            tools = await self._load_mcp_tools()
            self.graph = build_agent_graph(llm, tools)
            return self
        except Exception:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """Close all MCP client contexts and clear runtime state."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._clients.clear()
        self._tool_clients.clear()
        self.graph = None

    async def __aenter__(self) -> MCPAgentRuntime:
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call a discovered MCP tool and normalize its response to text."""
        client = self._tool_clients[tool_name]
        result = await client.call_tool(tool_name, arguments or {})
        return _stringify_mcp_result(result)

    async def _load_mcp_tools(self) -> list:
        """Start each MCP server and wrap discovered tools for LangChain."""
        if self._exit_stack is None:
            raise RuntimeError("MCP runtime has not been initialized")

        from langchain_core.tools import StructuredTool

        tools = []
        for server in MCP_SERVERS:
            transport = _make_transport(server.module)
            client = await self._exit_stack.enter_async_context(Client(transport))
            self._clients[server.name] = client

            discovered = await client.list_tools()
            by_name = {_tool_attr(tool, "name"): tool for tool in discovered}
            missing = [tool_name for tool_name in server.tools if tool_name not in by_name]
            if missing:
                raise RuntimeError(
                    f"MCP server {server.name} did not expose expected tools: {', '.join(missing)}"
                )

            for tool_name in server.tools:
                tool_meta = by_name[tool_name]
                self._tool_clients[tool_name] = client
                description = _tool_attr(tool_meta, "description") or f"MCP tool {tool_name}"
                input_schema = _tool_attr(tool_meta, "inputSchema", "input_schema") or {}
                args_schema = _build_args_schema(tool_name, input_schema)
                tools.append(
                    StructuredTool.from_function(
                        coroutine=self._make_tool_coroutine(tool_name),
                        name=tool_name,
                        description=description,
                        args_schema=args_schema,
                        infer_schema=False,
                    )
                )

        return tools

    def _make_tool_coroutine(self, tool_name: str):
        async def _invoke(**kwargs: Any) -> str:
            return await self.call_tool(tool_name, kwargs)

        _invoke.__name__ = f"call_{tool_name}"
        return _invoke


def _build_llm(model_name: str):
    """Create the configured chat model."""
    if "claude" in model_name or "anthropic" in model_name:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, api_key=settings.anthropic_api_key)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        api_key=settings.api_key,
        base_url=settings.api_base,
    )


def _make_transport(module: str):
    """Create a PythonStdioTransport for the given module."""
    from fastmcp.client.transports import PythonStdioTransport

    # Resolve the module to a script path
    # e.g. "agentrag.mcp_servers.docs_rag.server" -> actual file path
    import importlib
    mod = importlib.import_module(module)
    script_path = mod.__file__
    if script_path is None:
        raise RuntimeError(f"Cannot resolve module {module} to a file path")

    return PythonStdioTransport(
        script_path=script_path,
        python_cmd=sys.executable or "python",
    )


def _tool_attr(tool: Any, *names: str) -> Any:
    """Read a field from a mapping, pydantic model, or object."""
    if hasattr(tool, "model_dump"):
        tool = tool.model_dump()

    if isinstance(tool, dict):
        for name in names:
            if name in tool:
                return tool[name]
        return None

    for name in names:
        value = getattr(tool, name, None)
        if value is not None:
            return value

    return None


def _build_args_schema(tool_name: str, input_schema: dict[str, Any] | None):
    """Convert MCP JSON schema into a Pydantic model for StructuredTool."""
    schema = input_schema or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}

    for field_name, field_schema in properties.items():
        annotation = _annotation_from_json_schema(field_schema)
        default = ... if field_name in required else field_schema.get("default", None)
        description = field_schema.get("description")
        fields[field_name] = (annotation, Field(default=default, description=description))

    model_name = _sanitize_model_name(tool_name)
    return create_model(model_name, __base__=BaseModel, **fields)


def _sanitize_model_name(tool_name: str) -> str:
    """Create a valid Pydantic model name from a tool name."""
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", tool_name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"tool_{cleaned}"
    parts = [part.capitalize() for part in cleaned.split("_") if part]
    return "".join(parts) + "Args"


def _annotation_from_json_schema(schema: dict[str, Any] | None) -> Any:
    """Map a JSON schema fragment to a Python type annotation."""
    if not schema:
        return Any

    if "anyOf" in schema:
        return _annotation_from_union(schema["anyOf"])
    if "oneOf" in schema:
        return _annotation_from_union(schema["oneOf"])

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return _annotation_from_union([{"type": item} for item in schema_type])

    if schema_type == "null":
        return type(None)
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list[_annotation_from_json_schema(schema.get("items"))]
    if schema_type == "object":
        return dict[str, Any]

    return Any


def _annotation_from_union(options: list[dict[str, Any]]) -> Any:
    """Map oneOf/anyOf schema options to a Python union type."""
    annotations = [_annotation_from_json_schema(option) for option in options]
    filtered = [annotation for annotation in annotations if annotation is not type(None)]
    if len(filtered) != len(annotations):
        if not filtered:
            return Any | None
        union = filtered[0]
        for annotation in filtered[1:]:
            union = union | annotation
        return union | None

    union = filtered[0] if filtered else Any
    for annotation in filtered[1:]:
        union = union | annotation
    return union


def _stringify_mcp_result(result: Any) -> str:
    """Convert FastMCP tool results into plain text for LangChain."""
    if result is None:
        return ""

    content = getattr(result, "content", None)
    if content is not None:
        return _stringify_mcp_result(content)

    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
        content = dumped.get("content")
        if content is not None:
            return _stringify_mcp_result(content)
        return json.dumps(dumped, ensure_ascii=False)

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        if "content" in result:
            return _stringify_mcp_result(result["content"])
        if "text" in result:
            return str(result["text"])
        return json.dumps(result, ensure_ascii=False)

    if isinstance(result, list):
        parts = [_stringify_content_item(item) for item in result]
        return "\n".join(part for part in parts if part).strip()

    return str(result)


def _stringify_content_item(item: Any) -> str:
    """Convert one MCP content item into text."""
    if item is None:
        return ""

    if hasattr(item, "model_dump"):
        item = item.model_dump()

    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        if item.get("type") == "text" and "text" in item:
            return str(item["text"])
        if "text" in item:
            return str(item["text"])
        if "content" in item:
            return _stringify_mcp_result(item["content"])
        return json.dumps(item, ensure_ascii=False)

    text = getattr(item, "text", None)
    if text is not None:
        return str(text)

    return str(item)


async def create_agent(model_name: str | None = None) -> MCPAgentRuntime:
    """Create and start an MCP-backed agent runtime."""
    runtime = MCPAgentRuntime(model_name=model_name)
    return await runtime.start()


async def run_query(agent, query: str) -> dict:
    """Run a query through the agent.

    Returns a dict with `answer` and `sources`.
    """
    graph = getattr(agent, "graph", agent)
    if graph is None:
        raise RuntimeError("Agent graph is not initialized")
    return await run_agent(graph, query, system_prompt=SYSTEM_PROMPT)
