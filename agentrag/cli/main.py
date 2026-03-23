"""AgentRAG CLI — command-line interface for multi-source agentic RAG."""

from __future__ import annotations

import asyncio
import os
import sys

# Fix Windows GBK encoding for Rich unicode output
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(package_name="agentrag")
def cli():
    """AgentRAG — Multi-Source Agentic RAG with MCP."""


@cli.command()
@click.argument("query")
@click.option("--model", "-m", default=None, help="LLM model to use.")
@click.option("--verbose", "-v", is_flag=True, help="Show tool calls and sources.")
def ask(query: str, model: str | None, verbose: bool):
    """Ask a question — the agent retrieves from multiple sources and answers."""
    asyncio.run(_ask(query, model, verbose))


async def _ask(query: str, model: str | None, verbose: bool):
    from agentrag.agent.runner import create_agent, run_query

    console.print(Panel(f"[bold]Query:[/bold] {query}", style="blue"))

    runtime = None
    try:
        with console.status("[bold green]Thinking..."):
            runtime = await create_agent(model_name=model)
            result = await run_query(runtime, query)
    finally:
        if runtime is not None:
            await runtime.aclose()

    console.print()
    console.print(Markdown(result["answer"]))

    if verbose and result.get("sources"):
        console.print()
        console.print("[dim]Sources:[/dim]")
        for src in result["sources"]:
            console.print(f"  [dim]- {src['tool']}: {src['content'][:100]}[/dim]")


@cli.command()
@click.argument("path")
@click.option("--type", "-t", "doc_type", type=click.Choice(["doc", "code"]), default="doc")
def index(path: str, doc_type: str):
    """Index a file or directory for RAG retrieval."""
    asyncio.run(_index(path, doc_type))


async def _index(path: str, doc_type: str):
    from pathlib import Path

    target = Path(path).resolve()
    if not target.exists():
        console.print(f"[red]Error: {path} not found[/red]")
        sys.exit(1)

    console.print(f"Indexing [bold]{target}[/bold] as [bold]{doc_type}[/bold]...")

    if doc_type == "doc":
        from agentrag.mcp_servers.docs_rag.server import index_document

        if target.is_file():
            result = await index_document(str(target))
            console.print(result)
        elif target.is_dir():
            count = 0
            for f in target.rglob("*"):
                if f.is_file() and f.suffix in {".txt", ".md", ".pdf", ".json", ".html", ".csv"}:
                    result = await index_document(str(f))
                    console.print(f"  {result}")
                    count += 1
            console.print(f"[green]Indexed {count} files.[/green]")
    else:
        from agentrag.mcp_servers.code_index.server import index_repo

        result = await index_repo(str(target))
        console.print(result)


@cli.command()
@click.argument("query")
@click.option("--source", "-s", type=click.Choice(["docs", "code", "memory", "all"]), default="all")
@click.option("--top-k", "-k", default=5)
def search(query: str, source: str, top_k: int):
    """Search indexed content without agent reasoning."""
    asyncio.run(_search(query, source, top_k))


async def _search(query: str, source: str, top_k: int):
    results = []

    if source in ("docs", "all"):
        from agentrag.mcp_servers.docs_rag.server import search_documents

        r = await search_documents(query, top_k)
        results.append(("Documents", r))

    if source in ("code", "all"):
        from agentrag.mcp_servers.code_index.server import search_code

        r = await search_code(query, top_k)
        results.append(("Code", r))

    if source in ("memory", "all"):
        from agentrag.mcp_servers.memory.server import recall

        r = await recall(query, top_k)
        results.append(("Memory", r))

    for title, content in results:
        console.print(Panel(content, title=f"[bold]{title}[/bold]", style="cyan"))


@cli.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", "-p", default=8000, type=int)
def serve(host: str, port: int):
    """Start the REST API server."""
    console.print(f"Starting API server at [bold]http://{host}:{port}[/bold]")
    import uvicorn

    uvicorn.run("agentrag.api.app:app", host=host, port=port, reload=True)


@cli.command()
def status():
    """Show AgentRAG status — indexed docs, config, etc."""
    asyncio.run(_status())


async def _status():
    from agentrag.config import settings

    console.print(Panel("[bold]AgentRAG Status[/bold]", style="green"))
    console.print(f"  Model: {settings.model}")
    console.print(f"  Embedding: {settings.embedding_model}")
    console.print(f"  Data dir: {settings.data_dir}")

    from agentrag.storage.chroma import get_collection_count

    for name in ["documents", "code", "memories"]:
        count = get_collection_count(name)
        if count > 0:
            console.print(f"  {name}: {count} items")
        else:
            console.print(f"  {name}: [dim]empty[/dim]")


if __name__ == "__main__":
    cli()
