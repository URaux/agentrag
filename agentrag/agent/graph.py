"""LangGraph agent core — ReAct agent with MCP tool routing."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_count: int


def build_agent_graph(llm, tools: list, max_tool_rounds: int = 3):
    """Build a ReAct agent with a hard limit on tool-calling rounds."""
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        count = state.get("tool_call_count", 0)
        if count >= max_tool_rounds:
            # Force final answer without tools
            response = llm.invoke(state["messages"])
            return {"messages": [response], "tool_call_count": count}
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "tool_call_count": count}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    async def tools_with_count(state: AgentState) -> dict:
        result = await tool_node.ainvoke(state)
        new_count = state.get("tool_call_count", 0) + 1
        return {"messages": result["messages"], "tool_call_count": new_count}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_with_count)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


async def run_agent(graph, query: str, system_prompt: str | None = None) -> dict:
    """Run the agent graph with a query."""
    messages: list[BaseMessage] = []
    if system_prompt:
        from langchain_core.messages import SystemMessage
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=query))

    result = await graph.ainvoke(
        {"messages": messages, "tool_call_count": 0},
        config={"recursion_limit": 20},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    answer = ai_messages[-1].content if ai_messages else ""

    sources = []
    for msg in result["messages"]:
        if hasattr(msg, "name") and hasattr(msg, "content"):
            if msg.name and msg.name != "agent":
                sources.append({"tool": msg.name, "content": str(msg.content)[:200]})

    return {"answer": answer, "sources": sources}
