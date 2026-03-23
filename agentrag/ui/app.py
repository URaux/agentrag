"""AgentRAG Web UI — Streamlit application."""

from __future__ import annotations

import asyncio

import streamlit as st

st.set_page_config(page_title="AgentRAG", page_icon="🔍", layout="wide")

st.title("🔍 AgentRAG")
st.caption("Multi-Source Agentic RAG with MCP — Ask anything, retrieve from everywhere.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"], index=0)
    verbose = st.checkbox("Show sources", value=True)

    st.divider()
    st.header("Index Documents")
    upload_path = st.text_input("File or directory path")
    doc_type = st.radio("Type", ["Document", "Code"], horizontal=True)
    if st.button("Index") and upload_path:
        with st.spinner("Indexing..."):
            if doc_type == "Document":
                from agentrag.mcp_servers.docs_rag.server import index_document

                result = asyncio.run(index_document(upload_path))
            else:
                from agentrag.mcp_servers.code_index.server import index_repo

                result = asyncio.run(index_repo(upload_path))
            st.success(result)

    st.divider()
    st.header("Status")
    if st.button("Refresh"):
        try:
            from agentrag.storage.chroma import get_collection_count

            for name in ["documents", "code", "memories"]:
                try:
                    st.metric(name.title(), get_collection_count(name))
                except Exception:
                    st.metric(name.title(), 0)
        except Exception as e:
            st.error(str(e))

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.text(f"{src['tool']}: {src['content'][:150]}")

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from agentrag.agent.runner import create_agent, run_query

                agent = asyncio.run(create_agent(model_name=model))
                result = asyncio.run(run_query(agent, prompt))

                st.markdown(result["answer"])
                if verbose and result.get("sources"):
                    with st.expander("Sources"):
                        for src in result["sources"]:
                            st.text(f"{src['tool']}: {src['content'][:150]}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                })
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
