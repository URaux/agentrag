"""AgentRAG Web UI — NotebookLM-style Streamlit application."""

from __future__ import annotations

import asyncio

import streamlit as st

st.set_page_config(page_title="AgentRAG", page_icon="📚", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Clean, NotebookLM-inspired theme */
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    border-right: 1px solid #e0e0e0;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1a73e8;
}
/* Source cards */
.source-card {
    background: #f1f3f4;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    border-left: 3px solid #1a73e8;
    font-size: 0.85rem;
    line-height: 1.4;
}
.source-card .source-tool {
    color: #1a73e8;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.source-card .source-content {
    color: #5f6368;
}
/* Collection stat pills */
.stat-pill {
    display: inline-block;
    background: #e8f0fe;
    color: #1a73e8;
    border-radius: 16px;
    padding: 4px 14px;
    margin: 3px 4px;
    font-size: 0.82rem;
    font-weight: 500;
}
/* Header area */
.header-subtitle {
    color: #5f6368;
    font-size: 0.95rem;
    margin-top: -8px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_list" not in st.session_state:
    st.session_state.sources_list = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ── Sidebar: Sources Panel ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📚 Sources")

    # Collection stats
    try:
        from agentrag.storage.chroma import get_collection_count
        docs_n = get_collection_count("documents")
        code_n = get_collection_count("code")
        mem_n = get_collection_count("memories")
    except Exception:
        docs_n = code_n = mem_n = 0

    stats_html = (
        f'<span class="stat-pill">📄 {docs_n} docs</span>'
        f'<span class="stat-pill">💻 {code_n} code</span>'
        f'<span class="stat-pill">🧠 {mem_n} mem</span>'
    )
    st.markdown(stats_html, unsafe_allow_html=True)
    st.caption("")

    # Add sources
    with st.expander("➕ Add Source", expanded=len(st.session_state.indexed_files) == 0):
        upload_path = st.text_input("File or directory path", placeholder="/path/to/document.md")
        col1, col2 = st.columns(2)
        with col1:
            doc_type = st.radio("Type", ["Document", "Code"], horizontal=True, label_visibility="collapsed")
        with col2:
            if st.button("Add", use_container_width=True, type="primary"):
                if upload_path:
                    with st.spinner("Indexing..."):
                        if doc_type == "Document":
                            from agentrag.mcp_servers.docs_rag.server import index_document
                            result = asyncio.run(index_document(upload_path))
                        else:
                            from agentrag.mcp_servers.code_index.server import index_repo
                            result = asyncio.run(index_repo(upload_path))
                        st.session_state.indexed_files.append({"path": upload_path, "type": doc_type, "result": result})
                    st.rerun()

    # Show indexed sources as cards
    if st.session_state.indexed_files:
        for i, f in enumerate(st.session_state.indexed_files):
            icon = "📄" if f["type"] == "Document" else "💻"
            name = f["path"].replace("\\", "/").split("/")[-1]
            st.markdown(f"""<div class="source-card">
                <div class="source-tool">{icon} {f['type']}</div>
                <div class="source-content"><b>{name}</b><br/>{f['result']}</div>
            </div>""", unsafe_allow_html=True)

    # Also show from vector DB
    if docs_n > 0 and not st.session_state.indexed_files:
        try:
            from agentrag.storage.chroma import get_collection_data
            data = get_collection_data("documents")
            sources_set = sorted(set(m.get("source", "?") for m in data["metadatas"]))
            for s in sources_set[:10]:
                name = s.replace("\\", "/").split("/")[-1]
                st.markdown(f"""<div class="source-card">
                    <div class="source-tool">📄 Indexed</div>
                    <div class="source-content">{name}</div>
                </div>""", unsafe_allow_html=True)
        except Exception:
            pass

    st.divider()

    # Settings
    with st.expander("⚙️ Settings"):
        model = st.selectbox("Model", ["deepseek-chat", "gpt-4o", "gpt-4o-mini"], index=0)
        show_sources = st.toggle("Show source citations", value=True)

# ── Main: Chat Interface ────────────────────────────────────────────────────
st.markdown("# AgentRAG")
st.markdown('<div class="header-subtitle">Multi-source knowledge assistant — ask anything about your documents, code, or the web.</div>', unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            _render_sources(msg["sources"]) if False else None  # placeholder, rendered below

# Source rendering helper
def _render_sources_html(sources: list[dict]) -> str:
    cards = []
    for src in sources:
        tool = src.get("tool", "unknown")
        content = src.get("content", "")[:200]
        # Map tool names to friendly labels
        labels = {
            "search_documents": "📄 Document Search",
            "search_code": "💻 Code Search",
            "web_search": "🌐 Web Search",
            "tavily_search": "🔍 Tavily Search",
            "fetch_page": "🌐 Web Page",
            "recall": "🧠 Memory",
            "list_indexed": "📋 Index",
        }
        label = labels.get(tool, f"🔧 {tool}")
        cards.append(f"""<div class="source-card">
            <div class="source-tool">{label}</div>
            <div class="source-content">{content}</div>
        </div>""")
    return "".join(cards)

# Re-render with sources
for msg in st.session_state.messages:
    if msg["role"] == "assistant" and show_sources and msg.get("sources"):
        # Sources are rendered inline after the message via the expander below
        pass

# Chat input
if prompt := st.chat_input("Ask about your sources..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Researching your sources..."):
            try:
                from agentrag.agent.runner import create_agent, run_query

                runtime = asyncio.run(create_agent(model_name=model))
                try:
                    result = asyncio.run(run_query(runtime, prompt))
                finally:
                    asyncio.run(runtime.aclose())

                st.markdown(result["answer"])

                sources = result.get("sources", [])
                if show_sources and sources:
                    with st.expander(f"📎 {len(sources)} sources cited", expanded=False):
                        st.markdown(_render_sources_html(sources), unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                })
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
