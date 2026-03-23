# AgentRAG — 多源智能体 RAG + MCP

[English](README.md) | **中文**

一个智能研究助手，结合**检索增强生成 (RAG)** 与**自主智能体**，通过 **Model Context Protocol (MCP)** 驱动工具调用。

Agent 会推理你的问题，自动选择检索工具，从多个数据源获取信息，最终生成连贯的回答 —— 支持 CLI、REST API 和 Web UI 三种交互方式。

## 架构

```
用户提问
    │
    ▼
┌─────────────────────────────────┐
│  LangGraph ReAct Agent          │
│  (DeepSeek / GPT / Claude)      │
│                                 │
│  思考 → 选择工具 → 执行         │
└────────────┬────────────────────┘
             │ MCP 工具调用
             ▼
┌────────────────────────────────────────────┐
│  MCP Servers (FastMCP)                     │
│                                            │
│  docs-rag     — 文档索引与检索             │
│  web-search   — DuckDuckGo 搜索 + 网页抓取│
│  code-index   — 代码语义搜索               │
│  memory       — 持久化记忆                 │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│  向量数据库 (Qdrant 本地模式)              │
│  Embeddings: text-embedding-3-large        │
└────────────────────────────────────────────┘
```

## 特性

- **多源检索**: 文档、代码、网页、持久化记忆
- **自主智能体**: LangGraph ReAct 模式 —— Agent 自主决定使用哪些工具
- **MCP 集成**: 每个检索源都是独立的 MCP 服务器 (FastMCP)，可组合、可扩展
- **多种接口**: CLI (`click`)、REST API (`FastAPI`)、Web UI (`Streamlit`)
- **灵活的 LLM 后端**: 兼容任何 OpenAI 格式 API (DeepSeek、GPT、Qwen 等)
- **本地向量存储**: Qdrant 本地模式，无需外部服务

## 快速开始

```bash
# 克隆并安装
git clone https://github.com/URaux/agentrag.git
cd agentrag
pip install -e ".[dev]"

# 配置 API 密钥
cp .env.example .env
# 编辑 .env 填入你的 API 密钥

# 索引文档
agentrag index ./pyproject.toml --type doc

# 提问（Agent 自动检索）
agentrag ask "这个项目用了哪些依赖？"

# 直接搜索（不经过 Agent 推理）
agentrag search "dependencies" --source docs

# 启动 API 服务
agentrag serve
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `agentrag ask <query>` | 向 Agent 提问，自动检索并推理 |
| `agentrag index <path>` | 索引文件或目录 |
| `agentrag search <query>` | 直接语义搜索（不经过 Agent） |
| `agentrag serve` | 启动 REST API 服务 |
| `agentrag status` | 查看系统状态 |

## REST API

```bash
# 启动服务
agentrag serve --port 8000

# 向 Agent 提问
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "这个项目是做什么的？"}'

# 索引文档
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"path": "./README.md", "doc_type": "doc"}'

# 直接搜索
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "dependencies", "source": "docs"}'
```

## MCP 服务器

每个检索源作为独立 MCP 服务器运行：

| 服务器 | 工具 | 说明 |
|--------|------|------|
| `docs-rag` | `index_document`, `search_documents`, `list_indexed` | 本地文档索引与检索 |
| `web-search` | `web_search`, `fetch_page` | DuckDuckGo 实时网页搜索 |
| `code-index` | `index_repo`, `search_code` | 代码仓库语义搜索 |
| `memory` | `save_memory`, `recall`, `list_memories` | 持久化对话记忆 |

## 技术栈

- **Agent**: LangGraph (ReAct 模式)
- **LLM**: 兼容 OpenAI 格式的任意 API
- **MCP**: FastMCP 2.x
- **向量数据库**: Qdrant (本地模式)
- **Embeddings**: text-embedding-3-large (3072 维)
- **CLI**: Click + Rich
- **API**: FastAPI + Uvicorn
- **Web UI**: Streamlit

## 配置

所有配置通过环境变量设置（前缀 `AGENTRAG_`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AGENTRAG_API_KEY` | — | LLM API 密钥 |
| `AGENTRAG_API_BASE` | `https://api.deepseek.com/v1` | LLM API 地址 |
| `AGENTRAG_MODEL` | `deepseek-chat` | 对话模型名称 |
| `AGENTRAG_EMBEDDING_API_KEY` | — | Embedding API 密钥（未设置时使用 API_KEY） |
| `AGENTRAG_EMBEDDING_API_BASE` | — | Embedding API 地址（未设置时使用 API_BASE） |
| `AGENTRAG_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding 模型 |

## 许可证

MIT
