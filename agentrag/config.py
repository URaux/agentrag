"""Central configuration for AgentRAG."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "AGENTRAG_", "env_file": ".env", "extra": "ignore"}

    # LLM
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"

    # API (OpenAI-compatible endpoint)
    api_key: str = Field(default="", alias="AGENTRAG_API_KEY")
    api_base: str = Field(default="https://api.xiaocaseai.cloud/v1", alias="AGENTRAG_API_BASE")

    # Embedding (can use separate key/base)
    embedding_api_key: str = Field(default="", alias="AGENTRAG_EMBEDDING_API_KEY")
    embedding_api_base: str = Field(default="https://api.xiaocaseai.cloud/v1", alias="AGENTRAG_EMBEDDING_API_BASE")

    # Anthropic (optional)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Storage
    chroma_path: str = "./data/chroma"
    sqlite_path: str = "./data/metadata.db"
    data_dir: str = "./data"

    # MCP
    mcp_config: str = "./mcp_servers/config.json"

    # API Server
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def effective_embedding_key(self) -> str:
        return self.embedding_api_key or self.api_key

    @property
    def effective_embedding_base(self) -> str:
        return self.embedding_api_base or self.api_base


settings = Settings()
