"""Shared vector store using Qdrant (local) + OpenAI-compatible embeddings."""

from __future__ import annotations

from pathlib import Path

import httpx
from qdrant_client import QdrantClient, models

_client: QdrantClient | None = None
_embed_cache: dict = {}


def _get_settings():
    from agentrag.config import settings
    return settings


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI-compatible API."""
    s = _get_settings()
    resp = httpx.post(
        f"{s.effective_embedding_base}/embeddings",
        headers={"Authorization": f"Bearer {s.effective_embedding_key}"},
        json={"input": texts, "model": s.embedding_model},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        s = _get_settings()
        db_path = str(Path(s.data_dir) / "qdrant")
        Path(db_path).mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=db_path)
    return _client


def _ensure_collection(name: str, dim: int = 3072):
    client = get_client()
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )


def embed_and_upsert(collection_name: str, ids: list[str], documents: list[str], metadatas: list[dict]):
    """Embed documents and upsert into Qdrant."""
    embeddings = get_embeddings(documents)
    dim = len(embeddings[0])
    _ensure_collection(collection_name, dim)

    client = get_client()
    points = [
        models.PointStruct(
            id=abs(hash(id_)) % (2**63),  # Qdrant needs int or uuid
            vector=emb,
            payload={"document": doc, "doc_id": id_, **meta},
        )
        for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas)
    ]
    client.upsert(collection_name, points)


def query_collection(collection_name: str, query: str, n_results: int = 5, where: dict | None = None):
    """Query collection with semantic search."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return None

    count = client.count(collection_name).count
    if count == 0:
        return None

    query_embedding = get_embeddings([query])[0]

    query_filter = None
    if where:
        conditions = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in where.items()
        ]
        query_filter = models.Filter(must=conditions)

    results = client.query_points(
        collection_name,
        query=query_embedding,
        limit=min(n_results, count),
        query_filter=query_filter,
    )

    # Convert to ChromaDB-like format for compatibility
    documents = [[p.payload.get("document", "") for p in results.points]]
    metadatas = [[{k: v for k, v in p.payload.items() if k != "document"} for p in results.points]]
    distances = [[1 - p.score for p in results.points]]  # Convert similarity to distance

    return {"documents": documents, "metadatas": metadatas, "distances": distances}


def get_collection_count(collection_name: str) -> int:
    client = get_client()
    if not client.collection_exists(collection_name):
        return 0
    return client.count(collection_name).count


def get_collection_data(collection_name: str, where: dict | None = None):
    """Get all data from a collection."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return {"ids": [], "documents": [], "metadatas": []}

    query_filter = None
    if where:
        conditions = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in where.items()
        ]
        query_filter = models.Filter(must=conditions)

    results = client.scroll(collection_name, scroll_filter=query_filter, limit=1000)[0]

    return {
        "ids": [p.payload.get("doc_id", str(p.id)) for p in results],
        "documents": [p.payload.get("document", "") for p in results],
        "metadatas": [{k: v for k, v in p.payload.items() if k not in ("document", "doc_id")} for p in results],
    }
