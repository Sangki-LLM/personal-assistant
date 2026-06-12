import logging
import time

import chromadb
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_chroma_client = None


def _client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return _chroma_client


def _collection(user_id: str):
    safe_id = user_id.replace("-", "_")
    return _client().get_or_create_collection(
        name=f"memory_{safe_id}",
        metadata={"hnsw:space": "cosine"},
    )


async def _embed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.ollama_host}/api/embed",
            json={"model": settings.ollama_embed_model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


async def store_memory(user_id: str, text: str) -> None:
    """대화 내용이나 중요 정보를 ChromaDB에 저장한다."""
    try:
        embeddings = await _embed([text[:2000]])
        col = _collection(user_id)
        doc_id = f"mem_{int(time.time() * 1000)}"
        col.upsert(
            ids=[doc_id],
            documents=[text[:2000]],
            embeddings=embeddings,
            metadatas=[{"timestamp": str(int(time.time()))}],
        )
        logger.info("[memory] stored id=%s user=%s", doc_id, user_id)
    except Exception as e:
        logger.warning("[memory] store failed: %s", e)


async def search_memory(user_id: str, query: str, n: int = 3) -> list[str]:
    """과거 대화/메모에서 유사한 내용을 검색한다."""
    try:
        col = _collection(user_id)
        count = col.count()
        if count == 0:
            return []
        embeddings = await _embed([query[:2000]])
        results = col.query(
            query_embeddings=embeddings,
            n_results=min(n, count),
        )
        docs = results["documents"][0]
        logger.info("[memory] search user=%s found=%d", user_id, len(docs))
        return docs
    except Exception as e:
        logger.warning("[memory] search failed: %s", e)
        return []
