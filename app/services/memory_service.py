import logging
import re
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

        # 유사도 95% 이상인 기억이 이미 있으면 첫 번째는 update, 나머지는 delete (중복 제거)
        count = col.count()
        if count > 0:
            results = col.query(
                query_embeddings=embeddings,
                n_results=min(3, count),
                include=["distances", "documents"],
            )
            distances = results["distances"][0]
            ids = results["ids"][0]
            docs = results["documents"][0]

            similar_ids = [ids[i] for i, d in enumerate(distances) if d < 0.05]
            if similar_ids:
                if docs[0] == text[:2000]:
                    logger.info("[memory] skip exact duplicate user=%s", user_id)
                    return
                # 첫 번째 문서를 새 내용으로 update
                col.update(
                    ids=[similar_ids[0]],
                    documents=[text[:2000]],
                    embeddings=embeddings,
                    metadatas=[{"timestamp": str(int(time.time()))}],
                )
                logger.info("[memory] updated id=%s user=%s", similar_ids[0], user_id)
                # 나머지 유사 문서는 삭제 (중복 제거)
                if len(similar_ids) > 1:
                    col.delete(ids=similar_ids[1:])
                    logger.info("[memory] deleted %d duplicates user=%s", len(similar_ids) - 1, user_id)
                return

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


_JUNK_PATTERNS = re.compile(
    r'(핵심\s*정보[^:]*:|없으면\s*빈\s*문자열|내용\s*없음|없음$|없습니다$|\(내용\s*없음\))',
    re.IGNORECASE,
)


async def purge_junk_memories(user_id: str) -> int:
    """오염된 기억(prefix 포함, 무의미한 내용)을 ChromaDB에서 삭제하고 삭제 건수를 반환한다."""
    try:
        col = _collection(user_id)
        count = col.count()
        if count == 0:
            return 0
        all_data = col.get(limit=min(count, 2000), include=["documents"])
        junk_ids = [
            doc_id
            for doc_id, doc in zip(all_data["ids"], all_data["documents"])
            if _JUNK_PATTERNS.search(doc) or len(doc.strip()) <= 5
        ]
        if junk_ids:
            col.delete(ids=junk_ids)
            logger.info("[memory] purged %d junk docs user=%s", len(junk_ids), user_id)
        return len(junk_ids)
    except Exception as e:
        logger.warning("[memory] purge_junk failed: %s", e)
        return 0


async def find_similar(user_id: str, text: str) -> tuple[str | None, str | None]:
    """유사한 기억이 있으면 (existing_id, existing_doc) 반환, 없으면 (None, None)."""
    try:
        embeddings = await _embed([text[:2000]])
        col = _collection(user_id)
        count = col.count()
        if count == 0:
            return None, None
        results = col.query(
            query_embeddings=embeddings,
            n_results=1,
            include=["distances", "documents"],
        )
        if results["distances"][0] and results["distances"][0][0] < 0.05:
            return results["ids"][0][0], results["documents"][0][0]
        return None, None
    except Exception as e:
        logger.warning("[memory] find_similar failed: %s", e)
        return None, None


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z가-힣0-9]+", text.lower())
    korean_words = re.findall(r"[가-힣]+", text)
    extra: list[str] = []
    for word in korean_words:
        # 2-gram: 복합어 분해 ("결혼정장" → "결혼","혼정","정장", 조사 분리 "업체가" → "업체")
        for i in range(len(word) - 1):
            extra.append(word[i : i + 2])
        # 단일 음절: 1자 검색 대응 ("내차가" → "차" 매칭)
        extra.extend(list(word))
    return tokens + extra


async def search_memory(user_id: str, query: str, n: int = 3) -> list[str]:
    """벡터 + BM25 하이브리드 검색 (RRF 융합)으로 관련 기억을 반환한다."""
    try:
        from rank_bm25 import BM25Okapi

        col = _collection(user_id)
        count = col.count()
        if count == 0:
            return []

        candidates = min(max(n * 4, 10), count)

        # --- 벡터 검색 ---
        embeddings = await _embed([query[:2000]])
        vec_results = col.query(
            query_embeddings=embeddings,
            n_results=candidates,
            include=["documents"],
        )
        vec_ids: list[str] = vec_results["ids"][0]
        vec_docs: list[str] = vec_results["documents"][0]

        # --- BM25 검색 ---
        all_data = col.get(limit=min(count, 2000), include=["documents"])
        all_ids: list[str] = all_data["ids"]
        all_docs: list[str] = all_data["documents"]

        tokenized_corpus = [_tokenize(doc) for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(_tokenize(query))
        bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidates]

        # --- RRF 융합 (K=60) ---
        K = 60
        rrf: dict[str, float] = {}
        id_to_doc: dict[str, str] = {}

        for rank, (doc_id, doc) in enumerate(zip(vec_ids, vec_docs)):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1 / (K + rank + 1)
            id_to_doc[doc_id] = doc

        for rank, idx in enumerate(bm25_top_idx):
            doc_id = all_ids[idx]
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1 / (K + rank + 1)
            id_to_doc[doc_id] = all_docs[idx]

        sorted_ids = sorted(rrf, key=lambda d: rrf[d], reverse=True)
        docs = [id_to_doc[doc_id] for doc_id in sorted_ids[:n]]

        logger.info("[memory] hybrid search user=%s found=%d (vec+bm25 rrf)", user_id, len(docs))
        logger.info("[memory] search results: %s", [d[:50] for d in docs])
        return docs
    except Exception as e:
        logger.warning("[memory] search failed: %s", e)
        return []
