import logging
import re
import time

import chromadb
from app.core import kiwi as _kiwi_mod
from llama_index.core import Document, Settings as LlamaSettings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.config import settings

logger = logging.getLogger(__name__)

# LlamaIndex 전역 임베딩 설정 (LLM은 사용하지 않음)
LlamaSettings.embed_model = OllamaEmbedding(
    model_name=settings.ollama_embed_model,
    base_url=settings.ollama_host,
)
LlamaSettings.llm = None

_chroma_client = None
_indexes: dict[str, VectorStoreIndex] = {}


def _client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return _chroma_client


def _get_index(user_id: str) -> VectorStoreIndex:
    """사용자별 VectorStoreIndex 싱글톤 반환."""
    if user_id not in _indexes:
        safe_id = user_id.replace("-", "_")
        collection = _client().get_or_create_collection(
            name=f"memory_{safe_id}",
            metadata={"hnsw:space": "cosine"},
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _indexes[user_id] = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    return _indexes[user_id]


def _raw_collection(user_id: str):
    """중복 체크 / purge 등 raw 접근이 필요한 경우 ChromaDB 컬렉션 직접 반환."""
    safe_id = user_id.replace("-", "_")
    return _client().get_or_create_collection(
        name=f"memory_{safe_id}",
        metadata={"hnsw:space": "cosine"},
    )


_BM25_KEEP_TAGS = {"NNG", "NNP", "VV", "VA", "SL", "XR"}


def _tokenize(text: str) -> list[str]:
    try:
        morphs = _kiwi_mod.get().tokenize(text)
        result = [t.form.lower() for t in morphs if t.tag in _BM25_KEEP_TAGS and len(t.form) > 1]
    except Exception:
        result = []
    result += re.findall(r"[a-z0-9]+", text.lower())
    return list(dict.fromkeys(result))


_JUNK_PATTERNS = re.compile(
    r'(핵심\s*정보[^:]*:|없으면\s*빈\s*문자열|내용\s*없음|없음$|없습니다$|\(내용\s*없음\))',
    re.IGNORECASE,
)
_REALTIME_PATTERNS = re.compile(
    r'(주가\s*[:：]|코스피|코스닥|나스닥|s&p|환율\s*[:：]|달러\s*[:：]|원\s*/\s*달러|'
    r'비트코인|이더리움|코인\s*가격|주식\s*가격|지수\s*[:：]|포인트\s*$|'
    r'\d+\s*(달러|원|엔|유로|위안)\s*$)',
    re.IGNORECASE,
)
_JUNK_NEG_STEMS = {"없", "모르", "불명"}


def _is_junk_doc(doc: str) -> bool:
    if _JUNK_PATTERNS.search(doc) or _REALTIME_PATTERNS.search(doc) or len(doc.strip()) <= 5:
        return True
    # 짧은 부정 응답 감지 ("해당 정보가 없어요", "잘 모르겠습니다" 등)
    if len(doc) <= 30:
        try:
            tokens = _kiwi_mod.get().tokenize(doc)
            if any(t.form in _JUNK_NEG_STEMS for t in tokens):
                return True
        except Exception:
            pass
    return False


async def store_memory(user_id: str, text: str) -> None:
    """기억을 VectorStoreIndex(ChromaDB)에 저장한다. 유사 문서 중복 제거 포함."""
    try:
        text = text[:2000]
        col = _raw_collection(user_id)
        count = col.count()

        if count > 0:
            # 유사도 95% 이상(거리 < 0.05) 중복 체크
            index = _get_index(user_id)
            vec_retriever = index.as_retriever(similarity_top_k=min(3, count))
            nodes = await vec_retriever.aretrieve(text)

            similar = [n for n in nodes if (1 - n.score) < 0.05 if n.score is not None]
            if similar:
                top = similar[0]
                if top.node.text.strip() == text.strip():
                    logger.info("[memory] skip exact duplicate user=%s", user_id)
                    return
                # 기존 문서 삭제 후 새 내용으로 재삽입 (update)
                all_ids = [n.node.node_id for n in similar]
                col.delete(ids=all_ids)
                # 인덱스 캐시 무효화
                _indexes.pop(user_id, None)
                logger.info("[memory] replaced %d similar docs user=%s", len(all_ids), user_id)

        doc_id = f"mem_{int(time.time() * 1000)}"
        doc = Document(
            text=text,
            id_=doc_id,
            metadata={"timestamp": str(int(time.time())), "user_id": user_id},
        )
        index = _get_index(user_id)
        index.insert(doc)
        logger.info("[memory] stored id=%s user=%s", doc_id, user_id)

        # Knowledge Graph에도 저장 (엔티티-속성-값 추출)
        try:
            from app.services import graph_service
            import asyncio as _asyncio
            _asyncio.create_task(graph_service.save_to_graph(user_id, text))
        except Exception:
            pass
    except Exception as e:
        logger.warning("[memory] store failed: %s", e)


async def find_similar(user_id: str, text: str) -> tuple[str | None, str | None]:
    """유사한 기억이 있으면 (node_id, text) 반환, 없으면 (None, None)."""
    try:
        col = _raw_collection(user_id)
        if col.count() == 0:
            return None, None
        index = _get_index(user_id)
        retriever = index.as_retriever(similarity_top_k=1)
        nodes = await retriever.aretrieve(text[:2000])
        if nodes and nodes[0].score is not None and (1 - nodes[0].score) < 0.05:
            return nodes[0].node.node_id, nodes[0].node.text
        return None, None
    except Exception as e:
        logger.warning("[memory] find_similar failed: %s", e)
        return None, None


async def search_memory(user_id: str, query: str, n: int = 3) -> list[str]:
    """벡터 + BM25 하이브리드 검색 (QueryFusionRetriever RRF)으로 관련 기억을 반환한다."""
    try:
        col = _raw_collection(user_id)
        count = col.count()
        if count == 0:
            return []

        candidates = min(max(n * 4, 10), count)
        index = _get_index(user_id)

        # 벡터 리트리버
        vector_retriever = index.as_retriever(similarity_top_k=candidates)

        # BM25 리트리버 — ChromaDB에서 전체 텍스트 가져와 초기화
        all_data = col.get(limit=min(count, 2000), include=["documents"])
        bm25_nodes = [
            TextNode(id_=doc_id, text=doc)
            for doc_id, doc in zip(all_data["ids"], all_data["documents"])
            if doc
        ]
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=bm25_nodes,
            similarity_top_k=candidates,
            tokenizer=_tokenize,
        )

        # QueryFusionRetriever (RRF 융합)
        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=n,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=False,
        )

        nodes = await retriever.aretrieve(query[:2000])
        docs = [node.node.text for node in nodes if node.node.text]

        logger.info("[memory] hybrid search user=%s found=%d", user_id, len(docs))
        logger.info("[memory] results: %s", [d[:50] for d in docs])
        return docs
    except Exception as e:
        logger.warning("[memory] search failed: %s", e)
        return []


async def purge_junk_memories(user_id: str) -> int:
    """오염된 기억을 ChromaDB에서 삭제하고 삭제 건수를 반환한다."""
    try:
        col = _raw_collection(user_id)
        count = col.count()
        if count == 0:
            return 0
        all_data = col.get(limit=min(count, 2000), include=["documents"])
        junk_ids = [
            doc_id
            for doc_id, doc in zip(all_data["ids"], all_data["documents"])
            if _is_junk_doc(doc)
        ]
        if junk_ids:
            col.delete(ids=junk_ids)
            _indexes.pop(user_id, None)  # 캐시 무효화
            logger.info("[memory] purged %d junk docs user=%s", len(junk_ids), user_id)
        return len(junk_ids)
    except Exception as e:
        logger.warning("[memory] purge_junk failed: %s", e)
        return 0
