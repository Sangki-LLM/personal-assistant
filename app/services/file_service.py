import logging
import time
from datetime import datetime
from pathlib import Path

import chromadb
import httpx
from llama_index.core import Document, Settings as LlamaSettings, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user_file import UserFile

logger = logging.getLogger(__name__)

_chroma_client = None
_indexes: dict[str, VectorStoreIndex] = {}

LlamaSettings.llm = None


def _client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return _chroma_client


def _get_index(user_id: str) -> VectorStoreIndex:
    LlamaSettings.embed_model = OllamaEmbedding(
        model_name=settings.ollama_embed_model,
        base_url=settings.ollama_host,
    )
    if user_id not in _indexes:
        safe_id = user_id.replace("-", "_")
        collection = _client().get_or_create_collection(
            name=f"files_{safe_id}",
            metadata={"hnsw:space": "cosine"},
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _indexes[user_id] = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    return _indexes[user_id]


def _storage_dir(user_id: str) -> Path:
    path = Path(settings.file_storage_path) / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_embed_text(filename: str, mimetype: str, dt: datetime) -> str:
    return f"파일명: {filename} | 종류: {mimetype} | 날짜: {dt.strftime('%Y-%m-%d')}"


def _index_file(user_id: str, chroma_id: str, filename: str, mimetype: str, dt: datetime) -> None:
    try:
        doc = Document(
            text=_make_embed_text(filename, mimetype, dt),
            id_=chroma_id,
            metadata={"user_id": user_id, "filename": filename, "mimetype": mimetype},
        )
        _get_index(user_id).insert(doc)
    except Exception as e:
        logger.warning("[file] chroma index failed: %s", e)


def _reindex_file(user_id: str, chroma_id: str, filename: str, mimetype: str, dt: datetime) -> None:
    try:
        safe_id = user_id.replace("-", "_")
        col = _client().get_or_create_collection(
            name=f"files_{safe_id}", metadata={"hnsw:space": "cosine"}
        )
        col.delete(ids=[chroma_id])
        _indexes.pop(user_id, None)
        _index_file(user_id, chroma_id, filename, mimetype, dt)
    except Exception as e:
        logger.warning("[file] chroma reindex failed: %s", e)


async def handle_slack_file(db: AsyncSession, user_id: str, slack_file: dict, channel_id: str) -> str:
    """Slack에서 수신한 파일 메타데이터를 받아 다운로드 후 저장한다."""
    filename = slack_file.get("name", "unknown_file")
    mimetype = slack_file.get("mimetype", "application/octet-stream")
    url = slack_file.get("url_private") or slack_file.get("url_private_download", "")

    if not url:
        return "파일 URL을 가져오지 못했습니다."

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {settings.slack_bot_token}"}
            )
            resp.raise_for_status()
            content_bytes = resp.content
    except Exception as e:
        logger.warning("[file] download failed filename=%s: %s", filename, e)
        return f"파일 다운로드 실패: {e}"

    file_record = await save_file(db, user_id, filename, content_bytes, mimetype)
    size_str = _fmt_size(file_record.size_bytes)
    action = "업데이트" if file_record.updated_at != file_record.created_at else "저장"
    return f"✅ *{filename}* {action}했습니다. ({size_str})"


async def save_file(
    db: AsyncSession, user_id: str, filename: str, content_bytes: bytes, mimetype: str
) -> UserFile:
    """파일을 로컬 + DB + ChromaDB에 저장한다. 같은 이름이 있으면 덮어쓴다."""
    result = await db.execute(
        select(UserFile).where(UserFile.user_id == user_id, UserFile.original_name == filename)
    )
    existing = result.scalar_one_or_none()
    now = datetime.now()

    if existing:
        with open(existing.stored_path, "wb") as f:
            f.write(content_bytes)
        existing.size_bytes = len(content_bytes)
        existing.updated_at = now
        await db.commit()
        await db.refresh(existing)
        _reindex_file(user_id, existing.chroma_id, filename, mimetype, now)
        logger.info("[file] updated filename=%s user=%s", filename, user_id)
        return existing

    safe_name = f"{int(time.time())}_{filename}"
    stored_path = str(_storage_dir(user_id) / safe_name)
    with open(stored_path, "wb") as f:
        f.write(content_bytes)

    chroma_id = f"file_{int(time.time() * 1000)}"
    file_record = UserFile(
        user_id=user_id,
        original_name=filename,
        stored_path=stored_path,
        mimetype=mimetype,
        size_bytes=len(content_bytes),
        chroma_id=chroma_id,
        created_at=now,
        updated_at=now,
    )
    db.add(file_record)
    await db.commit()
    await db.refresh(file_record)
    _index_file(user_id, chroma_id, filename, mimetype, now)
    logger.info("[file] saved filename=%s user=%s", filename, user_id)
    return file_record


async def search_files(user_id: str, query: str, n: int = 5) -> list[str]:
    """ChromaDB 벡터 유사도로 파일명을 검색해 파일명 목록을 반환한다."""
    try:
        safe_id = user_id.replace("-", "_")
        col = _client().get_or_create_collection(
            name=f"files_{safe_id}", metadata={"hnsw:space": "cosine"}
        )
        if col.count() == 0:
            return []
        index = _get_index(user_id)
        retriever = index.as_retriever(similarity_top_k=n)
        nodes = await retriever.aretrieve(query)
        return [node.node.metadata.get("filename", "") for node in nodes if node.node.metadata.get("filename")]
    except Exception as e:
        logger.warning("[file] search failed: %s", e)
        return []


async def get_file_by_name(db: AsyncSession, user_id: str, filename: str) -> UserFile | None:
    result = await db.execute(
        select(UserFile).where(UserFile.user_id == user_id, UserFile.original_name == filename)
    )
    return result.scalar_one_or_none()


async def list_all_files(db: AsyncSession, user_id: str) -> list[UserFile]:
    result = await db.execute(
        select(UserFile).where(UserFile.user_id == user_id).order_by(UserFile.updated_at.desc())
    )
    return list(result.scalars().all())


def read_file_bytes(stored_path: str) -> bytes:
    with open(stored_path, "rb") as f:
        return f.read()


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f}MB"
    return f"{size_bytes // 1024}KB"
