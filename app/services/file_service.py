import io
import logging
import os
import re
import time
import zipfile
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
from app.models.user_file import FileBundle, UserFile

logger = logging.getLogger(__name__)

_chroma_client = None
_indexes: dict[str, VectorStoreIndex] = {}

LlamaSettings.llm = None

# 카테고리 추론: 짧은 단어 직접 입력 또는 "X로 저장/분류" 문장 패턴
_CATEGORY_PATTERN = re.compile(r'^[가-힣A-Za-z0-9 _\-]{1,20}$')
_IGNORE_WORDS = {"저장", "해줘", "이거", "파일", "문서", "보내", "올려", "넣어", "주세요", "좀"}
# "KB태양광 업무로 저장해줘" → "KB태양광 업무"
_CATEGORY_SENTENCE_PATTERN = re.compile(
    r'([가-힣A-Za-z0-9 _\-]{1,20})(?:으로|로|에)\s*(?:저장|분류|파일|묶어)'
)


def _extract_category(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

    # "X로 저장/분류" 문장 패턴 우선 시도 (짧은 텍스트도 포함)
    m = _CATEGORY_SENTENCE_PATTERN.search(text)
    if m:
        cat = m.group(1).strip()
        if cat:
            return cat

    # 짧고 단순한 단어 직접 입력
    if len(text) <= 15:
        words = set(re.findall(r'[가-힣]+', text))
        if not (words & _IGNORE_WORDS) and _CATEGORY_PATTERN.match(text):
            return text

    return None


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


def _make_embed_text(filename: str, mimetype: str, dt: datetime, category: str | None) -> str:
    parts = [f"파일명: {filename}", f"종류: {mimetype}", f"날짜: {dt.strftime('%Y-%m-%d')}"]
    if category:
        parts.insert(1, f"카테고리: {category}")
    return " | ".join(parts)


def _index_file(user_id: str, chroma_id: str, filename: str, mimetype: str, dt: datetime, category: str | None) -> None:
    try:
        # 기존 항목 먼저 삭제 (중복 방지)
        safe_id = user_id.replace("-", "_")
        col = _client().get_or_create_collection(name=f"files_{safe_id}", metadata={"hnsw:space": "cosine"})
        try:
            col.delete(ids=[chroma_id])
        except Exception:
            pass
        _indexes.pop(user_id, None)
        doc = Document(
            text=_make_embed_text(filename, mimetype, dt, category),
            id_=chroma_id,
            metadata={"user_id": user_id, "filename": filename, "mimetype": mimetype, "category": category or ""},
        )
        _get_index(user_id).insert(doc)
    except Exception as e:
        logger.warning("[file] chroma index failed: %s", e)


def _reindex_file(user_id: str, chroma_id: str, filename: str, mimetype: str, dt: datetime, category: str | None) -> None:
    try:
        safe_id = user_id.replace("-", "_")
        col = _client().get_or_create_collection(name=f"files_{safe_id}", metadata={"hnsw:space": "cosine"})
        col.delete(ids=[chroma_id])
        _indexes.pop(user_id, None)
        _index_file(user_id, chroma_id, filename, mimetype, dt, category)
    except Exception as e:
        logger.warning("[file] chroma reindex failed: %s", e)


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {settings.slack_bot_token}"})
        resp.raise_for_status()
        return resp.content


async def handle_slack_file(db: AsyncSession, user_id: str, slack_file: dict, channel_id: str, text: str = "") -> str:
    """단일 파일 수신 처리."""
    filename = slack_file.get("name", "unknown_file")
    mimetype = slack_file.get("mimetype", "application/octet-stream")
    url = slack_file.get("url_private") or slack_file.get("url_private_download", "")

    if not url:
        return "파일 URL을 가져오지 못했습니다."

    try:
        content_bytes = await _download(url)
    except Exception as e:
        logger.warning("[file] download failed filename=%s: %s", filename, e)
        return f"파일 다운로드 실패: {e}"

    category = _extract_category(text)
    file_record = await save_file(db, user_id, filename, content_bytes, mimetype, category=category)
    size_str = _fmt_size(file_record.size_bytes)
    action = "업데이트" if file_record.updated_at != file_record.created_at else "저장"
    cat_str = f" (카테고리: *{category}*)" if category else ""
    return f"✅ *{filename}* {action}했습니다. ({size_str}){cat_str}"


async def handle_slack_files(db: AsyncSession, user_id: str, slack_files: list[dict], channel_id: str, text: str = "") -> str:
    """여러 파일을 번들로 묶어 수신 처리."""
    category = _extract_category(text)

    first_name = slack_files[0].get("name", "파일") if slack_files else "파일"
    bundle_name = text.strip() if (text and len(text) <= 30 and not _extract_category(text) is None) else \
                  (category or f"{first_name} 외 {len(slack_files)-1}개")
    if not bundle_name or bundle_name == category:
        bundle_name = f"{first_name} 외 {len(slack_files)-1}개"

    bundle = FileBundle(user_id=user_id, name=bundle_name, category=category, created_at=datetime.now())
    db.add(bundle)
    await db.flush()

    saved, failed = [], []
    for sf in slack_files:
        filename = sf.get("name", "unknown")
        mimetype = sf.get("mimetype", "application/octet-stream")
        url = sf.get("url_private") or sf.get("url_private_download", "")
        if not url:
            continue
        try:
            content_bytes = await _download(url)
            await save_file(db, user_id, filename, content_bytes, mimetype, category=category, bundle_id=bundle.id)
            saved.append(filename)
        except Exception as e:
            logger.warning("[file] bundle item failed %s: %s", filename, e)
            failed.append(filename)

    await db.commit()

    names = "\n".join(f"• {n}" for n in saved)
    cat_str = f"\n카테고리: *{category}*" if category else ""
    fail_str = f"\n실패: {', '.join(failed)}" if failed else ""
    return f"✅ *{bundle_name}* 번들로 {len(saved)}개 저장했습니다.{cat_str}\n{names}{fail_str}"


async def save_file(
    db: AsyncSession,
    user_id: str,
    filename: str,
    content_bytes: bytes,
    mimetype: str,
    category: str | None = None,
    bundle_id: int | None = None,
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
        if category is not None:
            existing.category = category
        if bundle_id is not None:
            existing.bundle_id = bundle_id
        await db.commit()
        await db.refresh(existing)
        _reindex_file(user_id, existing.chroma_id, filename, existing.mimetype, now, existing.category)
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
        category=category,
        bundle_id=bundle_id,
        created_at=now,
        updated_at=now,
    )
    db.add(file_record)
    await db.commit()
    await db.refresh(file_record)
    _index_file(user_id, chroma_id, filename, mimetype, now, category)
    logger.info("[file] saved filename=%s category=%s user=%s", filename, category, user_id)
    return file_record


async def set_file_category(db: AsyncSession, user_id: str, filename: str, category: str) -> str:
    """파일 카테고리를 설정/변경한다."""
    result = await db.execute(
        select(UserFile).where(UserFile.user_id == user_id, UserFile.original_name == filename)
    )
    file_record = result.scalar_one_or_none()
    if not file_record:
        return f"'{filename}' 파일을 찾을 수 없습니다."
    file_record.category = category
    await db.commit()
    _reindex_file(user_id, file_record.chroma_id, filename, file_record.mimetype, file_record.updated_at, category)
    return f"✅ *{filename}* 카테고리를 *{category}* 로 설정했습니다."


async def search_files(user_id: str, query: str, n: int = 5) -> list[str]:
    """ChromaDB 벡터 유사도로 파일명을 검색해 반환한다."""
    try:
        safe_id = user_id.replace("-", "_")
        col = _client().get_or_create_collection(name=f"files_{safe_id}", metadata={"hnsw:space": "cosine"})
        if col.count() == 0:
            return []
        index = _get_index(user_id)
        retriever = index.as_retriever(similarity_top_k=n)
        nodes = await retriever.aretrieve(query)
        return [node.node.metadata.get("filename", "") for node in nodes if node.node.metadata.get("filename")]
    except Exception as e:
        logger.warning("[file] search failed: %s", e)
        return []


async def find_by_category(db: AsyncSession, user_id: str, category: str) -> list[UserFile]:
    """카테고리로 파일 목록을 DB에서 조회한다."""
    result = await db.execute(
        select(UserFile)
        .where(UserFile.user_id == user_id, UserFile.category == category)
        .order_by(UserFile.updated_at.desc())
    )
    return list(result.scalars().all())


async def list_categories(db: AsyncSession, user_id: str) -> list[str]:
    """사용 중인 카테고리 목록을 반환한다."""
    from sqlalchemy import distinct
    result = await db.execute(
        select(distinct(UserFile.category))
        .where(UserFile.user_id == user_id, UserFile.category.isnot(None))
    )
    return [row[0] for row in result.all() if row[0]]


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


async def delete_file(db: AsyncSession, user_id: str, filename: str) -> str:
    """파일을 로컬 + DB + ChromaDB에서 삭제한다."""
    result = await db.execute(
        select(UserFile).where(UserFile.user_id == user_id, UserFile.original_name == filename)
    )
    file_record = result.scalar_one_or_none()
    if not file_record:
        return f"'{filename}' 파일을 찾을 수 없습니다."

    try:
        os.remove(file_record.stored_path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("[file] local delete failed %s: %s", filename, e)

    try:
        safe_id = user_id.replace("-", "_")
        col = _client().get_or_create_collection(name=f"files_{safe_id}", metadata={"hnsw:space": "cosine"})
        col.delete(ids=[file_record.chroma_id])
        _indexes.pop(user_id, None)
    except Exception as e:
        logger.warning("[file] chroma delete failed %s: %s", filename, e)

    await db.delete(file_record)
    await db.commit()
    logger.info("[file] deleted filename=%s user=%s", filename, user_id)
    return f"🗑️ *{filename}* 삭제했습니다."


def read_file_bytes(stored_path: str) -> bytes:
    with open(stored_path, "rb") as f:
        return f.read()


def create_zip(files: list[tuple[str, bytes]]) -> bytes:
    """(파일명, 바이트) 목록을 zip으로 묶어 반환한다."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files:
            zf.writestr(filename, content)
    return buf.getvalue()


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f}MB"
    return f"{size_bytes // 1024}KB"
