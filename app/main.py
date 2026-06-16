import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import slack
from app.core.config import settings
from app.core.database import init_db
from app.middleware.logging import HttpLoggingMiddleware
from app.services import briefing_service, reminder_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    force=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # models import so Base.metadata includes all tables
    from app.models import reminder, todo  # noqa: F401
    await init_db()
    reminder_service.start_scheduler()
    briefing_service.start_briefing_scheduler()
    yield
    reminder_service.stop_scheduler()
    briefing_service.stop_briefing_scheduler()


app = FastAPI(
    title="Personal Assistant",
    description="AI 개인 비서 — Slack + LangGraph + Ollama",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(HttpLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(slack.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "ollama_host": settings.ollama_host,
    }


@app.post("/admin/memory/purge/{user_id}")
async def purge_memory(user_id: str):
    """오염된 ChromaDB 기억을 정리한다. (관리용)"""
    from app.services import memory_service
    deleted = await memory_service.purge_junk_memories(user_id)
    return {"deleted": deleted, "user_id": user_id}


@app.get("/admin/memory/list/{user_id}")
async def list_memory(user_id: str, limit: int = 100):
    """ChromaDB에 저장된 전체 기억 목록을 반환한다. (관리용)"""
    from app.services import memory_service
    col = memory_service._raw_collection(user_id)
    count = col.count()
    if count == 0:
        return {"count": 0, "docs": []}
    data = col.get(limit=min(limit, count), include=["documents", "metadatas"])
    items = [
        {"id": doc_id, "doc": doc, "meta": meta}
        for doc_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    return {"count": count, "docs": items}


@app.post("/admin/news/send")
async def trigger_news_briefing():
    """뉴스 브리핑을 즉시 실행한다. (관리용)"""
    from app.services import news_service
    await news_service.send_news_briefing()
    return {"ok": True}


@app.delete("/admin/memory/{user_id}/{doc_id}")
async def delete_memory(user_id: str, doc_id: str):
    """특정 기억을 ChromaDB에서 삭제한다. (관리용)"""
    from app.services import memory_service
    col = memory_service._raw_collection(user_id)
    col.delete(ids=[doc_id])
    return {"deleted": doc_id, "user_id": user_id}
