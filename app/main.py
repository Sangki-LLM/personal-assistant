import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import slack
from app.core.config import settings
from app.core.database import init_db
from app.middleware.logging import HttpLoggingMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    force=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


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
