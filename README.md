# AI Personal Assistant

Slack DM으로 일정·지출·할 일·리마인더·파일 저장·뉴스 브리핑을 처리하는 LLM 기반 AI 개인 비서 플랫폼

---

## 기술 스택

| 영역 | 기술 |
|---|---|
| 백엔드 | FastAPI, Python 3.11 |
| LLM (기본) | Google Gemini API (gemini-2.0-flash) |
| LLM (폴백) | Ollama |
| 에이전트 | LangGraph (ReAct) |
| 기억 저장소 | ChromaDB + LlamaIndex VectorStoreIndex |
| Knowledge Graph | LlamaIndex SimplePropertyGraphStore |
| DB | MySQL (aiomysql + SQLAlchemy) |
| 스케줄러 | APScheduler |
| 한국어 처리 | kiwipiepy (형태소 분석) |
| 외부 연동 | Google Calendar API, Google Sheets API |
| 파일 저장 | 로컬 디스크 + ChromaDB (메타데이터 임베딩) |
| 알림 | Slack SDK + Block Kit |

---

## 아키텍처

```
Slack DM
    ↓
FastAPI (Slack Events API 수신)
    ↓
LangGraph ReAct 에이전트 (Gemini → 할당량 초과 시 Ollama)
    ↓
도구 선택 및 실행
    ├── Google Calendar API       — 일정 추가·조회
    ├── Google Sheets API         — 지출 기록·조회
    ├── MySQL                     — Todo / Reminder
    ├── ChromaDB + LlamaIndex     — 장기 기억 + Knowledge Graph
    ├── DuckDuckGo                — 웹 검색
    └── 로컬 디스크 + ChromaDB    — 파일 저장·검색
    ↓
Slack DM 응답
```

---

## 주요 기능

- **장기 기억**: 대화 내용을 ChromaDB에 저장, 벡터 + BM25 하이브리드 검색(RRF)으로 조회
- **Knowledge Graph**: 대화에서 엔티티-속성-값을 자동 추출해 그래프로 저장
- **일정 관리**: "다음 주 화요일 오후 2시 치과 예약해줘" → Google Calendar 자동 등록
- **지출 기록**: "스타벅스 6500원" → Google Sheets 자동 기록
- **리마인더**: "30분 후 약 먹으라고 알려줘" → APScheduler로 예약 후 DM 발송
- **파일 저장소**: Slack으로 보낸 파일을 저장하고 자연어로 검색·전송
- **뉴스 브리핑**: 설정한 뉴스 사이트를 LLM으로 요약해 매일 DM 전송

---

## 환경 변수

```env
# Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_MY_USER_ID=U...

# LLM
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text

# DB
DATABASE_URL=mysql+aiomysql://user:password@localhost:3306/assistant

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Google
GOOGLE_CREDENTIALS_PATH=/app/credentials.json
GOOGLE_CALENDAR_ID=your@gmail.com
GOOGLE_SHEET_ID=...

# 파일 저장
FILE_STORAGE_PATH=/app/user_files

# 뉴스
NEWS_URLS=https://news.example.com
```

---

## 실행

```bash
# 의존성 설치
uv sync

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```yaml
# docker-compose.yml 예시
services:
  app:
    build: .
    volumes:
      - ./credentials.json:/app/credentials.json:ro
      - /var/data/user_files:/app/user_files
  chromadb:
    image: chromadb/chroma
  mysql:
    image: mysql:8
```

---

## 블로그

구축 과정 전체를 시리즈로 정리했습니다.

- [전체 인덱스 — FastAPI + LangGraph로 만든 AI 에이전트](https://sangkihan.github.io/posts/personal-assistant-intro)
- [장기 기억 — ChromaDB + Ollama 임베딩](https://sangkihan.github.io/posts/personal-assistant-memory)
- [Google Calendar·Sheets 연동](https://sangkihan.github.io/posts/personal-assistant-google)
- [할 일 목록과 리마인더](https://sangkihan.github.io/posts/personal-assistant-todo-reminder)
- [LLM 전략 — Gemini 우선, Ollama 폴백](https://sangkihan.github.io/posts/personal-assistant-llm-strategy)
- [기억 시스템 디버깅](https://sangkihan.github.io/posts/personal-assistant-memory-debug)
- [BM25 한국어 복합어 문제](https://sangkihan.github.io/posts/personal-assistant-korean-bm25)
- [LlamaIndex 도입 — 기억 검색 + Knowledge Graph](https://sangkihan.github.io/posts/personal-assistant-llamaindex)
- [Slack 파일 저장소](https://sangkihan.github.io/posts/personal-assistant-file-storage)
- [한국어 처리 개선 — kiwipiepy 형태소 분석기 도입](https://sangkihan.github.io/posts/personal-assistant-kiwipiepy)
