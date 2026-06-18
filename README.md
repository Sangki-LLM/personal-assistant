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

## 구현 기능

### 1. 기초 구조 — FastAPI + LangGraph + Slack
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-intro)

- Slack Events API 수신 후 `BackgroundTasks`로 즉시 200 반환, 백그라운드에서 에이전트 실행
- `@langchain_tool` 데코레이터로 도구 정의, `user_id`는 클로저로 바인딩
- bot_id 체크로 봇 응답 이벤트 무한루프 방지

### 2. 장기 기억 — ChromaDB + Ollama 임베딩
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-memory)

- 대화 내용을 ChromaDB에 벡터로 저장, 매 대화 전 자동 검색 후 메시지 앞에 주입
- ChromaDB HttpClient 싱글톤 관리로 `ResourceWarning` 방지

### 3. Google Calendar·Sheets 연동
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-google)

- 서비스 계정 방식으로 Google API 연동, 동기 라이브러리는 `run_in_executor`로 스레드풀 실행
- "다음 주 화요일 오후 2시 치과 예약해줘" → Calendar 자동 등록
- "스타벅스 6500원" → Sheets 자동 기록

### 4. 할 일 목록과 리마인더
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-todo-reminder)

- APScheduler 1분 polling 기반 리마인더, 취소 시 `fired=True` 처리로 이력 보존
- "30분 후 약 먹으라고 알려줘" → 자동 예약 후 DM 발송

### 5. LLM 전략 — Gemini 우선, Ollama 폴백
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-llm-strategy)

- Gemini 할당량 초과(429) 감지 시 Ollama로 자동 폴백하는 이중화 전략
- `think=False`로 Ollama 내부 추론 출력 제거

### 6. 기억 시스템 디버깅
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-memory-debug)

- LLM 프롬프트 오염으로 ChromaDB에 쓰레기 데이터 저장되는 문제 발견·수정
- Slack Block Kit 확인 버튼으로 저장 전 사용자 확인 UX 구현

### 7. BM25 한국어 복합어 문제
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-korean-bm25)

- "내차가"를 "차"로 매칭하지 못하는 BM25 한국어 문제 해결
- 2-gram + 음절 단위 토큰 추가로 형태소 분석기 없이 부분 매칭 구현

### 8. LlamaIndex 도입 — 기억 검색 + Knowledge Graph
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-llamaindex)

- `QueryFusionRetriever(mode="reciprocal_rerank")`로 벡터 + BM25 RRF 하이브리드 검색
- 대화 5턴 초과 시 오래된 대화를 LLM으로 요약, 최근 3턴 원문 유지
- `SimplePropertyGraphStore`로 엔티티-속성-값 Knowledge Graph 저장

### 9. Slack 파일 저장소
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-file-storage)

- 파일 내용이 아닌 파일명·카테고리·날짜를 ChromaDB에 임베딩 — 이진 파일 파싱 불필요
- "회의록 파일 줘"처럼 자연어로 파일 검색·전송
- `difflib.get_close_matches`로 파일명 퍼지 매칭 삭제 지원

### 10. 한국어 처리 개선 — kiwipiepy 형태소 분석기
> [📝 블로그](https://sangkihan.github.io/posts/personal-assistant-kiwipiepy)

- 카테고리 추출·BM25 토크나이저·긍정/부정 판별의 정규식을 형태소 분석으로 교체
- `app/core/kiwi.py` 싱글톤으로 세 서비스가 Kiwi 인스턴스 공유

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

