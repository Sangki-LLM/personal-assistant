import asyncio
import logging
import warnings
from datetime import date

from langchain_core.tools import tool as langchain_tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from app.core.config import settings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
warnings.filterwarnings("ignore", category=ResourceWarning)

logger = logging.getLogger(__name__)

# 확인 대기 중인 지출 기록 요청: user_id → {amount, category, memo}
_pending_expense: dict[str, dict] = {}
# 삭제 확인 대기 중인 파일 후보: user_id → [UserFile, ...]
_pending_delete: dict[str, list] = {}
# 대화 히스토리: user_id → [(user_msg, assistant_reply), ...]
_chat_histories: dict[str, list[tuple[str, str]]] = {}

_MAX_HISTORY_TURNS = 5   # 이 이상이면 오래된 대화 압축
_KEEP_RECENT_TURNS = 3   # 최근 N턴은 원문 그대로 유지


def _is_quota_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("quota", "429", "resource exhausted", "rate limit"))


def _make_ollama():
    return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host, think=False)


def _make_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
    )

_TOOL_TIMEOUT = 45


def _build_system_prompt() -> str:
    today = date.today().isoformat()
    return f"""당신은 유능한 개인 비서입니다. 오늘 날짜: {today}. 항상 한국어로 답변하세요.

**⛔ 절대 금지 — 리마인더와 할 일 혼용:**
- 사용자가 "알림 설정해줘", "리마인더 설정해줘" 라고 하면 → `set_reminder`만 호출
- `add_todo`를 함께 호출하면 절대 안 됩니다
- 사용자가 "할일에도 추가해줘", "todo에도 넣어줘"라고 명시적으로 말할 때만 `add_todo` 추가 호출

**도구 호출 규칙 — 반드시 준수:**

| 상황 | 호출할 도구 |
|---|---|
| "추가해줘", "등록해줘", "예약해줘", "캘린더에 넣어줘" + 날짜/시간 | add_calendar_event |
| "기억해줘", "메모해줘", "저장해줘" → 맥락/이벤트/상황 정보 | save_memory |
| 이름, 전화번호, 이메일, 생일, 주소 등 정형 속성 | save_fact(entity=이름, attribute=속성, value=값) |
| 일정 조회 요청 | list_calendar_events |
| 지출·비용·결제 언급 | add_expense |
| 지출 조회·요약 요청 | get_expense_summary |
| 알림·리마인더 설정 | set_reminder 단독 호출 (add_todo 금지) |
| 리마인더 목록 조회 | list_reminders |
| 리마인더 취소 | cancel_reminder |
| 할 일 추가 — 개인 (명시적 요청 시에만) | add_todo(content=내용, due_date=YYYY-MM-DD) |
| 할 일 추가 — 업무 ("XX 업무에 할일 추가") | add_todo(content=내용, category=카테고리명) |
| 할 일 전체 목록 조회 | list_todos |
| 특정 업무 카테고리 할 일 조회 | list_todos_by_category(category=카테고리명) |
| 업무 카테고리 목록 조회 | list_todo_categories |
| 할 일 완료 처리 | complete_todo |
| 과거 대화·정보 질문 | search_memory |
| 사용자 개인 정보 질문 ("내 ~~이 뭐야", "~~이 언제야") | search_memory |
| 특정 사람·장소·사물 관련 속성 조회 ("또리 생일", "김철수 전화") | query_knowledge_graph(entity=이름) |
| 날씨 질문 ("~~ 날씨", "날씨 어때") | get_weather(location=지역명) |
| URL이 포함된 메시지 | summarize_url |
| 최신 정보·검색 필요 | web_search |
| 파일 목록만 보여달라 ("파일 목록", "어떤 파일", "뭐 있어") | list_files 또는 find_files_by_category(list_only=True) |
| 파일 전송 요청 ("파일 줘", "보내줘", "전송해줘") | find_file 또는 find_files_by_category |
| 특정 카테고리 목록만 요청 ("업무 파일 목록", "개인파일 뭐 있어") | find_files_by_category(list_only=True) |
| 특정 카테고리 파일 전송 요청 ("업무 파일 다 보내줘") | find_files_by_category(list_only=False) |
| 파일 카테고리 설정·변경 | set_file_category |
| 저장된 카테고리 목록 조회 | list_categories |
| 파일 삭제 ("파일 지워줘", "삭제해줘") — 후보 목록 표시 후 정확한 파일명 확인 | delete_file |
| "이력서 작성해줘 [회사명]" + 채용공고 내용 | create_resume(company_name=회사명, job_posting=공고내용) |
| "이력서 틀 저장되어있어?" 또는 이력서 템플릿 여부 확인 | save_resume_template |
| "SEMS 프로젝트에 XX 추가해줘", "PLANIN에서 XX를 YY로 수정해줘" | edit_resume_project(project_name=프로젝트명, instruction=수정내용) |
| "이력서 프로젝트 목록 보여줘", 어떤 프로젝트 있는지 확인 | list_resume_projects |

**파일 목록 vs 파일 전송 구분 — 반드시 준수:**
- "목록", "뭐 있어", "어떤 거", "보여줘" → list_only=True (파일 전송 금지)
- "줘", "보내줘", "전송해줘", "올려줘" → list_only=False (파일 실제 전송)

**날짜 변환 (오늘={today} 기준):**
- "다음주 토요일" → 실제 YYYY-MM-DD 계산 후 전달
- "10시" → "10:00", "오후 3시" → "15:00"

**Slack mrkdwn 형식 — 반드시 준수:**
- 굵게: `*텍스트*` (별표 1개) — `**텍스트**` 사용 금지
- 목록: `• 항목` 또는 `- 항목` — `*   항목` 사용 금지
- 헤더 없음: `*굵게*` 로 대체 — `###`, `##`, `#` 사용 금지
- 이탤릭: `_텍스트_`, 취소선: `~텍스트~`, 코드: `` `코드` ``

**응답 방식:**
- 도구 결과에 `[AGENT_ONLY ...]` 또는 `[AGENT_ONLY - 사용자에게 표시 금지]` 섹션이 있으면 내부 참조용으로만 쓰고 절대 사용자에게 노출하지 마세요
- 메시지 앞에 `[⚠️ 사전 검색된 정보]` 블록이 있으면 참고하세요. KG 결과가 있으면 ChromaDB보다 우선합니다. 두 결과가 충돌하면 KG를 신뢰하세요. 위 정보로 답할 수 없거나 확신이 없으면 search_memory나 query_knowledge_graph를 재호출하세요
- search_memory 도구 결과가 반환되면 그 내용을 **반드시 신뢰**하고 "없다"고 하지 마세요. 형식이 어색해도 내용 안에 답이 있으면 답변하세요
- 도구 실행 결과를 먼저 확인한 뒤 간결하게 알려주세요
- 일정/지출/할 일은 반드시 도구로 기록하고 "등록했어요" 형식으로 답변하세요
- 도구 없이 텍스트만 답변하지 마세요 (기억·기록이 필요한 요청은 반드시 도구 호출)
- `list_todos` 결과는 ✅/⬜ 체크 표시를 *반드시* 포함하여 출력하세요 — `• 항목` 형태로 재포맷 금지, 완료 여부를 항상 표시
- save_memory 도구를 호출하면 즉시 저장됩니다. 저장 완료 후 간결하게 "기억했습니다" 형식으로 알려주세요."""


def _make_tools(user_id: str, channel_id: str = ""):
    @langchain_tool
    async def search_memory(query: str) -> str:
        """과거 대화나 기억에서 관련 내용을 검색합니다."""
        logger.info("[tool] search_memory user=%s query=%s", user_id, query[:50])
        from app.services import memory_service
        results = await memory_service.search_memory(user_id, query)
        logger.info("[tool] search_memory done count=%d docs=%s", len(results), [r[:40] for r in results])
        if not results:
            return "관련 기억이 없습니다."
        body = "\n---\n".join(results)
        return (
            f"[기억 검색 완료 — 아래 내용이 실제 저장된 사실입니다. 반드시 이 내용을 바탕으로 답변하세요]\n"
            f"{body}"
        )

    @langchain_tool
    async def save_memory(text: str) -> str:
        """중요한 정보를 즉시 기억합니다."""
        logger.info("[tool] save_memory user=%s text_len=%d", user_id, len(text))
        from app.services import memory_service

        existing_id, existing_doc = await memory_service.find_similar(user_id, text)
        if existing_id and existing_doc and existing_doc.strip() == text.strip():
            return "이미 동일한 내용을 기억하고 있습니다."

        await memory_service.store_memory(user_id, text)
        return "기억했습니다."

    @langchain_tool
    async def get_weather(location: str) -> str:
        """날씨 정보를 가져옵니다. location: 도시명 (예: 서울, 수원, Busan)"""
        logger.info("[tool] get_weather location=%s", location)
        from app.services import weather_service
        return await weather_service.get_weather(location)

    @langchain_tool
    async def summarize_url(url: str) -> str:
        """URL의 내용을 가져와 요약합니다."""
        logger.info("[tool] summarize_url url=%s", url[:80])

        from urllib.parse import urlparse
        import socket
        import ipaddress

        try:
            parsed = urlparse(url)
            if not parsed.hostname:
                return "유효하지 않은 URL 형식입니다."

            if parsed.scheme not in ("http", "https"):
                return f"지원하지 않는 프로토콜입니다 ({parsed.scheme}). http 또는 https만 가능합니다."

            # 1. 호스트명을 IP로 변환 (DNS Resolution)
            loop = asyncio.get_event_loop()
            try:
                # gethostbyname은 블로킹 함수이므로 스레드 풀에서 실행
                ip_str = await loop.run_in_executor(None, socket.gethostbyname, parsed.hostname)
                ip = ipaddress.ip_address(ip_str)
            except socket.gaierror:
                return "URL 주소를 확인할 수 없습니다 (DNS 조회 실패)."
            except ValueError:
                return "유효하지 않은 IP 주소 형식입니다."

            # 2. 내부 네트워크 대역 체크 (SSRF 방지)
            if any([
                ip.is_private,      # 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                ip.is_loopback,     # 127.0.0.1
                ip.is_link_local,   # 169.254.0.0/16 (클라우드 메타데이터 등)
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified
            ]):
                logger.warning("[security] SSRF attempt blocked: %s (Resolved IP: %s)", url, ip_str)
                return "보안 정책상 내부 네트워크 주소에는 접근할 수 없습니다."

            import httpx
            from html.parser import HTMLParser

            class _TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.texts: list[str] = []
                    self.img_srcs: list[str] = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "head", "nav", "footer"):
                        self._skip = True
                    if tag == "img":
                        d = dict(attrs)
                        src = d.get("src") or d.get("data-src") or d.get("data-lazy-src")
                        if src:
                            self.img_srcs.append(src)

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "head", "nav", "footer"):
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip and data.strip():
                        self.texts.append(data.strip())

            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                html_body = resp.text

            extractor = _TextExtractor()
            extractor.feed(html_body)
            raw_text = " ".join(extractor.texts)[:3000]

            # 텍스트가 빈약하면 페이지 내 이미지를 Vision으로 보완
            if len(raw_text.strip()) < 300 and extractor.img_srcs and settings.gemini_api_key:
                import base64
                from urllib.parse import urljoin
                vision_parts: list[str] = []
                async with httpx.AsyncClient(timeout=10, follow_redirects=True) as img_client:
                    for src in extractor.img_srcs[:6]:
                        try:
                            img_url = urljoin(url, src)
                            ir = await img_client.get(img_url, headers={"User-Agent": "Mozilla/5.0"})
                            img_bytes = ir.content
                            if len(img_bytes) < 8000:  # 아이콘·썸네일 제외
                                continue
                            mime = "image/png" if img_bytes[:8] == b'\x89PNG\r\n\x1a\n' else "image/jpeg"
                            b64 = base64.b64encode(img_bytes).decode()
                            from langchain_google_genai import ChatGoogleGenerativeAI
                            from langchain_core.messages import HumanMessage as _HM
                            vlm = ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.gemini_api_key)
                            vr = await vlm.ainvoke([_HM(content=[
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                                {"type": "text", "text": "이 이미지에서 채용공고 텍스트를 모두 추출해줘. 회사명, 담당업무, 자격요건, 우대사항 전부 포함."},
                            ])])
                            extracted = _extract_text(vr.content).strip()
                            if extracted:
                                vision_parts.append(extracted)
                        except Exception as ve:
                            logger.debug("[tool] vision img failed: %s", ve)
                if vision_parts:
                    logger.info("[tool] summarize_url vision fallback images=%d", len(vision_parts))
                    raw_text = "\n\n".join(vision_parts)[:3000]

            if not raw_text.strip():
                return "페이지 내용을 가져왔지만 텍스트를 추출하지 못했습니다."

            from langchain_core.messages import HumanMessage, SystemMessage
            try:
                llm = _make_gemini() if settings.gemini_api_key else _make_ollama()
                resp_llm = await llm.ainvoke([
                    SystemMessage(content="주어진 웹페이지 내용을 한국어로 3-5문장으로 요약해줘."),
                    HumanMessage(content=f"URL: {url}\n\n내용:\n{raw_text}"),
                ])
                return f"📄 *URL 요약*\n{resp_llm.content}"
            except Exception as e:
                logger.warning("[tool] summarize_url LLM failed: %s", e)
                return f"📄 *URL 내용 (요약 실패)*\n{raw_text[:500]}..."
        except Exception as e:
            logger.warning("[tool] summarize_url failed url=%s: %s", url[:50], e)
            return f"URL 내용을 가져오지 못했습니다: {e}"

    @langchain_tool
    async def add_calendar_event(title: str, date: str, time: str, description: str = "") -> str:
        """Google Calendar에 일정을 추가합니다. date: YYYY-MM-DD, time: HH:MM"""
        logger.info("[tool] add_calendar_event title=%s date=%s time=%s", title, date, time)
        from app.services import calendar_service
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: calendar_service.add_event(title, date, time, description)),
                timeout=_TOOL_TIMEOUT,
            )
            logger.info("[tool] add_calendar_event done: %s", result[:60])
            return result
        except asyncio.TimeoutError:
            logger.warning("[tool] add_calendar_event timeout")
            return "일정 등록 시간 초과. Google Calendar API에 연결할 수 없습니다."

    @langchain_tool
    async def list_calendar_events(date: str) -> str:
        """특정 날의 일정을 조회합니다. date: YYYY-MM-DD"""
        logger.info("[tool] list_calendar_events date=%s", date)
        from app.services import calendar_service
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: calendar_service.list_events(date)),
                timeout=_TOOL_TIMEOUT,
            )
            logger.info("[tool] list_calendar_events done")
            return result
        except asyncio.TimeoutError:
            logger.warning("[tool] list_calendar_events timeout")
            return "일정 조회 시간 초과. Google Calendar API에 연결할 수 없습니다."

    @langchain_tool
    async def add_expense(amount: int, category: str, memo: str = "") -> str:
        """지출을 Google Sheets에 기록합니다. amount: 금액(정수), category: 카테고리, memo: 메모"""
        logger.info("[tool] add_expense amount=%d category=%s", amount, category)
        from app.services import expense_service
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: expense_service.add_expense(amount, category, memo)),
                timeout=_TOOL_TIMEOUT,
            )
            logger.info("[tool] add_expense done: %s", result[:60])
            return result
        except asyncio.TimeoutError:
            logger.warning("[tool] add_expense timeout")
            return "지출 기록 시간 초과. Google Sheets API에 연결할 수 없습니다."

    @langchain_tool
    async def get_expense_summary(year_month: str = "") -> str:
        """월별 카테고리별 지출 요약을 조회합니다. year_month: YYYY-MM (빈값이면 이번달)"""
        logger.info("[tool] get_expense_summary year_month=%s", year_month)
        from app.services import expense_service
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: expense_service.get_monthly_summary(year_month)),
                timeout=_TOOL_TIMEOUT,
            )
            logger.info("[tool] get_expense_summary done")
            return result
        except asyncio.TimeoutError:
            logger.warning("[tool] get_expense_summary timeout")
            return "지출 조회 시간 초과. Google Sheets API에 연결할 수 없습니다."

    @langchain_tool
    async def set_reminder(message: str, datetime_str: str) -> str:
        """특정 시간에 Slack 알림을 설정합니다. datetime_str: '30분 후', '다음주 월요일 오전 10시', '내일 오후 3시', 'YYYY-MM-DD HH:MM' 등 자연어 표현 가능"""
        import dateparser
        from datetime import datetime
        from app.core.database import AsyncSessionLocal
        from app.services import reminder_service as rs

        logger.info("[tool] set_reminder message=%s datetime=%s", message[:40], datetime_str)
        now = datetime.now()
        run_at = dateparser.parse(
            datetime_str,
            languages=["ko", "en"],
            settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now, "RETURN_AS_TIMEZONE_AWARE": False},
        )
        if not run_at:
            return "시간 형식을 인식하지 못했습니다. 예: '30분 후', '다음주 월요일 오전 10시', '2026-06-12 15:00'"

        async with AsyncSessionLocal() as db:
            result = await rs.set_reminder(db, user_id, message, run_at)
        logger.info("[tool] set_reminder done")
        return result

    @langchain_tool
    async def list_reminders() -> str:
        """등록된 리마인더 목록을 조회합니다. (아직 울리지 않은 것만)"""
        logger.info("[tool] list_reminders user=%s", user_id)
        from app.core.database import AsyncSessionLocal
        from app.services import reminder_service as rs
        async with AsyncSessionLocal() as db:
            return await rs.list_reminders(db, user_id)

    @langchain_tool
    async def cancel_reminder(reminder_id: int) -> str:
        """리마인더를 취소합니다. reminder_id: list_reminders로 확인한 번호"""
        logger.info("[tool] cancel_reminder id=%d", reminder_id)
        from app.core.database import AsyncSessionLocal
        from app.services import reminder_service as rs
        async with AsyncSessionLocal() as db:
            return await rs.cancel_reminder(db, user_id, reminder_id)

    @langchain_tool
    async def add_todo(content: str, due_date: str = "", category: str = "") -> str:
        """할 일을 추가합니다.
        - 개인 할일: due_date(YYYY-MM-DD) 지정, category 없음
        - 업무 할일: category(예: 코오롱 업무) 지정, due_date 없어도 됨
        """
        logger.info("[tool] add_todo content=%s due=%s category=%s", content[:40], due_date, category)
        from app.core.database import AsyncSessionLocal
        from app.services import todo_service
        async with AsyncSessionLocal() as db:
            return await todo_service.add_todo(db, user_id, content, due_date or None, category or None)

    @langchain_tool
    async def list_todos() -> str:
        """미완료 할 일 전체 목록을 조회합니다. 업무 카테고리별 + 날짜별로 표시됩니다."""
        logger.info("[tool] list_todos user=%s", user_id)
        from app.core.database import AsyncSessionLocal
        from app.services import todo_service
        async with AsyncSessionLocal() as db:
            return await todo_service.list_todos(db, user_id)

    @langchain_tool
    async def list_todos_by_category(category: str) -> str:
        """특정 업무 카테고리의 할 일 목록을 조회합니다. category: 카테고리명 (예: 코오롱 업무)"""
        logger.info("[tool] list_todos_by_category category=%s", category)
        from app.core.database import AsyncSessionLocal
        from app.services import todo_service
        async with AsyncSessionLocal() as db:
            return await todo_service.list_todos_by_category(db, user_id, category)

    @langchain_tool
    async def list_todo_categories() -> str:
        """등록된 업무 카테고리 목록을 조회합니다."""
        logger.info("[tool] list_todo_categories user=%s", user_id)
        from app.core.database import AsyncSessionLocal
        from app.services import todo_service
        async with AsyncSessionLocal() as db:
            return await todo_service.list_todo_categories(db, user_id)

    @langchain_tool
    async def complete_todo(todo_id: int) -> str:
        """할 일을 완료 처리합니다. todo_id: 할 일 번호"""
        logger.info("[tool] complete_todo id=%d", todo_id)
        from app.core.database import AsyncSessionLocal
        from app.services import todo_service
        async with AsyncSessionLocal() as db:
            return await todo_service.complete_todo(db, user_id, todo_id)

    @langchain_tool
    async def query_knowledge_graph(entity: str) -> str:
        """특정 사람·장소·사물에 대해 Knowledge Graph에서 속성과 관계를 조회합니다. 예: '또리', '김철수'"""
        logger.info("[tool] query_knowledge_graph entity=%s", entity)
        from app.services import graph_service
        results = graph_service.query_graph(user_id, entity)
        if not results:
            return f"'{entity}'에 대한 Knowledge Graph 정보가 없습니다."
        return f"[Knowledge Graph: {entity}]\n" + "\n".join(results)

    @langchain_tool
    async def save_fact(entity: str, attribute: str, value: str) -> str:
        """이름·전화번호·생일·주소 등 정형 속성을 Knowledge Graph에 저장합니다.
        예: entity='또리', attribute='생일', value='2024-03-30'
        맥락·이벤트·상황 정보는 save_memory를 사용하세요."""
        logger.info("[tool] save_fact entity=%s attr=%s val=%s", entity, attribute, value)
        from app.services import graph_service
        graph_service.save_triplet(user_id, entity, attribute, value)
        return f"기억했습니다: {entity}의 {attribute} = {value}"

    @langchain_tool
    async def web_search(query: str) -> str:
        """인터넷에서 최신 정보를 검색합니다. LLM이 모르는 최신 정보나 뉴스를 찾을 때 사용하세요."""
        logger.info("[tool] web_search query=%s", query[:50])
        from app.services import search_service
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: search_service.web_search(query)),
                timeout=_TOOL_TIMEOUT,
            )
            logger.info("[tool] web_search done")
            return result
        except asyncio.TimeoutError:
            logger.warning("[tool] web_search timeout")
            return "웹 검색 시간 초과."

    @langchain_tool
    async def find_file(query: str, list_only: bool = False, as_zip: bool = False) -> str:
        """저장된 파일을 검색합니다. list_only=True면 목록만 반환(파일 전송 안 함). as_zip=True면 zip으로 묶어 전송."""
        logger.info("[tool] find_file user=%s query=%s list_only=%s as_zip=%s", user_id, query[:50], list_only, as_zip)
        from app.core.database import AsyncSessionLocal
        from app.services import file_service, slack_service

        filenames = await file_service.search_files(user_id, query, n=5)
        if not filenames:
            return "관련 파일을 찾지 못했습니다."

        async with AsyncSessionLocal() as db:
            matched = [
                f for fn in filenames
                for f in [await file_service.get_file_by_name(db, user_id, fn)]
                if f
            ]

        if not matched:
            return "관련 파일을 찾지 못했습니다."

        if list_only:
            lines = [f"• {f.original_name} ({f.updated_at.strftime('%Y-%m-%d')})" for f in matched]
            return f"검색된 파일 ({len(matched)}개):\n" + "\n".join(lines)

        if len(matched) == 1:
            f = matched[0]
            content = await asyncio.to_thread(file_service.read_file_bytes, f.stored_path)
            if channel_id:
                await slack_service.upload_file(channel_id, f.original_name, content, "요청하신 파일입니다.")
            return f"*{f.original_name}* 파일을 보냈습니다."

        if as_zip:
            file_pairs = [
                (f.original_name, await asyncio.to_thread(file_service.read_file_bytes, f.stored_path))
                for f in matched
            ]
            zip_bytes = await asyncio.to_thread(file_service.create_zip, file_pairs)
            if channel_id:
                await slack_service.upload_file(channel_id, f"{query}_묶음.zip", zip_bytes, f"파일 {len(file_pairs)}개를 zip으로 묶었습니다.")
            names = "\n".join(f"• {f.original_name}" for f in matched)
            return f"파일 {len(file_pairs)}개를 zip으로 묶어 보냈습니다.\n{names}"

        sent = []
        for f in matched:
            content = await asyncio.to_thread(file_service.read_file_bytes, f.stored_path)
            if channel_id:
                await slack_service.upload_file(channel_id, f.original_name, content)
            sent.append(f.original_name)
        names = "\n".join(f"• {n}" for n in sent)
        return f"파일 {len(sent)}개를 보냈습니다.\n{names}"

    @langchain_tool
    async def list_files() -> str:
        """저장된 파일 전체 목록을 조회합니다."""
        logger.info("[tool] list_files user=%s", user_id)
        from app.core.database import AsyncSessionLocal
        from app.services import file_service

        async with AsyncSessionLocal() as db:
            files = await file_service.list_all_files(db, user_id)
        if not files:
            return "저장된 파일이 없습니다."

        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for f in files:
            key = f.category or "미분류"
            groups[key].append(f)

        lines = [f"저장된 파일 ({len(files)}개):"]
        for cat, cat_files in sorted(groups.items()):
            lines.append(f"\n📁 *{cat}* ({len(cat_files)}개)")
            for f in cat_files:
                lines.append(f"  • {f.original_name} ({f.updated_at.strftime('%Y-%m-%d')})")
        return "\n".join(lines)

    @langchain_tool
    async def find_files_by_category(category: str, list_only: bool = False, as_zip: bool = False) -> str:
        """특정 카테고리의 파일을 조회합니다. list_only=True면 목록만 반환(파일 전송 안 함). as_zip=True면 zip으로 묶어 전송."""
        logger.info("[tool] find_files_by_category user=%s category=%s list_only=%s as_zip=%s", user_id, category, list_only, as_zip)
        from app.core.database import AsyncSessionLocal
        from app.services import file_service, slack_service

        async with AsyncSessionLocal() as db:
            files = await file_service.find_by_category(db, user_id, category)

        if not files:
            return f"*{category}* 카테고리에 저장된 파일이 없습니다."

        if list_only:
            lines = [f"• {f.original_name} ({f.updated_at.strftime('%Y-%m-%d')})" for f in files]
            return f"*{category}* 카테고리 파일 ({len(files)}개):\n" + "\n".join(lines)

        if len(files) == 1:
            f = files[0]
            content = await asyncio.to_thread(file_service.read_file_bytes, f.stored_path)
            if channel_id:
                await slack_service.upload_file(channel_id, f.original_name, content, f"{category} 카테고리 파일입니다.")
            return f"*{category}* 카테고리 파일 1개를 보냈습니다: {f.original_name}"

        if as_zip:
            file_pairs = []
            for f in files:
                try:
                    content = await asyncio.to_thread(file_service.read_file_bytes, f.stored_path)
                    file_pairs.append((f.original_name, content))
                except Exception as e:
                    logger.warning("[tool] find_files_by_category read failed %s: %s", f.original_name, e)
            if not file_pairs:
                return "파일을 읽는 중 오류가 발생했습니다."
            zip_bytes = await asyncio.to_thread(file_service.create_zip, file_pairs)
            if channel_id:
                await slack_service.upload_file(channel_id, f"{category}_묶음.zip", zip_bytes, f"{category} 카테고리 파일 {len(file_pairs)}개입니다.")
            names = "\n".join(f"• {n}" for n, _ in file_pairs)
            return f"*{category}* 카테고리 파일 {len(file_pairs)}개를 zip으로 묶어 보냈습니다.\n{names}"

        sent = []
        for f in files:
            try:
                content = await asyncio.to_thread(file_service.read_file_bytes, f.stored_path)
                if channel_id:
                    await slack_service.upload_file(channel_id, f.original_name, content)
                sent.append(f.original_name)
            except Exception as e:
                logger.warning("[tool] find_files_by_category read failed %s: %s", f.original_name, e)

        if not sent:
            return "파일을 읽는 중 오류가 발생했습니다."

        names = "\n".join(f"• {n}" for n in sent)
        return f"*{category}* 카테고리 파일 {len(sent)}개를 보냈습니다.\n{names}"

    @langchain_tool
    async def set_file_category(filename: str, category: str) -> str:
        """파일의 카테고리를 설정하거나 변경합니다. filename: 파일명, category: 카테고리명"""
        logger.info("[tool] set_file_category user=%s filename=%s category=%s", user_id, filename, category)
        from app.core.database import AsyncSessionLocal
        from app.services import file_service

        async with AsyncSessionLocal() as db:
            return await file_service.set_file_category(db, user_id, filename, category)

    @langchain_tool
    async def list_categories() -> str:
        """저장된 파일의 카테고리 목록을 조회합니다."""
        logger.info("[tool] list_categories user=%s", user_id)
        from app.core.database import AsyncSessionLocal
        from app.services import file_service

        async with AsyncSessionLocal() as db:
            categories = await file_service.list_categories(db, user_id)
        if not categories:
            return "설정된 카테고리가 없습니다."
        return "카테고리 목록:\n" + "\n".join(f"• {c}" for c in categories)

    @langchain_tool
    async def edit_resume_project(project_name: str, instruction: str) -> str:
        """이력서 특정 프로젝트 내용을 수정하거나 업무를 추가합니다.
        project_name: 수정할 프로젝트명 (부분 일치 가능, 예: 'SEMS', 'PLANIN', 'PuppyNote')
        instruction: 수정 내용 (예: '배포 자동화 업무 추가해줘', 'Redis 항목을 ~로 수정해줘')"""
        from app.services import resume_service

        if not resume_service.template_exists():
            return "이력서 템플릿이 없습니다. 먼저 HTML 파일을 업로드하면서 '내 이력서 틀이야 저장해줘'라고 해주세요."

        if not channel_id:
            return "Slack 채널 정보가 없어 파일을 전송할 수 없습니다."

        from app.services import slack_service
        await slack_service.send_message(channel_id, f"✏️ *{project_name}* 프로젝트 수정 중... (잠시만 기다려주세요)")

        try:
            matched_title, html_bytes = await asyncio.wait_for(
                resume_service.edit_resume_project(project_name, instruction),
                timeout=180,
            )
        except FileNotFoundError as e:
            return str(e)
        except ValueError as e:
            return str(e)
        except asyncio.TimeoutError:
            return "수정 시간 초과. 다시 시도해주세요."
        except Exception as e:
            logger.warning("[tool] edit_resume_project failed: %s", e)
            return f"수정 중 오류: {e}"

        await slack_service.upload_file(
            channel_id,
            "한상기_이력서_수정본.html",
            html_bytes,
            f"✅ *{matched_title}* 수정 완료! 확인 후 마음에 들면 이 파일을 '내 이력서 틀이야 저장해줘'로 저장하세요.",
        )
        return f"*{matched_title}* 수정본 HTML 파일 전송했습니다."

    @langchain_tool
    async def list_resume_projects() -> str:
        """이력서 템플릿에 등록된 프로젝트 목록을 조회합니다."""
        from app.services import resume_service
        titles = resume_service.list_project_titles()
        if not titles:
            return "이력서 템플릿이 없거나 프로젝트가 없습니다."
        lines = "\n".join(f"• {t}" for t in titles)
        return f"이력서 프로젝트 목록 ({len(titles)}개):\n{lines}"

    @langchain_tool
    async def save_resume_template(confirmed: bool = True) -> str:
        """이력서 HTML 템플릿을 저장합니다. 사용자가 HTML 파일과 함께 '이력서 틀이야 저장해줘'라고 할 때 호출하세요."""
        from app.services import resume_service
        if resume_service.template_exists():
            return "이력서 템플릿이 이미 저장되어 있습니다. 새 HTML 파일을 업로드하면 덮어씁니다."
        return "이력서 HTML 파일을 Slack에 업로드하면서 '이력서 틀이야 저장해줘'라고 말해주세요."

    @langchain_tool
    async def create_resume(company_name: str, job_posting: str) -> str:
        """채용공고를 분석해 자기소개를 생성하고 이력서 PDF를 만듭니다.
        company_name: 회사명, job_posting: 채용공고 전문 또는 요약"""
        from app.core.database import AsyncSessionLocal
        from app.services import resume_service, file_service, slack_service

        if not resume_service.template_exists():
            return "이력서 템플릿이 없습니다. 먼저 HTML 파일을 업로드하면서 '내 이력서 틀이야 저장해줘'라고 해주세요."

        if not channel_id:
            return "Slack 채널 정보가 없어 PDF를 전송할 수 없습니다."

        await slack_service.send_message(channel_id, f"📝 *{company_name}* 이력서 생성 중... (Gemma4가 자기소개 작성 중, 1-2분 소요)")

        try:
            pdf_bytes = await asyncio.wait_for(
                resume_service.generate_resume(company_name, job_posting),
                timeout=300,
            )
        except FileNotFoundError as e:
            return str(e)
        except asyncio.TimeoutError:
            return "이력서 생성 시간 초과. 다시 시도해주세요."
        except Exception as e:
            logger.warning("[tool] create_resume failed: %s", e)
            return f"이력서 생성 중 오류: {e}"

        filename = f"한상기 이력서 ({company_name}).pdf"
        async with AsyncSessionLocal() as db:
            await file_service.save_file(db, user_id, filename, pdf_bytes, "application/pdf", category="이력서")

        await slack_service.upload_file(channel_id, filename, pdf_bytes, f"*{company_name}* 맞춤 이력서입니다.")
        return f"✅ *{company_name}* 이력서 PDF 생성 완료! 이력서 카테고리에도 저장했습니다."

    @langchain_tool
    async def delete_file(query: str) -> str:
        """파일 삭제 요청 시 후보 목록을 먼저 보여줍니다. 사용자가 정확한 파일명을 입력하면 삭제됩니다. query: 파일명·카테고리·날짜 등. '전체' 또는 '모두' 입력 시 전체 목록 표시."""
        logger.info("[tool] delete_file user=%s query=%s", user_id, query[:50])
        from app.core.database import AsyncSessionLocal
        from app.services import file_service

        _ALL_KEYWORDS = {"전체", "모두", "모든", "다", "all", "ALL"}
        is_all = any(kw in query for kw in _ALL_KEYWORDS)

        async with AsyncSessionLocal() as db:
            if is_all:
                candidates = await file_service.list_all_files(db, user_id)
            else:
                filenames = await file_service.search_files(user_id, query, n=10)
                seen: set[str] = set()
                candidates = []
                for fn in filenames:
                    if fn in seen:
                        continue
                    seen.add(fn)
                    f = await file_service.get_file_by_name(db, user_id, fn)
                    if f:
                        candidates.append(f)

        if not candidates:
            return "관련 파일을 찾지 못했습니다."

        # DB 기준으로 중복 제거 (같은 original_name 여러 번 나올 경우)
        seen_names: set[str] = set()
        deduped = []
        for f in candidates:
            if f.original_name not in seen_names:
                seen_names.add(f.original_name)
                deduped.append(f)
        candidates = deduped

        _pending_delete[user_id] = candidates
        lines = [f"• {f.original_name} ({f.updated_at.strftime('%Y-%m-%d')})" for f in candidates]
        return (
            "🗑️ 다음 파일이 검색됐습니다. *삭제할 파일명을 정확히 입력*해주세요.\n\n"
            + "\n".join(lines)
            + "\n\n취소하려면 '취소'를 입력하세요."
        )

    return [
        search_memory, save_memory, save_fact, query_knowledge_graph, get_weather, summarize_url,
        add_calendar_event, list_calendar_events,
        add_expense, get_expense_summary,
        set_reminder, list_reminders, cancel_reminder,
        add_todo, list_todos, list_todos_by_category, list_todo_categories, complete_todo,
        web_search, find_file, list_files,
        find_files_by_category, set_file_category, list_categories,
        delete_file,
        save_resume_template, create_resume,
        edit_resume_project, list_resume_projects,
    ]


async def _invoke_graph(llm, tools, message: str, timeout: int):
    graph = create_react_agent(model=llm, tools=tools, prompt=_build_system_prompt())
    return await asyncio.wait_for(
        graph.ainvoke(
            {"messages": [("user", message)]},
            config={"recursion_limit": 20},
        ),
        timeout=timeout,
    )


async def _get_history_context(user_id: str) -> str:
    """대화 히스토리를 반환한다. 5턴 초과 시 오래된 대화를 LLM으로 요약해 압축한다."""
    history = _chat_histories.get(user_id, [])
    if not history:
        return ""

    if len(history) <= _MAX_HISTORY_TURNS:
        lines = [f"사용자: {u}\n비서: {a}" for u, a in history]
        return "[이전 대화]\n" + "\n\n".join(lines) + "\n"

    # 오래된 대화 요약 + 최근 N턴 원문 유지
    old = history[:-_KEEP_RECENT_TURNS]
    recent = history[-_KEEP_RECENT_TURNS:]
    old_text = "\n".join(f"사용자: {u}\n비서: {a}" for u, a in old)
    try:
        llm = _make_gemini() if settings.gemini_api_key else _make_ollama()
        resp = await asyncio.wait_for(
            llm.ainvoke(
                f"다음 대화를 핵심 사실 위주로 2-3문장으로 요약해줘. 한국어로.\n\n{old_text}"
            ),
            timeout=15,
        )
        summary = (resp.content or "").strip()
        # 히스토리를 요약 1개 + 최근 N턴으로 교체
        _chat_histories[user_id] = [("(이전 대화 요약)", summary)] + list(recent)
        logger.info("[agent] history compressed old=%d turns", len(old))
    except Exception as e:
        logger.warning("[agent] history compress failed: %s", e)
        summary = "(이전 대화 요약 실패)"

    recent_lines = [f"사용자: {u}\n비서: {a}" for u, a in recent]
    return (
        f"[이전 대화 요약]\n{summary}\n\n"
        f"[최근 대화]\n" + "\n\n".join(recent_lines) + "\n"
    )


async def _prefetch_memory(user_id: str, message: str) -> str:
    """ChromaDB + KG 병렬 사전 검색 후 컨텍스트로 주입한다."""
    try:
        from app.services import memory_service, graph_service
        from app.core import kiwi as _kiwi_mod

        # ChromaDB 검색
        memories_task = asyncio.create_task(
            asyncio.wait_for(memory_service.search_memory(user_id, message), timeout=10)
        )

        # Kiwi로 고유명사/일반명사 추출 → KG 조회
        kg_results: list[str] = []
        try:
            morphs = _kiwi_mod.get().tokenize(message)
            entities = list(dict.fromkeys(
                t.form for t in morphs if t.tag in ("NNP", "NNG") and len(t.form) > 1
            ))
            for entity in entities:
                results = graph_service.query_graph(user_id, entity)
                kg_results.extend(results)
            kg_results = list(dict.fromkeys(kg_results))
        except Exception as e:
            logger.warning("[agent] prefetch kg failed: %s", e)

        memories = await memories_task

        parts: list[str] = []
        if memories:
            parts.append("[ChromaDB 검색 결과]\n" + "\n".join(f"- {m}" for m in memories))
        if kg_results:
            parts.append("[Knowledge Graph 검색 결과 — 구조화된 사실 정보]\n" + "\n".join(f"- {r}" for r in kg_results))

        if not parts:
            return message

        logger.info("[agent] prefetch_memory chroma=%d kg=%d", len(memories), len(kg_results))
        combined = "\n\n".join(parts)
        return (
            f"[⚠️ 사전 검색된 정보]\n"
            f"{combined}\n"
            f"[주의] KG 결과가 있으면 ChromaDB보다 우선합니다. 두 정보가 충돌하면 KG를 신뢰하세요. "
            f"위 정보로 답할 수 없으면 search_memory나 query_knowledge_graph를 재호출하세요.\n\n"
            f"[사용자 메시지]\n{message}"
        )
    except Exception as e:
        logger.warning("[agent] prefetch_memory failed: %s", e)
        return message



async def chat(user_id: str, message: str, channel_id: str = "") -> str:
    """Gemini 우선 사용, 할당량 초과 시 Ollama로 자동 전환."""
    logger.info("[agent] chat user=%s message_len=%d", user_id, len(message))

    use_gemini = bool(settings.gemini_api_key)
    llm = _make_gemini() if use_gemini else _make_ollama()
    llm_name = settings.gemini_model if use_gemini else settings.ollama_model
    logger.info("[agent] using llm=%s", llm_name)

    tools = _make_tools(user_id, channel_id)
    history_context = await _get_history_context(user_id)
    augmented_message = await _prefetch_memory(user_id, message)
    if history_context:
        augmented_message = history_context + "\n" + augmented_message

    try:
        timeout = 300 if not use_gemini else 120
        result = await _invoke_graph(llm, tools, augmented_message, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("[agent] timeout llm=%s", llm_name)
        return "죄송합니다, 응답 시간이 초과되었습니다. 다시 시도해 주세요."
    except Exception as e:
        if use_gemini and _is_quota_error(e):
            logger.warning("[agent] Gemini 할당량 초과 → Ollama로 전환: %s", e)
            llm = _make_ollama()
            try:
                result = await _invoke_graph(llm, tools, augmented_message, timeout=300)
            except asyncio.TimeoutError:
                logger.warning("[agent] Ollama timeout")
                return "죄송합니다, 응답 시간이 초과되었습니다. 다시 시도해 주세요."
            except Exception as e2:
                logger.warning("[agent] Ollama error: %s", e2)
                return "죄송합니다, 오류가 발생했습니다."
        else:
            logger.warning("[agent] error llm=%s: %s", llm_name, e)
            return "죄송합니다, 오류가 발생했습니다."

    def _extract_text(content) -> str:
        if isinstance(content, list):
            return " ".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")
        return content or ""

    for i, msg in enumerate(result["messages"]):
        tc = getattr(msg, "tool_calls", [])
        logger.info("[agent] msg[%d] type=%s content_len=%d tool_calls=%d",
                    i, type(msg).__name__, len(str(msg.content)), len(tc))

    reply = _extract_text(result["messages"][-1].content)
    if not reply.strip():
        logger.warning("[agent] empty reply — scanning previous messages")
        for msg in reversed(result["messages"][:-1]):
            reply = _extract_text(msg.content)
            if reply.strip():
                break
        else:
            reply = "처리가 완료되었습니다."

    logger.info("[agent] reply_len=%d", len(reply))

    # 대화 히스토리 저장
    if reply.strip():
        history = _chat_histories.setdefault(user_id, [])
        history.append((message[:500], reply[:500]))

    return reply


