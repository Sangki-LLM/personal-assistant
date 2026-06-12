import asyncio
import logging

from langchain_core.tools import tool as langchain_tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from app.core.config import settings

logger = logging.getLogger(__name__)

_TOOL_TIMEOUT = 45  # 도구별 최대 대기 시간 (초)

SYSTEM_PROMPT = """당신은 친절하고 유능한 개인 비서입니다. 항상 한국어로 답변하세요.

사용 가능한 도구:
- search_memory: 과거 대화나 메모에서 관련 내용을 검색합니다
- save_memory: 중요한 정보를 기억합니다
- add_calendar_event: Google Calendar에 일정을 추가합니다
- list_calendar_events: 특정 날의 일정을 조회합니다
- add_expense: 지출을 Google Sheets에 기록합니다
- get_expense_summary: 월별 지출 요약을 조회합니다
- set_reminder: 특정 시간에 알림을 설정합니다
- add_todo: 할 일을 추가합니다
- list_todos: 할 일 목록을 조회합니다
- complete_todo: 할 일을 완료 처리합니다
- web_search: 인터넷에서 최신 정보를 검색합니다

대화 흐름:
1. 사용자가 과거 정보를 물어보면 search_memory를 먼저 호출하세요
2. 사용자가 기억해달라고 하면 save_memory를 호출하세요
3. 일정 추가/조회 요청에는 calendar 도구를 사용하세요
4. 날짜는 YYYY-MM-DD, 시간은 HH:MM 형식으로 변환해서 전달하세요
5. 간결하고 친근하게 답변하세요"""


def _make_tools(user_id: str):
    @langchain_tool
    async def search_memory(query: str) -> str:
        """과거 대화나 기억에서 관련 내용을 검색합니다."""
        logger.info("[tool] search_memory user=%s query=%s", user_id, query[:50])
        from app.services import memory_service
        results = await memory_service.search_memory(user_id, query)
        logger.info("[tool] search_memory done count=%d", len(results))
        if not results:
            return "관련 기억이 없습니다."
        return "\n---\n".join(results)

    @langchain_tool
    async def save_memory(text: str) -> str:
        """중요한 정보를 기억합니다."""
        logger.info("[tool] save_memory user=%s text_len=%d", user_id, len(text))
        from app.services import memory_service
        await memory_service.store_memory(user_id, text)
        logger.info("[tool] save_memory done")
        return "기억했습니다."

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
        """특정 시간에 Slack 알림을 설정합니다. datetime_str: 'YYYY-MM-DD HH:MM' 또는 '30분 후', '1시간 후'"""
        import re
        from datetime import datetime, timedelta
        from app.core.database import AsyncSessionLocal
        from app.services import reminder_service as rs

        logger.info("[tool] set_reminder message=%s datetime=%s", message[:40], datetime_str)
        now = datetime.now()
        m = re.search(r"(\d+)\s*(분|시간)", datetime_str)
        if m:
            val, unit = int(m.group(1)), m.group(2)
            run_at = now + (timedelta(minutes=val) if unit == "분" else timedelta(hours=val))
        else:
            try:
                run_at = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            except ValueError:
                return "시간 형식을 인식하지 못했습니다. 예: '30분 후', '2026-06-12 15:00'"

        async with AsyncSessionLocal() as db:
            result = await rs.set_reminder(db, user_id, message, run_at)
        logger.info("[tool] set_reminder done")
        return result

    @langchain_tool
    async def add_todo(content: str) -> str:
        """할 일을 추가합니다."""
        logger.info("[tool] add_todo content=%s", content[:40])
        from app.core.database import AsyncSessionLocal
        from app.services import todo_service
        async with AsyncSessionLocal() as db:
            return await todo_service.add_todo(db, user_id, content)

    @langchain_tool
    async def list_todos() -> str:
        """완료되지 않은 할 일 목록을 조회합니다."""
        logger.info("[tool] list_todos user=%s", user_id)
        from app.core.database import AsyncSessionLocal
        from app.services import todo_service
        async with AsyncSessionLocal() as db:
            return await todo_service.list_todos(db, user_id)

    @langchain_tool
    async def complete_todo(todo_id: int) -> str:
        """할 일을 완료 처리합니다. todo_id: 할 일 번호"""
        logger.info("[tool] complete_todo id=%d", todo_id)
        from app.core.database import AsyncSessionLocal
        from app.services import todo_service
        async with AsyncSessionLocal() as db:
            return await todo_service.complete_todo(db, user_id, todo_id)

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

    return [search_memory, save_memory, add_calendar_event, list_calendar_events,
            add_expense, get_expense_summary, set_reminder, add_todo, list_todos,
            complete_todo, web_search]


async def chat(user_id: str, message: str) -> str:
    """사용자 메시지를 받아 LangGraph ReAct 에이전트로 처리한다."""
    logger.info("[agent] chat user=%s message_len=%d", user_id, len(message))

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_host,
        think=False,
    )
    tools = _make_tools(user_id)
    graph = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(
                {"messages": [("user", message)]},
                config={"recursion_limit": 10},
            ),
            timeout=300,
        )
        reply = result["messages"][-1].content
        logger.info("[agent] reply_len=%d", len(reply))
        return reply
    except asyncio.TimeoutError:
        logger.warning("[agent] timeout after 300s")
        return "죄송합니다, 응답 시간이 초과되었습니다. 다시 시도해 주세요."
    except Exception as e:
        logger.warning("[agent] error: %s", e)
        return "죄송합니다, 오류가 발생했습니다."
