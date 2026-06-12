import asyncio
import logging

from langchain_core.tools import tool as langchain_tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 친절하고 유능한 개인 비서입니다. 항상 한국어로 답변하세요.

사용 가능한 도구:
- search_memory: 과거 대화나 메모에서 관련 내용을 검색합니다
- save_memory: 중요한 정보를 기억합니다
- add_calendar_event: Google Calendar에 일정을 추가합니다
- list_calendar_events: 특정 날의 일정을 조회합니다
- add_expense: 지출을 Google Sheets에 기록합니다
- get_expense_summary: 월별 지출 요약을 조회합니다

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
        from app.services import memory_service
        results = await memory_service.search_memory(user_id, query)
        if not results:
            return "관련 기억이 없습니다."
        return "\n---\n".join(results)

    @langchain_tool
    async def save_memory(text: str) -> str:
        """중요한 정보를 기억합니다."""
        from app.services import memory_service
        await memory_service.store_memory(user_id, text)
        return "기억했습니다."

    @langchain_tool
    def add_calendar_event(title: str, date: str, time: str, description: str = "") -> str:
        """Google Calendar에 일정을 추가합니다. date: YYYY-MM-DD, time: HH:MM"""
        from app.services import calendar_service
        return calendar_service.add_event(title, date, time, description)

    @langchain_tool
    def list_calendar_events(date: str) -> str:
        """특정 날의 일정을 조회합니다. date: YYYY-MM-DD"""
        from app.services import calendar_service
        return calendar_service.list_events(date)

    @langchain_tool
    def add_expense(amount: int, category: str, memo: str = "") -> str:
        """지출을 Google Sheets에 기록합니다. amount: 금액(정수), category: 카테고리, memo: 메모"""
        from app.services import expense_service
        return expense_service.add_expense(amount, category, memo)

    @langchain_tool
    def get_expense_summary(year_month: str = "") -> str:
        """월별 카테고리별 지출 요약을 조회합니다. year_month: YYYY-MM (빈값이면 이번달)"""
        from app.services import expense_service
        return expense_service.get_monthly_summary(year_month)

    return [search_memory, save_memory, add_calendar_event, list_calendar_events,
            add_expense, get_expense_summary]


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
                config={"recursion_limit": 20},
            ),
            timeout=120,
        )
        reply = result["messages"][-1].content
        logger.info("[agent] reply_len=%d", len(reply))
        return reply
    except asyncio.TimeoutError:
        logger.warning("[agent] timeout")
        return "죄송합니다, 응답 시간이 초과되었습니다. 다시 시도해 주세요."
    except Exception as e:
        logger.warning("[agent] error: %s", e)
        return "죄송합니다, 오류가 발생했습니다."
