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

대화 흐름:
1. 사용자가 과거 정보를 물어보면 search_memory를 먼저 호출하세요
2. 사용자가 기억해달라고 하면 save_memory를 호출하세요
3. 중요한 일정, 약속, 개인 정보는 자동으로 save_memory로 저장하세요
4. 간결하고 친근하게 답변하세요"""


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

    return [search_memory, save_memory]


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
