import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 친절하고 유능한 개인 비서입니다. 항상 한국어로 답변하세요.
간결하고 명확하게 답변해 주세요. 질문에 직접적으로 답하세요."""


async def chat(user_id: str, message: str) -> str:
    """사용자 메시지를 받아 LLM 응답을 반환한다."""
    logger.info("[agent] chat user=%s message_len=%d", user_id, len(message))
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_host,
        think=False,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message),
    ]
    response = await llm.ainvoke(messages)
    logger.info("[agent] response_len=%d", len(response.content))
    return response.content
