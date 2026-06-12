import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.core.config import settings

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _send_morning_briefing() -> None:
    """매일 아침 9시에 오늘 일정 + 할 일 브리핑을 Slack DM으로 전송한다."""
    from app.core.database import AsyncSessionLocal
    from app.services import calendar_service, slack_service, todo_service

    if not settings.slack_my_user_id:
        return

    today = datetime.now().strftime("%Y-%m-%d")

    # 오늘 일정 조회
    calendar_info = calendar_service.list_events(today)

    # 오늘 할 일 조회
    async with AsyncSessionLocal() as db:
        todos_info = await todo_service.list_todos(db, settings.slack_my_user_id)

    # LLM으로 브리핑 생성
    prompt = f"""오늘({today}) 아침 브리핑을 간결하게 작성해줘.

오늘 일정:
{calendar_info}

할 일 목록:
{todos_info}

친근하고 동기부여가 되는 톤으로 100자 이내로 작성해줘."""

    try:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_host,
            think=False,
        )
        response = await llm.ainvoke([
            SystemMessage(content="당신은 친절한 개인 비서입니다. 한국어로 답변하세요."),
            HumanMessage(content=prompt),
        ])
        briefing_text = response.content
    except Exception as e:
        logger.warning("[briefing] LLM failed: %s", e)
        briefing_text = f"{calendar_info}\n\n{todos_info}"

    message = f"🌅 *좋은 아침이에요!*\n\n{briefing_text}\n\n{calendar_info}\n\n{todos_info}"
    await slack_service.send_dm(settings.slack_my_user_id, message)
    logger.info("[briefing] sent to user=%s", settings.slack_my_user_id)


def start_briefing_scheduler() -> None:
    global _scheduler
    _scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    _scheduler.add_job(_send_morning_briefing, "cron", hour=9, minute=0)
    _scheduler.start()
    logger.info("[briefing] scheduler started (daily 09:00)")


def stop_briefing_scheduler() -> None:
    if _scheduler:
        _scheduler.shutdown()
