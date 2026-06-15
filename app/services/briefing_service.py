import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.core.config import settings

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def _make_llm():
    if settings.gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.gemini_api_key)
    return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host, think=False)


async def _send_morning_briefing() -> None:
    """매일 아침 9시에 오늘 일정 + 할 일 + 지출 패턴 경고 브리핑을 Slack DM으로 전송한다."""
    from app.core.database import AsyncSessionLocal
    from app.services import calendar_service, expense_service, slack_service, todo_service

    if not settings.slack_my_user_id:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    calendar_info = calendar_service.list_events(today)

    async with AsyncSessionLocal() as db:
        todos_info = await todo_service.list_todos(db, settings.slack_my_user_id)

    spending_alert = expense_service.get_spending_alert()

    prompt = f"""오늘({today}) 아침 브리핑을 간결하게 작성해줘.

오늘 일정:
{calendar_info}

할 일 목록:
{todos_info}

친근하고 동기부여가 되는 톤으로 100자 이내로 작성해줘."""

    try:
        llm = _make_llm()
        response = await llm.ainvoke([
            SystemMessage(content="당신은 친절한 개인 비서입니다. 한국어로 답변하세요."),
            HumanMessage(content=prompt),
        ])
        briefing_text = response.content
    except Exception as e:
        logger.warning("[briefing] LLM failed: %s", e)
        briefing_text = ""

    parts = ["🌅 *좋은 아침이에요!*"]
    if briefing_text:
        parts.append(briefing_text)
    if spending_alert:
        parts.append(spending_alert)
    parts.extend([calendar_info, todos_info])

    message = "\n\n".join(p for p in parts if p.strip())
    await slack_service.send_dm(settings.slack_my_user_id, message)
    logger.info("[briefing] sent to user=%s", settings.slack_my_user_id)


async def _send_weekly_report() -> None:
    """매주 월요일 아침 9시에 지난주 요약 리포트를 Slack DM으로 전송한다."""
    from app.core.database import AsyncSessionLocal
    from app.services import calendar_service, expense_service, slack_service, todo_service

    if not settings.slack_my_user_id:
        return

    from datetime import date, timedelta
    today = date.today()
    week_ago = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    weekly_expense = expense_service.get_weekly_summary()

    async with AsyncSessionLocal() as db:
        completed_todos = await todo_service.list_completed_todos(db, settings.slack_my_user_id, days=7)

    this_week_calendar = calendar_service.list_events(today_str)

    prompt = f"""지난 한 주를 정리하고 이번 주를 응원하는 주간 리포트를 작성해줘.

지난 7일 지출:
{weekly_expense}

지난 7일 완료한 일:
{completed_todos}

이번 주 일정:
{this_week_calendar}

친근하고 격려하는 톤으로 150자 이내로 작성해줘."""

    try:
        llm = _make_llm()
        response = await llm.ainvoke([
            SystemMessage(content="당신은 친절한 개인 비서입니다. 한국어로 답변하세요."),
            HumanMessage(content=prompt),
        ])
        summary = response.content
    except Exception as e:
        logger.warning("[briefing] weekly LLM failed: %s", e)
        summary = ""

    parts = [f"📅 *주간 리포트 ({week_ago} ~ {today_str})*"]
    if summary:
        parts.append(summary)
    parts.extend([weekly_expense, completed_todos, f"*이번 주 일정*\n{this_week_calendar}"])

    message = "\n\n".join(p for p in parts if p.strip())
    await slack_service.send_dm(settings.slack_my_user_id, message)
    logger.info("[briefing] weekly report sent to user=%s", settings.slack_my_user_id)


def start_briefing_scheduler() -> None:
    global _scheduler
    _scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    _scheduler.add_job(_send_morning_briefing, "cron", hour=9, minute=0)
    _scheduler.add_job(_send_weekly_report, "cron", day_of_week="mon", hour=9, minute=0)
    _scheduler.start()
    logger.info("[briefing] scheduler started (daily 09:00, weekly Mon 09:00)")


def stop_briefing_scheduler() -> None:
    if _scheduler:
        _scheduler.shutdown()
