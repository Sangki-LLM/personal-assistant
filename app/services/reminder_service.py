import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _fire_reminders() -> None:
    """만료된 리마인더를 Slack DM으로 전송한다."""
    from app.models.reminder import Reminder
    from app.services import slack_service

    async with AsyncSessionLocal() as db:
        now = datetime.now()
        result = await db.execute(
            select(Reminder).where(Reminder.run_at <= now, Reminder.fired == False)
        )
        reminders = result.scalars().all()
        for r in reminders:
            await slack_service.send_dm(r.slack_user_id, f"⏰ 리마인더: {r.message}")
            r.fired = True
            logger.info("[reminder] fired id=%d user=%s", r.id, r.slack_user_id)
        if reminders:
            await db.commit()


def start_scheduler() -> None:
    global _scheduler
    _scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    _scheduler.add_job(_fire_reminders, "interval", minutes=1)
    _scheduler.start()
    logger.info("[reminder] scheduler started")


def stop_scheduler() -> None:
    if _scheduler:
        _scheduler.shutdown()


async def set_reminder(db, slack_user_id: str, message: str, run_at: datetime) -> str:
    from app.models.reminder import Reminder
    r = Reminder(slack_user_id=slack_user_id, message=message, run_at=run_at)
    db.add(r)
    await db.commit()
    logger.info("[reminder] set id=%d run_at=%s", r.id, run_at)
    return f"리마인더 설정: {run_at.strftime('%Y-%m-%d %H:%M')}에 '{message}' 알림"
