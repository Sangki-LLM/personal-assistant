import hashlib
import hmac
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Request, Response

from app.core.config import settings
from app.services import agent_service, slack_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/slack", tags=["slack"])


def _verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    if abs(time.time() - float(timestamp)) > 300:
        return False
    base = f"v0:{timestamp}:{body.decode('utf-8')}"
    expected = "v0=" + hmac.new(
        settings.slack_signing_secret.encode(),
        base.encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def _handle_event(event: dict, channel_id: str) -> None:
    user_id = event.get("user", "")
    text = event.get("text", "").strip()

    # 봇 자신의 메시지 무시
    if event.get("bot_id") or not text:
        return

    # 허용된 사용자만 처리
    if settings.slack_my_user_id and user_id != settings.slack_my_user_id:
        logger.info("[slack] ignored message from unauthorized user=%s", user_id)
        return

    # <@BOTID> 멘션 제거
    import re
    text = re.sub(r"<@\w+>", "", text).strip()
    if not text:
        return

    logger.info("[slack] processing user=%s text=%r", user_id, text[:100])
    reply = await agent_service.chat(user_id, text)
    await slack_service.send_message(channel_id, reply)


@router.post("/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if settings.slack_signing_secret and not _verify_slack_signature(body, timestamp, signature):
        return Response(content="invalid signature", status_code=403)

    payload = await request.json()

    # Slack URL 검증 챌린지
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    event = payload.get("event", {})
    event_type = event.get("type", "")
    channel_id = event.get("channel", "")

    # DM 또는 앱 멘션 이벤트 처리
    if event_type in ("message", "app_mention") and channel_id:
        background_tasks.add_task(_handle_event, event, channel_id)

    return Response(status_code=200)
