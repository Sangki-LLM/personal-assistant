import hashlib
import hmac
import json
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Request, Response

from app.core.config import settings
from app.services import agent_service, slack_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/slack", tags=["slack"])

_YES = {"예", "네", "ㅇㅇ", "응", "좋아", "맞아", "ㅇ", "ok", "OK", "그래"}
_NO = {"아니오", "아니", "아냐", "ㄴ", "취소", "no", "No", "NO", "싫어"}


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

    if event.get("bot_id") or not user_id:
        return

    if settings.slack_my_user_id and user_id != settings.slack_my_user_id:
        logger.info("[slack] ignored message from unauthorized user=%s", user_id)
        return

    # 이미지 파일 이벤트 처리
    files = event.get("files", [])
    if files:
        for f in files:
            mime = f.get("mimetype", "")
            if mime.startswith("image/"):
                url = f.get("url_private") or f.get("url_private_download", "")
                if url:
                    await agent_service.handle_receipt_image(user_id, url, channel_id)
                return

    if not text:
        return

    import re
    text = re.sub(r"<@\w+>", "", text).strip()
    if not text:
        return

    # 지출 기록 대기 중이면 예/아니오 처리
    if user_id in agent_service._pending_expense:
        pending = agent_service._pending_expense.pop(user_id)
        if text in _YES:
            from app.services import expense_service
            result = expense_service.add_expense(
                pending["amount"], pending["category"], pending["memo"]
            )
            await slack_service.send_message(channel_id, f"✅ {result}")
        else:
            await slack_service.send_message(channel_id, "취소했습니다.")
        return

    logger.info("[slack] processing user=%s text=%r", user_id, text[:100])
    reply = await agent_service.chat(user_id, text, channel_id)
    if reply:
        await slack_service.send_message(channel_id, reply)


@router.post("/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if settings.slack_signing_secret and not _verify_slack_signature(body, timestamp, signature):
        return Response(content="invalid signature", status_code=403)

    payload = await request.json()

    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    event = payload.get("event", {})
    event_type = event.get("type", "")
    channel_id = event.get("channel", "")

    if event_type in ("message", "app_mention") and channel_id:
        background_tasks.add_task(_handle_event, event, channel_id)

    return Response(status_code=200)


@router.post("/actions")
async def slack_actions(request: Request):
    """Block Kit 버튼 클릭 콜백을 처리한다."""
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if settings.slack_signing_secret and not _verify_slack_signature(body, timestamp, signature):
        return Response(content="invalid signature", status_code=403)

    # Slack은 actions를 application/x-www-form-urlencoded payload 필드로 전송
    from urllib.parse import parse_qs
    form = parse_qs(body.decode("utf-8"))
    raw_payload = form.get("payload", ["{}"])[0]
    payload = json.loads(raw_payload)

    actions = payload.get("actions", [])
    if not actions:
        return Response(status_code=200)

    action = actions[0]
    action_id = action.get("action_id", "")
    user_id = action.get("value", "")
    channel_id = payload.get("container", {}).get("channel_id", "") or \
                 payload.get("channel", {}).get("id", "")

    logger.info("[slack/actions] action_id=%s user_id=%s", action_id, user_id)

    if action_id == "expense_confirm":
        pending = agent_service._pending_expense.pop(user_id, None)
        if pending:
            from app.services import expense_service
            result = expense_service.add_expense(
                pending["amount"], pending["category"], pending["memo"]
            )
            await slack_service.send_message(channel_id, f"✅ {result}")
        else:
            await slack_service.send_message(channel_id, "이미 처리된 요청입니다.")

    elif action_id == "expense_cancel":
        agent_service._pending_expense.pop(user_id, None)
        await slack_service.send_message(channel_id, "취소했습니다.")

    return Response(status_code=200)
