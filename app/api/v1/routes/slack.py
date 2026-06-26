import hashlib
import hmac
import json
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Request, Response

from app.core.config import settings
from app.core import kiwi as _kiwi_mod
from app.services import agent_service, slack_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/slack", tags=["slack"])

_YES = {"예", "네", "ㅇㅇ", "응", "좋아", "맞아", "ㅇ", "ok", "OK", "그래"}
_NO = {"아니오", "아니", "아냐", "ㄴ", "취소", "no", "No", "NO", "싫어"}
_YES_STEMS = {"응", "네", "맞", "좋"}
_NO_STEMS = {"아니", "싫"}


def _is_affirmative(text: str) -> bool:
    if text in _YES or text.lower() in {"ok", "yes"}:
        return True
    try:
        tokens = _kiwi_mod.get().tokenize(text)
        return any(t.form in _YES_STEMS for t in tokens)
    except Exception:
        return False


def _is_negative(text: str) -> bool:
    if text in _NO:
        return True
    try:
        tokens = _kiwi_mod.get().tokenize(text)
        return any(t.form in _NO_STEMS for t in tokens)
    except Exception:
        return False


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

    # 파일 이벤트 처리
    files = event.get("files", [])
    if files:
        # 이력서 HTML 템플릿 저장 감지: HTML 파일 1개 + "이력서 틀" 키워드
        _RESUME_KEYWORDS = {"이력서 틀", "이력서틀", "resume template", "이력서 템플릿"}
        _f0 = files[0] if files else {}
        is_resume_template = (
            len(files) == 1
            and (
                _f0.get("mimetype", "").startswith("text/html")
                or _f0.get("filetype") == "html"
            )
            and any(kw in text for kw in _RESUME_KEYWORDS)
        )

        # 이력서 사진 저장 감지: 이미지 파일 + "이력서 사진" 키워드
        _PHOTO_KEYWORDS = {"이력서 사진", "이력서사진", "resume photo", "이력서 사진이야", "프로필 사진이야"}
        is_resume_photo = (
            len(files) == 1
            and files[0].get("mimetype", "").startswith("image/")
            and any(kw in text for kw in _PHOTO_KEYWORDS)
        )

        if is_resume_photo:
            from app.services import file_service as _fs
            from pathlib import Path
            _PHOTO_DEST = Path(__file__).parent.parent.parent.parent / "templates" / "photo.jpg"
            try:
                img_bytes = await _fs._download(
                    files[0].get("url_private") or files[0].get("url_private_download", "")
                )
                _PHOTO_DEST.parent.mkdir(parents=True, exist_ok=True)
                _PHOTO_DEST.write_bytes(img_bytes)
                await slack_service.send_message(channel_id, "✅ 이력서 사진 저장 완료! 다음 이력서 생성 시 자동으로 포함됩니다.")
            except Exception as e:
                logger.warning("[slack] resume photo save failed: %s", e)
                await slack_service.send_message(channel_id, f"사진 저장 실패: {e}")
            return

        if is_resume_template:
            from app.services import file_service as _fs, resume_service
            try:
                html_bytes = await _fs._download(
                    files[0].get("url_private") or files[0].get("url_private_download", "")
                )
                msg = resume_service.save_template(html_bytes)
                await slack_service.send_message(channel_id, f"✅ {msg} — 이제 '이력서 작성해줘 [회사명]\\n[채용공고]' 형식으로 사용하세요.")
            except Exception as e:
                logger.warning("[slack] resume template save failed: %s", e)
                await slack_service.send_message(channel_id, f"이력서 템플릿 저장 실패: {e}")
            return

        from app.core.database import AsyncSessionLocal
        from app.services import file_service

        async with AsyncSessionLocal() as db:
            if len(files) == 1:
                result = await file_service.handle_slack_file(db, user_id, files[0], channel_id, text)
            else:
                result = await file_service.handle_slack_files(db, user_id, files, channel_id, text)
        await slack_service.send_message(channel_id, result)
        return

    if not text:
        return

    import re
    text = re.sub(r"<@\w+>", "", text).strip()
    if not text:
        return

    # 파일 삭제 확인 대기 중이면 파일명 일치 여부 처리
    if user_id in agent_service._pending_delete:
        candidates = agent_service._pending_delete[user_id]
        candidate_names = [f.original_name for f in candidates]

        if _is_negative(text):
            agent_service._pending_delete.pop(user_id)
            await slack_service.send_message(channel_id, "삭제를 취소했습니다.")
            return

        # 정확히 일치하는 파일 먼저 탐색
        matched_name = None
        if text in candidate_names:
            matched_name = text
        else:
            # 퍼지 매칭 — 유사도 0.6 이상 중 가장 가까운 것
            import difflib
            close = difflib.get_close_matches(text, candidate_names, n=1, cutoff=0.6)
            if close:
                matched_name = close[0]

        if matched_name:
            agent_service._pending_delete.pop(user_id)
            from app.core.database import AsyncSessionLocal
            from app.services import file_service
            async with AsyncSessionLocal() as db:
                result = await file_service.delete_file(db, user_id, matched_name)
            await slack_service.send_message(channel_id, result)
            return

        # 매칭 실패 시 목록 재안내
        lines = "\n".join(f"• {n}" for n in candidate_names)
        await slack_service.send_message(
            channel_id,
            f"파일을 특정하지 못했습니다. 아래 목록에서 파일명 일부를 입력하거나 번호를 말해주세요.\n\n{lines}\n\n취소: '취소'",
        )
        return

    # 지출 기록 대기 중이면 예/아니오 처리
    if user_id in agent_service._pending_expense:
        pending = agent_service._pending_expense.pop(user_id)
        if _is_affirmative(text):
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
