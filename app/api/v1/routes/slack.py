import hashlib
import hmac
import json
import logging
import re
import time

from fastapi import APIRouter, BackgroundTasks, Request, Response

from app.core.config import settings
from app.core import kiwi as _kiwi_mod
from app.services import agent_service, slack_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/slack", tags=["slack"])

_RESUME_TRIGGER_RE = re.compile(r"이력서\s*(?:써줘|작성해줘|만들어줘|써|작성해|만들어|작성)\s+(\S+)")
# 마지막으로 언급된 이력서 지원 회사명: user_id → company_name (자기소개만 나중에 따로 보낼 때 사용)
_last_resume_company: dict[str, str] = {}

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
        # event.text + 각 파일의 initial_comment 합산 (Slack이 둘 중 하나에만 텍스트를 넣는 경우 대응)
        _file_comments = " ".join(
            (f.get("initial_comment") or {}).get("comment", "")
            for f in files
            if isinstance(f.get("initial_comment"), dict)
        )
        _full_text = f"{text} {_file_comments}".strip()

        # 이력서 HTML 템플릿 저장 감지: HTML 파일 1개 + "이력서 틀" 키워드
        _RESUME_KEYWORDS = {"이력서 틀", "이력서틀", "resume template", "이력서 템플릿"}
        _f0 = files[0] if files else {}
        is_resume_template = (
            len(files) == 1
            and (
                _f0.get("mimetype", "").startswith("text/html")
                or _f0.get("filetype") == "html"
            )
            and any(kw in _full_text for kw in _RESUME_KEYWORDS)
        )

        # 이력서 사진 저장 감지: 이미지 파일 + "이력서 사진" 키워드
        _PHOTO_KEYWORDS = {"이력서 사진", "이력서사진", "resume photo", "이력서 사진이야", "프로필 사진이야"}
        is_resume_photo = (
            len(files) == 1
            and files[0].get("mimetype", "").startswith("image/")
            and any(kw in _full_text for kw in _PHOTO_KEYWORDS)
        )

        # 채용공고 이미지 감지: 이미지 파일(1장 이상) + "이력서" 관련 키워드
        _JOB_IMG_KEYWORDS = {"이력서 작성해줘", "이력서 만들어줘", "이력서 써줘", "이력서 작성", "이력서 써", "이력서 만들어"}
        _img_files = [f for f in files if f.get("mimetype", "").startswith("image/")]
        is_job_posting_image = (
            not is_resume_photo
            and not is_resume_template
            and len(_img_files) >= 1
            and any(kw in _full_text for kw in _JOB_IMG_KEYWORDS)
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

        if is_job_posting_image:
            from app.services import file_service as _fs
            from app.core.config import settings as _s
            try:
                import base64
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.messages import HumanMessage as _HM

                vlm = ChatGoogleGenerativeAI(model=_s.gemini_model, google_api_key=_s.gemini_api_key)

                # 이미지 여러 장을 한 번에 Vision에 전달
                content_parts: list = []
                for f in _img_files:
                    img_bytes = await _fs._download(f.get("url_private") or f.get("url_private_download", ""))
                    mime = "image/png" if img_bytes[:8] == b'\x89PNG\r\n\x1a\n' else "image/jpeg"
                    b64 = base64.b64encode(img_bytes).decode()
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
                content_parts.append({"type": "text", "text": f"이 이미지{'들' if len(_img_files) > 1 else ''}에서 채용공고 내용을 모두 추출해줘. 여러 장이면 순서대로 합쳐서 정리해줘. 회사명, 담당업무, 자격요건, 우대사항, 근무조건 전부 포함."})

                vr = await vlm.ainvoke([_HM(content=content_parts)])
                c = vr.content
                extracted = (" ".join(b["text"] for b in c if isinstance(b, dict) and b.get("type") == "text") if isinstance(c, list) else c or "").strip()
                combined = f"{text}\n\n[채용공고 이미지 내용]\n{extracted}"
                logger.info("[slack] job posting images=%d extracted_len=%d", len(_img_files), len(extracted))
                reply = await agent_service.chat(user_id, combined, channel_id)
                if reply:
                    await slack_service.send_message(channel_id, reply)
            except Exception as e:
                logger.warning("[slack] job posting image failed: %s", e)
                await slack_service.send_message(channel_id, f"채용공고 이미지 읽기 실패: {e}")
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

    text = re.sub(r"<@\w+>", "", text).strip()
    if not text:
        return

    # 이력서 회사명 언급 기억 (자기소개만 나중에 따로 보낼 때 사용)
    _m_company = _RESUME_TRIGGER_RE.search(text)
    if _m_company:
        _last_resume_company[user_id] = _m_company.group(1)

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

    # 자기소개 직접 제공 감지: LLM 라우팅을 거치지 않고 원문 그대로 이력서를 만든다
    user_intro, _ = agent_service._extract_user_intro(text)
    if user_intro:
        company_name = (_m_company.group(1) if _m_company else _last_resume_company.get(user_id, ""))
        if not company_name:
            await slack_service.send_message(channel_id, "어느 회사 이력서인지 알려주세요. 예: '이력서 써줘 [회사명]'")
            return

        from app.services import resume_service, file_service
        from app.core.database import AsyncSessionLocal

        if not resume_service.template_exists():
            await slack_service.send_message(channel_id, "이력서 템플릿이 없습니다. 먼저 HTML 파일을 업로드하면서 '내 이력서 틀이야 저장해줘'라고 해주세요.")
            return

        _last_resume_company[user_id] = company_name
        await slack_service.send_message(channel_id, f"📝 *{company_name}* 이력서 생성 중... (제공하신 자기소개를 그대로 사용합니다)")

        try:
            pdf_bytes = await resume_service.generate_resume(company_name, "", user_intro)
        except Exception as e:
            logger.warning("[slack] verbatim intro resume failed: %s", e)
            await slack_service.send_message(channel_id, f"이력서 생성 중 오류: {e}")
            return

        filename = f"한상기 이력서 ({company_name}).pdf"
        async with AsyncSessionLocal() as db:
            await file_service.save_file(db, user_id, filename, pdf_bytes, "application/pdf", category="이력서")
        await slack_service.upload_file(channel_id, filename, pdf_bytes, f"*{company_name}* 맞춤 이력서입니다.")
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
