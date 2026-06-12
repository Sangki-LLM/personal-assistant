import json
import logging
from datetime import datetime, timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.core.config import settings

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_TIMEZONE = "Asia/Seoul"


def _load_credentials_data() -> dict:
    """파일 경로 우선, 없으면 환경변수 JSON 문자열 사용."""
    import os
    path = settings.google_credentials_path
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return json.loads(settings.google_credentials_json)


def _service():
    creds = service_account.Credentials.from_service_account_info(_load_credentials_data(), scopes=_SCOPES)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _parse_datetime(date_str: str, time_str: str) -> tuple[str, str]:
    """날짜/시간 문자열을 RFC3339 형식으로 변환한다."""
    now = datetime.now()

    # 날짜 파싱
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d", "%m-%d"):
        try:
            d = datetime.strptime(date_str, fmt)
            if d.year == 1900:
                d = d.replace(year=now.year)
            break
        except ValueError:
            continue
    else:
        d = now

    # 시간 파싱
    for fmt in ("%H:%M", "%H시%M분", "%I:%M %p", "%H"):
        try:
            t = datetime.strptime(time_str.strip(), fmt)
            break
        except ValueError:
            continue
    else:
        t = now

    start = d.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    end = start + timedelta(hours=1)
    fmt = "%Y-%m-%dT%H:%M:%S"
    return start.strftime(fmt), end.strftime(fmt)


def add_event(title: str, date_str: str, time_str: str, description: str = "") -> str:
    """Google Calendar에 일정을 추가한다."""
    try:
        start_dt, end_dt = _parse_datetime(date_str, time_str)
        service = _service()
        event = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_dt, "timeZone": _TIMEZONE},
            "end": {"dateTime": end_dt, "timeZone": _TIMEZONE},
        }
        created = service.events().insert(calendarId="primary", body=event).execute()
        logger.info("[calendar] event created: %s", created.get("id"))
        return f"일정 등록 완료: {title} ({start_dt[:16]})"
    except Exception as e:
        logger.warning("[calendar] add_event failed: %s", e)
        return f"일정 등록 실패: {e}"


def list_events(date_str: str) -> str:
    """특정 날의 일정을 조회한다."""
    try:
        now = datetime.now()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d", "%m-%d"):
            try:
                d = datetime.strptime(date_str, fmt)
                if d.year == 1900:
                    d = d.replace(year=now.year)
                break
            except ValueError:
                continue
        else:
            d = now

        start = d.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        service = _service()
        events_result = service.events().list(
            calendarId="primary",
            timeMin=start.isoformat() + "+09:00",
            timeMax=end.isoformat() + "+09:00",
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        items = events_result.get("items", [])
        if not items:
            return f"{date_str} 일정이 없습니다."

        lines = [f"{date_str} 일정:"]
        for item in items:
            start_time = item["start"].get("dateTime", item["start"].get("date", ""))
            if "T" in start_time:
                start_time = start_time[11:16]
            lines.append(f"• {start_time} {item['summary']}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("[calendar] list_events failed: %s", e)
        return f"일정 조회 실패: {e}"
