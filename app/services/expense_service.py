import json
import logging
from collections import defaultdict
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.core.config import settings

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_HEADER = ["날짜", "금액", "카테고리", "메모"]


def _load_credentials_data() -> dict:
    import os
    path = settings.google_credentials_path
    if path and os.path.exists(path):
        with open(path) as f:
            raw = f.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return json.loads(raw, strict=False)
    return json.loads(settings.google_credentials_json)


def _service():
    creds = service_account.Credentials.from_service_account_info(_load_credentials_data(), scopes=_SCOPES)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _ensure_header() -> None:
    """시트 첫 행에 헤더가 없으면 추가한다."""
    try:
        service = _service()
        result = service.spreadsheets().values().get(
            spreadsheetId=settings.expense_sheet_id,
            range="A1:D1",
        ).execute()
        if not result.get("values"):
            service.spreadsheets().values().update(
                spreadsheetId=settings.expense_sheet_id,
                range="A1",
                valueInputOption="RAW",
                body={"values": [_HEADER]},
            ).execute()
    except Exception as e:
        logger.warning("[expense] ensure_header failed: %s", e)


def add_expense(amount: int, category: str, memo: str = "") -> str:
    """Google Sheets에 지출을 기록한다."""
    try:
        _ensure_header()
        today = datetime.now().strftime("%Y-%m-%d")
        row = [today, amount, category, memo]
        _service().spreadsheets().values().append(
            spreadsheetId=settings.expense_sheet_id,
            range="A:D",
            valueInputOption="RAW",
            body={"values": [row]},
        ).execute()
        logger.info("[expense] added %d %s %s", amount, category, memo)
        return f"지출 기록 완료: {category} {amount:,}원 ({memo})"
    except Exception as e:
        logger.warning("[expense] add_expense failed: %s", e)
        return f"지출 기록 실패: {e}"


def get_monthly_summary(year_month: str = "") -> str:
    """월별 카테고리별 지출 합계를 반환한다. year_month: YYYY-MM"""
    try:
        if not year_month:
            year_month = datetime.now().strftime("%Y-%m")

        result = _service().spreadsheets().values().get(
            spreadsheetId=settings.expense_sheet_id,
            range="A:D",
        ).execute()
        rows = result.get("values", [])
        if len(rows) <= 1:
            return f"{year_month} 지출 내역이 없습니다."

        totals: dict[str, int] = defaultdict(int)
        grand_total = 0
        for row in rows[1:]:  # 헤더 제외
            if len(row) < 3:
                continue
            date, amount_str, category = row[0], row[1], row[2]
            if not date.startswith(year_month):
                continue
            try:
                amount = int(str(amount_str).replace(",", ""))
                totals[category] += amount
                grand_total += amount
            except ValueError:
                continue

        if not totals:
            return f"{year_month} 지출 내역이 없습니다."

        lines = [f"📊 {year_month} 지출 요약"]
        for cat, total in sorted(totals.items(), key=lambda x: -x[1]):
            lines.append(f"• {cat}: {total:,}원")
        lines.append(f"합계: {grand_total:,}원")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("[expense] get_monthly_summary failed: %s", e)
        return f"지출 요약 실패: {e}"
