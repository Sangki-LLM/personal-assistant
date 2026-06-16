import asyncio
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.core.config import settings

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}


def _make_llm():
    if settings.gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.gemini_api_key)
    return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host, think=False)


class _LinkExtractor(HTMLParser):
    """메인 페이지에서 기사 링크(텍스트+href)를 추출한다."""

    def __init__(self, base_url: str):
        super().__init__()
        self._base = base_url
        self._base_netloc = urlparse(base_url).netloc
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "") or ""
            if not href or href.startswith(("#", "javascript")):
                return
            full = urljoin(self._base, href)
            # 같은 도메인 링크만
            if urlparse(full).netloc == self._base_netloc:
                self._href = full
                self._buf = []

    def handle_endtag(self, tag):
        if tag == "a" and self._href:
            text = " ".join(t for t in self._buf if t).strip()
            if 10 <= len(text) <= 120:
                self.links.append((text, self._href))
            self._href = None
            self._buf = []

    def handle_data(self, data):
        if self._href:
            stripped = data.strip()
            if stripped:
                self._buf.append(stripped)


def _strip_html(html: str) -> str:
    """HTML 태그를 제거하고 평문 텍스트만 반환한다."""
    html = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


async def _fetch_article_links(client: httpx.AsyncClient, url: str) -> list[tuple[str, str]]:
    """메인 페이지에서 기사 링크 (title, url) 목록을 반환한다."""
    try:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()
        parser = _LinkExtractor(url)
        parser.feed(resp.text)
        # 중복 URL 제거, 순서 유지
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for title, href in parser.links:
            if href not in seen:
                seen.add(href)
                unique.append((title, href))
        return unique
    except Exception as e:
        logger.warning("[news] fetch_links failed url=%s: %s", url, e)
        return []


async def _fetch_article_body(client: httpx.AsyncClient, url: str) -> str:
    """기사 URL의 본문 텍스트 앞부분을 반환한다."""
    try:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        text = _strip_html(resp.text)
        return text[:1200]
    except Exception:
        return ""


async def _summarize_site(site_url: str) -> str:
    """한 뉴스 사이트에서 기사를 수집하고 LLM 요약 텍스트를 반환한다."""
    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        links = await _fetch_article_links(client, site_url)

    if not links:
        return f"⚠️ {site_url} 에서 기사를 가져올 수 없었습니다."

    # 상위 후보 10개만 본문 병렬 fetch
    candidates = links[:10]
    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        bodies = await asyncio.gather(
            *[_fetch_article_body(client, href) for _, href in candidates],
            return_exceptions=True,
        )

    # 프롬프트용 기사 목록 구성
    articles_text = ""
    for i, ((title, href), body) in enumerate(zip(candidates, bodies), 1):
        body_str = body if isinstance(body, str) else ""
        articles_text += f"\n[{i}] 제목: {title}\nURL: {href}\n내용: {body_str[:300]}\n"

    prompt = (
        f"다음은 뉴스 사이트({site_url})에서 수집한 기사 목록입니다.\n"
        "이 중 중요하고 흥미로운 뉴스 기사 10개를 선별해서 아래 형식으로 정리해줘.\n"
        "중복되거나 광고/이벤트 성격이면 제외해.\n\n"
        "형식:\n"
        "1️⃣ *기사 제목 (20자 이내 요약)*\n"
        "   한 줄 내용 요약 (40자 이내)\n"
        "   🔗 URL\n\n"
        f"기사 목록:\n{articles_text}"
    )

    try:
        llm = _make_llm()
        resp = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content="당신은 뉴스 편집자입니다. 한국어로 간결하게 답변하세요."),
                HumanMessage(content=prompt),
            ]),
            timeout=180,
        )
        raw = resp.content
        if isinstance(raw, list):
            content = " ".join(b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text")
        else:
            content = str(raw or "")
        content = content.strip()
        logger.info("[news] LLM response len=%d preview=%r", len(content), content[:80])
        if not content:
            raise ValueError("LLM returned empty content")
        return content
    except Exception as e:
        logger.warning("[news] LLM failed or empty: %s — falling back to title list", e)
        fallback = "\n".join(f"• {t}\n  🔗 {h}" for t, h in candidates[:10])
        return f"(요약 실패 — 제목만 표시)\n{fallback}"


async def send_news_briefing() -> None:
    """설정된 뉴스 사이트를 크롤링해 기사 요약을 Slack DM으로 전송한다."""
    from app.services import slack_service

    if not settings.slack_my_user_id:
        return

    urls = [u.strip() for u in settings.news_urls.split(",") if u.strip()]
    logger.info("[news] news_urls raw=%r parsed=%d sites", settings.news_urls[:100], len(urls))
    if not urls:
        logger.info("[news] no news_urls configured, skipping")
        return

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"📰 *뉴스 브리핑 ({now})*"

    parts = [header]
    for url in urls:
        logger.info("[news] fetching site=%s", url)
        summary = await _summarize_site(url)
        domain = urlparse(url).netloc
        parts.append(f"\n*[ {domain} ]*\n{summary}")

    message = "\n\n".join(parts)
    await slack_service.send_dm(settings.slack_my_user_id, message)
    logger.info("[news] briefing sent to user=%s sites=%d", settings.slack_my_user_id, len(urls))
