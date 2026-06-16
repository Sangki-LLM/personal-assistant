import asyncio
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

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


async def _llm_summarize(title: str, body: str) -> str:
    """기사 제목+본문으로 한 줄 요약을 반환한다. Gemini 우선, 없으면 Ollama 직접 호출."""
    prompt = (
        f"다음 기사를 한 줄(30자 이내)로 요약해줘. 요약문만 출력해.\n\n"
        f"제목: {title}\n내용: {body[:400]}"
    )
    try:
        if settings.gemini_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            llm = ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.gemini_api_key)
            resp = await asyncio.wait_for(llm.ainvoke([HumanMessage(content=prompt)]), timeout=30)
            raw = resp.content
            if isinstance(raw, list):
                return " ".join(b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text").strip()
            return str(raw).strip()
        else:
            # Ollama API 직접 호출 — langchain 우회로 think=false 확실히 적용
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{settings.ollama_host}/api/generate",
                    json={
                        "model": settings.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "think": False,
                        "options": {"num_predict": 100, "temperature": 0.3},
                    },
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("[news] summarize failed title=%r: %s", title[:30], e)
        return ""


class _LinkExtractor(HTMLParser):
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
    html = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


async def _fetch_article_links(client: httpx.AsyncClient, url: str) -> list[tuple[str, str]]:
    try:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()
        parser = _LinkExtractor(url)
        parser.feed(resp.text)
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


async def _fetch_and_summarize_article(client: httpx.AsyncClient, title: str, href: str) -> tuple[str, str, str]:
    """(title, summary, href) 반환."""
    try:
        resp = await client.get(href, timeout=10)
        resp.raise_for_status()
        body = _strip_html(resp.text)
    except Exception:
        body = ""

    summary = await _llm_summarize(title, body)
    return title, summary or title, href


async def _summarize_site(site_url: str) -> str:
    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        links = await _fetch_article_links(client, site_url)

    if not links:
        return f"⚠️ {site_url} 에서 기사를 가져올 수 없었습니다."

    candidates = links[:10]

    # 기사 fetch + 개별 LLM 요약 병렬 실행
    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        results = await asyncio.gather(
            *[_fetch_and_summarize_article(client, title, href) for title, href in candidates],
            return_exceptions=True,
        )

    lines = []
    for i, r in enumerate(results, 1):
        if isinstance(r, Exception):
            continue
        title, summary, href = r
        lines.append(f"{i}. *{title}*\n   {summary}\n   🔗 {href}")

    return "\n\n".join(lines) if lines else "기사를 가져올 수 없었습니다."


async def send_news_briefing() -> None:
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
