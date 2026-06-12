import logging

from ddgs import DDGS

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 3) -> str:
    """DuckDuckGo로 웹을 검색하고 상위 결과를 반환한다."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "검색 결과가 없습니다."
        lines = [f"🔍 '{query}' 검색 결과:"]
        for r in results:
            lines.append(f"\n[{r['title']}]\n{r['body']}\n출처: {r['href']}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("[search] web_search failed: %s", e)
        return f"검색 실패: {e}"
