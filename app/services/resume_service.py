import asyncio
import difflib
import logging
import re
import tempfile
from pathlib import Path

from jinja2 import Template

from app.core.config import settings

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "resume_template.html"
_PHOTO_PATH = Path(__file__).parent.parent / "templates" / "photo.jpg"


def save_template(html_bytes: bytes) -> str:
    _TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TEMPLATE_PATH.write_bytes(html_bytes)
    logger.info("[resume] template saved size=%d", len(html_bytes))
    return "이력서 템플릿 저장 완료"


def template_exists() -> bool:
    return _TEMPLATE_PATH.exists()


async def generate_resume(company_name: str, job_posting: str) -> bytes:
    if not template_exists():
        raise FileNotFoundError("이력서 템플릿이 없습니다.")

    intro = await _generate_intro(company_name, job_posting)
    logger.info("[resume] intro generated len=%d", len(intro))

    html = _render(intro)
    pdf = await _html_to_pdf(html)
    logger.info("[resume] pdf generated size=%d", len(pdf))
    return pdf


def _make_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
    )


def _make_ollama():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host, think=False)


def _is_quota_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("quota", "429", "resource exhausted", "rate limit"))


def _extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")
    return content or ""


async def _invoke_llm(messages: list) -> str:
    use_gemini = bool(settings.gemini_api_key)
    llm = _make_gemini() if use_gemini else _make_ollama()
    try:
        resp = await llm.ainvoke(messages)
        return _extract_text(resp.content).strip()
    except Exception as e:
        if use_gemini and _is_quota_error(e):
            logger.warning("[resume] Gemini 할당량 초과 → Ollama 전환")
            resp = await _make_ollama().ainvoke(messages)
            return _extract_text(resp.content).strip()
        raise


def _calc_career() -> tuple[str, str]:
    """(경력기간, 년차) 반환 — 예: ('4년 5개월', '5년차')"""
    from datetime import date
    today = date.today()
    experiences = [
        (date(2021, 8, 1), date(2024, 3, 31)),   # 모빌씨앤씨
        (date(2024, 5, 1), date(2024, 12, 31)),   # 큐텐테크놀로지
        (date(2025, 5, 1), today),                 # 티앤엠테크
    ]
    total = sum((e.year - s.year) * 12 + e.month - s.month for s, e in experiences)
    years, months = total // 12, total % 12
    duration = f"{years}년 {months}개월" if months else f"{years}년"
    year_level = f"{years + 1}년차"
    return duration, year_level


async def _generate_intro(company_name: str, job_posting: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    career, year_level = _calc_career()

    prompt = f"""다음은 {company_name} 채용공고입니다.

{job_posting}

위 채용공고에 맞는 한상기의 이력서 자기소개를 작성해줘.

한상기 정보:
- 총 경력: {year_level} — 모빌씨앤씨 → 큐텐테크놀로지 → 티앤엠테크
- 백엔드 개발자, Java/Spring 메인, Python/FastAPI 경험
- 현재 티앤엠테크에서 GS25·GSFRESH 점포관제 시스템(SEMS) 유지보수·고도화
- 30억 건 데이터 MySQL→MongoDB 이관으로 조회 API 40배 성능 개선
- 600만 건 테이블 파티셔닝으로 쿼리 11배 개선
- Kafka 비동기 파이프라인, LLM/RAG, Redis, Elasticsearch 경험
- 성능 최적화와 실측 데이터 기반 개선을 즐김

작성 방식:
1. 채용공고에서 담당업무 1~2개를 구체적으로 파악한다
2. 한상기의 경험 중 그 업무와 직접 연결되는 것을 꺼낸다
3. "나는 [내 경험]을 했는데, 그게 [회사의 이 업무]에 맞는다"는 흐름으로 쓴다
4. 회사 이름과 실제 직무명을 자연스럽게 언급한다

작성 규칙:
- "안녕하세요"로 시작
- 경력 표기는 "{year_level} 개발자" 형태만 사용, 개월수 언급 금지
- 2~3 문단으로 나눠서 작성, 문단 사이는 빈 줄(\\n\\n)로 구분
- 문단당 2~3문장, 전체 250자 내외
- 실제 수치(40배, 11배 등)는 사용해도 되지만 과장 금지

절대 쓰지 말 것 (AI 문체):
- "~에 집중할 생각입니다", "~을 풀어내겠습니다", "~을 극대화하겠습니다"
- "기술적 지평을 넓혀", "도전 과제", "치명적인 결과"
- "~을 바탕으로", "~을 통해 기여", "함께하겠습니다"
- 과도한 다짐·포부, 회사 미래를 함께 만들겠다는 식의 문장
- 거창한 수식어 ("고도의", "실질적인", "핵심적인")

좋은 예시 (담당업무 매칭):
"안녕하세요. 백엔드 개발자 한상기입니다. 티앤엠테크에서 30억 건 데이터를 MySQL에서 MongoDB로 이관해 조회 성능을 40배 개선했고, Kafka로 대용량 이벤트 파이프라인을 구성했습니다.

{company_name}의 [채용공고에서 읽은 구체적인 업무명]과 직접 연관된 경험이라 지원하게 됐습니다. [내 기술/경험]이 해당 업무에 바로 붙을 수 있을 것 같습니다." """

    return await _invoke_llm([
        SystemMessage(content="이력서 자기소개 작성가입니다. 자연스럽고 담담한 한국어로 씁니다. AI가 쓴 티가 나는 표현은 절대 사용하지 않습니다."),
        HumanMessage(content=prompt),
    ])


def _render(intro: str) -> str:
    template_html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    photo_path = _PHOTO_PATH.as_uri() if _PHOTO_PATH.exists() else ""

    paragraphs = [p.strip().replace("\n", " ") for p in intro.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        parts = [f'<p style="margin:0 0 9px 0;">{p}</p>' for p in paragraphs[:-1]]
        parts.append(f'<p style="margin:0;">{paragraphs[-1]}</p>')
        intro_html = "".join(parts)
    else:
        intro_html = intro.replace("\n", "<br>")

    return Template(template_html).render(intro=intro_html, photo_path=photo_path)


def _split_projects(html: str) -> tuple[str, list[str]]:
    """HTML을 preamble + 프로젝트 청크 목록으로 분리한다."""
    parts = html.split('<div class="project">')
    preamble = parts[0]
    chunks = ['<div class="project">' + p for p in parts[1:]]
    return preamble, chunks


def _get_project_title(chunk: str) -> str:
    m = re.search(r'class="project-title"[^>]*>(.*?)</div>', chunk, re.DOTALL)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""


def list_project_titles() -> list[str]:
    if not template_exists():
        return []
    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    _, chunks = _split_projects(html)
    return [_get_project_title(c) for c in chunks if _get_project_title(c)]


async def edit_resume_project(project_name: str, instruction: str) -> tuple[str, bytes]:
    """특정 프로젝트 섹션을 LLM으로 수정하고 수정된 HTML을 반환한다. (템플릿에 저장하지 않음)"""
    if not template_exists():
        raise FileNotFoundError("이력서 템플릿이 없습니다.")

    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    preamble, chunks = _split_projects(html)
    titles = [_get_project_title(c) for c in chunks]

    matches = difflib.get_close_matches(project_name, titles, n=1, cutoff=0.25)
    if not matches:
        matches = [t for t in titles if project_name.lower() in t.lower()]
    if not matches:
        available = "\n".join(f"- {t}" for t in titles if t)
        raise ValueError(f"'{project_name}' 프로젝트를 찾지 못했습니다.\n\n등록된 프로젝트:\n{available}")

    matched_title = matches[0]
    idx = titles.index(matched_title)
    original_chunk = chunks[idx]

    edited_chunk = await _llm_edit_section(original_chunk, instruction)

    chunks[idx] = edited_chunk
    new_html = preamble + "".join(chunks)
    logger.info("[resume] project edited (not saved) title=%s", matched_title)
    return matched_title, new_html.encode("utf-8")


async def _llm_edit_section(chunk: str, instruction: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    return await _invoke_llm([
        SystemMessage(content=(
            "당신은 HTML 이력서 편집 전문가입니다. "
            "주어진 HTML 섹션을 수정 지시에 따라 편집하고, "
            "수정된 HTML만 반환하세요. 코드블록(```) 없이 순수 HTML만 반환하세요."
        )),
        HumanMessage(content=(
            f"수정 지시: {instruction}\n\n"
            f"규칙:\n"
            f"- HTML 태그와 class명은 절대 변경하지 말 것\n"
            f"- <ul class=\"ach\"> 내 항목 추가/수정/삭제만 허용\n"
            f"- <strong> 태그로 핵심 수치·기술명 강조 유지\n"
            f"- 수정된 HTML 전체를 그대로 반환\n\n"
            f"현재 HTML:\n{chunk}"
        )),
    ])
    result = re.sub(r"^```[a-z]*\n?", "", result)
    result = re.sub(r"\n?```$", "", result)
    return result.strip()


async def _html_to_pdf(html: str) -> bytes:
    from playwright.async_api import async_playwright

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html)
        tmp_path = f.name

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=[
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--disable-setuid-sandbox",
                ])
            page = await browser.new_page()
            await page.goto(f"file:///{tmp_path}", wait_until="networkidle")
            pdf_bytes = await page.pdf(
                format="A4",
                print_background=True,
                margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
            )
            await browser.close()
        return pdf_bytes
    finally:
        Path(tmp_path).unlink(missing_ok=True)
