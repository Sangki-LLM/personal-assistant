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


async def generate_resume(company_name: str, job_posting: str, user_intro: str | None = None) -> bytes:
    if not template_exists():
        raise FileNotFoundError("이력서 템플릿이 없습니다.")

    if user_intro:
        intro = user_intro
        logger.info("[resume] user intro used verbatim len=%d", len(intro))
    else:
        if not job_posting:
            raise ValueError("자기소개를 직접 주시거나, 채용공고 내용을 알려주세요.")
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


def _extract_resume_text(html: str) -> str:
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def _generate_intro(company_name: str, job_posting: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    _, year_level = _calc_career()

    resume_html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    resume_text = _extract_resume_text(resume_html)

    prompt = f"""다음은 {company_name} 채용공고입니다.

{job_posting}

다음은 한상기의 이력서 전문입니다. 이 내용만을 근거로 자기소개를 작성하세요. 이력서에 없는 경험은 절대 지어내지 마세요.

[한상기 이력서]
{resume_text}

반드시 2문단으로만 작성 (3문단 금지):
[1문단] "안녕하세요. {year_level} 백엔드 개발자 한상기입니다." + 이력서에 실제로 적힌 수치·성과 포함 (2~3문장)
[2문단] 채용공고 담당업무·우대사항 중 1~2개를 직접 언급하며, 이력서에 실제 기재된 경험과 매칭. 마지막 문장은 반드시 "{company_name}에 합류하게 된다면 [담당업무 관련 구체적 표현]에 기여하겠습니다."로 마무리 (2~3문장)

작성 규칙:
- "안녕하세요"로 시작
- 경력 표기: "{year_level} 개발자" 형태만, 개월수 언급 금지
- 문단 사이 빈 줄(\\n\\n)로 구분
- 문단당 2~3문장, 전체 220자 내외
- 기술명과 수치는 이력서에 실제로 있는 것만 사용

절대 쓰지 말 것:
- 이력서에 없는 경험·기술·수치 지어내기
- "역량", "경험을 살려", "실력", "능력", "역량을 발휘"
- "관심이 많습니다", "관심을 갖고", "~에 익숙하며"
- "~을 바로 적용하겠습니다", "~을 극대화하겠습니다", "~을 풀어내겠습니다"
- "이번 합류를 통해", "함께하겠습니다", "함께 성장", "실질적인 결과를 만들겠습니다"
- "~에 자신 있습니다", "~에 강점이 있습니다", "~이 연결된다고 생각해", "~라 판단했습니다"
- "즐깁니다", "집중해왔습니다" 식의 자기 서술, 3번째 문단"""

    return await _invoke_llm([
        SystemMessage(content="이력서 자기소개 작성가입니다. 반드시 제공된 이력서 내용만 근거로 씁니다. 이력서에 없는 내용은 절대 추가하지 않습니다. 자연스럽고 담담한 한국어로 씁니다."),
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
