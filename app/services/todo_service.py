import logging
from datetime import date, datetime, timedelta

from sqlalchemy import func, select, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.todo import Todo

logger = logging.getLogger(__name__)


async def add_todo(db: AsyncSession, user_id: str, content: str, due_date: str | None = None) -> str:
    parsed_date: date | None = None
    if due_date:
        try:
            parsed_date = date.fromisoformat(due_date)
        except ValueError:
            return f"날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요. (입력값: {due_date})"

    todo = Todo(user_id=user_id, content=content, due_date=parsed_date)
    db.add(todo)
    await db.commit()
    await db.refresh(todo)
    logger.info("[todo] added id=%d user=%s due=%s", todo.id, user_id, parsed_date)
    date_str = f" (기한: {parsed_date})" if parsed_date else ""
    return f"할 일 추가됨: {content}{date_str} [AGENT_ONLY id={todo.id}]"


async def list_todos(db: AsyncSession, user_id: str) -> str:
    """미완료 할 일 전체를 기한 순으로 반환한다."""
    result = await db.execute(
        select(Todo)
        .where(Todo.user_id == user_id, Todo.done == False)
        .order_by(func.isnull(Todo.due_date), Todo.due_date.asc(), Todo.created_at.asc())
    )
    todos = result.scalars().all()
    if not todos:
        return "등록된 할 일이 없습니다."

    today = date.today()

    # 날짜별 그룹핑
    from collections import defaultdict
    groups: dict[str, list[Todo]] = defaultdict(list)
    for t in todos:
        if t.due_date is None:
            groups["날짜 미정"].append(t)
        elif t.due_date < today:
            groups[f"⚠️ 기한 초과 — {t.due_date}"].append(t)
        elif t.due_date == today:
            groups[f"오늘 — {t.due_date}"].append(t)
        else:
            groups[str(t.due_date)].append(t)

    display = ["*할 일 목록*"]
    refs = []
    for label, items in groups.items():
        display.append(f"\n*{label}*")
        for t in items:
            display.append(f"⬜ {t.content}")
            refs.append(f"id={t.id}: {t.content}")

    display.append("\n[AGENT_ONLY - 사용자에게 표시 금지]\n" + "\n".join(refs))
    return "\n".join(display)


async def list_today_todos(db: AsyncSession, user_id: str) -> str:
    """오늘 기한인 할 일 목록 (브리핑용)."""
    today = date.today()
    result = await db.execute(
        select(Todo)
        .where(Todo.user_id == user_id, Todo.done == False, Todo.due_date == today)
        .order_by(Todo.created_at)
    )
    todos = result.scalars().all()
    if not todos:
        return ""
    lines = [f"*오늘 할 일 ({today})*"]
    for t in todos:
        lines.append(f"⬜ {t.content}")
    return "\n".join(lines)


async def list_completed_todos(db: AsyncSession, user_id: str, days: int = 7) -> str:
    since = datetime.now() - timedelta(days=days)
    result = await db.execute(
        select(Todo)
        .where(Todo.user_id == user_id, Todo.done == True, Todo.created_at >= since)
        .order_by(Todo.created_at)
    )
    todos = result.scalars().all()
    if not todos:
        return f"지난 {days}일 동안 완료한 할 일이 없습니다."
    lines = [f"✅ 지난 {days}일 완료한 일 ({len(todos)}개)"]
    for t in todos:
        lines.append(f"• {t.content}")
    return "\n".join(lines)


async def complete_todo(db: AsyncSession, user_id: str, todo_id: int) -> str:
    result = await db.execute(
        select(Todo).where(Todo.id == todo_id, Todo.user_id == user_id)
    )
    todo = result.scalar_one_or_none()
    if not todo:
        return f"할 일 [{todo_id}]을 찾을 수 없습니다."
    todo.done = True
    await db.commit()
    logger.info("[todo] completed id=%d", todo_id)
    return f"완료: [{todo_id}] {todo.content}"
