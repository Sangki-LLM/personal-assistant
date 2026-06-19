import logging
from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.todo import Todo

logger = logging.getLogger(__name__)


async def add_todo(
    db: AsyncSession,
    user_id: str,
    content: str,
    due_date: str | None = None,
    category: str | None = None,
) -> str:
    parsed_date: date | None = None
    if due_date:
        try:
            parsed_date = date.fromisoformat(due_date)
        except ValueError:
            return f"날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요. (입력값: {due_date})"

    todo = Todo(user_id=user_id, content=content, due_date=parsed_date, category=category)
    db.add(todo)
    await db.commit()
    await db.refresh(todo)
    logger.info("[todo] added id=%d user=%s due=%s category=%s", todo.id, user_id, parsed_date, category)

    suffix = ""
    if category:
        suffix += f" (카테고리: {category})"
    if parsed_date:
        suffix += f" (기한: {parsed_date})"
    return f"할 일 추가됨: {content}{suffix} [AGENT_ONLY id={todo.id}]"


async def list_todos(db: AsyncSession, user_id: str) -> str:
    """미완료 할 일 전체를 카테고리별 + 날짜별로 반환한다."""
    result = await db.execute(
        select(Todo)
        .where(Todo.user_id == user_id, Todo.done == False)
        .order_by(func.isnull(Todo.category), Todo.category.asc(), func.isnull(Todo.due_date), Todo.due_date.asc(), Todo.created_at.asc())
    )
    todos = result.scalars().all()
    if not todos:
        return "등록된 할 일이 없습니다."

    today = date.today()
    display = ["*할 일 목록*"]
    refs = []

    # 카테고리 있는 것 — 카테고리별 그룹
    from collections import defaultdict
    category_groups: dict[str, list[Todo]] = defaultdict(list)
    date_todos: list[Todo] = []

    for t in todos:
        if t.category:
            category_groups[t.category].append(t)
        else:
            date_todos.append(t)

    if category_groups:
        display.append("")
        for cat, items in sorted(category_groups.items()):
            display.append(f"📁 *{cat}*")
            for t in items:
                due_str = f" `{t.due_date}`" if t.due_date else ""
                display.append(f"⬜ {t.content}{due_str}")
                refs.append(f"id={t.id}: {t.content}")

    # 카테고리 없는 것 — 날짜별 그룹
    if date_todos:
        display.append("")
        date_groups: dict[str, list[Todo]] = defaultdict(list)
        for t in date_todos:
            if t.due_date is None:
                date_groups["날짜 미정"].append(t)
            elif t.due_date < today:
                date_groups[f"⚠️ 기한 초과 — {t.due_date}"].append(t)
            elif t.due_date == today:
                date_groups[f"오늘 — {t.due_date}"].append(t)
            else:
                date_groups[str(t.due_date)].append(t)

        for label, items in date_groups.items():
            display.append(f"*{label}*")
            for t in items:
                display.append(f"⬜ {t.content}")
                refs.append(f"id={t.id}: {t.content}")

    display.append("\n[AGENT_ONLY - 사용자에게 표시 금지]\n" + "\n".join(refs))
    return "\n".join(display)


async def list_todos_by_category(db: AsyncSession, user_id: str, category: str) -> str:
    """특정 카테고리의 미완료 할 일 목록을 반환한다."""
    result = await db.execute(
        select(Todo)
        .where(Todo.user_id == user_id, Todo.done == False, Todo.category == category)
        .order_by(Todo.created_at.asc())
    )
    todos = result.scalars().all()
    if not todos:
        return f"*{category}* 카테고리에 등록된 할 일이 없습니다."

    lines = [f"📁 *{category}* 할 일 목록"]
    refs = []
    for t in todos:
        due_str = f" `{t.due_date}`" if t.due_date else ""
        lines.append(f"⬜ {t.content}{due_str}")
        refs.append(f"id={t.id}: {t.content}")

    lines.append("\n[AGENT_ONLY - 사용자에게 표시 금지]\n" + "\n".join(refs))
    return "\n".join(lines)


async def list_todo_categories(db: AsyncSession, user_id: str) -> str:
    """사용 중인 카테고리 목록을 반환한다."""
    from sqlalchemy import distinct
    result = await db.execute(
        select(distinct(Todo.category))
        .where(Todo.user_id == user_id, Todo.done == False, Todo.category.isnot(None))
    )
    categories = [row[0] for row in result.all() if row[0]]
    if not categories:
        return "등록된 업무 카테고리가 없습니다."
    lines = ["*업무 카테고리 목록*"] + [f"• {c}" for c in sorted(categories)]
    return "\n".join(lines)


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
