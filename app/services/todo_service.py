import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.todo import Todo

logger = logging.getLogger(__name__)


async def add_todo(db: AsyncSession, user_id: str, content: str) -> str:
    todo = Todo(user_id=user_id, content=content)
    db.add(todo)
    await db.commit()
    await db.refresh(todo)
    logger.info("[todo] added id=%d user=%s", todo.id, user_id)
    return f"할 일 추가됨: {content} [AGENT_ONLY id={todo.id}]"


async def list_todos(db: AsyncSession, user_id: str) -> str:
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(Todo)
        .where(Todo.user_id == user_id, Todo.created_at >= today_start)
        .order_by(Todo.created_at)
    )
    todos = result.scalars().all()
    if not todos:
        return "오늘 등록된 할 일이 없습니다."
    display = ["*오늘 할 일*"]
    refs = []
    for t in todos:
        check = "✅" if t.done else "⬜"
        display.append(f"{check} {t.content}")
        if not t.done:
            refs.append(f"id={t.id}: {t.content}")
    if refs:
        display.append("\n[AGENT_ONLY - 사용자에게 표시 금지]\n" + "\n".join(refs))
    return "\n".join(display)


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
