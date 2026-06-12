import logging

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
    return f"할 일 추가: [{todo.id}] {content}"


async def list_todos(db: AsyncSession, user_id: str) -> str:
    result = await db.execute(
        select(Todo)
        .where(Todo.user_id == user_id, Todo.done == False)
        .order_by(Todo.created_at)
    )
    todos = result.scalars().all()
    if not todos:
        return "할 일 목록이 없습니다."
    lines = ["📋 할 일 목록"]
    for t in todos:
        lines.append(f"• [{t.id}] {t.content}")
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
