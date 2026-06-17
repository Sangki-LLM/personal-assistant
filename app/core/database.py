from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

_is_mysql = "mysql" in settings.database_url

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=1800,
    **(
        {"connect_args": {"ssl": None, "auth_plugin": ""}}
        if _is_mysql
        else {}
    ),
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    from app.models import user_file  # noqa: F401 — UserFile, FileBundle 모델 등록

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        _sql = __import__("sqlalchemy").text
        migrations = [
            "ALTER TABLE todos ADD COLUMN due_date DATE NULL",
            "ALTER TABLE user_files ADD COLUMN category VARCHAR(100) NULL",
            "ALTER TABLE user_files ADD COLUMN bundle_id INTEGER NULL",
        ]
        for stmt in migrations:
            try:
                await conn.execute(_sql(stmt))
            except Exception:
                pass
