from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import Request
from sqlalchemy import event, insert, select, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base, SCHEMA_VERSION, SchemaVersion


class Database:
    """Own the asynchronous SQLite engine, sessions, and schema lifecycle."""

    def __init__(
        self,
        database_file: Path,
        *,
        busy_timeout_ms: int,
        echo: bool = False,
    ) -> None:
        self.database_file = database_file.resolve()
        self.busy_timeout_ms = busy_timeout_ms
        url = URL.create(
            drivername="sqlite+aiosqlite",
            database=str(self.database_file),
        )
        self.engine: AsyncEngine = create_async_engine(
            url,
            echo=echo,
            connect_args={"timeout": busy_timeout_ms / 1000},
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._configure_sqlite_connections()

    def _configure_sqlite_connections(self) -> None:
        busy_timeout_ms = self.busy_timeout_ms

        @event.listens_for(self.engine.sync_engine, "connect")
        def set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.fetchone()
                cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            finally:
                cursor.close()

    async def initialize(self) -> None:
        """Create the initial schema idempotently and verify its version/queryability."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            current = await connection.scalar(
                select(SchemaVersion.version).where(SchemaVersion.id == 1)
            )
            if current is None:
                await connection.execute(
                    insert(SchemaVersion).values(
                        id=1,
                        version=SCHEMA_VERSION,
                        applied_at=datetime.now(timezone.utc),
                    )
                )
            elif current != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported database schema version {current}; "
                    f"expected {SCHEMA_VERSION}"
                )
        await self.check_health()

    async def check_health(self) -> None:
        async with self.engine.connect() as connection:
            value = await connection.scalar(text("SELECT 1"))
        if value != 1:
            raise RuntimeError("SQLite health query returned an unexpected result")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()


async def get_database_session(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI session dependency foundation for future API tasks."""
    database: Database = request.app.state.database
    async with database.session() as session:
        yield session
