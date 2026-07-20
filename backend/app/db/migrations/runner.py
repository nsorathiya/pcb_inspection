from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Sequence

from sqlalchemy import inspect, select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.db.migrations import (
    migration_001_initial,
    migration_002_validation_results,
    migration_003_processing_results,
)
from app.db.models import SCHEMA_VERSION, SchemaVersion


@dataclass(frozen=True)
class Migration:
    version: int
    identifier: str
    required_table_names: frozenset[str]
    upgrade: Callable[[AsyncConnection], Awaitable[None]]


DEFAULT_MIGRATIONS = (
    Migration(
        migration_001_initial.VERSION,
        migration_001_initial.IDENTIFIER,
        migration_001_initial.REQUIRED_TABLE_NAMES,
        migration_001_initial.upgrade,
    ),
    Migration(
        migration_002_validation_results.VERSION,
        migration_002_validation_results.IDENTIFIER,
        migration_002_validation_results.REQUIRED_TABLE_NAMES,
        migration_002_validation_results.upgrade,
    ),
    Migration(
        migration_003_processing_results.VERSION,
        migration_003_processing_results.IDENTIFIER,
        migration_003_processing_results.REQUIRED_TABLE_NAMES,
        migration_003_processing_results.upgrade,
    ),
)


class MigrationRunner:
    def __init__(
        self,
        engine: AsyncEngine,
        migrations: Sequence[Migration] = DEFAULT_MIGRATIONS,
    ) -> None:
        self._engine = engine
        self._migrations = tuple(migrations)
        versions = tuple(item.version for item in self._migrations)
        if versions != tuple(range(1, len(versions) + 1)):
            raise ValueError("database migrations must be contiguous and ordered from version 1")
        identifiers = tuple(item.identifier for item in self._migrations)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("database migration identifiers must be unique")

    async def run(self, *, target_version: int = SCHEMA_VERSION) -> None:
        if target_version < 1 or target_version > len(self._migrations):
            raise RuntimeError(f"Unsupported target schema version {target_version}")
        current, tables = await self._current_state()
        if current > target_version:
            raise RuntimeError(
                f"Unsupported database schema version {current}; expected at most {target_version}"
            )
        if current > 0:
            self._require_tables(current, tables)

        for migration in self._migrations:
            if not current < migration.version <= target_version:
                continue
            async with self._engine.begin() as connection:
                await migration.upgrade(connection)
                now = datetime.now(timezone.utc)
                if migration.version == 1:
                    await connection.execute(
                        SchemaVersion.__table__.insert().values(
                            id=1,
                            version=1,
                            applied_at=now,
                        )
                    )
                else:
                    result = await connection.execute(
                        update(SchemaVersion)
                        .where(
                            SchemaVersion.id == 1,
                            SchemaVersion.version == current,
                        )
                        .values(version=migration.version, applied_at=now)
                    )
                    if result.rowcount != 1:
                        raise RuntimeError("schema version changed during migration")
            current = migration.version
            _, tables = await self._current_state()
            self._require_tables(current, tables)

        final, tables = await self._current_state()
        if final != target_version:
            raise RuntimeError(
                f"Database schema initialization stopped at version {final}; expected {target_version}"
            )
        self._require_tables(final, tables)

    async def _current_state(self) -> tuple[int, frozenset[str]]:
        async with self._engine.connect() as connection:
            tables = frozenset(
                await connection.run_sync(
                    lambda sync_connection: inspect(sync_connection).get_table_names()
                )
            )
            if "schema_version" not in tables:
                if tables:
                    raise RuntimeError(
                        "Database contains tables but has no authoritative schema version"
                    )
                return 0, tables
            rows = list(
                (
                    await connection.execute(
                        select(SchemaVersion.id, SchemaVersion.version)
                    )
                ).all()
            )
        if len(rows) != 1 or rows[0].id != 1:
            raise RuntimeError("Database schema version record is invalid")
        version = rows[0].version
        if not isinstance(version, int) or version < 1 or version > len(self._migrations):
            raise RuntimeError(f"Unsupported database schema version {version}")
        return version, tables

    def _require_tables(self, version: int, tables: frozenset[str]) -> None:
        required = frozenset().union(
            *(item.required_table_names for item in self._migrations if item.version <= version)
        )
        missing = required.difference(tables)
        if missing:
            raise RuntimeError(
                f"Database schema version {version} is missing required tables"
            )
