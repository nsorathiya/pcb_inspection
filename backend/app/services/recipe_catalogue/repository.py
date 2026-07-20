from __future__ import annotations

from sqlalchemy import Select, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Recipe

from .models import (
    RecipeCatalogueCursorBoundary,
    RecipeCatalogueFilters,
    RecipeCatalogueRow,
)


class RecipeCatalogueRepository:
    """Column-projected, read-only recipe catalogue persistence access."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _statement(
        filters: RecipeCatalogueFilters,
        boundary: RecipeCatalogueCursorBoundary | None,
        fetch_limit: int,
    ) -> Select:
        statement = select(
            Recipe.id,
            Recipe.recipe_id,
            Recipe.recipe_version,
            Recipe.name,
            Recipe.status,
            Recipe.created_at,
            Recipe.updated_at,
        )
        conditions = []
        for column, value in (
            (Recipe.recipe_id, filters.recipe_id),
            (Recipe.recipe_version, filters.recipe_version),
            (Recipe.name, filters.name),
            (Recipe.status, filters.status),
        ):
            if value is not None:
                conditions.append(column == value)
        if boundary is not None:
            conditions.append(
                or_(
                    Recipe.created_at < boundary.created_at,
                    and_(
                        Recipe.created_at == boundary.created_at,
                        Recipe.id < boundary.row_id,
                    ),
                )
            )
        if conditions:
            statement = statement.where(*conditions)
        return statement.order_by(Recipe.created_at.desc(), Recipe.id.desc()).limit(
            fetch_limit
        )

    async def fetch_page(
        self,
        filters: RecipeCatalogueFilters,
        boundary: RecipeCatalogueCursorBoundary | None,
        fetch_limit: int,
    ) -> list[RecipeCatalogueRow]:
        async with self._session_factory() as session:
            result = await session.execute(
                self._statement(filters, boundary, fetch_limit)
            )
            return [
                RecipeCatalogueRow(
                    row_id=row.id,
                    recipe_id=row.recipe_id,
                    recipe_version=row.recipe_version,
                    name=row.name,
                    status=row.status,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                for row in result.all()
            ]
