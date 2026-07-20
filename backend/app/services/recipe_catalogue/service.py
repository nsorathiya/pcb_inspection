from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import RecipeStatus

from .cursor import (
    canonical_filter_payload,
    decode_cursor,
    encode_cursor,
    filter_digest,
)
from .exceptions import (
    RecipeCatalogueConsistencyError,
    RecipeCatalogueCursorFilterMismatchError,
    RecipeCatalogueFilterError,
    RecipeCatalogueRetrievalError,
)
from .models import (
    RecipeCatalogueCursorBoundary,
    RecipeCatalogueFilterInput,
    RecipeCatalogueFilters,
    RecipeCatalogueItem,
    RecipeCatalogueResult,
    RecipeCatalogueRow,
)
from .repository import RecipeCatalogueRepository


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class RecipeCatalogueService:
    """Normalize, retrieve, and safely project persisted recipe rows."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._repository = RecipeCatalogueRepository(session_factory)

    @property
    def repository(self) -> RecipeCatalogueRepository:
        return self._repository

    @staticmethod
    def _filter_text(value: str | None, field: str, maximum: int) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if (
            not normalized
            or len(normalized) > maximum
            or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
        ):
            raise RecipeCatalogueFilterError(f"{field} is invalid")
        return normalized

    def normalize_filters(
        self, source: RecipeCatalogueFilterInput
    ) -> RecipeCatalogueFilters:
        status = None
        if source.status is not None:
            try:
                status = RecipeStatus(source.status.strip().upper())
            except (ValueError, AttributeError) as exc:
                raise RecipeCatalogueFilterError("status is unsupported") from exc
        return RecipeCatalogueFilters(
            recipe_id=self._filter_text(source.recipe_id, "recipe_id", 128),
            recipe_version=self._filter_text(
                source.recipe_version, "recipe_version", 128
            ),
            name=self._filter_text(source.name, "name", 256),
            status=status,
        )

    @staticmethod
    def _validate_row(row: RecipeCatalogueRow) -> None:
        try:
            canonical_id = str(UUID(row.row_id))
        except ValueError as exc:
            raise RecipeCatalogueConsistencyError(
                "Persisted recipe identity is invalid"
            ) from exc
        if canonical_id != row.row_id:
            raise RecipeCatalogueConsistencyError(
                "Persisted recipe identity is invalid"
            )
        for value, maximum in (
            (row.recipe_id, 128),
            (row.recipe_version, 128),
            (row.name, 256),
        ):
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or len(value) > maximum
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                raise RecipeCatalogueConsistencyError(
                    "Persisted recipe display metadata is invalid"
                )

    async def list_catalogue(
        self,
        *,
        limit: int,
        cursor: str | None,
        filters: RecipeCatalogueFilterInput,
    ) -> RecipeCatalogueResult:
        normalized = self.normalize_filters(filters)
        digest = filter_digest(normalized)
        boundary = None
        if cursor is not None:
            boundary, cursor_digest = decode_cursor(cursor)
            if cursor_digest != digest:
                raise RecipeCatalogueCursorFilterMismatchError(
                    "The recipe cursor does not match the current filters"
                )
        try:
            rows = await self._repository.fetch_page(normalized, boundary, limit + 1)
        except SQLAlchemyError as exc:
            raise RecipeCatalogueRetrievalError(
                "The recipe catalogue could not be read"
            ) from exc

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        for row in page_rows:
            self._validate_row(row)
        items = [
            RecipeCatalogueItem(
                recipe_id=row.recipe_id,
                recipe_version=row.recipe_version,
                name=row.name,
                status=row.status.value,
                created_at=_as_utc(row.created_at),
                updated_at=_as_utc(row.updated_at),
            )
            for row in page_rows
        ]
        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_cursor(
                RecipeCatalogueCursorBoundary(
                    created_at=_as_utc(last.created_at),
                    row_id=last.row_id,
                ),
                digest,
            )
        return RecipeCatalogueResult(
            items=items,
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
            applied_filters={
                key: value
                for key, value in canonical_filter_payload(normalized).items()
                if value is not None
            },
        )
