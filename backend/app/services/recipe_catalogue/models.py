from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.db.models import RecipeStatus


@dataclass(frozen=True)
class RecipeCatalogueFilterInput:
    recipe_id: str | None = None
    recipe_version: str | None = None
    name: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class RecipeCatalogueFilters:
    recipe_id: str | None
    recipe_version: str | None
    name: str | None
    status: RecipeStatus | None


@dataclass(frozen=True)
class RecipeCatalogueCursorBoundary:
    created_at: datetime
    row_id: str


@dataclass(frozen=True)
class RecipeCatalogueRow:
    row_id: str
    recipe_id: str
    recipe_version: str
    name: str
    status: RecipeStatus
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class RecipeCatalogueItem:
    recipe_id: str
    recipe_version: str
    name: str
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class RecipeCatalogueResult:
    items: list[RecipeCatalogueItem]
    limit: int
    has_more: bool
    next_cursor: str | None
    applied_filters: dict[str, str]
