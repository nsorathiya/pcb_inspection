from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict

from app.api.errors import ApiError, ApiErrorResponse
from app.services.recipe_catalogue import (
    RecipeCatalogueConsistencyError,
    RecipeCatalogueCursorError,
    RecipeCatalogueCursorFilterMismatchError,
    RecipeCatalogueCursorVersionError,
    RecipeCatalogueFilterError,
    RecipeCatalogueFilterInput,
    RecipeCatalogueRetrievalError,
    RecipeCatalogueService,
)

router = APIRouter(tags=["recipes"])


class RecipeCatalogueItemResponse(BaseModel):
    recipe_id: str
    recipe_version: str
    name: str
    status: str
    created_at: datetime
    updated_at: datetime


class RecipeCataloguePageResponse(BaseModel):
    limit: int
    has_more: bool
    next_cursor: str | None


class RecipeCatalogueResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Read-only persisted recipe identities and display metadata for "
                "inspection-intake selection. Listing does not prove model, "
                "calibration, or production approval."
            )
        }
    )

    items: list[RecipeCatalogueItemResponse]
    page: RecipeCataloguePageResponse
    applied_filters: dict[str, str]
    request_id: str


@router.get(
    "/recipes",
    response_model=RecipeCatalogueResponse,
    responses={
        400: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    summary="List the read-only recipe catalogue",
    description=(
        "Returns a deterministic cursor page of persisted recipe identities, "
        "names, statuses, and timestamps for future inspection-intake selection. "
        "Exact filters use AND semantics. The route selects database columns only; "
        "it does not expose configuration JSON, mutate recipes, write audit events, "
        "read files, or execute validation, preprocessing, or inference. Every "
        "persisted version remains independently selectable. A listed recipe or "
        "ACTIVE status does not prove model compatibility, calibration validity, "
        "production readiness, approval, or preference over another version."
    ),
)
async def list_recipe_catalogue(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    cursor: str | None = None,
    recipe_id: str | None = None,
    recipe_version: str | None = None,
    name: str | None = None,
    recipe_status: Annotated[str | None, Query(alias="status")] = None,
) -> RecipeCatalogueResponse:
    service: RecipeCatalogueService = request.app.state.recipe_catalogue
    try:
        result = await service.list_catalogue(
            limit=limit,
            cursor=cursor,
            filters=RecipeCatalogueFilterInput(
                recipe_id=recipe_id,
                recipe_version=recipe_version,
                name=name,
                status=recipe_status,
            ),
        )
    except RecipeCatalogueCursorFilterMismatchError as exc:
        raise ApiError(
            400,
            "RECIPE_CURSOR_FILTER_MISMATCH",
            "The recipe cursor does not match the current filters.",
        ) from exc
    except RecipeCatalogueCursorVersionError as exc:
        raise ApiError(
            400,
            "UNSUPPORTED_RECIPE_CURSOR_VERSION",
            "The recipe cursor version is unsupported.",
        ) from exc
    except RecipeCatalogueCursorError as exc:
        raise ApiError(
            400,
            "INVALID_RECIPE_CURSOR",
            "The recipe cursor is invalid.",
        ) from exc
    except RecipeCatalogueFilterError as exc:
        raise ApiError(
            400,
            "INVALID_RECIPE_FILTER",
            "One or more recipe catalogue filters are invalid.",
        ) from exc
    except RecipeCatalogueConsistencyError as exc:
        request.app.state.logger.exception("Recipe catalogue data is inconsistent")
        raise ApiError(
            500,
            "RECIPE_CATALOGUE_INCONSISTENT",
            "The recipe catalogue could not be represented safely.",
        ) from exc
    except RecipeCatalogueRetrievalError as exc:
        request.app.state.logger.exception("Recipe catalogue read failed")
        raise ApiError(
            500,
            "RECIPE_CATALOGUE_READ_FAILED",
            "The recipe catalogue could not be retrieved.",
        ) from exc
    except Exception as exc:
        request.app.state.logger.exception("Unexpected recipe catalogue read failure")
        raise ApiError(
            500,
            "RECIPE_CATALOGUE_READ_FAILED",
            "The recipe catalogue could not be retrieved.",
        ) from exc

    return RecipeCatalogueResponse(
        items=[RecipeCatalogueItemResponse(**asdict(item)) for item in result.items],
        page=RecipeCataloguePageResponse(
            limit=result.limit,
            has_more=result.has_more,
            next_cursor=result.next_cursor,
        ),
        applied_filters=result.applied_filters,
        request_id=request.state.request_id,
    )
