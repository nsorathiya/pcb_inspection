from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.api.errors import ApiError, ApiErrorResponse
from app.services.demo_workspace import (
    DemoWorkspaceConsistencyError,
    DemoWorkspaceDisabledError,
    DemoWorkspaceNotConfiguredError,
    DemoWorkspaceService,
    DemoWorkspaceState,
)

router = APIRouter(tags=["development"])


class DemoInspectionResponse(BaseModel):
    key: str
    inspection_id: str
    board_id: str
    status: str | None
    validation_outcome: str | None
    processing_status: str | None
    preprocessing_outcome: str | None
    mock_decision: str | None
    complete: bool


class DemoWorkspaceResponse(BaseModel):
    enabled: bool
    available: bool
    loaded: bool
    recipes_ready: bool
    inspections: list[DemoInspectionResponse]
    synthetic: bool
    production_approved: bool
    idempotent_existing: bool | None
    request_id: str


def _response(state: DemoWorkspaceState, request_id: str) -> DemoWorkspaceResponse:
    return DemoWorkspaceResponse(
        enabled=state.enabled,
        available=state.available,
        loaded=state.loaded,
        recipes_ready=state.recipes_ready,
        inspections=[
            DemoInspectionResponse(**inspection.__dict__)
            for inspection in state.inspections
        ],
        synthetic=state.synthetic,
        production_approved=state.production_approved,
        idempotent_existing=state.idempotent_existing,
        request_id=request_id,
    )


def _map_error(error: Exception) -> ApiError:
    if isinstance(error, DemoWorkspaceDisabledError):
        return ApiError(
            404,
            "DEMO_WORKSPACE_DISABLED",
            "The development demo workspace is disabled.",
        )
    if isinstance(error, DemoWorkspaceNotConfiguredError):
        return ApiError(
            503,
            "DEMO_WORKSPACE_NOT_CONFIGURED",
            "The development demo workspace is not configured safely.",
        )
    if isinstance(error, DemoWorkspaceConsistencyError):
        return ApiError(
            409,
            "DEMO_WORKSPACE_CONFLICT",
            "Existing data conflicts with the reserved development demo workspace.",
        )
    return ApiError(
        500,
        "DEMO_WORKSPACE_LOAD_FAILED",
        "The development demo workspace could not be loaded.",
    )


@router.get(
    "/development/demo-workspace",
    response_model=DemoWorkspaceResponse,
    responses={500: {"model": ApiErrorResponse}},
    summary="Read development demo-workspace availability and state",
    description=(
        "Returns path-free state for the explicitly configured synthetic demo "
        "workspace. It does not generate fixtures, create records, execute "
        "validation or processing, or delete inspections. Results are synthetic "
        "development evidence and are never production-approved."
    ),
)
async def get_demo_workspace(request: Request) -> DemoWorkspaceResponse:
    service: DemoWorkspaceService = request.app.state.demo_workspace
    try:
        state = await service.get_state()
    except Exception as exc:
        request.app.state.logger.exception("Demo workspace state read failed")
        raise _map_error(exc) from exc
    return _response(state, request.state.request_id)


@router.post(
    "/development/demo-workspace/load",
    response_model=DemoWorkspaceResponse,
    responses={
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        503: {"model": ApiErrorResponse},
    },
    summary="Explicitly load the persistent development demo workspace",
    description=(
        "Development-only, operator-triggered loading of repository-owned "
        "synthetic fixtures. The loader uses the existing paired intake, technical "
        "validation, preprocessing, mock inference, and guarded lifecycle services. "
        "Reserved deterministic identities make exact and concurrent retries "
        "idempotent. Existing inspections are never deleted. Mock PASS, FAIL, and "
        "UNCERTAIN values are not image-based predictions or production decisions."
    ),
)
async def load_demo_workspace(request: Request) -> DemoWorkspaceResponse:
    service: DemoWorkspaceService = request.app.state.demo_workspace
    try:
        state = await service.load(request_id=request.state.request_id)
    except Exception as exc:
        mapped = _map_error(exc)
        if mapped.status_code == 500:
            request.app.state.logger.exception("Demo workspace load failed")
        raise mapped from exc
    return _response(state, request.state.request_id)
