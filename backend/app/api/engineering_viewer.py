from __future__ import annotations

from dataclasses import asdict
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from app.api.errors import ApiError, ApiErrorResponse
from app.services.engineering_viewer import (
    EngineeringArtifactIntegrityError,
    EngineeringArtifactPairError,
    EngineeringEvidenceReadError,
    EngineeringFormatUnsupportedError,
    EngineeringInspectionNotFoundError,
    EngineeringRasterTooLargeError,
    EngineeringRoiBoundsError,
    EngineeringSampleBoundsError,
    EngineeringViewerDisabledError,
    EngineeringViewerService,
)

router = APIRouter(tags=["engineering-viewer"])


class EngineeringRasterMetadataResponse(BaseModel):
    artifact_type: Literal["RGB_RAW", "HEIGHT_RAW"]
    detected_format: str
    width: int
    height: int
    channels: int
    bit_depth: int
    color_mode: str
    storage_data_type: str | None
    sha256: str
    byte_size: int


class HeightHistogramResponse(BaseModel):
    bin_count: Literal[64]
    native_min: int | float
    native_max: int | float
    counts: list[int]


class HeightStatisticsResponse(BaseModel):
    native_min: int | float
    native_max: int | float
    valid_count: int
    invalid_count: int
    histogram: HeightHistogramResponse


class ValidationEvidenceResponse(BaseModel):
    available: bool
    validation_id: str | None
    outcome: str | None
    policy_id: str | None
    policy_version: str | None
    technically_ready: bool | None
    finding_codes: list[str]


class ProcessingEvidenceResponse(BaseModel):
    available: bool
    processing_run_id: str | None
    processing_status: str | None
    preprocessing_outcome: str | None
    mock_decision: str | None
    production_approved: bool | None
    synthetic_input_verified: bool | None
    finding_codes: list[str]


class EngineeringViewResponse(BaseModel):
    inspection_id: str
    inspection_status: str
    rgb: EngineeringRasterMetadataResponse
    height: EngineeringRasterMetadataResponse
    height_statistics: HeightStatisticsResponse
    calibration_status: str
    registration_status: str
    physical_height_unit: None
    validation: ValidationEvidenceResponse
    processing: ProcessingEvidenceResponse
    warnings: list[str]
    synthetic_input_verified: bool
    production_approved: Literal[False]
    request_id: str


class RgbSampleResponse(BaseModel):
    x: int
    y: int
    storage_data_type: str | None
    values: list[int]


class HeightSampleResponse(BaseModel):
    x: int
    y: int
    storage_data_type: str | None
    value: int | float | None
    valid: bool
    physical_unit: None


class EngineeringSampleResponse(BaseModel):
    inspection_id: str
    rgb: RgbSampleResponse
    height: HeightSampleResponse
    warnings: list[str]
    request_id: str


class EngineeringHeightRoiStatisticsResponse(BaseModel):
    inspection_id: str
    x: int
    y: int
    width: int
    height: int
    storage_data_type: str | None
    native_min: int | float
    native_max: int | float
    native_mean: float
    valid_count: int
    invalid_count: int
    physical_unit: None
    warnings: list[str]
    request_id: str


def _canonical_inspection_id(value: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ApiError(
            400,
            "INVALID_INSPECTION_ID",
            "Inspection ID must be a canonical UUID.",
        ) from exc
    if str(parsed) != value:
        raise ApiError(
            400,
            "INVALID_INSPECTION_ID",
            "Inspection ID must be a canonical UUID.",
        )
    return value


def _map_error(error: Exception) -> ApiError:
    if isinstance(error, EngineeringViewerDisabledError):
        return ApiError(
            404,
            "ENGINEERING_VIEWER_DISABLED",
            "The engineering viewer is disabled.",
        )
    if isinstance(error, EngineeringInspectionNotFoundError):
        return ApiError(404, "INSPECTION_NOT_FOUND", "Inspection was not found.")
    if isinstance(error, EngineeringArtifactPairError):
        return ApiError(
            409,
            "ENGINEERING_ARTIFACT_PAIR_UNAVAILABLE",
            "The inspection does not have one unambiguous paired RGB and height source.",
        )
    if isinstance(error, EngineeringArtifactIntegrityError):
        return ApiError(
            409,
            "ENGINEERING_ARTIFACT_INTEGRITY_FAILED",
            "Registered artifact ownership or byte integrity could not be verified.",
        )
    if isinstance(error, EngineeringFormatUnsupportedError):
        return ApiError(
            422,
            "ENGINEERING_FORMAT_UNSUPPORTED",
            "The artifact pair is outside the supported synthetic engineering-view formats.",
        )
    if isinstance(error, EngineeringRasterTooLargeError):
        return ApiError(
            413,
            "ENGINEERING_RASTER_TOO_LARGE",
            "The raster exceeds the bounded engineering-view pixel limit.",
        )
    if isinstance(error, EngineeringSampleBoundsError):
        return ApiError(
            422,
            "ENGINEERING_SAMPLE_OUT_OF_BOUNDS",
            "RGB or height sample coordinates are outside their respective raster bounds.",
        )
    if isinstance(error, EngineeringRoiBoundsError):
        return ApiError(
            422,
            "ENGINEERING_HEIGHT_ROI_OUT_OF_BOUNDS",
            "The height ROI is outside raster bounds or exceeds the bounded pixel limit.",
        )
    if isinstance(error, EngineeringEvidenceReadError):
        return ApiError(
            500,
            "ENGINEERING_EVIDENCE_READ_FAILED",
            "Persisted engineering evidence could not be represented safely.",
        )
    return ApiError(
        500,
        "ENGINEERING_VIEW_READ_FAILED",
        "The engineering view could not be generated safely.",
    )


async def _call(request: Request, operation):
    try:
        return await operation
    except Exception as exc:
        mapped = _map_error(exc)
        if mapped.status_code == 500:
            request.app.state.logger.exception("Engineering viewer read failed")
        raise mapped from exc


@router.get(
    "/inspections/{inspection_id}/engineering-view",
    response_model=EngineeringViewResponse,
    responses={
        400: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        413: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    summary="Read integrity-verified synthetic engineering metadata",
    description=(
        "Reads only the requested inspection's registered raw pair, verifies path "
        "confinement, ownership, SHA-256, and byte size, then returns native metadata "
        "and bounded height statistics. It creates no files or audit records and does "
        "not execute validation, preprocessing, or inference."
    ),
)
async def get_engineering_view(
    inspection_id: str,
    request: Request,
) -> EngineeringViewResponse:
    canonical_id = _canonical_inspection_id(inspection_id)
    service: EngineeringViewerService = request.app.state.engineering_viewer
    view = await _call(request, service.get_view(canonical_id))
    return EngineeringViewResponse(
        **asdict(view),
        request_id=request.state.request_id,
    )


def _preview_response(preview) -> Response:
    return Response(
        content=preview.content,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-PCB-AOI-Preview-Derived": "true",
            "X-PCB-AOI-Preview-Persisted": "false",
            "X-PCB-AOI-Preview-Kind": preview.preview_kind,
            "X-PCB-AOI-Preview-Transform": preview.transform,
            "X-PCB-AOI-Physical-Units": "unavailable",
        },
    )


@router.get(
    "/inspections/{inspection_id}/engineering-view/rgb-preview",
    responses={200: {"content": {"image/png": {}}}, 404: {"model": ApiErrorResponse}},
    summary="Generate an in-memory RGB PNG preview",
)
async def get_rgb_preview(inspection_id: str, request: Request) -> Response:
    canonical_id = _canonical_inspection_id(inspection_id)
    service: EngineeringViewerService = request.app.state.engineering_viewer
    preview = await _call(request, service.rgb_preview(canonical_id))
    return _preview_response(preview)


@router.get(
    "/inspections/{inspection_id}/engineering-view/height-preview",
    responses={200: {"content": {"image/png": {}}}, 404: {"model": ApiErrorResponse}},
    summary="Generate an in-memory derived height PNG preview",
)
async def get_height_preview(inspection_id: str, request: Request) -> Response:
    canonical_id = _canonical_inspection_id(inspection_id)
    service: EngineeringViewerService = request.app.state.engineering_viewer
    preview = await _call(request, service.height_preview(canonical_id))
    return _preview_response(preview)


@router.get(
    "/inspections/{inspection_id}/engineering-view/sample",
    response_model=EngineeringSampleResponse,
    responses={
        400: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    summary="Sample separate native RGB and height coordinates",
)
async def get_engineering_sample(
    inspection_id: str,
    request: Request,
    rgb_x: int,
    rgb_y: int,
    height_x: int,
    height_y: int,
) -> EngineeringSampleResponse:
    canonical_id = _canonical_inspection_id(inspection_id)
    service: EngineeringViewerService = request.app.state.engineering_viewer
    sample = await _call(
        request,
        service.sample(
            canonical_id,
            rgb_x=rgb_x,
            rgb_y=rgb_y,
            height_x=height_x,
            height_y=height_y,
        ),
    )
    return EngineeringSampleResponse(
        inspection_id=sample.inspection_id,
        rgb=RgbSampleResponse(
            x=sample.rgb.x,
            y=sample.rgb.y,
            storage_data_type=sample.rgb.storage_data_type,
            values=list(sample.rgb.values or ()),
        ),
        height=HeightSampleResponse(
            x=sample.height.x,
            y=sample.height.y,
            storage_data_type=sample.height.storage_data_type,
            value=sample.height.value,
            valid=bool(sample.height.valid),
            physical_unit=None,
        ),
        warnings=list(sample.warnings),
        request_id=request.state.request_id,
    )


@router.get(
    "/inspections/{inspection_id}/engineering-view/height-roi",
    response_model=EngineeringHeightRoiStatisticsResponse,
    responses={
        400: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        413: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    summary="Read bounded native height ROI statistics",
    description=(
        "Verifies the registered artifact pair, then calculates native min, max, "
        "mean, and valid/invalid counts for a bounded rectangular height ROI. "
        "It does not persist the ROI, create audit records, or fabricate units."
    ),
)
async def get_engineering_height_roi(
    inspection_id: str,
    request: Request,
    x: int,
    y: int,
    width: int,
    height: int,
) -> EngineeringHeightRoiStatisticsResponse:
    canonical_id = _canonical_inspection_id(inspection_id)
    service: EngineeringViewerService = request.app.state.engineering_viewer
    statistics = await _call(
        request,
        service.height_roi_statistics(
            canonical_id,
            x=x,
            y=y,
            width=width,
            height=height,
        ),
    )
    return EngineeringHeightRoiStatisticsResponse(
        **asdict(statistics),
        request_id=request.state.request_id,
    )
