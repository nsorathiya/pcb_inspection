from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Annotated, AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.api.errors import ApiError, ApiErrorResponse
from app.api.validation_responses import (
    InspectionValidationResponse,
    map_validation_response,
)
from app.api.processing_responses import (
    InspectionProcessingResponse,
    map_processing_response,
)
from app.db.models import ArtifactType, InspectionStatus
from app.services.artifact_storage import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactRegistrationError,
    ArtifactSizeLimitError,
    ArtifactStorageError,
    validate_intake_file,
)
from app.services.inspection_intake import (
    InspectionIntakeCommand,
    InspectionIntakeCoordinator,
    InspectionIntakeFailure,
    IntakeArtifactSource,
)
from app.services.inspection_validation.orchestrator import (
    InspectionValidationOrchestrator,
    InvalidValidationPolicySelectionError,
    ValidationExecutionConflictError,
    ValidationExecutionConsistencyError,
    ValidationInspectionNotFoundError,
    ValidationPolicyNotFoundError,
    ValidationPolicyVersionUnsupportedError,
    ValidationResultNotFoundError,
)
from app.services.inspection_inference.exceptions import InferencePolicyLoadError
from app.services.inspection_preprocessing.exceptions import PreprocessingPolicyLoadError
from app.services.inspection_processing import (
    InspectionProcessingApiService,
    ProcessingExecutionArtifactPairError,
    ProcessingExecutionConflictError,
    ProcessingExecutionConsistencyError,
    ProcessingExecutionInProgressError,
    ProcessingExecutionInspectionNotFoundError,
    ProcessingExecutionInspectionNotReadyError,
    ProcessingExecutionOptionalEvidenceUnsupportedError,
    ProcessingExecutionOrchestrationError,
    ProcessingExecutionPolicyError,
    ProcessingExecutionReprocessingUnsupportedError,
    ProcessingExecutionRecoveryRequiredError,
    ProcessingExecutionResultNotFoundError,
    ProcessingExecutionValidationMissingError,
    ProcessingExecutionValidationNotPassedError,
    SyntheticProcessingNotConfiguredError,
    SyntheticProvenanceMismatchError,
    SyntheticProvenanceUnavailableError,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MULTIPART_FIELDS = (
    "rgb_image",
    "height_map",
    "board_id",
    "recipe_id",
    "recipe_version",
    "lot_id",
    "operator_id",
    "station_id",
    "rgb_sha256",
    "height_sha256",
    "rgb_byte_size",
    "height_byte_size",
)


class _InspectionIntakeMetadata(BaseModel):
    board_id: str
    recipe_id: str
    recipe_version: str
    lot_id: str | None = None
    operator_id: str | None = None
    station_id: str | None = None
    rgb_sha256: str | None = None
    height_sha256: str | None = None
    rgb_byte_size: int | None = None
    height_byte_size: int | None = None

    @field_validator("board_id", "recipe_id", "recipe_version", mode="before")
    @classmethod
    def validate_required_identifier(cls, value: object) -> str:
        return cls._validated_identifier(value, required=True)

    @field_validator("lot_id", "operator_id", "station_id", mode="before")
    @classmethod
    def validate_optional_identifier(cls, value: object) -> str | None:
        return cls._validated_identifier(value, required=False)

    @staticmethod
    def _validated_identifier(value: object, *, required: bool) -> str | None:
        if not isinstance(value, str):
            raise ValueError("identifier must be text")
        if any(unicodedata.category(character) == "Cc" for character in value):
            raise ValueError("identifier must not contain control characters")
        normalized = value.strip()
        if not normalized:
            if required:
                raise ValueError("required identifier must not be empty")
            return None
        if len(normalized) > 128:
            raise ValueError("identifier must not exceed 128 characters")
        return normalized

    @field_validator("rgb_sha256", "height_sha256", mode="before")
    @classmethod
    def validate_expected_hash(cls, value: object) -> str | None:
        if value is None or value == "":
            return None
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value.strip()):
            raise ValueError("expected hash must be lowercase SHA-256")
        return value.strip()

    @field_validator("rgb_byte_size", "height_byte_size", mode="before")
    @classmethod
    def validate_expected_size(cls, value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("expected size must be a non-negative integer")
        text = str(value).strip()
        if not text.isdecimal():
            raise ValueError("expected size must be a non-negative integer")
        return int(text)


class IntakeArtifactResponse(BaseModel):
    artifact_type: ArtifactType
    sha256: str
    byte_size: int
    media_type: str | None


class InspectionIntakeResponse(BaseModel):
    inspection_id: str
    status: InspectionStatus
    board_id: str
    recipe_id: str
    recipe_version: str
    lot_id: str | None
    request_id: str
    created_at: datetime
    artifacts: list[IntakeArtifactResponse]


class InspectionErrorResponse(BaseModel):
    code: str
    message: str


class InspectionArtifactDetailResponse(IntakeArtifactResponse):
    created_at: datetime


class InspectionDetailResponse(BaseModel):
    inspection_id: str
    status: InspectionStatus
    board_id: str
    recipe_id: str
    recipe_version: str
    lot_id: str | None
    intake_request_id: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: InspectionErrorResponse | None
    artifacts: list[InspectionArtifactDetailResponse]


class InspectionValidationRequest(BaseModel):
    policy_id: str = Field(max_length=128)
    policy_version: str = Field(max_length=64)

    model_config = ConfigDict(extra="forbid")


class InspectionProcessingRequest(BaseModel):
    preprocessing_policy_id: str = Field(max_length=128)
    preprocessing_policy_version: str = Field(max_length=64)
    inference_policy_id: str = Field(max_length=128)
    inference_policy_version: str = Field(max_length=64)

    model_config = ConfigDict(extra="forbid")


class _PairedUploads:
    def __init__(self, rgb_image: UploadFile, height_map: UploadFile) -> None:
        self.rgb_image = rgb_image
        self.height_map = height_map


async def _paired_uploads(
    rgb_image: Annotated[UploadFile, File(description="Original 2D RGB image")],
    height_map: Annotated[
        UploadFile,
        File(description="Original native height/depth raster or NumPy array"),
    ],
) -> AsyncIterator[_PairedUploads]:
    try:
        yield _PairedUploads(rgb_image, height_map)
    finally:
        await rgb_image.close()
        await height_map.close()


async def _reject_duplicate_fields(request: Request) -> None:
    form = await request.form()
    if form.getlist("inspection_id"):
        raise ApiError(
            status_code=400,
            code="CLIENT_INSPECTION_ID_NOT_ALLOWED",
            message="Inspection IDs are generated by the server.",
        )
    duplicates = [name for name in MULTIPART_FIELDS if len(form.getlist(name)) > 1]
    if duplicates:
        raise ApiError(
            status_code=400,
            code="DUPLICATE_MULTIPART_FIELD",
            message="Multipart fields must be supplied at most once.",
        )


def _map_intake_failure(failure: InspectionIntakeFailure) -> ApiError:
    cause = failure.cause
    if isinstance(cause, ArtifactSizeLimitError):
        return ApiError(413, "ARTIFACT_SIZE_LIMIT_EXCEEDED", "An artifact is too large.")
    if isinstance(cause, ArtifactConflictError):
        return ApiError(409, "IMMUTABLE_ARTIFACT_CONFLICT", "Artifact storage conflict.")
    if isinstance(cause, ArtifactIntegrityError):
        return ApiError(400, "ARTIFACT_INTEGRITY_MISMATCH", "Artifact integrity check failed.")
    if isinstance(cause, ArtifactRegistrationError):
        return ApiError(500, "ARTIFACT_REGISTRATION_FAILED", "Artifact registration failed.")
    if isinstance(cause, ArtifactStorageError):
        return ApiError(500, "ARTIFACT_STORAGE_FAILED", "Artifact storage failed.")
    return ApiError(500, "INSPECTION_INTAKE_FAILED", "Inspection intake failed.")


router = APIRouter(tags=["inspections"])


def _canonical_inspection_id(value: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError) as exc:
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


def _public_inspection_error(
    inspection_status: InspectionStatus,
    error_code: str | None,
    error_message: str | None,
) -> InspectionErrorResponse | None:
    if inspection_status not in {
        InspectionStatus.ERROR,
        InspectionStatus.VALIDATION_FAILED,
    }:
        return None

    safe_known_errors = {
        "INSPECTION_INTAKE_FAILED": "Paired artifact intake did not complete.",
    }
    if (
        error_code in safe_known_errors
        and error_message == safe_known_errors[error_code]
    ):
        return InspectionErrorResponse(code=error_code, message=error_message)
    if inspection_status is InspectionStatus.VALIDATION_FAILED:
        return InspectionErrorResponse(
            code="INSPECTION_VALIDATION_FAILED",
            message="Inspection validation failed.",
        )
    return InspectionErrorResponse(
        code="INSPECTION_ERROR",
        message="Inspection processing failed.",
    )


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _map_validation_error(error: Exception) -> ApiError:
    if isinstance(error, InvalidValidationPolicySelectionError):
        return ApiError(
            400,
            "INVALID_VALIDATION_POLICY_SELECTION",
            "Validation policy ID or version is invalid.",
        )
    if isinstance(error, ValidationPolicyNotFoundError):
        return ApiError(
            404,
            "VALIDATION_POLICY_NOT_FOUND",
            "The selected validation policy was not found.",
        )
    if isinstance(error, ValidationPolicyVersionUnsupportedError):
        return ApiError(
            404,
            "VALIDATION_POLICY_VERSION_UNSUPPORTED",
            "The selected validation policy version is unsupported.",
        )
    if isinstance(error, ValidationInspectionNotFoundError):
        return ApiError(404, "INSPECTION_NOT_FOUND", "Inspection was not found.")
    if isinstance(error, ValidationResultNotFoundError):
        return ApiError(
            404,
            "INSPECTION_VALIDATION_NOT_FOUND",
            "No validation result exists for this inspection.",
        )
    if isinstance(error, ValidationExecutionConsistencyError):
        return ApiError(
            409,
            "VALIDATION_LIFECYCLE_CONFLICT",
            "Persisted validation evidence conflicts with inspection state.",
        )
    if isinstance(error, ValidationExecutionConflictError):
        return ApiError(
            409,
            "INSPECTION_NOT_ELIGIBLE_FOR_VALIDATION",
            "The inspection is not eligible for a new validation. Revalidation is unsupported.",
        )
    return ApiError(
        500,
        "VALIDATION_ORCHESTRATION_FAILED",
        "Inspection validation could not be completed reliably.",
    )


def _validated_processing_selection(
    selection: InspectionProcessingRequest,
) -> InspectionProcessingRequest:
    identity = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    version = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
    if (
        identity.fullmatch(selection.preprocessing_policy_id) is None
        or version.fullmatch(selection.preprocessing_policy_version) is None
        or identity.fullmatch(selection.inference_policy_id) is None
        or version.fullmatch(selection.inference_policy_version) is None
    ):
        raise ApiError(
            400,
            "INVALID_PROCESSING_POLICY_SELECTION",
            "Processing policy IDs or versions are invalid.",
        )
    return selection


def _map_processing_error(error: Exception) -> ApiError:
    if isinstance(error, SyntheticProcessingNotConfiguredError):
        return ApiError(
            503,
            "SYNTHETIC_PROCESSING_NOT_CONFIGURED",
            "Synthetic processing execution is not configured.",
        )
    if isinstance(error, ProcessingExecutionPolicyError):
        cause = error.__cause__
        if isinstance(cause, PreprocessingPolicyLoadError):
            if cause.finding_code == "PREPROCESSING_POLICY_NOT_FOUND":
                return ApiError(
                    404,
                    "PREPROCESSING_POLICY_NOT_FOUND",
                    "The selected preprocessing policy was not found.",
                )
            if cause.finding_code == "PREPROCESSING_POLICY_VERSION_UNSUPPORTED":
                return ApiError(
                    404,
                    "PREPROCESSING_POLICY_VERSION_UNSUPPORTED",
                    "The selected preprocessing policy version is unsupported.",
                )
        if isinstance(cause, InferencePolicyLoadError):
            if cause.finding_code == "INFERENCE_POLICY_NOT_FOUND":
                return ApiError(
                    404,
                    "INFERENCE_POLICY_NOT_FOUND",
                    "The selected inference policy was not found.",
                )
            if cause.finding_code == "INFERENCE_POLICY_VERSION_UNSUPPORTED":
                return ApiError(
                    404,
                    "INFERENCE_POLICY_VERSION_UNSUPPORTED",
                    "The selected inference policy version is unsupported.",
                )
        return ApiError(
            500,
            "PROCESSING_POLICY_UNAVAILABLE",
            "The configured processing policy could not be loaded safely.",
        )
    if isinstance(error, ProcessingExecutionInspectionNotFoundError):
        return ApiError(404, "INSPECTION_NOT_FOUND", "Inspection was not found.")
    if isinstance(error, ProcessingExecutionResultNotFoundError):
        return ApiError(
            404,
            "INSPECTION_PROCESSING_NOT_FOUND",
            "No processing result exists for this inspection.",
        )
    if isinstance(error, ProcessingExecutionInProgressError):
        return ApiError(
            409,
            "PROCESSING_ALREADY_IN_PROGRESS",
            "Inspection processing is already in progress.",
        )
    if isinstance(error, ProcessingExecutionOptionalEvidenceUnsupportedError):
        return ApiError(
            409,
            "OPTIONAL_EVIDENCE_PROCESSING_UNSUPPORTED",
            "Synthetic processing does not yet support optional mask or calibration evidence.",
        )
    if isinstance(error, ProcessingExecutionReprocessingUnsupportedError):
        return ApiError(
            409,
            "REPROCESSING_UNSUPPORTED",
            "The inspection already has a final status. Reprocessing is unsupported.",
        )
    if isinstance(error, ProcessingExecutionInspectionNotReadyError):
        return ApiError(
            409,
            "INSPECTION_NOT_READY",
            "The inspection is not READY for processing.",
        )
    if isinstance(error, ProcessingExecutionValidationMissingError):
        return ApiError(
            409,
            "INSPECTION_VALIDATION_REQUIRED",
            "A persisted passed validation is required before processing.",
        )
    if isinstance(error, ProcessingExecutionValidationNotPassedError):
        return ApiError(
            409,
            "INSPECTION_VALIDATION_NOT_PASSED",
            "The persisted inspection validation did not pass.",
        )
    if isinstance(error, ProcessingExecutionArtifactPairError):
        return ApiError(
            409,
            "INCOMPLETE_ARTIFACT_PAIR",
            "The registered RGB and height artifact pair is incomplete or ambiguous.",
        )
    if isinstance(error, SyntheticProvenanceMismatchError):
        return ApiError(
            409,
            "SYNTHETIC_PROVENANCE_MISMATCH",
            "Trusted synthetic fixture provenance did not match the inspection.",
        )
    if isinstance(error, SyntheticProvenanceUnavailableError):
        return ApiError(
            409,
            "SYNTHETIC_PROVENANCE_UNAVAILABLE",
            "Trusted synthetic fixture provenance is unavailable.",
        )
    if isinstance(error, ProcessingExecutionConsistencyError):
        return ApiError(
            500,
            "PROCESSING_DATA_INCONSISTENT",
            "Persisted processing evidence is internally inconsistent.",
        )
    if isinstance(error, ProcessingExecutionRecoveryRequiredError):
        return ApiError(
            500,
            "PROCESSING_RECOVERY_REQUIRED",
            "Processing could not be finalized and requires operational recovery.",
        )
    if isinstance(error, ProcessingExecutionConflictError):
        return ApiError(
            409,
            "PROCESSING_LIFECYCLE_CONFLICT",
            "Processing conflicts with the current inspection lifecycle.",
        )
    if isinstance(error, ProcessingExecutionOrchestrationError):
        return ApiError(
            500,
            "PROCESSING_ORCHESTRATION_FAILED",
            "Inspection processing could not be completed reliably.",
        )
    return ApiError(
        500,
        "PROCESSING_ORCHESTRATION_FAILED",
        "Inspection processing could not be completed reliably.",
    )


@router.post(
    "/inspections/{inspection_id}/process",
    response_model=InspectionProcessingResponse,
    responses={
        400: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        503: {"model": ApiErrorResponse},
    },
    summary="Development-only trusted synthetic processing execution",
    description=(
        "Development-only endpoint for generator-owned synthetic fixtures. It "
        "delegates execution exclusively to the trusted synthetic processing "
        "orchestrator, which verifies fixture provenance before beginning a new "
        "lifecycle. PASS, FAIL, and UNCERTAIN are deterministic mock workflow "
        "decisions, not image-based predictions or production PCB dispositions. "
        "The digest bucket does not analyze PCB defects, confidence is unavailable, "
        "and no real AI model is executed. Exact completed retries return persisted "
        "evidence without rerunning. Reprocessing is not supported."
    ),
)
async def process_inspection(
    inspection_id: str,
    selection: InspectionProcessingRequest,
    request: Request,
) -> InspectionProcessingResponse:
    canonical_id = _canonical_inspection_id(inspection_id)
    selected = _validated_processing_selection(selection)
    service: InspectionProcessingApiService = request.app.state.inspection_processing
    try:
        execution = await service.execute_processing(
            canonical_id,
            selected.preprocessing_policy_id,
            selected.preprocessing_policy_version,
            selected.inference_policy_id,
            selected.inference_policy_version,
            request_id=request.state.request_id,
        )
    except Exception as exc:
        mapped = _map_processing_error(exc)
        if mapped.status_code == 500:
            request.app.state.logger.exception(
                "Processing orchestration failed inspection_id=%s",
                canonical_id,
            )
        raise mapped from exc
    return map_processing_response(execution, request_id=request.state.request_id)


@router.get(
    "/inspections/{inspection_id}/processing",
    response_model=InspectionProcessingResponse,
    responses={
        400: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    summary="Retrieve latest persisted synthetic processing evidence",
    description=(
        "Returns the deterministic latest persisted processing lifecycle, typed "
        "summaries, and ordered findings without rerunning preprocessing or mock "
        "inference, reading artifact files, verifying fixture manifests, writing "
        "database state, or appending audit events. Mock decisions are development "
        "workflow evidence, not real predictions or production PCB dispositions; "
        "confidence is unavailable."
    ),
)
async def get_inspection_processing(
    inspection_id: str,
    request: Request,
) -> InspectionProcessingResponse:
    canonical_id = _canonical_inspection_id(inspection_id)
    service: InspectionProcessingApiService = request.app.state.inspection_processing
    try:
        execution = await service.get_latest_processing(canonical_id)
    except Exception as exc:
        mapped = _map_processing_error(exc)
        if mapped.status_code == 500:
            request.app.state.logger.exception(
                "Processing result read failed inspection_id=%s",
                canonical_id,
            )
        raise mapped from exc
    return map_processing_response(execution, request_id=request.state.request_id)


@router.post(
    "/inspections/{inspection_id}/validate",
    response_model=InspectionValidationResponse,
    responses={
        400: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    summary="Execute technical validation for one received inspection",
    description=(
        "Performs technical semantic validation of the registered RGB and native "
        "height pair under an explicitly selected policy. It does not perform PCB "
        "defect classification and does not run AI inference. VALIDATION_PASSED "
        "means technically ready for future preprocessing only, never PCB PASS. "
        "Revalidation is not supported. Exact retries are system-idempotent and "
        "return the already committed lifecycle result."
    ),
)
async def validate_inspection(
    inspection_id: str,
    selection: InspectionValidationRequest,
    request: Request,
) -> InspectionValidationResponse:
    canonical_id = _canonical_inspection_id(inspection_id)
    orchestrator: InspectionValidationOrchestrator = (
        request.app.state.inspection_validation
    )
    try:
        execution = await orchestrator.execute_validation(
            canonical_id,
            selection.policy_id,
            selection.policy_version,
            actor_id=None,
            request_id=request.state.request_id,
        )
    except Exception as exc:
        mapped = _map_validation_error(exc)
        if mapped.status_code == 500:
            request.app.state.logger.exception(
                "Validation orchestration failed inspection_id=%s",
                canonical_id,
            )
        raise mapped from exc
    return map_validation_response(
        execution,
        request_id=request.state.request_id,
    )


@router.get(
    "/inspections/{inspection_id}/validation",
    response_model=InspectionValidationResponse,
    responses={
        400: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    summary="Retrieve the latest persisted technical validation result",
    description=(
        "Returns the latest persisted technical validation result and ordered "
        "findings. It does not rerun validation, verify current file availability, "
        "or read artifact bytes. It does not run AI inference and does not "
        "classify the PCB."
    ),
)
async def get_inspection_validation(
    inspection_id: str,
    request: Request,
) -> InspectionValidationResponse:
    canonical_id = _canonical_inspection_id(inspection_id)
    orchestrator: InspectionValidationOrchestrator = (
        request.app.state.inspection_validation
    )
    try:
        execution = await orchestrator.get_latest_validation(canonical_id)
    except Exception as exc:
        mapped = _map_validation_error(exc)
        if mapped.status_code == 500:
            request.app.state.logger.exception(
                "Validation result read failed inspection_id=%s",
                canonical_id,
            )
        raise mapped from exc
    return map_validation_response(
        execution,
        request_id=request.state.request_id,
    )


@router.get(
    "/inspections/{inspection_id}",
    response_model=InspectionDetailResponse,
    responses={
        400: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    summary="Retrieve one inspection and its artifact metadata",
    description=(
        "Returns persisted inspection state and registered artifact integrity "
        "metadata. It does not read artifact bytes, expose storage paths, "
        "semantically validate files, or classify the inspection."
    ),
)
async def get_inspection(
    inspection_id: str,
    request: Request,
) -> InspectionDetailResponse:
    canonical_id = _canonical_inspection_id(inspection_id)
    repositories = request.app.state.repositories
    try:
        inspection = await repositories.inspections.get(canonical_id)
        if inspection is None:
            raise ApiError(
                404,
                "INSPECTION_NOT_FOUND",
                "Inspection was not found.",
            )
        artifacts = await repositories.artifacts.list_for_inspection(canonical_id)
    except ApiError:
        raise
    except Exception as exc:
        request.app.state.logger.exception(
            "Inspection detail read failed inspection_id=%s",
            canonical_id,
        )
        raise ApiError(
            500,
            "INSPECTION_READ_FAILED",
            "Inspection details could not be retrieved.",
        ) from exc

    return InspectionDetailResponse(
        inspection_id=inspection.id,
        status=inspection.status,
        board_id=inspection.board_id,
        recipe_id=inspection.recipe_id,
        recipe_version=inspection.recipe_version,
        lot_id=inspection.lot_id,
        intake_request_id=inspection.request_id,
        created_at=_as_utc(inspection.created_at),
        started_at=_as_utc(inspection.started_at),
        completed_at=_as_utc(inspection.completed_at),
        error=_public_inspection_error(
            inspection.status,
            inspection.error_code,
            inspection.error_message,
        ),
        artifacts=[
            InspectionArtifactDetailResponse(
                artifact_type=artifact.artifact_type,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                media_type=artifact.media_type,
                created_at=_as_utc(artifact.created_at),
            )
            for artifact in artifacts
        ],
    )


@router.post(
    "/inspections",
    response_model=InspectionIntakeResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        413: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    summary="Receive one paired RGB and height-map inspection",
    description=(
        "Receives and immutably stores exactly one RGB file and one height/depth "
        "file for the same physical inspection instance. Files are not decoded or "
        "semantically validated, and alignment is not proven. RECEIVED does not "
        "mean PASS, FAIL, UNCERTAIN, ready for training, or production accepted."
    ),
)
async def create_inspection(
    request: Request,
    board_id: Annotated[str, Form(description="Board or assembly identifier")],
    recipe_id: Annotated[str, Form(description="Inspection recipe identifier")],
    recipe_version: Annotated[str, Form(description="Inspection recipe version")],
    uploads: Annotated[_PairedUploads, Depends(_paired_uploads)],
    lot_id: Annotated[str | None, Form()] = None,
    operator_id: Annotated[str | None, Form()] = None,
    station_id: Annotated[str | None, Form()] = None,
    rgb_sha256: Annotated[str | None, Form()] = None,
    height_sha256: Annotated[str | None, Form()] = None,
    rgb_byte_size: Annotated[str | None, Form()] = None,
    height_byte_size: Annotated[str | None, Form()] = None,
    _duplicates: Annotated[None, Depends(_reject_duplicate_fields)] = None,
) -> InspectionIntakeResponse:
    rgb_image = uploads.rgb_image
    height_map = uploads.height_map
    try:
        metadata = _InspectionIntakeMetadata(
            board_id=board_id,
            recipe_id=recipe_id,
            recipe_version=recipe_version,
            lot_id=lot_id,
            operator_id=operator_id,
            station_id=station_id,
            rgb_sha256=rgb_sha256,
            height_sha256=height_sha256,
            rgb_byte_size=rgb_byte_size,
            height_byte_size=height_byte_size,
        )
    except ValidationError as exc:
        raise ApiError(
            400,
            "INVALID_INTAKE_METADATA",
            "One or more intake metadata fields are invalid.",
        ) from exc

    try:
        validate_intake_file(
            ArtifactType.RGB_RAW,
            rgb_image.filename,
            rgb_image.content_type,
        )
        validate_intake_file(
            ArtifactType.HEIGHT_RAW,
            height_map.filename,
            height_map.content_type,
        )
    except ArtifactStorageError as exc:
        raise ApiError(
            400,
            "UNSUPPORTED_INTAKE_FORMAT",
            "An upload filename extension or media type is not accepted.",
        ) from exc

    coordinator: InspectionIntakeCoordinator = request.app.state.inspection_intake
    try:
        result = await coordinator.receive_pair(
            InspectionIntakeCommand(
                board_id=metadata.board_id,
                recipe_id=metadata.recipe_id,
                recipe_version=metadata.recipe_version,
                lot_id=metadata.lot_id,
                operator_id=metadata.operator_id,
                station_id=metadata.station_id,
                request_id=request.state.request_id,
                rgb=IntakeArtifactSource(
                    source=rgb_image.file,
                    original_filename=rgb_image.filename,
                    media_type=rgb_image.content_type,
                    expected_sha256=metadata.rgb_sha256,
                    expected_byte_size=metadata.rgb_byte_size,
                ),
                height=IntakeArtifactSource(
                    source=height_map.file,
                    original_filename=height_map.filename,
                    media_type=height_map.content_type,
                    expected_sha256=metadata.height_sha256,
                    expected_byte_size=metadata.height_byte_size,
                ),
            )
        )
    except InspectionIntakeFailure as exc:
        raise _map_intake_failure(exc) from exc

    inspection = result.inspection
    return InspectionIntakeResponse(
        inspection_id=inspection.id,
        status=inspection.status,
        board_id=inspection.board_id,
        recipe_id=inspection.recipe_id,
        recipe_version=inspection.recipe_version,
        lot_id=inspection.lot_id,
        request_id=inspection.request_id or request.state.request_id,
        created_at=inspection.created_at,
        artifacts=[
            IntakeArtifactResponse(
                artifact_type=artifact.artifact_type,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                media_type=artifact.media_type,
            )
            for artifact in result.artifacts
        ],
    )
