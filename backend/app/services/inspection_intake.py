from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Callable
from uuid import uuid4

from app.db.models import ArtifactType, Inspection, InspectionArtifact, InspectionStatus
from app.db.repositories import AuditEventCreate, InspectionCreate, Repositories
from app.services.artifact_storage import (
    ArtifactConflictError,
    ArtifactInput,
    ArtifactIntegrityError,
    ArtifactRegistrationError,
    ArtifactSizeLimitError,
    ArtifactStorageError,
)
from app.services.artifact_storage.service import (
    ArtifactRegistrationService,
    _RegistrationOperation,
)
from app.services.audit_actions import INSPECTION_INTAKE_FAILED, INSPECTION_RECEIVED


@dataclass(frozen=True)
class IntakeArtifactSource:
    source: BinaryIO
    original_filename: str | None
    media_type: str | None
    expected_sha256: str | None
    expected_byte_size: int | None


@dataclass(frozen=True)
class InspectionIntakeCommand:
    board_id: str
    recipe_id: str
    recipe_version: str
    request_id: str
    rgb: IntakeArtifactSource
    height: IntakeArtifactSource
    lot_id: str | None = None
    operator_id: str | None = None
    station_id: str | None = None


@dataclass(frozen=True)
class InspectionIntakeResult:
    inspection: Inspection
    artifacts: tuple[InspectionArtifact, InspectionArtifact]


class InspectionIntakeFailure(Exception):
    def __init__(
        self,
        *,
        category: str,
        cause: Exception,
        inspection_id: str | None,
        compensation_complete: bool,
    ) -> None:
        super().__init__("paired inspection intake failed")
        self.category = category
        self.cause = cause
        self.inspection_id = inspection_id
        self.compensation_complete = compensation_complete


def _failure_category(error: Exception) -> str:
    if isinstance(error, ArtifactSizeLimitError):
        return "SIZE_LIMIT_EXCEEDED"
    if isinstance(error, ArtifactIntegrityError):
        return "INTEGRITY_MISMATCH"
    if isinstance(error, ArtifactConflictError):
        return "IMMUTABLE_CONFLICT"
    if isinstance(error, ArtifactRegistrationError):
        return "DATABASE_REGISTRATION_FAILED"
    if isinstance(error, ArtifactStorageError):
        return "ARTIFACT_STORAGE_FAILED"
    return "INTERNAL_INTAKE_FAILURE"


class InspectionIntakeCoordinator:
    """Create and compensate exactly one raw RGB/height inspection pair."""

    def __init__(
        self,
        repositories: Repositories,
        artifact_registration: ArtifactRegistrationService,
        *,
        inspection_id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._repositories = repositories
        self._artifact_registration = artifact_registration
        self._inspection_id = inspection_id_generator or (lambda: str(uuid4()))

    async def receive_pair(
        self,
        command: InspectionIntakeCommand,
    ) -> InspectionIntakeResult:
        inspection_id = self._inspection_id()
        try:
            inspection = await self._repositories.inspections.create(
                InspectionCreate(
                    id=inspection_id,
                    status=InspectionStatus.RECEIVED,
                    board_id=command.board_id,
                    recipe_id=command.recipe_id,
                    recipe_version=command.recipe_version,
                    lot_id=command.lot_id,
                    operator_id=command.operator_id,
                    request_id=command.request_id,
                )
            )
        except Exception as exc:
            raise InspectionIntakeFailure(
                category="DATABASE_INSPECTION_CREATE_FAILED",
                cause=exc,
                inspection_id=None,
                compensation_complete=True,
            ) from exc

        operations: list[_RegistrationOperation] = []
        try:
            rgb_operation = await self._artifact_registration._store_and_register_operation(
                ArtifactInput(
                    inspection_id=inspection_id,
                    artifact_type=ArtifactType.RGB_RAW,
                    source=command.rgb.source,
                    original_filename=command.rgb.original_filename,
                    media_type=command.rgb.media_type,
                    expected_sha256=command.rgb.expected_sha256,
                    expected_byte_size=command.rgb.expected_byte_size,
                )
            )
            operations.append(rgb_operation)

            height_operation = (
                await self._artifact_registration._store_and_register_operation(
                    ArtifactInput(
                        inspection_id=inspection_id,
                        artifact_type=ArtifactType.HEIGHT_RAW,
                        source=command.height.source,
                        original_filename=command.height.original_filename,
                        media_type=command.height.media_type,
                        expected_sha256=command.height.expected_sha256,
                        expected_byte_size=command.height.expected_byte_size,
                    )
                )
            )
            operations.append(height_operation)

            await self._repositories.audit_events.append(
                AuditEventCreate(
                    entity_type="inspection",
                    entity_id=inspection_id,
                    action=INSPECTION_RECEIVED,
                    actor_id=command.operator_id,
                    request_id=command.request_id,
                    details={
                        "artifact_types": [
                            ArtifactType.RGB_RAW.value,
                            ArtifactType.HEIGHT_RAW.value,
                        ],
                        "byte_sizes": {
                            ArtifactType.RGB_RAW.value: rgb_operation.record.byte_size,
                            ArtifactType.HEIGHT_RAW.value: (
                                height_operation.record.byte_size
                            ),
                        },
                        "station_id": command.station_id,
                    },
                )
            )
            return InspectionIntakeResult(
                inspection=inspection,
                artifacts=(rgb_operation.record, height_operation.record),
            )
        except Exception as exc:
            category = _failure_category(exc)
            compensation_complete = True
            for operation in reversed(operations):
                try:
                    compensated = (
                        await self._artifact_registration._compensate_operation(
                            operation
                        )
                    )
                    compensation_complete = compensation_complete and compensated
                except Exception:
                    compensation_complete = False

            try:
                await self._repositories.inspections.mark_intake_failed(
                    inspection_id,
                    error_code="INSPECTION_INTAKE_FAILED",
                    error_message="Paired artifact intake did not complete.",
                )
            except Exception:
                compensation_complete = False

            try:
                await self._repositories.audit_events.append(
                    AuditEventCreate(
                        entity_type="inspection",
                        entity_id=inspection_id,
                        action=INSPECTION_INTAKE_FAILED,
                        actor_id=command.operator_id,
                        request_id=command.request_id,
                        details={
                            "failure_category": category,
                            "compensation_complete": compensation_complete,
                        },
                    )
                )
            except Exception:
                compensation_complete = False

            raise InspectionIntakeFailure(
                category=category,
                cause=exc,
                inspection_id=inspection_id,
                compensation_complete=compensation_complete,
            ) from exc
