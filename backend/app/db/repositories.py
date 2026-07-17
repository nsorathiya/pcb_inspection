from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Mapping
from uuid import UUID, uuid4

from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    ArtifactType,
    AuditEvent,
    Inspection,
    InspectionArtifact,
    InspectionStatus,
    ModelCompatibilityStatus,
    ModelStatus,
    ModelVersion,
    Recipe,
    RecipeStatus,
)

if TYPE_CHECKING:
    from app.services.inspection_validation.persistence import InspectionValidationRepository

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FINAL_INSPECTION_STATUSES = {
    InspectionStatus.PASS,
    InspectionStatus.FAIL,
    InspectionStatus.UNCERTAIN,
}


def _new_or_valid_uuid(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    if str(UUID(value)) != value:
        raise ValueError("id must be a canonical UUID string")
    return value


def _json_text(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("JSON details must contain serializable values") from exc


def _validate_sha256(value: str | None, *, required: bool) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


def _validate_relative_path(value: str | None, *, required: bool) -> None:
    if value is None and not required:
        return
    if (
        not isinstance(value, str)
        or not value
        or value == "."
        or "\\" in value
        or ":" in value
    ):
        raise ValueError("artifact path must be a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must not be absolute or contain '..'")


@dataclass(frozen=True)
class InspectionCreate:
    status: InspectionStatus
    board_id: str
    recipe_id: str
    recipe_version: str
    id: str | None = None
    model_id: str | None = None
    model_version: str | None = None
    confidence: float | None = None
    processing_ms: float | None = None
    lot_id: str | None = None
    operator_id: str | None = None
    request_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        status = InspectionStatus(self.status)
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be between 0 and 1")
        if self.processing_ms is not None and (
            not math.isfinite(self.processing_ms) or self.processing_ms < 0
        ):
            raise ValueError("processing_ms must be non-negative")
        if (self.model_id is None) != (self.model_version is None):
            raise ValueError("model_id and model_version must be supplied together")
        if status in FINAL_INSPECTION_STATUSES and self.completed_at is None:
            raise ValueError("PASS, FAIL, and UNCERTAIN require completed_at")


@dataclass(frozen=True)
class InspectionArtifactCreate:
    inspection_id: str
    artifact_type: ArtifactType
    relative_path: str
    sha256: str
    byte_size: int
    id: str | None = None
    media_type: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        ArtifactType(self.artifact_type)
        _validate_relative_path(self.relative_path, required=True)
        _validate_sha256(self.sha256, required=True)
        if self.byte_size < 0:
            raise ValueError("byte_size must be non-negative")


@dataclass(frozen=True)
class RecipeCreate:
    recipe_id: str
    recipe_version: str
    name: str
    configuration: Mapping[str, Any]
    status: RecipeStatus = RecipeStatus.DRAFT
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        RecipeStatus(self.status)
        _json_text(self.configuration)


@dataclass(frozen=True)
class ModelVersionCreate:
    model_id: str
    model_version: str
    engine_type: str
    class_label_contract_version: str
    defect_taxonomy_version: str
    compatibility_status: ModelCompatibilityStatus
    status: ModelStatus = ModelStatus.REGISTERED
    id: str | None = None
    artifact_relative_path: str | None = None
    sha256: str | None = None
    created_at: datetime | None = None
    activated_at: datetime | None = None

    def __post_init__(self) -> None:
        ModelCompatibilityStatus(self.compatibility_status)
        ModelStatus(self.status)
        _validate_relative_path(self.artifact_relative_path, required=False)
        _validate_sha256(self.sha256, required=False)


@dataclass(frozen=True)
class AuditEventCreate:
    entity_type: str
    entity_id: str
    action: str
    details: Mapping[str, Any] = field(default_factory=dict)
    id: str | None = None
    actor_id: str | None = None
    request_id: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        _json_text(self.details)


class InspectionRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def create(self, data: InspectionCreate) -> Inspection:
        record = Inspection(
            id=_new_or_valid_uuid(data.id),
            status=InspectionStatus(data.status),
            board_id=data.board_id,
            recipe_id=data.recipe_id,
            recipe_version=data.recipe_version,
            model_id=data.model_id,
            model_version=data.model_version,
            confidence=data.confidence,
            processing_ms=data.processing_ms,
            lot_id=data.lot_id,
            operator_id=data.operator_id,
            request_id=data.request_id,
            error_code=data.error_code,
            error_message=data.error_message,
            started_at=data.started_at,
            completed_at=data.completed_at,
        )
        if data.created_at is not None:
            record.created_at = data.created_at
        async with self._sessions() as session, session.begin():
            session.add(record)
            await session.flush()
        return record

    async def get(self, inspection_id: str) -> Inspection | None:
        async with self._sessions() as session:
            return await session.get(Inspection, inspection_id)

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[Inspection]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        statement = (
            select(Inspection)
            .order_by(Inspection.created_at.desc(), Inspection.id.asc())
            .limit(limit)
            .offset(offset)
        )
        async with self._sessions() as session:
            result = await session.scalars(statement)
            return list(result)

    async def mark_intake_failed(
        self,
        inspection_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> Inspection:
        """Apply only the RECEIVED-to-ERROR transition used by intake compensation."""
        async with self._sessions() as session, session.begin():
            record = await session.get(Inspection, inspection_id)
            if record is None:
                raise ValueError("inspection does not exist")
            if record.status is not InspectionStatus.RECEIVED:
                raise ValueError("only a RECEIVED inspection can fail during intake")
            record.status = InspectionStatus.ERROR
            record.error_code = error_code
            record.error_message = error_message
            record.completed_at = datetime.now(timezone.utc)
            await session.flush()
        return record


class InspectionArtifactRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def create(self, data: InspectionArtifactCreate) -> InspectionArtifact:
        record = InspectionArtifact(
            id=_new_or_valid_uuid(data.id),
            inspection_id=data.inspection_id,
            artifact_type=ArtifactType(data.artifact_type),
            relative_path=data.relative_path,
            sha256=data.sha256,
            byte_size=data.byte_size,
            media_type=data.media_type,
        )
        if data.created_at is not None:
            record.created_at = data.created_at
        async with self._sessions() as session, session.begin():
            session.add(record)
            await session.flush()
        return record

    async def get(self, artifact_id: str) -> InspectionArtifact | None:
        async with self._sessions() as session:
            return await session.get(InspectionArtifact, artifact_id)

    async def get_by_location(
        self,
        inspection_id: str,
        artifact_type: ArtifactType,
        relative_path: str,
    ) -> InspectionArtifact | None:
        statement = select(InspectionArtifact).where(
            InspectionArtifact.inspection_id == inspection_id,
            InspectionArtifact.artifact_type == ArtifactType(artifact_type),
            InspectionArtifact.relative_path == relative_path,
        )
        async with self._sessions() as session:
            return await session.scalar(statement)

    async def list_for_inspection(
        self,
        inspection_id: str,
    ) -> list[InspectionArtifact]:
        artifact_types = tuple(ArtifactType)
        artifact_type_order = case(
            *(
                (InspectionArtifact.artifact_type == artifact_type, position)
                for position, artifact_type in enumerate(artifact_types)
            ),
            else_=len(artifact_types),
        )
        statement = (
            select(InspectionArtifact)
            .where(InspectionArtifact.inspection_id == inspection_id)
            .order_by(
                artifact_type_order,
                InspectionArtifact.created_at.asc(),
                InspectionArtifact.id.asc(),
            )
        )
        async with self._sessions() as session:
            result = await session.scalars(statement)
            return list(result)

    async def _delete_for_intake_compensation(
        self,
        *,
        artifact_id: str,
        inspection_id: str,
        artifact_type: ArtifactType,
        relative_path: str,
        sha256: str,
        byte_size: int,
    ) -> bool:
        """Delete only an exact artifact row owned by a failed current intake."""
        async with self._sessions() as session, session.begin():
            record = await session.get(InspectionArtifact, artifact_id)
            if record is None:
                return False
            if (
                record.inspection_id != inspection_id
                or record.artifact_type is not ArtifactType(artifact_type)
                or record.relative_path != relative_path
                or record.sha256 != sha256
                or record.byte_size != byte_size
            ):
                return False
            await session.delete(record)
            await session.flush()
        return True


class RecipeRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def register(self, data: RecipeCreate) -> Recipe:
        record = Recipe(
            id=_new_or_valid_uuid(data.id),
            recipe_id=data.recipe_id,
            recipe_version=data.recipe_version,
            name=data.name,
            configuration_json=_json_text(data.configuration),
            status=RecipeStatus(data.status),
        )
        if data.created_at is not None:
            record.created_at = data.created_at
        if data.updated_at is not None:
            record.updated_at = data.updated_at
        async with self._sessions() as session, session.begin():
            session.add(record)
            await session.flush()
        return record


class ModelVersionRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def register(self, data: ModelVersionCreate) -> ModelVersion:
        record = ModelVersion(
            id=_new_or_valid_uuid(data.id),
            model_id=data.model_id,
            model_version=data.model_version,
            engine_type=data.engine_type,
            artifact_relative_path=data.artifact_relative_path,
            sha256=data.sha256,
            class_label_contract_version=data.class_label_contract_version,
            defect_taxonomy_version=data.defect_taxonomy_version,
            compatibility_status=ModelCompatibilityStatus(data.compatibility_status),
            status=ModelStatus(data.status),
            activated_at=data.activated_at,
        )
        if data.created_at is not None:
            record.created_at = data.created_at
        async with self._sessions() as session, session.begin():
            session.add(record)
            await session.flush()
        return record


class AuditEventRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def append(self, data: AuditEventCreate) -> AuditEvent:
        record = AuditEvent(
            id=_new_or_valid_uuid(data.id),
            entity_type=data.entity_type,
            entity_id=data.entity_id,
            action=data.action,
            actor_id=data.actor_id,
            request_id=data.request_id,
            details_json=_json_text(data.details),
        )
        if data.created_at is not None:
            record.created_at = data.created_at
        async with self._sessions() as session, session.begin():
            session.add(record)
            await session.flush()
        return record

    async def get(self, event_id: str) -> AuditEvent | None:
        async with self._sessions() as session:
            return await session.get(AuditEvent, event_id)

    async def list_for_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> list[AuditEvent]:
        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.entity_type == entity_type,
                AuditEvent.entity_id == entity_id,
            )
            .order_by(AuditEvent.created_at.asc(), AuditEvent.id.asc())
        )
        async with self._sessions() as session:
            result = await session.scalars(statement)
            return list(result)


@dataclass(frozen=True)
class Repositories:
    inspections: InspectionRepository
    artifacts: InspectionArtifactRepository
    recipes: RecipeRepository
    models: ModelVersionRepository
    audit_events: AuditEventRepository
    validations: "InspectionValidationRepository"

    @classmethod
    def from_session_factory(
        cls,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> "Repositories":
        from app.services.inspection_validation.persistence import (
            InspectionValidationRepository,
        )

        return cls(
            inspections=InspectionRepository(session_factory),
            artifacts=InspectionArtifactRepository(session_factory),
            recipes=RecipeRepository(session_factory),
            models=ModelVersionRepository(session_factory),
            audit_events=AuditEventRepository(session_factory),
            validations=InspectionValidationRepository(session_factory),
        )
