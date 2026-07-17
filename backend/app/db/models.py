from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as PythonEnum

from sqlalchemy import (
    CheckConstraint,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.db.validation_types import FindingCategory, FindingSeverity, ValidationOutcome

SCHEMA_VERSION = 2


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class InspectionStatus(str, PythonEnum):
    RECEIVED = "RECEIVED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    READY = "READY"
    PROCESSING = "PROCESSING"
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"


class ArtifactType(str, PythonEnum):
    RGB_RAW = "RGB_RAW"
    HEIGHT_RAW = "HEIGHT_RAW"
    VALIDITY_MASK = "VALIDITY_MASK"
    CALIBRATION = "CALIBRATION"
    RGB_PREVIEW = "RGB_PREVIEW"
    HEIGHT_PREVIEW = "HEIGHT_PREVIEW"
    RESULT_OVERLAY = "RESULT_OVERLAY"
    REPORT = "REPORT"


class RecipeStatus(str, PythonEnum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


class ModelCompatibilityStatus(str, PythonEnum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    INCOMPATIBLE = "INCOMPATIBLE"


class ModelStatus(str, PythonEnum):
    REGISTERED = "REGISTERED"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    BLOCKED = "BLOCKED"


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_schema_version_singleton"),
        CheckConstraint("version > 0", name="ck_schema_version_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class Inspection(Base):
    __tablename__ = "inspections"
    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_inspection_confidence_range",
        ),
        CheckConstraint(
            "processing_ms IS NULL OR processing_ms >= 0",
            name="ck_inspection_processing_nonnegative",
        ),
        CheckConstraint(
            "((model_id IS NULL AND model_version IS NULL) OR "
            "(model_id IS NOT NULL AND model_version IS NOT NULL))",
            name="ck_inspection_model_identity_pair",
        ),
        CheckConstraint(
            "status NOT IN ('PASS', 'FAIL', 'UNCERTAIN') OR completed_at IS NOT NULL",
            name="ck_inspection_final_completed_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[InspectionStatus] = mapped_column(
        Enum(
            InspectionStatus,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="inspection_status",
        ),
        nullable=False,
    )
    board_id: Mapped[str] = mapped_column(String(128), nullable=False)
    recipe_id: Mapped[str] = mapped_column(String(128), nullable=False)
    recipe_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(128))
    model_version: Mapped[str | None] = mapped_column(String(128))
    confidence: Mapped[float | None] = mapped_column(Float)
    processing_ms: Mapped[float | None] = mapped_column(Float)
    lot_id: Mapped[str | None] = mapped_column(String(128))
    operator_id: Mapped[str | None] = mapped_column(String(128))
    request_id: Mapped[str | None] = mapped_column(String(256))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    artifacts: Mapped[list["InspectionArtifact"]] = relationship(
        back_populates="inspection",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    validations: Mapped[list["InspectionValidation"]] = relationship(
        back_populates="inspection",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class InspectionArtifact(Base):
    __tablename__ = "inspection_artifacts"
    __table_args__ = (
        CheckConstraint("byte_size >= 0", name="ck_artifact_byte_size_nonnegative"),
        CheckConstraint(
            "length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_artifact_sha256",
        ),
        CheckConstraint(
            "length(relative_path) > 0 "
            "AND substr(relative_path, 1, 1) <> '/' "
            "AND relative_path NOT GLOB '[A-Za-z]:*' "
            "AND instr(relative_path, '\\') = 0 "
            "AND relative_path <> '..' "
            "AND relative_path NOT LIKE '../%' "
            "AND relative_path NOT LIKE '%/../%' "
            "AND relative_path NOT LIKE '%/..'",
            name="ck_artifact_relative_path",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    inspection_id: Mapped[str] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_type: Mapped[ArtifactType] = mapped_column(
        Enum(
            ArtifactType,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="artifact_type",
        ),
        nullable=False,
    )
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    inspection: Mapped[Inspection] = relationship(back_populates="artifacts")


class InspectionValidation(Base):
    __tablename__ = "inspection_validations"
    __table_args__ = (
        UniqueConstraint(
            "inspection_id",
            "validation_key",
            name="uq_inspection_validation_key",
        ),
        CheckConstraint(
            "length(id) = 36 AND lower(id) = id "
            "AND length(replace(id, '-', '')) = 32 "
            "AND substr(id, 9, 1) = '-' AND substr(id, 14, 1) = '-' "
            "AND substr(id, 19, 1) = '-' AND substr(id, 24, 1) = '-' "
            "AND id NOT GLOB '*[^0-9a-f-]*'",
            name="ck_inspection_validation_uuid",
        ),
        CheckConstraint(
            "length(validation_key) = 64 "
            "AND validation_key NOT GLOB '*[^0-9a-f]*'",
            name="ck_inspection_validation_key_sha256",
        ),
        CheckConstraint(
            "length(result_sha256) = 64 "
            "AND result_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_inspection_validation_result_sha256",
        ),
        CheckConstraint(
            "completed_at >= started_at",
            name="ck_inspection_validation_timestamp_order",
        ),
        CheckConstraint(
            "json_valid(rgb_summary_json) AND json_type(rgb_summary_json) = 'object'",
            name="ck_inspection_validation_rgb_json",
        ),
        CheckConstraint(
            "json_valid(height_summary_json) AND json_type(height_summary_json) = 'object'",
            name="ck_inspection_validation_height_json",
        ),
        CheckConstraint(
            "json_valid(summary_json) AND json_type(summary_json) = 'object'",
            name="ck_inspection_validation_summary_json",
        ),
        CheckConstraint(
            "json_valid(result_json) AND json_type(result_json) = 'object'",
            name="ck_inspection_validation_result_json",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    inspection_id: Mapped[str] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    contract_version: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    validator_version: Mapped[str] = mapped_column(String(64), nullable=False)
    validation_key: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[ValidationOutcome] = mapped_column(
        Enum(
            ValidationOutcome,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="validation_outcome",
        ),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    rgb_summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    height_summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )

    inspection: Mapped[Inspection] = relationship(back_populates="validations")
    findings: Mapped[list["InspectionValidationFinding"]] = relationship(
        back_populates="validation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class InspectionValidationFinding(Base):
    __tablename__ = "inspection_validation_findings"
    __table_args__ = (
        UniqueConstraint(
            "validation_id",
            "ordinal",
            name="uq_inspection_validation_finding_ordinal",
        ),
        CheckConstraint("ordinal >= 0", name="ck_validation_finding_ordinal"),
        CheckConstraint(
            "length(id) = 36 AND lower(id) = id "
            "AND length(replace(id, '-', '')) = 32 "
            "AND substr(id, 9, 1) = '-' AND substr(id, 14, 1) = '-' "
            "AND substr(id, 19, 1) = '-' AND substr(id, 24, 1) = '-' "
            "AND id NOT GLOB '*[^0-9a-f-]*'",
            name="ck_validation_finding_uuid",
        ),
        CheckConstraint(
            "json_valid(details_json) AND json_type(details_json) = 'object'",
            name="ck_validation_finding_details_json",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    validation_id: Mapped[str] = mapped_column(
        ForeignKey("inspection_validations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[FindingSeverity] = mapped_column(
        Enum(
            FindingSeverity,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="validation_finding_severity",
        ),
        nullable=False,
    )
    category: Mapped[FindingCategory] = mapped_column(
        Enum(
            FindingCategory,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="validation_finding_category",
        ),
        nullable=False,
    )
    message: Mapped[str] = mapped_column(String(512), nullable=False)
    artifact_type: Mapped[ArtifactType | None] = mapped_column(
        Enum(
            ArtifactType,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="validation_finding_artifact_type",
        )
    )
    field: Mapped[str | None] = mapped_column(String(128))
    blocking: Mapped[bool] = mapped_column(Boolean, nullable=False)
    details_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    validation: Mapped[InspectionValidation] = relationship(back_populates="findings")


class Recipe(Base):
    __tablename__ = "recipes"
    __table_args__ = (
        UniqueConstraint("recipe_id", "recipe_version", name="uq_recipe_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    recipe_id: Mapped[str] = mapped_column(String(128), nullable=False)
    recipe_version: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    configuration_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[RecipeStatus] = mapped_column(
        Enum(
            RecipeStatus,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="recipe_status",
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "model_version", name="uq_model_version"),
        CheckConstraint(
            "sha256 IS NULL OR (length(sha256) = 64 "
            "AND sha256 NOT GLOB '*[^0-9a-f]*')",
            name="ck_model_sha256",
        ),
        CheckConstraint(
            "artifact_relative_path IS NULL OR ("
            "length(artifact_relative_path) > 0 "
            "AND substr(artifact_relative_path, 1, 1) <> '/' "
            "AND artifact_relative_path NOT GLOB '[A-Za-z]:*' "
            "AND instr(artifact_relative_path, '\\') = 0 "
            "AND artifact_relative_path <> '..' "
            "AND artifact_relative_path NOT LIKE '../%' "
            "AND artifact_relative_path NOT LIKE '%/../%' "
            "AND artifact_relative_path NOT LIKE '%/..')",
            name="ck_model_relative_path",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    engine_type: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_relative_path: Mapped[str | None] = mapped_column(String(512))
    sha256: Mapped[str | None] = mapped_column(String(64))
    class_label_contract_version: Mapped[str] = mapped_column(String(64), nullable=False)
    defect_taxonomy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    compatibility_status: Mapped[ModelCompatibilityStatus] = mapped_column(
        Enum(
            ModelCompatibilityStatus,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="model_compatibility_status",
        ),
        nullable=False,
    )
    status: Mapped[ModelStatus] = mapped_column(
        Enum(
            ModelStatus,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="model_status",
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(128))
    request_id: Mapped[str | None] = mapped_column(String(256))
    details_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )
