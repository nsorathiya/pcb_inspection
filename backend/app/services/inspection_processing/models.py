from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Sequence
from uuid import UUID

from app.db.models import InspectionStatus
from app.db.processing_types import ProcessingFinalDecision, ProcessingRunStatus

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def canonical_uuid(value: str, field: str) -> str:
    try:
        canonical = str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a canonical UUID string") from exc
    if canonical != value:
        raise ValueError(f"{field} must be a canonical UUID string")
    return value


def lowercase_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def safe_identity(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_IDENTITY.fullmatch(value):
        raise ValueError(f"{field} is invalid")
    return value


@dataclass(frozen=True)
class ProcessingKeyArtifact:
    artifact_type: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        safe_identity(self.artifact_type, "artifact_type")
        lowercase_sha256(self.sha256, "artifact sha256")
        if isinstance(self.byte_size, bool) or not isinstance(self.byte_size, int) or self.byte_size < 0:
            raise ValueError("artifact byte_size must be a non-negative integer")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "artifact_type": self.artifact_type,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


def generate_processing_key(
    *,
    inspection_id: str,
    validation_id: str,
    validation_result_sha256: str,
    rgb_artifact: ProcessingKeyArtifact,
    height_artifact: ProcessingKeyArtifact,
    preprocessing_policy_id: str,
    preprocessing_policy_version: str,
    preprocessing_implementation_id: str,
    preprocessing_implementation_version: str,
    inference_policy_id: str,
    inference_policy_version: str,
    engine_id: str,
    engine_version: str,
    engine_type: str,
    evidence_artifacts: Sequence[ProcessingKeyArtifact] = (),
) -> str:
    canonical_uuid(inspection_id, "inspection_id")
    canonical_uuid(validation_id, "validation_id")
    lowercase_sha256(validation_result_sha256, "validation_result_sha256")
    if rgb_artifact.artifact_type != "RGB_RAW":
        raise ValueError("rgb_artifact must identify RGB_RAW")
    if height_artifact.artifact_type != "HEIGHT_RAW":
        raise ValueError("height_artifact must identify HEIGHT_RAW")
    identities = {
        "preprocessing_policy_id": preprocessing_policy_id,
        "preprocessing_policy_version": preprocessing_policy_version,
        "preprocessing_implementation_id": preprocessing_implementation_id,
        "preprocessing_implementation_version": preprocessing_implementation_version,
        "inference_policy_id": inference_policy_id,
        "inference_policy_version": inference_policy_version,
        "engine_id": engine_id,
        "engine_version": engine_version,
        "engine_type": engine_type,
    }
    for field, value in identities.items():
        safe_identity(value, field)
    if engine_type != "MOCK":
        raise ValueError("schema version 3 processing supports only MOCK engine_type")
    document = {
        "engine": {
            "engine_id": engine_id,
            "engine_type": engine_type,
            "engine_version": engine_version,
            "policy_id": inference_policy_id,
            "policy_version": inference_policy_version,
        },
        "evidence_artifacts": [
            item.to_dict()
            for item in sorted(
                evidence_artifacts,
                key=lambda item: (item.artifact_type, item.sha256, item.byte_size),
            )
        ],
        "height_artifact": height_artifact.to_dict(),
        "inspection_id": inspection_id,
        "preprocessing": {
            "implementation_id": preprocessing_implementation_id,
            "implementation_version": preprocessing_implementation_version,
            "policy_id": preprocessing_policy_id,
            "policy_version": preprocessing_policy_version,
        },
        "rgb_artifact": rgb_artifact.to_dict(),
        "validation_id": validation_id,
        "validation_result_sha256": validation_result_sha256,
    }
    payload = json.dumps(
        document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProcessingStartIdentity:
    processing_run_id: str
    inspection_id: str
    validation_id: str
    validation_result_sha256: str
    rgb_artifact: ProcessingKeyArtifact
    height_artifact: ProcessingKeyArtifact
    preprocessing_policy_id: str
    preprocessing_policy_version: str
    preprocessing_implementation_id: str
    preprocessing_implementation_version: str
    inference_policy_id: str
    inference_policy_version: str
    engine_id: str
    engine_version: str
    engine_type: str = "MOCK"
    evidence_artifacts: tuple[ProcessingKeyArtifact, ...] = ()

    def __post_init__(self) -> None:
        canonical_uuid(self.processing_run_id, "processing_run_id")
        self.generated_key()

    def generated_key(self) -> str:
        values = dict(self.__dict__)
        values.pop("processing_run_id")
        return generate_processing_key(**values)


@dataclass(frozen=True)
class BeginProcessingResult:
    processing_run_id: str
    inspection_id: str
    validation_id: str
    processing_key: str
    inspection_status: InspectionStatus
    processing_status: ProcessingRunStatus
    idempotent_existing: bool
    started_at: datetime


@dataclass(frozen=True)
class CompleteProcessingResult:
    processing_run_id: str
    inspection_id: str
    preprocessing_id: str
    inference_id: str | None
    processing_status: ProcessingRunStatus
    inspection_status: InspectionStatus
    final_decision: ProcessingFinalDecision | None
    preprocessing_outcome: str
    inference_execution_outcome: str | None
    idempotent_existing: bool
    completed_at: datetime
    audit_action: str
