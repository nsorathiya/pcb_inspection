from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence
from uuid import UUID

from app.db.models import ArtifactType

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


@dataclass(frozen=True)
class ValidationKeyArtifact:
    artifact_type: ArtifactType
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_type, ArtifactType):
            raise ValueError("validation-key artifact type is invalid")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("validation-key artifact SHA-256 must be lowercase hexadecimal")
        if (
            not isinstance(self.byte_size, int)
            or isinstance(self.byte_size, bool)
            or self.byte_size < 0
        ):
            raise ValueError("validation-key artifact byte size must be non-negative")


def _canonical_uuid(value: str) -> str:
    try:
        canonical = str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("inspection_id must be a canonical UUID string") from exc
    if canonical != value:
        raise ValueError("inspection_id must be a canonical UUID string")
    return value


def generate_validation_key(
    *,
    inspection_id: str,
    rgb_artifact: ValidationKeyArtifact,
    height_artifact: ValidationKeyArtifact,
    contract_version: str,
    policy_id: str,
    policy_version: str,
    validator_version: str,
    evidence_artifacts: Sequence[ValidationKeyArtifact] = (),
) -> str:
    _canonical_uuid(inspection_id)
    if rgb_artifact.artifact_type is not ArtifactType.RGB_RAW:
        raise ValueError("rgb_artifact must use RGB_RAW")
    if height_artifact.artifact_type is not ArtifactType.HEIGHT_RAW:
        raise ValueError("height_artifact must use HEIGHT_RAW")
    if not _IDENTIFIER.fullmatch(policy_id):
        raise ValueError("policy_id is invalid")
    if not _VERSION.fullmatch(policy_version) or not _VERSION.fullmatch(validator_version):
        raise ValueError("policy and validator versions must be explicit")
    if not contract_version or len(contract_version) > 128:
        raise ValueError("validation contract version must be explicit")

    def artifact_value(value: ValidationKeyArtifact) -> dict[str, object]:
        return {
            "artifact_type": value.artifact_type.value,
            "byte_size": value.byte_size,
            "sha256": value.sha256,
        }

    evidence = sorted(
        (artifact_value(item) for item in evidence_artifacts),
        key=lambda item: (
            str(item["artifact_type"]),
            str(item["sha256"]),
            int(item["byte_size"]),
        ),
    )
    payload = {
        "contract_version": contract_version,
        "evidence_artifacts": evidence,
        "height_artifact": artifact_value(height_artifact),
        "inspection_id": inspection_id,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "rgb_artifact": artifact_value(rgb_artifact),
        "validator_version": validator_version,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()
