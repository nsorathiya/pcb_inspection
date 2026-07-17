from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO, TypeAlias

from app.db.models import ArtifactType
from app.services.artifact_storage.exceptions import InvalidArtifactInputError

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
BinarySource: TypeAlias = bytes | bytearray | memoryview | BinaryIO


@dataclass(frozen=True)
class ArtifactInput:
    inspection_id: str
    artifact_type: ArtifactType
    source: BinarySource
    original_filename: str | None = None
    media_type: str | None = None
    expected_sha256: str | None = None
    expected_byte_size: int | None = None

    def __post_init__(self) -> None:
        try:
            ArtifactType(self.artifact_type)
        except (TypeError, ValueError) as exc:
            raise InvalidArtifactInputError("unknown artifact type") from exc
        if self.expected_sha256 is not None and not SHA256_PATTERN.fullmatch(
            self.expected_sha256
        ):
            raise InvalidArtifactInputError(
                "expected_sha256 must be 64 lowercase hexadecimal characters"
            )
        if self.expected_byte_size is not None and self.expected_byte_size < 0:
            raise InvalidArtifactInputError(
                "expected_byte_size must be non-negative"
            )
        if not isinstance(self.source, (bytes, bytearray, memoryview)) and not callable(
            getattr(self.source, "read", None)
        ):
            raise InvalidArtifactInputError(
                "source must be bytes or a binary file-like stream"
            )


@dataclass(frozen=True)
class ArtifactStorageResult:
    artifact_type: ArtifactType
    relative_path: str
    sha256: str
    byte_size: int
    media_type: str | None
    original_filename: str | None
    idempotent_existing: bool


@dataclass(frozen=True)
class ArtifactSizeLimits:
    rgb_bytes: int
    height_bytes: int
    mask_bytes: int
    calibration_bytes: int
    generated_artifact_bytes: int

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.rgb_bytes,
                self.height_bytes,
                self.mask_bytes,
                self.calibration_bytes,
                self.generated_artifact_bytes,
            )
        ):
            raise InvalidArtifactInputError("artifact size limits must be positive")

    def for_type(self, artifact_type: ArtifactType) -> int:
        artifact_type = ArtifactType(artifact_type)
        if artifact_type is ArtifactType.RGB_RAW:
            return self.rgb_bytes
        if artifact_type is ArtifactType.HEIGHT_RAW:
            return self.height_bytes
        if artifact_type is ArtifactType.VALIDITY_MASK:
            return self.mask_bytes
        if artifact_type is ArtifactType.CALIBRATION:
            return self.calibration_bytes
        return self.generated_artifact_bytes
