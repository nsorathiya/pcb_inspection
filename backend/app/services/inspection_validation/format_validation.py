from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import PurePosixPath

from app.db.models import ArtifactType
from app.services.dataset_validation.file_inspection import (
    FileInspectionError,
    InspectedRaster,
    inspect_height,
    inspect_rgb,
)
from app.services.inspection_validation.findings import FindingFactory
from app.services.inspection_validation.interfaces import (
    ArtifactIntegrityInspection,
    ArtifactTechnicalSummary,
    NativeFormatInspection,
    ReadabilityStatus,
    StoredArtifactReference,
)

_RESULT_STORAGE_TYPES = {"uint8", "uint16", "int16", "uint32", "float32", "float64"}
_FORMAT_EXTENSIONS = {
    "PNG": {".png"}, "JPEG": {".jpg", ".jpeg"}, "BMP": {".bmp"},
    "TIFF": {".tif", ".tiff"}, "NPY": {".npy"},
}
_FORMAT_MEDIA_TYPES = {
    "PNG": {"image/png", "application/octet-stream"},
    "JPEG": {"image/jpeg", "application/octet-stream"},
    "BMP": {"image/bmp", "image/x-ms-bmp", "application/octet-stream"},
    "TIFF": {"image/tiff", "application/octet-stream"},
    "NPY": {"application/x-npy", "application/x-numpy", "application/octet-stream"},
}


def _signature(path) -> bytes:
    with path.open("rb") as source:
        return source.read(8)


def _empty_summary(artifact: StoredArtifactReference, integrity: ArtifactIntegrityInspection) -> ArtifactTechnicalSummary:
    return ArtifactTechnicalSummary(
        artifact_type=artifact.artifact_type,
        sha256=artifact.registered_sha256,
        byte_size=artifact.registered_byte_size,
        declared_media_type=artifact.declared_media_type,
        detected_format=None, width=None, height=None, channels=None, bit_depth=None,
        storage_data_type=None, readability_status=integrity.readability_status,
        source_extension=PurePosixPath(artifact.relative_path).suffix.lower() or None,
    )


def _summary(artifact: StoredArtifactReference, raster: InspectedRaster) -> ArtifactTechnicalSummary:
    dtype = raster.storage_data_type
    return ArtifactTechnicalSummary(
        artifact_type=artifact.artifact_type,
        sha256=artifact.registered_sha256,
        byte_size=artifact.registered_byte_size,
        declared_media_type=artifact.declared_media_type,
        detected_format=raster.detected_format,
        width=raster.width,
        height=raster.height,
        channels=raster.channels,
        bit_depth=raster.bit_depth,
        storage_data_type=dtype if dtype in _RESULT_STORAGE_TYPES else None,
        readability_status=ReadabilityStatus.READABLE,
        color_mode=raster.color_mode,
        source_extension=PurePosixPath(artifact.relative_path).suffix.lower() or None,
        observed_storage_data_type=dtype,
    )


class PurposeSpecificNativeFormatInspector:
    def __init__(self, findings: FindingFactory) -> None:
        self._findings = findings

    async def inspect_native_format(self, artifact: StoredArtifactReference, integrity: ArtifactIntegrityInspection) -> NativeFormatInspection:
        if integrity.failure_code is not None:
            return NativeFormatInspection(
                _empty_summary(artifact, integrity),
                (self._findings.create(integrity.failure_code, artifact_type=artifact.artifact_type),),
            )
        if integrity.resolved_path is None:
            raise RuntimeError("readable artifact has no resolved file")
        if artifact.artifact_type is ArtifactType.RGB_RAW:
            return await self._inspect_rgb(artifact, integrity)
        if artifact.artifact_type is ArtifactType.HEIGHT_RAW:
            return await self._inspect_height(artifact, integrity)
        raise ValueError("native format inspection is limited to raw RGB and height")

    async def inspect_validity_mask(self, artifact: StoredArtifactReference, integrity: ArtifactIntegrityInspection, height: ArtifactTechnicalSummary) -> tuple:
        if integrity.failure_code is not None:
            return (self._findings.create(integrity.failure_code, artifact_type=ArtifactType.VALIDITY_MASK),)
        path = integrity.resolved_path
        if path is None:
            raise RuntimeError("readable validity mask has no resolved file")
        try:
            raster = await asyncio.to_thread(inspect_rgb, path)
        except (FileInspectionError, OSError, ValueError):
            return (self._findings.create("FILE_UNREADABLE", artifact_type=ArtifactType.VALIDITY_MASK),)
        if raster.channels != 1 or (height.width, height.height) != (raster.width, raster.height):
            return (self._findings.create("FILE_UNREADABLE", artifact_type=ArtifactType.VALIDITY_MASK, field="technical_shape", details={"channels": raster.channels, "width": raster.width, "height": raster.height}),)
        return ()

    async def _inspect_rgb(self, artifact: StoredArtifactReference, integrity: ArtifactIntegrityInspection) -> NativeFormatInspection:
        path = integrity.resolved_path
        assert path is not None
        try:
            raster = await asyncio.to_thread(inspect_rgb, path)
        except (FileInspectionError, OSError, ValueError):
            signature = await asyncio.to_thread(_signature, path)
            code = "RGB_FORMAT_UNSUPPORTED" if signature.startswith((b"\x93NUMPY", b"\x89HDF", b"v/1\x01")) else "FILE_UNREADABLE"
            summary = _empty_summary(artifact, integrity)
            return NativeFormatInspection(replace(summary, readability_status=ReadabilityStatus.UNREADABLE), (self._findings.create(code, artifact_type=ArtifactType.RGB_RAW),))
        return self._with_content_checks(artifact, _summary(artifact, raster))

    async def _inspect_height(self, artifact: StoredArtifactReference, integrity: ArtifactIntegrityInspection) -> NativeFormatInspection:
        path = integrity.resolved_path
        assert path is not None
        findings = []
        try:
            raster = await asyncio.to_thread(inspect_height, path)
        except (FileInspectionError, OSError, ValueError):
            try:
                raster = await asyncio.to_thread(inspect_rgb, path)
            except (FileInspectionError, OSError, ValueError):
                signature = await asyncio.to_thread(_signature, path)
                code = "HEIGHT_FORMAT_UNSUPPORTED" if signature.startswith((b"BM", b"\xff\xd8", b"\x89HDF", b"v/1\x01")) else "FILE_UNREADABLE"
                summary = _empty_summary(artifact, integrity)
                summary = replace(summary, readability_status=ReadabilityStatus.UNREADABLE)
                return NativeFormatInspection(summary, (self._findings.create(code, artifact_type=ArtifactType.HEIGHT_RAW),))
        summary = _summary(artifact, raster)
        if summary.channels is not None and summary.channels != 1:
            stem = PurePosixPath(artifact.relative_path).stem.lower()
            code = "HEIGHT_COLORIZED_PREVIEW_REJECTED" if "colorized" in stem or "preview" in stem else "HEIGHT_NOT_SINGLE_CHANNEL"
            findings.append(self._findings.create(code, artifact_type=ArtifactType.HEIGHT_RAW, field="channels", details={"observed": summary.channels}))
        return self._with_content_checks(artifact, summary, findings)

    def _with_content_checks(self, artifact: StoredArtifactReference, summary: ArtifactTechnicalSummary, findings: list | None = None) -> NativeFormatInspection:
        result = list(findings or [])
        detected = summary.detected_format
        if detected is not None:
            if summary.source_extension not in _FORMAT_EXTENSIONS[detected]:
                result.append(self._findings.create("EXTENSION_CONTENT_MISMATCH", artifact_type=artifact.artifact_type, field="extension", details={"detected_format": detected}))
            media = (summary.declared_media_type or "").lower()
            if media and media not in _FORMAT_MEDIA_TYPES[detected]:
                result.append(self._findings.create("MEDIA_TYPE_CONTENT_MISMATCH", artifact_type=artifact.artifact_type, field="declared_media_type", details={"detected_format": detected}))
        return NativeFormatInspection(summary, tuple(result))
