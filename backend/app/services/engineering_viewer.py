from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.db.models import ArtifactType, Inspection, InspectionArtifact
from app.db.repositories import Repositories
from app.services.dataset_validation.file_inspection import (
    DecodedRaster,
    FileInspectionError,
    decode_height_values,
    decode_rgb_values,
    inspect_height,
    inspect_rgb,
)
from app.services.inspection_processing.api_service import (
    InspectionProcessingApiService,
)
from app.services.inspection_processing.exceptions import (
    ProcessingExecutionResultNotFoundError,
)
from app.services.inspection_validation.artifact_reader import (
    ManagedArtifactPathResolver,
)
from app.services.inspection_validation.integrity_validation import (
    StreamingFilesystemIntegrityInspector,
)
from app.services.inspection_validation.interfaces import (
    ReadabilityStatus,
    StoredArtifactReference,
)
from app.services.inspection_validation.orchestrator import (
    InspectionValidationOrchestrator,
    ValidationResultNotFoundError,
)
from app.testing.synthetic_aoi.raster_generation import encode_png

HISTOGRAM_BIN_COUNT = 64
MAX_ENGINEERING_PIXEL_COUNT = 16_777_216

SAFE_ENGINEERING_WARNINGS = (
    "DEVELOPMENT_ENGINEERING_VIEW_ONLY",
    "NO_PHYSICAL_HEIGHT_UNITS_AVAILABLE",
    "NO_PHYSICAL_REGISTRATION_CLAIM",
    "PREVIEWS_ARE_DERIVED_AND_NOT_PERSISTED",
)


class EngineeringViewerError(Exception):
    """Base class for safe engineering-view failures."""


class EngineeringViewerDisabledError(EngineeringViewerError):
    pass


class EngineeringInspectionNotFoundError(EngineeringViewerError):
    pass


class EngineeringArtifactPairError(EngineeringViewerError):
    pass


class EngineeringArtifactIntegrityError(EngineeringViewerError):
    pass


class EngineeringFormatUnsupportedError(EngineeringViewerError):
    pass


class EngineeringRasterTooLargeError(EngineeringViewerError):
    pass


class EngineeringSampleBoundsError(EngineeringViewerError):
    pass


class EngineeringEvidenceReadError(EngineeringViewerError):
    pass


@dataclass(frozen=True)
class EngineeringRasterMetadata:
    artifact_type: str
    detected_format: str
    width: int
    height: int
    channels: int
    bit_depth: int
    color_mode: str
    storage_data_type: str | None
    sha256: str
    byte_size: int


@dataclass(frozen=True)
class HeightHistogram:
    bin_count: int
    native_min: int | float
    native_max: int | float
    counts: tuple[int, ...]


@dataclass(frozen=True)
class HeightStatistics:
    native_min: int | float
    native_max: int | float
    valid_count: int
    invalid_count: int
    histogram: HeightHistogram


@dataclass(frozen=True)
class ValidationEvidence:
    available: bool
    validation_id: str | None = None
    outcome: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    technically_ready: bool | None = None
    finding_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessingEvidence:
    available: bool
    processing_run_id: str | None = None
    processing_status: str | None = None
    preprocessing_outcome: str | None = None
    mock_decision: str | None = None
    production_approved: bool | None = None
    synthetic_input_verified: bool | None = None
    finding_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EngineeringView:
    inspection_id: str
    inspection_status: str
    rgb: EngineeringRasterMetadata
    height: EngineeringRasterMetadata
    height_statistics: HeightStatistics
    calibration_status: str
    registration_status: str
    physical_height_unit: None
    validation: ValidationEvidence
    processing: ProcessingEvidence
    warnings: tuple[str, ...]
    synthetic_input_verified: bool
    production_approved: bool = False


@dataclass(frozen=True)
class SampleValue:
    x: int
    y: int
    storage_data_type: str | None
    values: tuple[int, ...] | None = None
    value: int | float | None = None
    valid: bool | None = None
    physical_unit: None = None


@dataclass(frozen=True)
class EngineeringSample:
    inspection_id: str
    rgb: SampleValue
    height: SampleValue
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedPreview:
    content: bytes
    preview_kind: str
    transform: str


@dataclass(frozen=True)
class _VerifiedPair:
    inspection: Inspection
    rgb_record: InspectionArtifact
    height_record: InspectionArtifact
    rgb: DecodedRaster
    height: DecodedRaster
    calibration_status: str


class EngineeringViewerService:
    """Read-only, integrity-gated viewer for the supported synthetic subset."""

    def __init__(
        self,
        *,
        enabled: bool,
        repositories: Repositories,
        resolver: ManagedArtifactPathResolver,
        validation_reader: InspectionValidationOrchestrator,
        processing_reader: InspectionProcessingApiService,
    ) -> None:
        self._enabled = enabled
        self._repositories = repositories
        self._integrity = StreamingFilesystemIntegrityInspector(resolver)
        self._validation = validation_reader
        self._processing = processing_reader

    async def get_view(self, inspection_id: str) -> EngineeringView:
        pair = await self._verified_pair(inspection_id)
        statistics = await asyncio.to_thread(
            self._height_statistics,
            pair.height.values,
        )
        validation, processing = await self._persisted_evidence(inspection_id)
        registration_status = "NOT_ESTABLISHED"
        if "SYNTHETIC_IDENTITY_REGISTRATION_USED" in processing.finding_codes:
            registration_status = "SYNTHETIC_IDENTITY_ONLY"
        synthetic_input_verified = processing.synthetic_input_verified is True
        provenance_warning = (
            "SYNTHETIC_INPUT_VERIFIED_BY_PERSISTED_PROCESSING"
            if synthetic_input_verified
            else "SYNTHETIC_INPUT_PROVENANCE_NOT_VERIFIED"
        )
        return EngineeringView(
            inspection_id=inspection_id,
            inspection_status=pair.inspection.status.value,
            rgb=self._metadata(pair.rgb_record, pair.rgb),
            height=self._metadata(pair.height_record, pair.height),
            height_statistics=statistics,
            calibration_status=pair.calibration_status,
            registration_status=registration_status,
            physical_height_unit=None,
            validation=validation,
            processing=processing,
            warnings=(*SAFE_ENGINEERING_WARNINGS, provenance_warning),
            synthetic_input_verified=synthetic_input_verified,
        )

    async def rgb_preview(self, inspection_id: str) -> GeneratedPreview:
        pair = await self._verified_pair(inspection_id)
        content = await asyncio.to_thread(self._rgb_preview_png, pair.rgb)
        return GeneratedPreview(content, "RGB", "NATIVE_VALUES_TO_8_BIT_RGB")

    async def height_preview(self, inspection_id: str) -> GeneratedPreview:
        pair = await self._verified_pair(inspection_id)
        content = await asyncio.to_thread(self._height_preview_png, pair.height)
        return GeneratedPreview(content, "HEIGHT", "NATIVE_MIN_MAX_GRAYSCALE")

    async def sample(
        self,
        inspection_id: str,
        *,
        rgb_x: int,
        rgb_y: int,
        height_x: int,
        height_y: int,
    ) -> EngineeringSample:
        pair = await self._verified_pair(inspection_id)
        self._assert_coordinate(pair.rgb, rgb_x, rgb_y, "RGB")
        self._assert_coordinate(pair.height, height_x, height_y, "height")
        rgb_offset = (rgb_y * pair.rgb.metadata.width + rgb_x) * 3
        rgb_values = tuple(
            int(value) for value in pair.rgb.values[rgb_offset : rgb_offset + 3]
        )
        height_offset = height_y * pair.height.metadata.width + height_x
        native_height = pair.height.values[height_offset]
        height_valid = self._is_valid_height(native_height)
        return EngineeringSample(
            inspection_id=inspection_id,
            rgb=SampleValue(
                x=rgb_x,
                y=rgb_y,
                storage_data_type=pair.rgb.metadata.storage_data_type,
                values=rgb_values,
            ),
            height=SampleValue(
                x=height_x,
                y=height_y,
                storage_data_type=pair.height.metadata.storage_data_type,
                value=native_height if height_valid else None,
                valid=height_valid,
                physical_unit=None,
            ),
            warnings=SAFE_ENGINEERING_WARNINGS,
        )

    async def _verified_pair(self, inspection_id: str) -> _VerifiedPair:
        self._assert_enabled()
        inspection = await self._repositories.inspections.get(inspection_id)
        if inspection is None:
            raise EngineeringInspectionNotFoundError("inspection does not exist")
        records = await self._repositories.artifacts.list_for_inspection(inspection_id)
        rgb_record = self._one_artifact(records, ArtifactType.RGB_RAW)
        height_record = self._one_artifact(records, ArtifactType.HEIGHT_RAW)
        rgb_path = await self._verified_path(rgb_record, inspection_id)
        height_path = await self._verified_path(height_record, inspection_id)
        calibration_records = [
            record
            for record in records
            if record.artifact_type is ArtifactType.CALIBRATION
        ]
        if len(calibration_records) > 1:
            raise EngineeringArtifactPairError("calibration evidence is ambiguous")
        calibration_status = "NOT_PROVIDED"
        if calibration_records:
            await self._verified_path(calibration_records[0], inspection_id)
            calibration_status = "PRESENT_UNINTERPRETED"
        try:
            rgb_meta, height_meta = await asyncio.gather(
                asyncio.to_thread(inspect_rgb, rgb_path),
                asyncio.to_thread(inspect_height, height_path),
            )
            self._assert_supported_metadata(rgb_meta, ArtifactType.RGB_RAW)
            self._assert_supported_metadata(height_meta, ArtifactType.HEIGHT_RAW)
            rgb, height = await asyncio.gather(
                asyncio.to_thread(decode_rgb_values, rgb_path),
                asyncio.to_thread(decode_height_values, height_path),
            )
        except EngineeringViewerError:
            raise
        except (FileInspectionError, OSError, ValueError) as exc:
            raise EngineeringFormatUnsupportedError(
                "artifact format is not supported by the engineering viewer"
            ) from exc
        return _VerifiedPair(
            inspection,
            rgb_record,
            height_record,
            rgb,
            height,
            calibration_status,
        )

    def _assert_enabled(self) -> None:
        if not self._enabled:
            raise EngineeringViewerDisabledError("engineering viewer is disabled")

    @staticmethod
    def _one_artifact(
        records: Sequence[InspectionArtifact],
        artifact_type: ArtifactType,
    ) -> InspectionArtifact:
        selected = [record for record in records if record.artifact_type is artifact_type]
        if len(selected) != 1:
            raise EngineeringArtifactPairError(
                "inspection must own exactly one RGB_RAW and one HEIGHT_RAW artifact"
            )
        return selected[0]

    async def _verified_path(
        self,
        record: InspectionArtifact,
        inspection_id: str,
    ) -> Path:
        if record.inspection_id != inspection_id:
            raise EngineeringArtifactIntegrityError("artifact ownership mismatch")
        reference = StoredArtifactReference(
            inspection_id=inspection_id,
            artifact_type=record.artifact_type,
            relative_path=record.relative_path,
            registered_sha256=record.sha256,
            registered_byte_size=record.byte_size,
            declared_media_type=record.media_type,
        )
        inspected = await self._integrity.inspect_integrity(reference)
        if (
            inspected.readability_status is not ReadabilityStatus.READABLE
            or inspected.resolved_path is None
        ):
            raise EngineeringArtifactIntegrityError(
                inspected.failure_code or "artifact integrity verification failed"
            )
        return inspected.resolved_path

    @staticmethod
    def _assert_supported_metadata(metadata, artifact_type: ArtifactType) -> None:
        pixels = metadata.width * metadata.height
        if pixels <= 0 or pixels > MAX_ENGINEERING_PIXEL_COUNT:
            raise EngineeringRasterTooLargeError(
                "raster exceeds the engineering viewer pixel limit"
            )
        if artifact_type is ArtifactType.RGB_RAW:
            supported = (
                metadata.detected_format in {"PNG", "TIFF"}
                and metadata.color_mode == "RGB"
                and metadata.channels == 3
                and metadata.bit_depth in {8, 16}
            )
        else:
            supported = (
                metadata.detected_format in {"PNG", "TIFF", "NPY"}
                and metadata.channels == 1
                and metadata.storage_data_type in {"uint16", "float32"}
            )
        if not supported:
            raise EngineeringFormatUnsupportedError(
                "artifact format is outside the supported synthetic subset"
            )

    @staticmethod
    def _metadata(
        record: InspectionArtifact,
        raster: DecodedRaster,
    ) -> EngineeringRasterMetadata:
        metadata = raster.metadata
        return EngineeringRasterMetadata(
            artifact_type=record.artifact_type.value,
            detected_format=metadata.detected_format,
            width=metadata.width,
            height=metadata.height,
            channels=metadata.channels,
            bit_depth=metadata.bit_depth,
            color_mode=metadata.color_mode,
            storage_data_type=metadata.storage_data_type,
            sha256=record.sha256,
            byte_size=record.byte_size,
        )

    @staticmethod
    def _is_valid_height(value: int | float) -> bool:
        return not isinstance(value, float) or math.isfinite(value)

    @classmethod
    def _height_statistics(
        cls,
        values: Sequence[int | float],
    ) -> HeightStatistics:
        valid = [value for value in values if cls._is_valid_height(value)]
        invalid_count = len(values) - len(valid)
        if not valid:
            raise EngineeringFormatUnsupportedError(
                "height raster contains no finite native values"
            )
        native_min = min(valid)
        native_max = max(valid)
        counts = [0] * HISTOGRAM_BIN_COUNT
        if native_min == native_max:
            counts[0] = len(valid)
        else:
            width = (float(native_max) - float(native_min)) / HISTOGRAM_BIN_COUNT
            for value in valid:
                index = int((float(value) - float(native_min)) / width)
                counts[min(index, HISTOGRAM_BIN_COUNT - 1)] += 1
        return HeightStatistics(
            native_min=native_min,
            native_max=native_max,
            valid_count=len(valid),
            invalid_count=invalid_count,
            histogram=HeightHistogram(
                bin_count=HISTOGRAM_BIN_COUNT,
                native_min=native_min,
                native_max=native_max,
                counts=tuple(counts),
            ),
        )

    @staticmethod
    def _rgb_preview_png(raster: DecodedRaster) -> bytes:
        maximum = (1 << raster.metadata.bit_depth) - 1
        pixels = bytes(
            max(0, min(255, round(int(value) * 255 / maximum)))
            for value in raster.values
        )
        return encode_png(
            raster.metadata.width,
            raster.metadata.height,
            bit_depth=8,
            color_type=2,
            pixel_bytes=pixels,
        )

    @classmethod
    def _height_preview_png(cls, raster: DecodedRaster) -> bytes:
        valid = [value for value in raster.values if cls._is_valid_height(value)]
        if not valid:
            raise EngineeringFormatUnsupportedError(
                "height raster contains no finite native values"
            )
        low = float(min(valid))
        high = float(max(valid))
        span = high - low
        pixels = bytes(
            0
            if not cls._is_valid_height(value) or span == 0
            else max(0, min(255, round((float(value) - low) * 255 / span)))
            for value in raster.values
        )
        return encode_png(
            raster.metadata.width,
            raster.metadata.height,
            bit_depth=8,
            color_type=0,
            pixel_bytes=pixels,
        )

    @staticmethod
    def _assert_coordinate(
        raster: DecodedRaster,
        x: int,
        y: int,
        label: str,
    ) -> None:
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or x < 0
            or y < 0
            or x >= raster.metadata.width
            or y >= raster.metadata.height
        ):
            raise EngineeringSampleBoundsError(
                f"{label} sample coordinates are outside the raster bounds"
            )

    async def _persisted_evidence(
        self,
        inspection_id: str,
    ) -> tuple[ValidationEvidence, ProcessingEvidence]:
        try:
            validation_execution = await self._validation.get_latest_validation(
                inspection_id
            )
        except ValidationResultNotFoundError:
            validation = ValidationEvidence(available=False)
        except Exception as exc:
            raise EngineeringEvidenceReadError(
                "persisted validation evidence could not be read"
            ) from exc
        else:
            result = validation_execution.result
            validation = ValidationEvidence(
                available=True,
                validation_id=result.validation_id,
                outcome=result.outcome.value,
                policy_id=result.validation_policy_id,
                policy_version=result.validation_policy_version,
                technically_ready=result.summary.technically_ready,
                finding_codes=tuple(finding.code for finding in result.findings),
            )
        try:
            processing_execution = await self._processing.get_latest_processing(
                inspection_id
            )
        except ProcessingExecutionResultNotFoundError:
            processing = ProcessingEvidence(available=False)
        except Exception as exc:
            raise EngineeringEvidenceReadError(
                "persisted processing evidence could not be read"
            ) from exc
        else:
            finding_codes = tuple(
                finding.code
                for finding in processing_execution.preprocessing.findings
            )
            if processing_execution.inference is not None:
                finding_codes += tuple(
                    finding.code
                    for finding in processing_execution.inference.findings
                )
            processing = ProcessingEvidence(
                available=True,
                processing_run_id=processing_execution.processing_run_id,
                processing_status=processing_execution.processing_status.value,
                preprocessing_outcome=processing_execution.preprocessing_outcome,
                mock_decision=processing_execution.mock_decision,
                production_approved=processing_execution.production_approved,
                synthetic_input_verified=(
                    processing_execution.synthetic_input_verified
                ),
                finding_codes=finding_codes,
            )
        return validation, processing
