"""Typed, execution-free preprocessing contract models and semantic checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Any, Mapping, Sequence


class PreprocessingOutcome(str, Enum):
    SUCCEEDED = "PREPROCESSING_SUCCEEDED"
    FAILED = "PREPROCESSING_FAILED"
    ERROR = "PREPROCESSING_ERROR"


class FindingSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class FindingCategory(str, Enum):
    PREREQUISITE = "PREREQUISITE"
    POLICY = "POLICY"
    RGB = "RGB"
    HEIGHT = "HEIGHT"
    REGISTRATION = "REGISTRATION"
    OUTPUT = "OUTPUT"
    INTERNAL = "INTERNAL"


class OutputLayout(str, Enum):
    HW = "HW"
    HWC = "HWC"
    CHW = "CHW"
    NCHW = "NCHW"


class OutputDataType(str, Enum):
    UINT8 = "uint8"
    UINT16 = "uint16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class ROIMode(str, Enum):
    FULL_FRAME = "FULL_FRAME"
    STATIC_RECTANGLE = "STATIC_RECTANGLE"
    RECIPE_DEFINED = "RECIPE_DEFINED"


class ResizeMode(str, Enum):
    NONE = "NONE"
    RESIZE = "RESIZE"
    LETTERBOX = "LETTERBOX"
    CENTER_CROP = "CENTER_CROP"


class InterpolationMode(str, Enum):
    NEAREST = "NEAREST"
    BILINEAR = "BILINEAR"
    BICUBIC = "BICUBIC"


class RGBNormalizationMode(str, Enum):
    NONE = "NONE"
    UNIT_RANGE = "UNIT_RANGE"
    MEAN_STD = "MEAN_STD"


class HeightScalingMode(str, Enum):
    NONE = "NONE"
    DECLARED_PHYSICAL_SCALE = "DECLARED_PHYSICAL_SCALE"
    MIN_MAX = "MIN_MAX"
    STANDARD_SCORE = "STANDARD_SCORE"


class InvalidValueHandling(str, Enum):
    REJECT = "REJECT"
    MASK = "MASK"
    REPLACE_WITH_CONSTANT = "REPLACE_WITH_CONSTANT"
    PRESERVE_NAN = "PRESERVE_NAN"


class RegistrationMode(str, Enum):
    NOT_PERFORMED = "NOT_PERFORMED"
    USE_DECLARED_TRANSFORM = "USE_DECLARED_TRANSFORM"
    SYNTHETIC_IDENTITY_ONLY = "SYNTHETIC_IDENTITY_ONLY"


@dataclass(frozen=True)
class ROIRectangle:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PreprocessingPrerequisites:
    required_inspection_status: str
    required_validation_outcome: str
    accepted_rgb_formats: tuple[str, ...]
    accepted_height_formats: tuple[str, ...]
    accepted_rgb_channels: tuple[int, ...]
    accepted_rgb_bit_depths: tuple[int, ...]
    accepted_height_storage_types: tuple[str, ...]


@dataclass(frozen=True)
class RGBPreprocessingPolicy:
    roi_mode: ROIMode
    roi_rectangle: ROIRectangle | None
    resize_mode: ResizeMode
    target_width: int | None
    target_height: int | None
    output_channels: int
    output_data_type: OutputDataType
    output_layout: OutputLayout
    normalization_mode: RGBNormalizationMode
    normalization_parameters: Mapping[str, tuple[float, ...]]
    preserve_aspect_ratio: bool
    interpolation_mode: InterpolationMode


@dataclass(frozen=True)
class HeightPreprocessingPolicy:
    roi_mode: ROIMode
    roi_rectangle: ROIRectangle | None
    resize_mode: ResizeMode
    target_width: int | None
    target_height: int | None
    output_channels: int
    output_data_type: OutputDataType
    output_layout: OutputLayout
    scaling_mode: HeightScalingMode
    scaling_parameters: Mapping[str, str | int | float]
    invalid_value_handling: InvalidValueHandling
    requires_validity_mask_input: bool
    replacement_value: int | float | None
    interpolation_mode: InterpolationMode


@dataclass(frozen=True)
class RegistrationPolicy:
    registration_mode: RegistrationMode
    require_registration_evidence: bool
    output_coordinate_reference: str
    dimension_relationship: str
    transform_source: str | None
    allow_synthetic_identity_transform: bool


@dataclass(frozen=True)
class PreprocessingOutputPolicy:
    require_matching_output_dimensions: bool
    include_validity_mask: bool
    include_registration_mask: bool
    include_safe_summary_statistics: bool


@dataclass(frozen=True)
class PreprocessingSafetyPolicy:
    allow_mock_implementation: bool
    allow_synthetic_input: bool
    allow_real_input: bool
    allow_uncalibrated_height: bool


@dataclass(frozen=True)
class ArtifactInputIdentity:
    artifact_type: str
    sha256: str
    byte_size: int
    detected_format: str
    width: int
    height: int
    channels: int
    bit_depth: int
    storage_data_type: str | None


@dataclass(frozen=True)
class ValidatedInspectionInputs:
    """Metadata-only identities supplied after technical validation passes."""

    inspection_id: str
    validation_id: str
    rgb_input: ArtifactInputIdentity
    height_input: ArtifactInputIdentity
    validity_mask_input: ArtifactInputIdentity | None = None
    calibration_input: ArtifactInputIdentity | None = None
    synthetic_input: bool = False


@dataclass(frozen=True)
class ValidatedArtifactSource:
    """Internal-only association between public identity and a read-only source."""

    identity: ArtifactInputIdentity
    source_path: Path | None


@dataclass(frozen=True)
class ValidatedInspectionInput:
    inspection_id: str
    validation_id: str
    inspection_status: str
    validation_outcome: str | None
    synthetic_input: bool
    rgb: ValidatedArtifactSource
    height: ValidatedArtifactSource
    validity_mask: ValidatedArtifactSource | None = None
    calibration: ValidatedArtifactSource | None = None
    registration_evidence: str | None = None


@dataclass(frozen=True)
class PreprocessedBufferDescriptor:
    """Framework-neutral inference-boundary descriptor; it never contains bytes."""

    shape: tuple[int, ...]
    layout: OutputLayout
    data_type: OutputDataType
    channel_count: int
    width: int
    height: int
    byte_order: str
    contiguous: bool
    finite_values_verified: bool
    source_artifact_sha256: str


@dataclass(frozen=True)
class InternalPreprocessedBuffer:
    """Immutable standard-library float buffer kept outside public results."""

    descriptor: PreprocessedBufferDescriptor
    data: bytes
    element_count: int
    byte_size: int
    content_sha256: str

    @classmethod
    def from_bytes(
        cls, descriptor: PreprocessedBufferDescriptor, data: bytes
    ) -> "InternalPreprocessedBuffer":
        immutable = bytes(data)
        return cls(
            descriptor=descriptor,
            data=immutable,
            element_count=prod(descriptor.shape),
            byte_size=len(immutable),
            content_sha256=sha256(immutable).hexdigest(),
        )


@dataclass(frozen=True)
class BranchPreprocessingOutput:
    descriptor: PreprocessedBufferDescriptor
    roi_mode: str
    safe_statistics: Mapping[str, int | float | None] | None = None


@dataclass(frozen=True)
class RGBPreprocessingOutput(BranchPreprocessingOutput):
    normalization_mode: str = "NONE"


@dataclass(frozen=True)
class HeightPreprocessingOutput(BranchPreprocessingOutput):
    scaling_mode: str = "NONE"
    invalid_value_handling: str = "REJECT"
    physical_unit: str | None = None
    physical_scale_applied: bool = False


@dataclass(frozen=True)
class RGBProcessedBranch:
    buffer: InternalPreprocessedBuffer
    output: RGBPreprocessingOutput


@dataclass(frozen=True)
class HeightProcessedBranch:
    buffer: InternalPreprocessedBuffer
    output: HeightPreprocessingOutput


@dataclass(frozen=True)
class RegistrationProcessingResult:
    registration_mode: str
    registration_status: str
    transform_applied: bool
    transform_reference: str | None
    synthetic_identity: bool
    output_coordinate_reference: str
    registration_warning: str | None


@dataclass(frozen=True)
class PreprocessingFinding:
    code: str
    severity: FindingSeverity
    category: FindingCategory
    message: str
    blocking: bool
    branch: str | None = None
    field: str | None = None
    details: Mapping[str, str | int | float | bool | None] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingSummary:
    total_findings: int
    blocking_findings: int
    warnings: int
    errors: int


@dataclass(frozen=True)
class InspectionPreprocessingPolicy:
    """Typed policy document; implementations must advertise supported subsets."""

    contract_version: str
    policy_id: str
    policy_version: str
    name: str
    description: str
    development_only: bool
    production_approved: bool
    prerequisites: PreprocessingPrerequisites
    rgb: RGBPreprocessingPolicy
    height: HeightPreprocessingPolicy
    registration: RegistrationPolicy
    output: PreprocessingOutputPolicy
    safety: PreprocessingSafetyPolicy


@dataclass(frozen=True)
class InspectionPreprocessingResult:
    contract_version: str
    preprocessing_id: str
    inspection_id: str
    validation_id: str
    policy_id: str
    policy_version: str
    implementation_id: str
    implementation_version: str
    outcome: PreprocessingOutcome
    started_at: datetime
    completed_at: datetime
    synthetic_input: bool
    mock_implementation: bool
    production_approved: bool
    rgb_input: ArtifactInputIdentity
    height_input: ArtifactInputIdentity
    rgb_output: RGBPreprocessingOutput | None
    height_output: HeightPreprocessingOutput | None
    registration: RegistrationProcessingResult
    findings: tuple[PreprocessingFinding, ...]
    summary: PreprocessingSummary
    validity_mask_input: ArtifactInputIdentity | None = None
    calibration_input: ArtifactInputIdentity | None = None

    def to_dict(self) -> dict[str, Any]:
        return preprocessing_result_to_dict(self)


@dataclass(frozen=True)
class SyntheticPreprocessingExecution:
    result: InspectionPreprocessingResult
    rgb_buffer: InternalPreprocessedBuffer | None
    height_buffer: InternalPreprocessedBuffer | None
    validity_mask_buffer: InternalPreprocessedBuffer | None = None
    registration_mask_buffer: InternalPreprocessedBuffer | None = None


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("preprocessing timestamps must include timezone information")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact_document(value: ArtifactInputIdentity) -> dict[str, Any]:
    return {
        "artifact_type": value.artifact_type,
        "sha256": value.sha256,
        "byte_size": value.byte_size,
        "detected_format": value.detected_format,
        "width": value.width,
        "height": value.height,
        "channels": value.channels,
        "bit_depth": value.bit_depth,
        "storage_data_type": value.storage_data_type,
    }


def _descriptor_document(value: PreprocessedBufferDescriptor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "layout": value.layout.value,
        "data_type": value.data_type.value,
        "channel_count": value.channel_count,
        "width": value.width,
        "height": value.height,
        "byte_order": value.byte_order,
        "contiguous": value.contiguous,
        "finite_values_verified": value.finite_values_verified,
        "source_artifact_sha256": value.source_artifact_sha256,
    }


def _roi_document(value: BranchPreprocessingOutput) -> dict[str, Any]:
    descriptor = value.descriptor
    return {
        "mode": value.roi_mode,
        "x": 0,
        "y": 0,
        "width": descriptor.width,
        "height": descriptor.height,
    }


def _rgb_output_document(value: RGBPreprocessingOutput | None) -> dict[str, Any] | None:
    if value is None:
        return None
    result = _descriptor_document(value.descriptor)
    result.update(
        normalization_mode=value.normalization_mode,
        roi=_roi_document(value),
    )
    if value.safe_statistics is not None:
        result["safe_statistics"] = dict(value.safe_statistics)
    return result


def _height_output_document(value: HeightPreprocessingOutput | None) -> dict[str, Any] | None:
    if value is None:
        return None
    result = _descriptor_document(value.descriptor)
    result.update(
        scaling_mode=value.scaling_mode,
        invalid_value_handling=value.invalid_value_handling,
        physical_unit=value.physical_unit,
        physical_scale_applied=value.physical_scale_applied,
        roi=_roi_document(value),
    )
    if value.safe_statistics is not None:
        result["safe_statistics"] = dict(value.safe_statistics)
    return result


def _finding_document(value: PreprocessingFinding) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": value.code,
        "severity": value.severity.value,
        "category": value.category.value,
        "message": value.message,
        "blocking": value.blocking,
    }
    if value.branch is not None:
        result["branch"] = value.branch
    if value.field is not None:
        result["field"] = value.field
    if value.details:
        result["details"] = dict(value.details)
    return result


def preprocessing_result_to_dict(value: InspectionPreprocessingResult) -> dict[str, Any]:
    registration = value.registration
    result: dict[str, Any] = {
        "contract_version": value.contract_version,
        "preprocessing_id": value.preprocessing_id,
        "inspection_id": value.inspection_id,
        "validation_id": value.validation_id,
        "policy_id": value.policy_id,
        "policy_version": value.policy_version,
        "implementation_id": value.implementation_id,
        "implementation_version": value.implementation_version,
        "outcome": value.outcome.value,
        "started_at": _timestamp(value.started_at),
        "completed_at": _timestamp(value.completed_at),
        "synthetic_input": value.synthetic_input,
        "mock_implementation": value.mock_implementation,
        "production_approved": value.production_approved,
        "rgb_input": _artifact_document(value.rgb_input),
        "height_input": _artifact_document(value.height_input),
        "rgb_output": _rgb_output_document(value.rgb_output),
        "height_output": _height_output_document(value.height_output),
        "registration": {
            "registration_mode": registration.registration_mode,
            "registration_status": registration.registration_status,
            "transform_applied": registration.transform_applied,
            "transform_reference": registration.transform_reference,
            "synthetic_identity": registration.synthetic_identity,
            "output_coordinate_reference": registration.output_coordinate_reference,
            "registration_warning": registration.registration_warning,
        },
        "findings": [_finding_document(item) for item in value.findings],
        "summary": {
            "total_findings": value.summary.total_findings,
            "blocking_findings": value.summary.blocking_findings,
            "warnings": value.summary.warnings,
            "errors": value.summary.errors,
        },
    }
    if value.validity_mask_input is not None:
        result["validity_mask_input"] = _artifact_document(value.validity_mask_input)
    if value.calibration_input is not None:
        result["calibration_input"] = _artifact_document(value.calibration_input)
    return result


def preprocessing_result_json(value: InspectionPreprocessingResult) -> str:
    return json.dumps(
        preprocessing_result_to_dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_resize(branch: Mapping[str, Any], name: str) -> None:
    mode = branch["resize_mode"]
    dimensions = (branch["target_width"], branch["target_height"])
    if mode == "NONE":
        _require(dimensions == (None, None), f"{name} NONE resize requires null target dimensions")
    else:
        _require(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in dimensions), f"{name} {mode} requires positive target dimensions")


def _validate_roi(branch: Mapping[str, Any], name: str) -> None:
    rectangle = branch["roi_rectangle"]
    if branch["roi_mode"] == "STATIC_RECTANGLE":
        _require(isinstance(rectangle, Mapping), f"{name} static ROI requires a rectangle")
    elif branch["roi_mode"] == "FULL_FRAME":
        _require(rectangle is None, f"{name} full-frame ROI must not fabricate a rectangle")


def validate_preprocessing_policy_document(document: Mapping[str, Any]) -> None:
    """Apply cross-field rules after JSON Schema validation; never fills defaults."""

    rgb = document["rgb"]
    height = document["height"]
    registration = document["registration"]
    output = document["output"]
    safety = document["safety"]
    development = document["development_only"]
    production = document["production_approved"]

    _require(not (development and production), "development policy cannot claim production approval")
    if production:
        _require(not development, "production approval requires development_only=false")
        _require(not registration["allow_synthetic_identity_transform"], "production policy cannot enable synthetic identity registration")
    _validate_resize(rgb, "RGB")
    _validate_resize(height, "height")
    _validate_roi(rgb, "RGB")
    _validate_roi(height, "height")

    parameters = rgb["normalization_parameters"]
    if rgb["normalization_mode"] == "MEAN_STD":
        means = parameters.get("means")
        deviations = parameters.get("standard_deviations")
        channels = rgb["output_channels"]
        _require(isinstance(means, list) and len(means) == channels, "MEAN_STD requires one mean per output channel")
        _require(isinstance(deviations, list) and len(deviations) == channels and all(value > 0 for value in deviations), "MEAN_STD requires one positive standard deviation per output channel")

    scaling = height["scaling_parameters"]
    if height["scaling_mode"] == "DECLARED_PHYSICAL_SCALE":
        _require(all(isinstance(scaling.get(key), str) and scaling[key] for key in ("physical_unit", "scale_source", "offset_source")), "declared physical scaling requires unit, scale source, and offset source")
    if height["invalid_value_handling"] == "MASK":
        _require(height["requires_validity_mask_input"] is True and output["include_validity_mask"] is True, "MASK handling requires a validity-mask input and output")
    if height["invalid_value_handling"] == "REPLACE_WITH_CONSTANT":
        _require(height["replacement_value"] is not None, "replacement handling requires a replacement value")
    if height["invalid_value_handling"] == "PRESERVE_NAN":
        _require(height["output_data_type"] in {"float32", "float64"}, "PRESERVE_NAN requires floating-point output")

    if registration["registration_mode"] == "SYNTHETIC_IDENTITY_ONLY":
        _require(development is True, "synthetic identity registration requires development_only=true")
        _require(registration["allow_synthetic_identity_transform"] is True, "synthetic identity registration must be explicitly allowed")
        _require(safety["allow_synthetic_input"] is True and safety["allow_real_input"] is False, "synthetic identity registration is limited to synthetic input")
    if registration["registration_mode"] == "USE_DECLARED_TRANSFORM":
        _require(bool(registration["transform_source"]), "declared transform mode requires a transform source")
    _require(isinstance(output["require_matching_output_dimensions"], bool), "matching output dimensions must be explicit")


def finding_sort_key(finding: Mapping[str, Any], catalogue: Mapping[str, Any]) -> tuple[Any, ...]:
    orders = {entry["code"]: entry["order"] for entry in catalogue["findings"]}
    return (
        orders[finding["code"]],
        finding.get("branch", ""),
        finding.get("field", ""),
        finding["code"],
        finding["message"],
        json.dumps(finding.get("details", {}), sort_keys=True, separators=(",", ":")),
    )


def validate_preprocessing_result_document(
    document: Mapping[str, Any], catalogue: Mapping[str, Any]
) -> None:
    """Check catalogue consistency, deterministic ordering, and derived counts."""

    definitions = {entry["code"]: entry for entry in catalogue["findings"]}
    findings: Sequence[Mapping[str, Any]] = document["findings"]
    for finding in findings:
        definition = definitions.get(finding["code"])
        _require(definition is not None, "preprocessing result contains an unknown finding code")
        _require(finding["severity"] == definition["severity"], "finding severity disagrees with catalogue")
        _require(finding["category"] == definition["category"], "finding category disagrees with catalogue")
    _require(list(findings) == sorted(findings, key=lambda item: finding_sort_key(item, catalogue)), "findings are not in deterministic catalogue order")
    summary = document["summary"]
    expected = {
        "total_findings": len(findings),
        "blocking_findings": sum(item["blocking"] is True for item in findings),
        "warnings": sum(item["severity"] == "WARNING" for item in findings),
        "errors": sum(item["severity"] == "ERROR" for item in findings),
    }
    _require(all(summary[key] == value for key, value in expected.items()), "preprocessing summary counts do not match findings")
    outcome = document["outcome"]
    if outcome == "PREPROCESSING_SUCCEEDED":
        _require(expected["blocking_findings"] == 0, "successful preprocessing cannot contain blocking findings")
        _require(document["rgb_output"] is not None and document["height_output"] is not None, "successful preprocessing requires both branch outputs")
    elif outcome == "PREPROCESSING_FAILED":
        _require(expected["blocking_findings"] > 0, "failed preprocessing requires a blocking finding")
    else:
        _require(any(item["code"] == "PREPROCESSING_INTERNAL_ERROR" and item["blocking"] for item in findings), "preprocessing error requires a blocking internal-error finding")
