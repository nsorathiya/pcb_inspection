from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from app.db.models import ArtifactType
from app.services.inspection_validation.findings import FindingFactory
from app.services.inspection_validation.interfaces import (
    ArtifactTechnicalSummary,
    DimensionRelationship,
    FindingSeverity,
    InspectionValidationPolicy,
    ReadabilityStatus,
    StoredArtifactReference,
    ValidationFinding,
)

_SUPPORTED_RGB_COLOR_MODES = {"GRAY", "RGB", "RGBA"}


class ContractValidationPolicyEvaluator:
    def __init__(self, findings: FindingFactory) -> None:
        self._findings = findings

    def evaluate_policy(
        self,
        policy: InspectionValidationPolicy,
        artifacts: Sequence[ArtifactTechnicalSummary],
        *,
        registered_artifacts: Sequence[StoredArtifactReference] = (),
        registration_evidence_available: bool = False,
    ) -> Sequence[ValidationFinding]:
        result: list[ValidationFinding] = []
        by_type = {item.artifact_type: item for item in artifacts}
        counts = {kind: sum(item.artifact_type is kind for item in registered_artifacts) for kind in ArtifactType}
        rgb = by_type.get(ArtifactType.RGB_RAW)
        height = by_type.get(ArtifactType.HEIGHT_RAW)

        if rgb is not None and rgb.readability_status is ReadabilityStatus.READABLE:
            result.extend(self._rgb_findings(policy, rgb))
        if height is not None and height.readability_status is ReadabilityStatus.READABLE:
            result.extend(self._height_findings(policy, height))
        if rgb is not None and height is not None and rgb.readability_status is ReadabilityStatus.READABLE and height.readability_status is ReadabilityStatus.READABLE:
            same = (rgb.width, rgb.height) == (height.width, height.height)
            if policy.dimension_relationship is DimensionRelationship.SAME_DIMENSIONS_REQUIRED and not same:
                result.append(self._findings.create("DIMENSION_RELATIONSHIP_UNSUPPORTED", field="dimension_relationship", details={"expected": policy.dimension_relationship.value}))
            if policy.dimension_relationship is DimensionRelationship.REGISTERED_TRANSFORM_REQUIRED and not registration_evidence_available:
                result.append(self._findings.create("REGISTRATION_EVIDENCE_MISSING", field="dimension_relationship", blocking=True))

        if policy.require_validity_mask and counts[ArtifactType.VALIDITY_MASK] != 1:
            result.append(self._findings.create("VALIDITY_MASK_MISSING", artifact_type=ArtifactType.VALIDITY_MASK, details={"observed_count": counts[ArtifactType.VALIDITY_MASK]}))
        if policy.require_calibration_artifact and counts[ArtifactType.CALIBRATION] < 1:
            result.append(self._findings.create("CALIBRATION_EVIDENCE_MISSING", artifact_type=ArtifactType.CALIBRATION, blocking=True))
        if policy.require_registration_evidence and not registration_evidence_available and policy.dimension_relationship is not DimensionRelationship.REGISTERED_TRANSFORM_REQUIRED:
            result.append(self._findings.create("REGISTRATION_EVIDENCE_MISSING", blocking=True))

        if policy.warning_as_blocking:
            result = [replace(item, blocking=True) if item.severity is FindingSeverity.WARNING else item for item in result]
        return result

    def _rgb_findings(self, policy: InspectionValidationPolicy, item: ArtifactTechnicalSummary) -> list[ValidationFinding]:
        result = []
        if item.detected_format not in policy.allowed_rgb_formats:
            result.append(self._findings.create("RGB_FORMAT_UNSUPPORTED", artifact_type=ArtifactType.RGB_RAW, field="detected_format", details={"observed": item.detected_format}))
        if not self._dimensions_allowed(policy, item):
            result.append(self._findings.create("RGB_DIMENSIONS_INVALID", artifact_type=ArtifactType.RGB_RAW, field="dimensions", details={"width": item.width, "height": item.height}))
        if item.channels not in policy.allowed_rgb_channels:
            result.append(self._findings.create("RGB_CHANNELS_UNSUPPORTED", artifact_type=ArtifactType.RGB_RAW, field="channels", details={"observed": item.channels}))
        if item.bit_depth not in policy.allowed_rgb_bit_depths:
            result.append(self._findings.create("RGB_BIT_DEPTH_UNSUPPORTED", artifact_type=ArtifactType.RGB_RAW, field="bit_depth", details={"observed": item.bit_depth}))
        if item.color_mode not in _SUPPORTED_RGB_COLOR_MODES:
            result.append(self._findings.create("RGB_COLOR_MODE_UNSUPPORTED", artifact_type=ArtifactType.RGB_RAW, field="color_mode", details={"observed": item.color_mode}))
        return result

    def _height_findings(self, policy: InspectionValidationPolicy, item: ArtifactTechnicalSummary) -> list[ValidationFinding]:
        result = []
        if item.detected_format not in policy.allowed_height_formats:
            result.append(self._findings.create("HEIGHT_FORMAT_UNSUPPORTED", artifact_type=ArtifactType.HEIGHT_RAW, field="detected_format", details={"observed": item.detected_format}))
        if not self._dimensions_allowed(policy, item):
            result.append(self._findings.create("HEIGHT_DIMENSIONS_INVALID", artifact_type=ArtifactType.HEIGHT_RAW, field="dimensions", details={"width": item.width, "height": item.height}))
        if policy.require_single_channel_height and item.channels != 1:
            result.append(self._findings.create("HEIGHT_NOT_SINGLE_CHANNEL", artifact_type=ArtifactType.HEIGHT_RAW, field="channels", details={"observed": item.channels}))
            return result
        if item.bit_depth is not None and item.bit_depth < policy.minimum_height_bit_depth:
            result.append(self._findings.create("HEIGHT_BIT_DEPTH_TOO_LOW", artifact_type=ArtifactType.HEIGHT_RAW, field="bit_depth", details={"minimum": policy.minimum_height_bit_depth, "observed": item.bit_depth}))
            return result
        native_type = item.observed_storage_data_type or item.storage_data_type
        if native_type not in policy.allowed_height_storage_types:
            result.append(self._findings.create("HEIGHT_STORAGE_TYPE_UNSUPPORTED", artifact_type=ArtifactType.HEIGHT_RAW, field="storage_data_type", details={"observed": native_type}))
        return result

    @staticmethod
    def _dimensions_allowed(policy: InspectionValidationPolicy, item: ArtifactTechnicalSummary) -> bool:
        return item.width is not None and item.height is not None and policy.minimum_width <= item.width <= policy.maximum_width and policy.minimum_height <= item.height <= policy.maximum_height
