from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable, Sequence
from uuid import UUID, uuid4

from app.db.models import ArtifactType
from app.services.inspection_validation.exceptions import PolicyLoadError
from app.services.inspection_validation.findings import FindingFactory
from app.services.inspection_validation.interfaces import (
    ArtifactTechnicalSummary,
    FilesystemIntegrityInspector,
    FindingSeverity,
    InspectionValidationPolicy,
    InspectionValidationResult,
    NativeFormatInspector,
    ReadabilityStatus,
    RetrievedInspectionArtifacts,
    StoredArtifactReference,
    ValidationArtifactRetriever,
    ValidationFinding,
    ValidationOutcome,
    ValidationPolicyEvaluator,
    ValidationSummary,
)
from app.services.inspection_validation.policy_loader import ValidationPolicyLoader

VALIDATION_CONTRACT_VERSION = "pcb-aoi-inspection-validation/1.0"
DEFAULT_VALIDATOR_VERSION = "1.0.0"


def _canonical_uuid(value: str, field: str) -> str:
    try:
        canonical = str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a canonical UUID string") from exc
    if canonical != value:
        raise ValueError(f"{field} must be a canonical UUID string")
    return value


def _blank(artifact_type: ArtifactType, status: ReadabilityStatus, reference: StoredArtifactReference | None = None) -> ArtifactTechnicalSummary:
    return ArtifactTechnicalSummary(
        artifact_type=artifact_type,
        sha256=None if reference is None else reference.registered_sha256,
        byte_size=None if reference is None else reference.registered_byte_size,
        declared_media_type=None if reference is None else reference.declared_media_type,
        detected_format=None, width=None, height=None, channels=None, bit_depth=None,
        storage_data_type=None, readability_status=status,
    )


class InspectionValidationService:
    def __init__(
        self,
        artifacts: ValidationArtifactRetriever,
        integrity: FilesystemIntegrityInspector,
        formats: NativeFormatInspector,
        evaluator: ValidationPolicyEvaluator,
        findings: FindingFactory,
        *,
        clock: Callable[[], datetime] | None = None,
        validation_id_generator: Callable[[], str] | None = None,
        validator_version: str = DEFAULT_VALIDATOR_VERSION,
        policy_loader: ValidationPolicyLoader | None = None,
    ) -> None:
        self._artifacts = artifacts
        self._integrity = integrity
        self._formats = formats
        self._evaluator = evaluator
        self._findings = findings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._validation_id = validation_id_generator or (lambda: str(uuid4()))
        self._validator_version = validator_version
        self._policy_loader = policy_loader

    async def validate_inspection_pair(self, inspection_id: str, policy: InspectionValidationPolicy) -> InspectionValidationResult:
        _canonical_uuid(inspection_id, "inspection_id")
        validation_id = _canonical_uuid(self._validation_id(), "validation_id")
        started_at = self._clock()
        context: RetrievedInspectionArtifacts | None = None
        rgb = _blank(ArtifactType.RGB_RAW, ReadabilityStatus.UNINSPECTED)
        height = _blank(ArtifactType.HEIGHT_RAW, ReadabilityStatus.UNINSPECTED)
        try:
            context = await self._artifacts.get_validation_artifacts(inspection_id)
            references = tuple(context.artifacts)
            if context.inspection_id != inspection_id or any(item.inspection_id != inspection_id for item in references):
                raise RuntimeError("artifact retrieval returned a different inspection identity")
            findings: list[ValidationFinding] = []
            rgb_refs = tuple(item for item in references if item.artifact_type is ArtifactType.RGB_RAW)
            height_refs = tuple(item for item in references if item.artifact_type is ArtifactType.HEIGHT_RAW)
            findings.extend(self._pair_findings(rgb_refs, height_refs))

            rgb, rgb_findings = await self._inspect_required(rgb_refs, ArtifactType.RGB_RAW)
            height, height_findings = await self._inspect_required(height_refs, ArtifactType.HEIGHT_RAW)
            findings.extend(rgb_findings)
            findings.extend(height_findings)

            evidence_findings = await self._inspect_evidence(references, policy, height)
            findings.extend(evidence_findings)
            findings.extend(self._evaluator.evaluate_policy(
                policy, (rgb, height), registered_artifacts=references,
                registration_evidence_available=context.registration_evidence_available,
            ))
            findings = self._deduplicate(findings)
            if policy.warning_as_blocking:
                findings = [replace(item, blocking=True) if item.severity is FindingSeverity.WARNING else item for item in findings]
            ordered = self._findings.sort(findings)
            outcome = ValidationOutcome.VALIDATION_FAILED if any(item.blocking for item in ordered) else ValidationOutcome.VALIDATION_PASSED
        except Exception:
            ordered = (self._findings.create("VALIDATOR_INTERNAL_ERROR"),)
            outcome = ValidationOutcome.VALIDATION_ERROR
        completed_at = self._clock()
        if completed_at < started_at:
            completed_at = started_at
        return self._result(
            validation_id, inspection_id, policy.policy_id, policy.policy_version,
            outcome, started_at, completed_at, rgb, height, ordered,
            synthetic_example=False if context is None else context.synthetic_example,
        )

    async def validate_registered_policy(self, inspection_id: str, policy_id: str, policy_version: str) -> InspectionValidationResult:
        _canonical_uuid(inspection_id, "inspection_id")
        if self._policy_loader is None:
            raise RuntimeError("no validation policy loader is configured")
        try:
            policy = self._policy_loader.load(policy_id, policy_version)
        except PolicyLoadError as exc:
            validation_id = _canonical_uuid(self._validation_id(), "validation_id")
            started = self._clock()
            completed = self._clock()
            finding = self._findings.create(exc.finding_code, field="validation_policy")
            return self._result(
                validation_id, inspection_id, policy_id, policy_version,
                ValidationOutcome.VALIDATION_FAILED, started, max(started, completed),
                _blank(ArtifactType.RGB_RAW, ReadabilityStatus.UNINSPECTED),
                _blank(ArtifactType.HEIGHT_RAW, ReadabilityStatus.UNINSPECTED),
                (finding,), synthetic_example=False,
            )
        return await self.validate_inspection_pair(inspection_id, policy)

    async def _inspect_required(self, references: Sequence[StoredArtifactReference], artifact_type: ArtifactType) -> tuple[ArtifactTechnicalSummary, tuple[ValidationFinding, ...]]:
        if len(references) == 0:
            return _blank(artifact_type, ReadabilityStatus.MISSING), ()
        if len(references) != 1:
            return _blank(artifact_type, ReadabilityStatus.UNINSPECTED), ()
        reference = references[0]
        integrity = await self._integrity.inspect_integrity(reference)
        inspected = await self._formats.inspect_native_format(reference, integrity)
        return inspected.summary, inspected.findings

    async def _inspect_evidence(self, references: Sequence[StoredArtifactReference], policy: InspectionValidationPolicy, height: ArtifactTechnicalSummary) -> tuple[ValidationFinding, ...]:
        findings: list[ValidationFinding] = []
        for reference in references:
            if reference.artifact_type not in {ArtifactType.VALIDITY_MASK, ArtifactType.CALIBRATION}:
                continue
            integrity = await self._integrity.inspect_integrity(reference)
            if integrity.failure_code is not None:
                findings.append(self._findings.create(integrity.failure_code, artifact_type=reference.artifact_type))
            elif reference.artifact_type is ArtifactType.VALIDITY_MASK and policy.require_validity_mask:
                inspector = getattr(self._formats, "inspect_validity_mask", None)
                if inspector is None:
                    raise RuntimeError("format inspector has no validity-mask adapter")
                findings.extend(await inspector(reference, integrity, height))
        return tuple(findings)

    def _pair_findings(self, rgb: Sequence[StoredArtifactReference], height: Sequence[StoredArtifactReference]) -> list[ValidationFinding]:
        result = []
        if not rgb:
            result.append(self._findings.create("RGB_RAW_MISSING", artifact_type=ArtifactType.RGB_RAW))
        elif len(rgb) > 1:
            result.append(self._findings.create("DUPLICATE_RGB_RAW", artifact_type=ArtifactType.RGB_RAW, details={"observed_count": len(rgb)}))
        if not height:
            result.append(self._findings.create("HEIGHT_RAW_MISSING", artifact_type=ArtifactType.HEIGHT_RAW))
        elif len(height) > 1:
            result.append(self._findings.create("DUPLICATE_HEIGHT_RAW", artifact_type=ArtifactType.HEIGHT_RAW, details={"observed_count": len(height)}))
        if len(rgb) != 1 or len(height) != 1:
            result.append(self._findings.create("INCOMPLETE_RAW_PAIR", details={"rgb_count": len(rgb), "height_count": len(height)}))
        return result

    @staticmethod
    def _deduplicate(findings: Sequence[ValidationFinding]) -> list[ValidationFinding]:
        if any(item.code == "HEIGHT_COLORIZED_PREVIEW_REJECTED" for item in findings):
            findings = tuple(item for item in findings if item.code != "HEIGHT_NOT_SINGLE_CHANNEL")
        result = []
        seen = set()
        for finding in findings:
            key = (finding.code, finding.artifact_type, finding.field)
            if key not in seen:
                seen.add(key)
                result.append(finding)
        return result

    def _result(self, validation_id: str, inspection_id: str, policy_id: str, policy_version: str, outcome: ValidationOutcome, started: datetime, completed: datetime, rgb: ArtifactTechnicalSummary, height: ArtifactTechnicalSummary, findings: Sequence[ValidationFinding], *, synthetic_example: bool) -> InspectionValidationResult:
        findings = tuple(findings)
        summary = ValidationSummary(
            finding_count=len(findings),
            info_count=sum(item.severity is FindingSeverity.INFO for item in findings),
            warning_count=sum(item.severity is FindingSeverity.WARNING for item in findings),
            error_count=sum(item.severity is FindingSeverity.ERROR for item in findings),
            blocking_count=sum(item.blocking for item in findings),
            technically_ready=outcome is ValidationOutcome.VALIDATION_PASSED,
            synthetic_example=synthetic_example,
        )
        return InspectionValidationResult(
            contract_version=VALIDATION_CONTRACT_VERSION,
            validation_id=validation_id,
            inspection_id=inspection_id,
            validation_policy_id=policy_id,
            validation_policy_version=policy_version,
            outcome=outcome,
            started_at=started,
            completed_at=completed,
            validator_version=self._validator_version,
            rgb_artifact=rgb,
            height_artifact=height,
            findings=findings,
            summary=summary,
        )
