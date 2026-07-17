from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
from uuid import UUID, uuid4

from app.services.inspection_preprocessing.buffers import validate_internal_buffer
from app.services.inspection_preprocessing.exceptions import PreprocessingKnownFailure
from app.services.inspection_preprocessing.findings import PreprocessingFindingFactory
from app.services.inspection_preprocessing.height_processor import SyntheticHeightPreprocessor
from app.services.inspection_preprocessing.models import (
    FindingSeverity,
    HeightProcessedBranch,
    InspectionPreprocessingPolicy,
    InspectionPreprocessingResult,
    PreprocessingOutcome,
    PreprocessingSummary,
    RGBProcessedBranch,
    RegistrationMode,
    RegistrationProcessingResult,
    SyntheticPreprocessingExecution,
    ValidatedInspectionInput,
)
from app.services.inspection_preprocessing.registration import (
    SyntheticIdentityRegistrationProcessor,
)
from app.services.inspection_preprocessing.rgb_processor import SyntheticRGBPreprocessor

PREPROCESSING_CONTRACT_VERSION = "pcb-aoi-inspection-preprocessing/1.0"
IMPLEMENTATION_ID = "synthetic-mock-preprocessor"
IMPLEMENTATION_VERSION = "1.0.0"


def _canonical_uuid(value: str, field: str) -> str:
    try:
        canonical = str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a canonical UUID string") from exc
    if canonical != value:
        raise ValueError(f"{field} must be a canonical UUID string")
    return value


class SyntheticInspectionPreprocessingService:
    """Development-only, in-memory executor for validated generated fixtures."""

    def __init__(
        self,
        *,
        rgb_processor: SyntheticRGBPreprocessor | None = None,
        height_processor: SyntheticHeightPreprocessor | None = None,
        registration_processor: SyntheticIdentityRegistrationProcessor | None = None,
        findings: PreprocessingFindingFactory | None = None,
        clock: Callable[[], datetime] | None = None,
        preprocessing_id_generator: Callable[[], str] | None = None,
        implementation_id: str = IMPLEMENTATION_ID,
        implementation_version: str = IMPLEMENTATION_VERSION,
    ) -> None:
        self._rgb = rgb_processor or SyntheticRGBPreprocessor()
        self._height = height_processor or SyntheticHeightPreprocessor()
        self._registration = registration_processor or SyntheticIdentityRegistrationProcessor()
        self._findings = findings or PreprocessingFindingFactory()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._preprocessing_id = preprocessing_id_generator or (lambda: str(uuid4()))
        self._implementation_id = implementation_id
        self._implementation_version = implementation_version

    async def preprocess_inspection(
        self,
        validated_input: ValidatedInspectionInput,
        policy: InspectionPreprocessingPolicy,
    ) -> SyntheticPreprocessingExecution:
        preprocessing_id = _canonical_uuid(
            self._preprocessing_id(), "preprocessing_id"
        )
        _canonical_uuid(validated_input.inspection_id, "inspection_id")
        _canonical_uuid(validated_input.validation_id, "validation_id")
        started_at = self._clock()
        rgb: RGBProcessedBranch | None = None
        height: HeightProcessedBranch | None = None
        findings = []
        registration = self._not_performed_registration(policy)
        try:
            self._check_prerequisites(validated_input, policy)
            rgb = await self._rgb.preprocess_rgb(validated_input, policy)
            validate_internal_buffer(
                rgb.buffer,
                channel_count=3,
                width=validated_input.rgb.identity.width,
                height=validated_input.rgb.identity.height,
            )
            height = await self._height.preprocess_height(validated_input, policy)
            validate_internal_buffer(
                height.buffer,
                channel_count=1,
                width=validated_input.height.identity.width,
                height=validated_input.height.identity.height,
            )
            registration = await self._registration.coordinate_registration(
                rgb, height, validated_input, policy
            )
            findings.append(
                self._findings.create(
                    "SYNTHETIC_IDENTITY_REGISTRATION_USED",
                    branch="REGISTRATION",
                )
            )
            outcome = PreprocessingOutcome.SUCCEEDED
        except PreprocessingKnownFailure as exc:
            findings.append(
                self._findings.create(
                    exc.finding_code,
                    branch=exc.branch,
                    field=exc.field,
                    details=exc.details,
                )
            )
            outcome = PreprocessingOutcome.FAILED
        except Exception:
            findings = [self._findings.create("PREPROCESSING_INTERNAL_ERROR")]
            outcome = PreprocessingOutcome.ERROR
            registration = RegistrationProcessingResult(
                registration_mode=policy.registration.registration_mode.value,
                registration_status="FAILED",
                transform_applied=False,
                transform_reference=None,
                synthetic_identity=False,
                output_coordinate_reference=policy.registration.output_coordinate_reference,
                registration_warning=None,
            )

        ordered = self._findings.sort(findings)
        completed_at = self._clock()
        if completed_at < started_at:
            completed_at = started_at
        summary = PreprocessingSummary(
            total_findings=len(ordered),
            blocking_findings=sum(item.blocking for item in ordered),
            warnings=sum(item.severity is FindingSeverity.WARNING for item in ordered),
            errors=sum(item.severity is FindingSeverity.ERROR for item in ordered),
        )
        result = InspectionPreprocessingResult(
            contract_version=PREPROCESSING_CONTRACT_VERSION,
            preprocessing_id=preprocessing_id,
            inspection_id=validated_input.inspection_id,
            validation_id=validated_input.validation_id,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            implementation_id=self._implementation_id,
            implementation_version=self._implementation_version,
            outcome=outcome,
            started_at=started_at,
            completed_at=completed_at,
            synthetic_input=validated_input.synthetic_input,
            mock_implementation=True,
            production_approved=False,
            rgb_input=validated_input.rgb.identity,
            height_input=validated_input.height.identity,
            rgb_output=None if rgb is None else rgb.output,
            height_output=None if height is None else height.output,
            registration=registration,
            findings=ordered,
            summary=summary,
            validity_mask_input=(
                None
                if validated_input.validity_mask is None
                else validated_input.validity_mask.identity
            ),
            calibration_input=(
                None
                if validated_input.calibration is None
                else validated_input.calibration.identity
            ),
        )
        return SyntheticPreprocessingExecution(
            result=result,
            rgb_buffer=None if rgb is None else rgb.buffer,
            height_buffer=None if height is None else height.buffer,
        )

    def _check_prerequisites(
        self,
        inputs: ValidatedInspectionInput,
        policy: InspectionPreprocessingPolicy,
    ) -> None:
        if inputs.inspection_status != policy.prerequisites.required_inspection_status:
            raise PreprocessingKnownFailure("INSPECTION_NOT_READY")
        if inputs.validation_outcome is None:
            raise PreprocessingKnownFailure("VALIDATION_RESULT_REQUIRED")
        if inputs.validation_outcome != policy.prerequisites.required_validation_outcome:
            raise PreprocessingKnownFailure("VALIDATION_NOT_PASSED")
        if (
            not inputs.synthetic_input
            or not policy.safety.allow_synthetic_input
            or policy.safety.allow_real_input
            or not policy.safety.allow_mock_implementation
            or not policy.development_only
            or policy.production_approved
            or policy.contract_version
            != "pcb-aoi-inspection-preprocessing-policy/1.0"
            or policy.registration.registration_mode
            is not RegistrationMode.SYNTHETIC_IDENTITY_ONLY
        ):
            raise PreprocessingKnownFailure("PREPROCESSING_POLICY_INVALID")
        if not self._implementation_id.startswith("synthetic-"):
            raise PreprocessingKnownFailure("PREPROCESSING_POLICY_INVALID")

    @staticmethod
    def _not_performed_registration(
        policy: InspectionPreprocessingPolicy,
    ) -> RegistrationProcessingResult:
        return RegistrationProcessingResult(
            registration_mode=policy.registration.registration_mode.value,
            registration_status="NOT_PERFORMED",
            transform_applied=False,
            transform_reference=None,
            synthetic_identity=False,
            output_coordinate_reference=policy.registration.output_coordinate_reference,
            registration_warning=None,
        )
