import json
from datetime import datetime
from pathlib import Path

from app.db.models import ArtifactType
from app.services.inspection_inference.models import (
    InferenceExecutionOutcome,
    InferenceFinding,
    InferenceFindingCategory,
    InferenceFindingSeverity,
    InferenceInputIdentity,
    InferenceSummary,
    InspectionInferenceResult,
    MockDecision,
)
from app.services.inspection_preprocessing.models import (
    ArtifactInputIdentity,
    FindingCategory as PreprocessingFindingCategory,
    FindingSeverity as PreprocessingFindingSeverity,
    HeightPreprocessingOutput,
    InspectionPreprocessingResult,
    OutputDataType,
    OutputLayout,
    PreprocessedBufferDescriptor,
    PreprocessingFinding,
    PreprocessingOutcome,
    PreprocessingSummary,
    RegistrationProcessingResult,
    RGBPreprocessingOutput,
)
from app.services.inspection_validation.interfaces import (
    ArtifactTechnicalSummary,
    InspectionValidationResult,
    ReadabilityStatus,
    ValidationFinding,
    ValidationSummary,
)
from app.db.validation_types import FindingCategory, FindingSeverity, ValidationOutcome

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "contracts" / "examples"


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def document(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def artifact(value: dict) -> ArtifactInputIdentity:
    return ArtifactInputIdentity(**value)


def descriptor(value: dict) -> PreprocessedBufferDescriptor:
    return PreprocessedBufferDescriptor(
        shape=tuple(value["shape"]), layout=OutputLayout(value["layout"]),
        data_type=OutputDataType(value["data_type"]), channel_count=value["channel_count"],
        width=value["width"], height=value["height"], byte_order=value["byte_order"],
        contiguous=value["contiguous"], finite_values_verified=value["finite_values_verified"],
        source_artifact_sha256=value["source_artifact_sha256"],
    )


def preprocessing_result(
    name: str,
    *,
    inspection_id: str,
    validation_id: str,
    preprocessing_id: str,
) -> InspectionPreprocessingResult:
    value = document(name)

    def branch(item, kind):
        if item is None:
            return None
        common = dict(
            descriptor=descriptor(item), roi_mode=item["roi"]["mode"],
            safe_statistics=item.get("safe_statistics"),
        )
        return (
            RGBPreprocessingOutput(**common, normalization_mode=item["normalization_mode"])
            if kind == "rgb"
            else HeightPreprocessingOutput(
                **common, scaling_mode=item["scaling_mode"],
                invalid_value_handling=item["invalid_value_handling"],
                physical_unit=item["physical_unit"],
                physical_scale_applied=item["physical_scale_applied"],
            )
        )

    findings = tuple(PreprocessingFinding(
        code=item["code"], severity=PreprocessingFindingSeverity(item["severity"]),
        category=PreprocessingFindingCategory(item["category"]), message=item["message"],
        blocking=item["blocking"], branch=item.get("branch"), field=item.get("field"),
        details=item.get("details", {}),
    ) for item in value["findings"])
    return InspectionPreprocessingResult(
        contract_version=value["contract_version"], preprocessing_id=preprocessing_id,
        inspection_id=inspection_id, validation_id=validation_id,
        policy_id=value["policy_id"], policy_version=value["policy_version"],
        implementation_id=value["implementation_id"],
        implementation_version=value["implementation_version"],
        outcome=PreprocessingOutcome(value["outcome"]),
        started_at=timestamp(value["started_at"]), completed_at=timestamp(value["completed_at"]),
        synthetic_input=value["synthetic_input"], mock_implementation=value["mock_implementation"],
        production_approved=value["production_approved"], rgb_input=artifact(value["rgb_input"]),
        height_input=artifact(value["height_input"]),
        validity_mask_input=None if value.get("validity_mask_input") is None else artifact(value["validity_mask_input"]),
        calibration_input=None if value.get("calibration_input") is None else artifact(value["calibration_input"]),
        rgb_output=branch(value["rgb_output"], "rgb"),
        height_output=branch(value["height_output"], "height"),
        registration=RegistrationProcessingResult(**value["registration"]),
        findings=findings, summary=PreprocessingSummary(**value["summary"]),
    )


def inference_result(
    name: str,
    *,
    inspection_id: str,
    validation_id: str,
    preprocessing_id: str,
    inference_id: str,
) -> InspectionInferenceResult:
    value = document(name)

    def identity(item):
        return None if item is None else InferenceInputIdentity(**{**item, "shape": tuple(item["shape"])})

    findings = tuple(InferenceFinding(
        code=item["code"], severity=InferenceFindingSeverity(item["severity"]),
        category=InferenceFindingCategory(item["category"]), message=item["message"],
        blocking=item["blocking"], branch=item.get("branch"), field=item.get("field"),
        details=item.get("details", {}),
    ) for item in value["findings"])
    return InspectionInferenceResult(
        contract_version=value["contract_version"], inference_id=inference_id,
        inspection_id=inspection_id, validation_id=validation_id,
        preprocessing_id=preprocessing_id, policy_id=value["policy_id"],
        policy_version=value["policy_version"], engine_id=value["engine_id"],
        engine_version=value["engine_version"], engine_type=value["engine_type"],
        execution_outcome=InferenceExecutionOutcome(value["execution_outcome"]),
        started_at=timestamp(value["started_at"]), completed_at=timestamp(value["completed_at"]),
        synthetic_input=value["synthetic_input"], mock_preprocessing=value["mock_preprocessing"],
        mock_inference=value["mock_inference"], production_approved=value["production_approved"],
        rgb_input=identity(value["rgb_input"]), height_input=identity(value["height_input"]),
        decision=None if value["decision"] is None else MockDecision(value["decision"]),
        defect_type=value["defect_type"], confidence=None,
        decision_basis=value["decision_basis"], decision_digest=value["decision_digest"],
        findings=findings, summary=InferenceSummary(**value["summary"]),
    )


def validation_result(*, inspection_id: str, validation_id: str) -> InspectionValidationResult:
    value = document("inspection_validation_result.passed.json")

    def technical(item):
        return ArtifactTechnicalSummary(
            artifact_type=ArtifactType(item["artifact_type"]), sha256=item["sha256"],
            byte_size=item["byte_size"], declared_media_type=item["declared_media_type"],
            detected_format=item["detected_format"], width=item["width"], height=item["height"],
            channels=item["channels"], bit_depth=item["bit_depth"],
            storage_data_type=item["storage_data_type"],
            readability_status=ReadabilityStatus(item["readability_status"]),
        )

    findings = tuple(ValidationFinding(
        code=item["code"], severity=FindingSeverity(item["severity"]),
        category=FindingCategory(item["category"]), message=item["message"],
        blocking=item["blocking"],
        artifact_type=ArtifactType(item["artifact_type"]) if item.get("artifact_type") else None,
        field=item.get("field"), details=item.get("details"),
    ) for item in value["findings"])
    return InspectionValidationResult(
        contract_version=value["contract_version"], validation_id=validation_id,
        inspection_id=inspection_id, validation_policy_id=value["validation_policy_id"],
        validation_policy_version=value["validation_policy_version"],
        outcome=ValidationOutcome(value["outcome"]), started_at=timestamp(value["started_at"]),
        completed_at=timestamp(value["completed_at"]), validator_version=value["validator_version"],
        rgb_artifact=technical(value["rgb_artifact"]),
        height_artifact=technical(value["height_artifact"]), findings=findings,
        summary=ValidationSummary(**value["summary"]),
    )
