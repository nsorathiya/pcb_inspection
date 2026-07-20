"""Typed contracts for development-only deterministic mock inference."""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from app.services.inspection_preprocessing.models import InternalPreprocessedBuffer


class InferenceExecutionOutcome(str, Enum):
    SUCCEEDED = "INFERENCE_SUCCEEDED"
    FAILED = "INFERENCE_FAILED"
    ERROR = "INFERENCE_ERROR"


class MockDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class InferenceFindingSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class InferenceFindingCategory(str, Enum):
    PREREQUISITE = "PREREQUISITE"
    POLICY = "POLICY"
    RGB_INPUT = "RGB_INPUT"
    HEIGHT_INPUT = "HEIGHT_INPUT"
    PAIR = "PAIR"
    DECISION = "DECISION"
    INTERNAL = "INTERNAL"


@dataclass(frozen=True)
class InferencePrerequisites:
    required_preprocessing_outcome: str
    require_synthetic_input: bool
    require_mock_preprocessing: bool
    accepted_rgb_layouts: tuple[str, ...]
    accepted_height_layouts: tuple[str, ...]
    accepted_rgb_data_types: tuple[str, ...]
    accepted_height_data_types: tuple[str, ...]
    require_matching_spatial_dimensions: bool


@dataclass(frozen=True)
class MockEnginePolicy:
    engine_type: str
    decision_strategy: str
    decision_bucket_count: int
    pass_buckets: tuple[int, ...]
    fail_buckets: tuple[int, ...]
    uncertain_buckets: tuple[int, ...]
    defect_selection_strategy: str
    confidence_mode: str


@dataclass(frozen=True)
class InferenceSafetyPolicy:
    allow_mock_engine: bool
    allow_real_input: bool
    allow_model_accuracy_claim: bool
    allow_production_decision: bool


@dataclass(frozen=True)
class InspectionInferencePolicy:
    contract_version: str
    policy_id: str
    policy_version: str
    name: str
    description: str
    development_only: bool
    production_approved: bool
    prerequisites: InferencePrerequisites
    engine: MockEnginePolicy
    safety: InferenceSafetyPolicy


@dataclass(frozen=True)
class SyntheticInferenceInput:
    inspection_id: str
    validation_id: str
    preprocessing_id: str
    preprocessing_outcome: str | None
    synthetic_input: bool
    mock_preprocessing: bool
    rgb_buffer: InternalPreprocessedBuffer | None
    height_buffer: InternalPreprocessedBuffer | None


@dataclass(frozen=True)
class InferenceInputIdentity:
    buffer_sha256: str
    shape: tuple[int, ...]
    layout: str
    data_type: str
    channel_count: int
    width: int
    height: int
    byte_size: int
    source_artifact_sha256: str


@dataclass(frozen=True)
class ValidatedInferenceInput:
    source: SyntheticInferenceInput
    rgb_buffer: InternalPreprocessedBuffer
    height_buffer: InternalPreprocessedBuffer
    rgb_identity: InferenceInputIdentity
    height_identity: InferenceInputIdentity


@dataclass(frozen=True)
class MockEngineDecision:
    decision: MockDecision
    decision_digest: str
    defect_type: str | None


@dataclass(frozen=True)
class InferenceFinding:
    code: str
    severity: InferenceFindingSeverity
    category: InferenceFindingCategory
    message: str
    blocking: bool
    branch: str | None = None
    field: str | None = None
    details: Mapping[str, str | int | float | bool | None] = dataclass_field(
        default_factory=dict
    )


@dataclass(frozen=True)
class InferenceSummary:
    total_findings: int
    blocking_findings: int
    warnings: int
    errors: int


@dataclass(frozen=True)
class InspectionInferenceResult:
    contract_version: str
    inference_id: str
    inspection_id: str
    validation_id: str
    preprocessing_id: str
    policy_id: str
    policy_version: str
    engine_id: str
    engine_version: str
    engine_type: str
    execution_outcome: InferenceExecutionOutcome
    started_at: datetime
    completed_at: datetime
    synthetic_input: bool
    mock_preprocessing: bool
    mock_inference: bool
    production_approved: bool
    rgb_input: InferenceInputIdentity | None
    height_input: InferenceInputIdentity | None
    decision: MockDecision | None
    defect_type: str | None
    confidence: None
    decision_basis: str | None
    decision_digest: str | None
    findings: tuple[InferenceFinding, ...]
    summary: InferenceSummary

    def to_dict(self) -> dict[str, Any]:
        return inference_result_to_dict(self)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("inference timestamps must include timezone information")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _input_identity(value: InferenceInputIdentity | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "buffer_sha256": value.buffer_sha256,
        "shape": list(value.shape),
        "layout": value.layout,
        "data_type": value.data_type,
        "channel_count": value.channel_count,
        "width": value.width,
        "height": value.height,
        "byte_size": value.byte_size,
        "source_artifact_sha256": value.source_artifact_sha256,
    }


def _finding(value: InferenceFinding) -> dict[str, Any]:
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


def inference_result_to_dict(value: InspectionInferenceResult) -> dict[str, Any]:
    return {
        "contract_version": value.contract_version,
        "inference_id": value.inference_id,
        "inspection_id": value.inspection_id,
        "validation_id": value.validation_id,
        "preprocessing_id": value.preprocessing_id,
        "policy_id": value.policy_id,
        "policy_version": value.policy_version,
        "engine_id": value.engine_id,
        "engine_version": value.engine_version,
        "engine_type": value.engine_type,
        "execution_outcome": value.execution_outcome.value,
        "started_at": _timestamp(value.started_at),
        "completed_at": _timestamp(value.completed_at),
        "synthetic_input": value.synthetic_input,
        "mock_preprocessing": value.mock_preprocessing,
        "mock_inference": value.mock_inference,
        "production_approved": value.production_approved,
        "rgb_input": _input_identity(value.rgb_input),
        "height_input": _input_identity(value.height_input),
        "decision": None if value.decision is None else value.decision.value,
        "defect_type": value.defect_type,
        "confidence": None,
        "decision_basis": value.decision_basis,
        "decision_digest": value.decision_digest,
        "findings": [_finding(item) for item in value.findings],
        "summary": {
            "total_findings": value.summary.total_findings,
            "blocking_findings": value.summary.blocking_findings,
            "warnings": value.summary.warnings,
            "errors": value.summary.errors,
        },
    }


def inference_result_json(value: InspectionInferenceResult) -> str:
    return json.dumps(
        inference_result_to_dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_inference_policy_document(document: Mapping[str, Any]) -> None:
    """Apply version-1 cross-field rules without inserting defaults."""

    prerequisites = document["prerequisites"]
    engine = document["engine"]
    safety = document["safety"]
    _require(document["development_only"] is True, "mock policy must be development-only")
    _require(document["production_approved"] is False, "mock policy cannot claim production approval")
    _require(
        prerequisites["required_preprocessing_outcome"] == "PREPROCESSING_SUCCEEDED",
        "mock policy requires successful preprocessing",
    )
    _require(prerequisites["require_synthetic_input"] is True, "mock policy requires synthetic input")
    _require(prerequisites["require_mock_preprocessing"] is True, "mock policy requires mock preprocessing")
    _require(engine["engine_type"] == "MOCK", "contract version 1.0 supports only MOCK")
    _require(engine["decision_strategy"] == "DETERMINISTIC_HASH_BUCKET", "mock decision strategy is unsupported")
    _require(engine["defect_selection_strategy"] == "DETERMINISTIC_TAXONOMY_BUCKET", "mock defect selection strategy is unsupported")
    _require(engine["confidence_mode"] == "NONE", "contract version 1.0 confidence mode must be NONE")
    _require(safety["allow_mock_engine"] is True, "mock engine must be explicitly allowed")
    _require(safety["allow_real_input"] is False, "mock policy cannot enable real input")
    _require(safety["allow_model_accuracy_claim"] is False, "mock policy cannot permit accuracy claims")
    _require(safety["allow_production_decision"] is False, "mock policy cannot permit production decisions")

    count = engine["decision_bucket_count"]
    assignments = {
        "PASS": tuple(engine["pass_buckets"]),
        "FAIL": tuple(engine["fail_buckets"]),
        "UNCERTAIN": tuple(engine["uncertain_buckets"]),
    }
    combined = tuple(bucket for buckets in assignments.values() for bucket in buckets)
    _require(all(assignments.values()), "each mock decision requires at least one bucket")
    _require(len(combined) == len(set(combined)), "decision bucket assignments must not overlap")
    _require(all(0 <= bucket < count for bucket in combined), "decision bucket index is outside configured count")
    _require(set(combined) == set(range(count)), "decision bucket assignments must be complete")


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


def validate_inference_result_document(
    document: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> None:
    """Check catalogue consistency, counts, outcomes, and mock-decision semantics."""

    definitions = {entry["code"]: entry for entry in catalogue["findings"]}
    findings: Sequence[Mapping[str, Any]] = document["findings"]
    for finding in findings:
        definition = definitions.get(finding["code"])
        _require(definition is not None, "inference result contains an unknown finding code")
        _require(finding["severity"] == definition["severity"], "finding severity disagrees with catalogue")
        _require(finding["category"] == definition["category"], "finding category disagrees with catalogue")
    _require(
        list(findings) == sorted(findings, key=lambda item: finding_sort_key(item, catalogue)),
        "findings are not in deterministic catalogue order",
    )
    expected = {
        "total_findings": len(findings),
        "blocking_findings": sum(item["blocking"] is True for item in findings),
        "warnings": sum(item["severity"] == "WARNING" for item in findings),
        "errors": sum(item["severity"] == "ERROR" for item in findings),
    }
    _require(document["summary"] == expected, "inference summary counts do not match findings")

    outcome = document["execution_outcome"]
    decision = document["decision"]
    if outcome == "INFERENCE_SUCCEEDED":
        _require(expected["blocking_findings"] == 0, "successful inference cannot contain blocking findings")
        _require(decision in {"PASS", "FAIL", "UNCERTAIN"}, "successful inference requires a mock decision")
        _require(document["decision_basis"] == "DETERMINISTIC_HASH_BUCKET", "successful inference has an invalid decision basis")
        _require(isinstance(document["decision_digest"], str), "successful inference requires a decision digest")
        required = {"MOCK_INFERENCE_USED", "MOCK_DECISION_GENERATED", "CONFIDENCE_UNAVAILABLE"}
        codes = {item["code"] for item in findings}
        _require(required.issubset(codes), "successful inference is missing required mock findings")
    else:
        _require(all(document[key] is None for key in ("decision", "defect_type", "confidence", "decision_basis", "decision_digest")), "failed or error inference cannot contain a decision")
        _require(expected["blocking_findings"] > 0, "failed or error inference requires a blocking finding")
        if outcome == "INFERENCE_ERROR":
            _require(any(item["code"] == "INFERENCE_INTERNAL_ERROR" for item in findings), "inference error requires its internal finding")

    supported = set(taxonomy["$defs"]["supported_defect_type"]["enum"])
    if decision == "FAIL":
        _require(document["defect_type"] in supported, "mock FAIL requires a supported taxonomy defect")
        _require(any(item["code"] == "MOCK_FAIL_DEFECT_ASSIGNED" for item in findings), "mock FAIL requires its assignment finding")
    else:
        _require(document["defect_type"] is None, "PASS, UNCERTAIN, and technical failures cannot contain a defect type")
    _require(document["confidence"] is None, "mock inference confidence must be null")
