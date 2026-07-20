from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
from uuid import UUID, uuid4

from app.services.inspection_inference.exceptions import InferenceKnownFailure
from app.services.inspection_inference.findings import InferenceFindingFactory
from app.services.inspection_inference.interfaces import InferenceEngine, InferenceInputValidator
from app.services.inspection_inference.mock_engine import DeterministicMockInferenceEngine
from app.services.inspection_inference.models import (
    InferenceExecutionOutcome,
    InferenceFindingSeverity,
    InferenceSummary,
    InspectionInferencePolicy,
    InspectionInferenceResult,
    MockDecision,
    SyntheticInferenceInput,
    ValidatedInferenceInput,
)
from app.services.inspection_inference.validation import SyntheticInferenceInputValidator

INFERENCE_CONTRACT_VERSION = "pcb-aoi-inspection-inference/1.0"


def _canonical_uuid(value: str, field: str) -> str:
    try:
        canonical = str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a canonical UUID string") from exc
    if canonical != value:
        raise ValueError(f"{field} must be a canonical UUID string")
    return value


class SyntheticMockInferenceService:
    """In-memory orchestrator for development-only deterministic mock inference."""

    def __init__(
        self,
        *,
        input_validator: InferenceInputValidator | None = None,
        engine: InferenceEngine | None = None,
        findings: InferenceFindingFactory | None = None,
        clock: Callable[[], datetime] | None = None,
        inference_id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._validator = input_validator or SyntheticInferenceInputValidator()
        self._engine = engine or DeterministicMockInferenceEngine()
        self._findings = findings or InferenceFindingFactory()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._inference_id = inference_id_generator or (lambda: str(uuid4()))

    async def run_inference(
        self,
        inputs: SyntheticInferenceInput,
        policy: InspectionInferencePolicy,
    ) -> InspectionInferenceResult:
        inference_id = _canonical_uuid(self._inference_id(), "inference_id")
        _canonical_uuid(inputs.inspection_id, "inspection_id")
        _canonical_uuid(inputs.validation_id, "validation_id")
        _canonical_uuid(inputs.preprocessing_id, "preprocessing_id")
        started_at = self._clock()
        validated: ValidatedInferenceInput | None = None
        decision: MockDecision | None = None
        defect_type: str | None = None
        decision_digest: str | None = None
        decision_basis: str | None = None
        findings = []
        try:
            validated = self._validator.validate(inputs, policy)
            engine_decision = await self._engine.infer(validated, policy)
            decision = engine_decision.decision
            defect_type = engine_decision.defect_type
            decision_digest = engine_decision.decision_digest
            decision_basis = "DETERMINISTIC_HASH_BUCKET"
            findings.extend(
                (
                    self._findings.create("MOCK_INFERENCE_USED"),
                    self._findings.create("MOCK_DECISION_GENERATED"),
                    self._findings.create("CONFIDENCE_UNAVAILABLE"),
                )
            )
            if decision is MockDecision.FAIL:
                findings.append(self._findings.create("MOCK_FAIL_DEFECT_ASSIGNED"))
            outcome = InferenceExecutionOutcome.SUCCEEDED
        except InferenceKnownFailure as exc:
            findings.append(
                self._findings.create(
                    exc.finding_code,
                    branch=exc.branch,
                    field=exc.field,
                    details=exc.details,
                )
            )
            outcome = InferenceExecutionOutcome.FAILED
        except Exception:
            findings = [self._findings.create("INFERENCE_INTERNAL_ERROR")]
            outcome = InferenceExecutionOutcome.ERROR

        ordered = self._findings.sort(findings)
        completed_at = self._clock()
        if completed_at < started_at:
            completed_at = started_at
        summary = InferenceSummary(
            total_findings=len(ordered),
            blocking_findings=sum(item.blocking for item in ordered),
            warnings=sum(
                item.severity is InferenceFindingSeverity.WARNING for item in ordered
            ),
            errors=sum(
                item.severity is InferenceFindingSeverity.ERROR for item in ordered
            ),
        )
        return InspectionInferenceResult(
            contract_version=INFERENCE_CONTRACT_VERSION,
            inference_id=inference_id,
            inspection_id=inputs.inspection_id,
            validation_id=inputs.validation_id,
            preprocessing_id=inputs.preprocessing_id,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            engine_id=self._engine.engine_id,
            engine_version=self._engine.engine_version,
            engine_type="MOCK",
            execution_outcome=outcome,
            started_at=started_at,
            completed_at=completed_at,
            synthetic_input=inputs.synthetic_input,
            mock_preprocessing=inputs.mock_preprocessing,
            mock_inference=True,
            production_approved=False,
            rgb_input=None if validated is None else validated.rgb_identity,
            height_input=None if validated is None else validated.height_identity,
            decision=decision,
            defect_type=defect_type,
            confidence=None,
            decision_basis=decision_basis,
            decision_digest=decision_digest,
            findings=ordered,
            summary=summary,
        )
