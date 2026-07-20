"""Versioned, replaceable, development-only mock inference boundary."""

from app.services.inspection_inference.interfaces import (
    InferenceEngine,
    InferenceInputValidator,
    InferenceOrchestrator,
    InferencePolicyLoader,
    InferenceResultSink,
)
from app.services.inspection_inference.mock_engine import (
    DECISION_DIGEST_PREFIX_HEX_LENGTH,
    DeterministicMockInferenceEngine,
    canonical_decision_bytes,
    canonical_decision_document,
)
from app.services.inspection_inference.models import (
    InferenceExecutionOutcome,
    InferenceFinding,
    InferenceInputIdentity,
    InferencePrerequisites,
    InferenceSafetyPolicy,
    InferenceSummary,
    InspectionInferencePolicy,
    InspectionInferenceResult,
    MockDecision,
    MockEnginePolicy,
    SyntheticInferenceInput,
    ValidatedInferenceInput,
    inference_result_json,
    inference_result_to_dict,
    validate_inference_policy_document,
    validate_inference_result_document,
)
from app.services.inspection_inference.policy_loader import (
    SyntheticMockInferencePolicyLoader,
)
from app.services.inspection_inference.service import SyntheticMockInferenceService
from app.services.inspection_inference.validation import SyntheticInferenceInputValidator

__all__ = [
    "DECISION_DIGEST_PREFIX_HEX_LENGTH",
    "DeterministicMockInferenceEngine",
    "InferenceEngine",
    "InferenceExecutionOutcome",
    "InferenceFinding",
    "InferenceInputIdentity",
    "InferenceInputValidator",
    "InferenceOrchestrator",
    "InferencePolicyLoader",
    "InferencePrerequisites",
    "InferenceResultSink",
    "InferenceSafetyPolicy",
    "InferenceSummary",
    "InspectionInferencePolicy",
    "InspectionInferenceResult",
    "MockDecision",
    "MockEnginePolicy",
    "SyntheticInferenceInput",
    "SyntheticInferenceInputValidator",
    "SyntheticMockInferencePolicyLoader",
    "SyntheticMockInferenceService",
    "ValidatedInferenceInput",
    "canonical_decision_bytes",
    "canonical_decision_document",
    "inference_result_json",
    "inference_result_to_dict",
    "validate_inference_policy_document",
    "validate_inference_result_document",
]
