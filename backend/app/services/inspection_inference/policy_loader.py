from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from app.services.inspection_inference.exceptions import InferencePolicyLoadError
from app.services.inspection_inference.models import (
    InferencePrerequisites,
    InferenceSafetyPolicy,
    InspectionInferencePolicy,
    MockEnginePolicy,
    validate_inference_policy_document,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCHEMA_PATH = REPOSITORY_ROOT / "contracts" / "inspection_inference_policy.schema.json"
DEFAULT_POLICY_PATH = REPOSITORY_ROOT / "contracts" / "examples" / "inspection_inference_policy.mock.json"
MOCK_POLICY_ID = "synthetic-deterministic-mock-inference"
MOCK_POLICY_VERSION = "1.0"
POLICY_CONTRACT_VERSION = "pcb-aoi-inspection-inference-policy/1.0"


def _to_policy(document: Mapping[str, Any]) -> InspectionInferencePolicy:
    prerequisites = document["prerequisites"]
    engine = document["engine"]
    safety = document["safety"]
    return InspectionInferencePolicy(
        contract_version=document["contract_version"],
        policy_id=document["policy_id"],
        policy_version=document["policy_version"],
        name=document["name"],
        description=document["description"],
        development_only=document["development_only"],
        production_approved=document["production_approved"],
        prerequisites=InferencePrerequisites(
            required_preprocessing_outcome=prerequisites["required_preprocessing_outcome"],
            require_synthetic_input=prerequisites["require_synthetic_input"],
            require_mock_preprocessing=prerequisites["require_mock_preprocessing"],
            accepted_rgb_layouts=tuple(prerequisites["accepted_rgb_layouts"]),
            accepted_height_layouts=tuple(prerequisites["accepted_height_layouts"]),
            accepted_rgb_data_types=tuple(prerequisites["accepted_rgb_data_types"]),
            accepted_height_data_types=tuple(prerequisites["accepted_height_data_types"]),
            require_matching_spatial_dimensions=prerequisites["require_matching_spatial_dimensions"],
        ),
        engine=MockEnginePolicy(
            engine_type=engine["engine_type"],
            decision_strategy=engine["decision_strategy"],
            decision_bucket_count=engine["decision_bucket_count"],
            pass_buckets=tuple(engine["pass_buckets"]),
            fail_buckets=tuple(engine["fail_buckets"]),
            uncertain_buckets=tuple(engine["uncertain_buckets"]),
            defect_selection_strategy=engine["defect_selection_strategy"],
            confidence_mode=engine["confidence_mode"],
        ),
        safety=InferenceSafetyPolicy(**dict(safety)),
    )


class SyntheticMockInferencePolicyLoader:
    """Load only the repository-owned mock policy by exact identity."""

    def __init__(
        self,
        *,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
        policy_document: Mapping[str, Any] | None = None,
    ) -> None:
        self._schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self._injected_document = None if policy_document is None else dict(policy_document)

    def load(self, policy_id: str, policy_version: str) -> InspectionInferencePolicy:
        if policy_id != MOCK_POLICY_ID:
            raise InferencePolicyLoadError("INFERENCE_POLICY_NOT_FOUND")
        if policy_version != MOCK_POLICY_VERSION:
            raise InferencePolicyLoadError("INFERENCE_POLICY_VERSION_UNSUPPORTED")
        try:
            document = (
                json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
                if self._injected_document is None
                else dict(self._injected_document)
            )
            if document.get("contract_version") != POLICY_CONTRACT_VERSION:
                raise InferencePolicyLoadError("INFERENCE_POLICY_VERSION_UNSUPPORTED")
            Draft202012Validator(self._schema, format_checker=FormatChecker()).validate(document)
            validate_inference_policy_document(document)
            policy = _to_policy(document)
        except InferencePolicyLoadError:
            raise
        except Exception as exc:
            raise InferencePolicyLoadError("INFERENCE_POLICY_INVALID") from exc
        if policy.policy_id != policy_id or policy.policy_version != policy_version:
            raise InferencePolicyLoadError("INFERENCE_POLICY_INVALID")
        return policy
