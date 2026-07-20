from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from app.services.inspection_inference.exceptions import InferenceKnownFailure
from app.services.inspection_inference.models import (
    InspectionInferencePolicy,
    MockDecision,
    MockEngineDecision,
    ValidatedInferenceInput,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TAXONOMY_PATH = REPOSITORY_ROOT / "contracts" / "defect_taxonomy.json"
DECISION_DIGEST_PREFIX_HEX_LENGTH = 16
ENGINE_ID = "synthetic-deterministic-mock-engine"
ENGINE_VERSION = "1.0.0"


def canonical_decision_document(
    inputs: ValidatedInferenceInput,
    policy: InspectionInferencePolicy,
    *,
    engine_id: str,
    engine_version: str,
) -> dict[str, Any]:
    source = inputs.source
    return {
        "engine_id": engine_id,
        "engine_version": engine_version,
        "height": {
            "buffer_sha256": inputs.height_identity.buffer_sha256,
            "data_type": inputs.height_identity.data_type,
            "layout": inputs.height_identity.layout,
            "shape": list(inputs.height_identity.shape),
        },
        "inspection_id": source.inspection_id,
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "preprocessing_id": source.preprocessing_id,
        "rgb": {
            "buffer_sha256": inputs.rgb_identity.buffer_sha256,
            "data_type": inputs.rgb_identity.data_type,
            "layout": inputs.rgb_identity.layout,
            "shape": list(inputs.rgb_identity.shape),
        },
        "strategy": policy.engine.decision_strategy,
        "validation_id": source.validation_id,
    }


def canonical_decision_bytes(
    inputs: ValidatedInferenceInput,
    policy: InspectionInferencePolicy,
    *,
    engine_id: str,
    engine_version: str,
) -> bytes:
    return json.dumps(
        canonical_decision_document(
            inputs,
            policy,
            engine_id=engine_id,
            engine_version=engine_version,
        ),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class DeterministicMockInferenceEngine:
    """Select mock workflow decisions from identities; no image analysis occurs."""

    def __init__(
        self,
        *,
        engine_id: str = ENGINE_ID,
        engine_version: str = ENGINE_VERSION,
        taxonomy_document: Mapping[str, Any] | None = None,
    ) -> None:
        self._engine_id = engine_id
        self._engine_version = engine_version
        document = (
            json.loads(DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8"))
            if taxonomy_document is None
            else dict(taxonomy_document)
        )
        try:
            taxonomy_version = document["taxonomy_version"]
            defects = tuple(document["$defs"]["supported_defect_type"]["enum"])
        except (KeyError, TypeError) as exc:
            raise ValueError("authoritative defect taxonomy is invalid") from exc
        if (
            taxonomy_version != "pcb-aoi-defects/1.0"
            or not defects
            or len(defects) != len(set(defects))
            or "no_defect" in defects
            or any(not isinstance(item, str) or not item for item in defects)
        ):
            raise ValueError("authoritative defect taxonomy is invalid")
        self._taxonomy_version = taxonomy_version
        self._defects = defects

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def engine_version(self) -> str:
        return self._engine_version

    async def infer(
        self,
        inputs: ValidatedInferenceInput,
        policy: InspectionInferencePolicy,
    ) -> MockEngineDecision:
        if (
            policy.engine.engine_type != "MOCK"
            or policy.engine.decision_strategy != "DETERMINISTIC_HASH_BUCKET"
            or policy.engine.defect_selection_strategy
            != "DETERMINISTIC_TAXONOMY_BUCKET"
            or policy.engine.confidence_mode != "NONE"
        ):
            raise InferenceKnownFailure("INFERENCE_POLICY_INVALID")
        digest = sha256(
            canonical_decision_bytes(
                inputs,
                policy,
                engine_id=self.engine_id,
                engine_version=self.engine_version,
            )
        ).hexdigest()
        bucket = (
            int(digest[:DECISION_DIGEST_PREFIX_HEX_LENGTH], 16)
            % policy.engine.decision_bucket_count
        )
        if bucket in policy.engine.pass_buckets:
            decision = MockDecision.PASS
        elif bucket in policy.engine.fail_buckets:
            decision = MockDecision.FAIL
        elif bucket in policy.engine.uncertain_buckets:
            decision = MockDecision.UNCERTAIN
        else:
            raise InferenceKnownFailure("INFERENCE_POLICY_INVALID")
        defect_type = None
        if decision is MockDecision.FAIL:
            defect_digest = sha256(
                f"{digest}:{self._taxonomy_version}".encode("ascii")
            ).hexdigest()
            index = int(defect_digest[:DECISION_DIGEST_PREFIX_HEX_LENGTH], 16) % len(
                self._defects
            )
            defect_type = self._defects[index]
        return MockEngineDecision(
            decision=decision,
            decision_digest=digest,
            defect_type=defect_type,
        )
