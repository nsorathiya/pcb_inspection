"""Replaceable inference boundaries; no persistence or transport integration."""

from __future__ import annotations

from typing import Protocol

from app.services.inspection_inference.models import (
    InspectionInferencePolicy,
    InspectionInferenceResult,
    MockEngineDecision,
    SyntheticInferenceInput,
    ValidatedInferenceInput,
)


class InferencePolicyLoader(Protocol):
    def load(self, policy_id: str, policy_version: str) -> InspectionInferencePolicy: ...


class InferenceInputValidator(Protocol):
    def validate(self, inputs: SyntheticInferenceInput, policy: InspectionInferencePolicy) -> ValidatedInferenceInput: ...


class InferenceEngine(Protocol):
    @property
    def engine_id(self) -> str: ...

    @property
    def engine_version(self) -> str: ...

    async def infer(self, inputs: ValidatedInferenceInput, policy: InspectionInferencePolicy) -> MockEngineDecision: ...


class InferenceOrchestrator(Protocol):
    async def run_inference(self, inputs: SyntheticInferenceInput, policy: InspectionInferencePolicy) -> InspectionInferenceResult: ...


class InferenceResultSink(Protocol):
    """Future persistence boundary; this task intentionally provides no sink."""

    async def save_inference_result(self, result: InspectionInferenceResult) -> None: ...
