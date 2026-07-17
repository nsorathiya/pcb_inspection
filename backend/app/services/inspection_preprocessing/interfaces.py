"""Replaceable preprocessing boundaries. No implementation is provided here."""

from __future__ import annotations

from typing import Protocol

from app.services.inspection_preprocessing.models import (
    HeightProcessedBranch,
    InspectionPreprocessingPolicy,
    InspectionPreprocessingResult,
    RGBProcessedBranch,
    RegistrationProcessingResult,
    SyntheticPreprocessingExecution,
    ValidatedInspectionInput,
    ValidatedInspectionInputs,
)


class PreprocessingPolicyLoader(Protocol):
    def load(self, policy_id: str, policy_version: str) -> InspectionPreprocessingPolicy: ...


class ValidatedInspectionReader(Protocol):
    async def read_validated_inspection(self, inspection_id: str, validation_id: str) -> ValidatedInspectionInputs: ...


class RGBPreprocessor(Protocol):
    async def preprocess_rgb(self, inputs: ValidatedInspectionInput, policy: InspectionPreprocessingPolicy) -> RGBProcessedBranch: ...


class HeightPreprocessor(Protocol):
    async def preprocess_height(self, inputs: ValidatedInspectionInput, policy: InspectionPreprocessingPolicy) -> HeightProcessedBranch: ...


class RegistrationProcessor(Protocol):
    async def coordinate_registration(self, rgb: RGBProcessedBranch, height: HeightProcessedBranch, inputs: ValidatedInspectionInput, policy: InspectionPreprocessingPolicy) -> RegistrationProcessingResult: ...


class PreprocessingOrchestrator(Protocol):
    async def preprocess_inspection(self, validated_input: ValidatedInspectionInput, policy: InspectionPreprocessingPolicy) -> SyntheticPreprocessingExecution: ...


class PreprocessingResultSink(Protocol):
    """Future persistence boundary; this contract does not implement a sink."""

    async def save_preprocessing_result(self, result: InspectionPreprocessingResult) -> None: ...
