"""Versioned preprocessing contracts and replaceable, execution-free interfaces."""

from app.services.inspection_preprocessing.interfaces import (
    HeightPreprocessor,
    PreprocessingOrchestrator,
    PreprocessingPolicyLoader,
    PreprocessingResultSink,
    RGBPreprocessor,
    RegistrationProcessor,
    ValidatedInspectionReader,
)
from app.services.inspection_preprocessing.models import (
    ArtifactInputIdentity,
    HeightPreprocessingPolicy,
    HeightPreprocessingOutput,
    InspectionPreprocessingPolicy,
    InspectionPreprocessingResult,
    PreprocessedBufferDescriptor,
    PreprocessingFinding,
    PreprocessingOutcome,
    PreprocessingOutputPolicy,
    PreprocessingPrerequisites,
    PreprocessingSafetyPolicy,
    RGBPreprocessingPolicy,
    RGBPreprocessingOutput,
    RegistrationPolicy,
    RegistrationProcessingResult,
    ValidatedInspectionInputs,
    validate_preprocessing_policy_document,
    validate_preprocessing_result_document,
)

__all__ = [
    "ArtifactInputIdentity", "HeightPreprocessingOutput", "HeightPreprocessingPolicy", "HeightPreprocessor",
    "InspectionPreprocessingPolicy", "InspectionPreprocessingResult",
    "PreprocessedBufferDescriptor", "PreprocessingFinding", "PreprocessingOrchestrator",
    "PreprocessingOutcome", "PreprocessingOutputPolicy", "PreprocessingPolicyLoader",
    "PreprocessingPrerequisites", "PreprocessingResultSink", "PreprocessingSafetyPolicy",
    "RGBPreprocessingOutput", "RGBPreprocessingPolicy", "RGBPreprocessor",
    "RegistrationPolicy", "RegistrationProcessingResult",
    "RegistrationProcessor", "ValidatedInspectionInputs", "ValidatedInspectionReader",
    "validate_preprocessing_policy_document", "validate_preprocessing_result_document",
]
