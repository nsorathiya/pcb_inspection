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
    HeightProcessedBranch,
    HeightPreprocessingPolicy,
    HeightPreprocessingOutput,
    InternalPreprocessedBuffer,
    InspectionPreprocessingPolicy,
    InspectionPreprocessingResult,
    PreprocessedBufferDescriptor,
    PreprocessingFinding,
    PreprocessingOutcome,
    PreprocessingOutputPolicy,
    PreprocessingPrerequisites,
    PreprocessingSafetyPolicy,
    RGBProcessedBranch,
    RGBPreprocessingPolicy,
    RGBPreprocessingOutput,
    RegistrationPolicy,
    RegistrationProcessingResult,
    SyntheticPreprocessingExecution,
    ValidatedArtifactSource,
    ValidatedInspectionInput,
    ValidatedInspectionInputs,
    validate_preprocessing_policy_document,
    validate_preprocessing_result_document,
    preprocessing_result_json,
    preprocessing_result_to_dict,
)
from app.services.inspection_preprocessing.policy_loader import (
    SyntheticPreprocessingPolicyLoader,
)
from app.services.inspection_preprocessing.service import (
    SyntheticInspectionPreprocessingService,
)

__all__ = [
    "ArtifactInputIdentity", "HeightProcessedBranch", "HeightPreprocessingOutput", "HeightPreprocessingPolicy", "HeightPreprocessor",
    "InternalPreprocessedBuffer",
    "InspectionPreprocessingPolicy", "InspectionPreprocessingResult",
    "PreprocessedBufferDescriptor", "PreprocessingFinding", "PreprocessingOrchestrator",
    "PreprocessingOutcome", "PreprocessingOutputPolicy", "PreprocessingPolicyLoader",
    "PreprocessingPrerequisites", "PreprocessingResultSink", "PreprocessingSafetyPolicy",
    "RGBProcessedBranch", "RGBPreprocessingOutput", "RGBPreprocessingPolicy", "RGBPreprocessor",
    "RegistrationPolicy", "RegistrationProcessingResult", "RegistrationProcessor",
    "SyntheticInspectionPreprocessingService", "SyntheticPreprocessingExecution",
    "SyntheticPreprocessingPolicyLoader", "ValidatedArtifactSource",
    "ValidatedInspectionInput", "ValidatedInspectionInputs", "ValidatedInspectionReader",
    "preprocessing_result_json", "preprocessing_result_to_dict",
    "validate_preprocessing_policy_document", "validate_preprocessing_result_document",
]
