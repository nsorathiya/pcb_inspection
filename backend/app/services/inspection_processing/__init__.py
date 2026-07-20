"""Schema-v3 processing persistence and guarded lifecycle coordination."""

from app.db.processing_types import ProcessingFinalDecision, ProcessingRunStatus

from app.services.inspection_processing.canonical import (
    canonical_inference_result_bytes,
    canonical_inference_result_sha256,
    canonical_preprocessing_result_bytes,
    canonical_preprocessing_result_sha256,
)
from app.services.inspection_processing.lifecycle import (
    AUDIT_MOCK_FAIL,
    AUDIT_MOCK_PASS,
    AUDIT_MOCK_UNCERTAIN,
    AUDIT_PROCESSING_ERROR,
    AUDIT_PROCESSING_STARTED,
    InvalidProcessingTransitionError,
    ProcessingInspectionNotFoundError,
    ProcessingLifecycleConflictError,
    ProcessingLifecycleConsistencyError,
    ProcessingLifecycleError,
    ProcessingLifecycleService,
    ProcessingValidationNotFoundError,
)
from app.services.inspection_processing.models import (
    BeginProcessingResult,
    CompleteProcessingResult,
    ProcessingKeyArtifact,
    ProcessingStartIdentity,
    generate_processing_key,
)
from app.services.inspection_processing.persistence import (
    InspectionProcessingRepository,
    PersistedProcessingFinding,
    PersistedProcessingResult,
    PersistedProcessingRun,
    ProcessingPersistenceConflictError,
    ProcessingPersistenceError,
    ProcessingPersistenceIntegrityError,
)

__all__ = [name for name in globals() if not name.startswith("_")]
