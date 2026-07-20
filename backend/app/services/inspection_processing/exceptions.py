class ProcessingExecutionError(Exception):
    """Base class for safe internal processing-orchestration failures."""


class ProcessingExecutionInspectionNotFoundError(ProcessingExecutionError):
    pass


class ProcessingExecutionInspectionNotReadyError(ProcessingExecutionError):
    pass


class ProcessingExecutionValidationMissingError(ProcessingExecutionError):
    pass


class ProcessingExecutionValidationNotPassedError(ProcessingExecutionError):
    pass


class ProcessingExecutionArtifactPairError(ProcessingExecutionError):
    pass


class ProcessingExecutionPolicyError(ProcessingExecutionError):
    pass


class SyntheticProvenanceError(ProcessingExecutionError):
    pass


class SyntheticProvenanceUnavailableError(SyntheticProvenanceError):
    pass


class SyntheticProvenanceMismatchError(SyntheticProvenanceError):
    pass


class ProcessingArtifactPreflightError(ProcessingExecutionError):
    pass


class ProcessingExecutionConflictError(ProcessingExecutionError):
    pass


class ProcessingExecutionInProgressError(ProcessingExecutionConflictError):
    pass


class ProcessingExecutionConsistencyError(ProcessingExecutionConflictError):
    pass


class ProcessingExecutionOrchestrationError(ProcessingExecutionError):
    pass


class ProcessingExecutionRecoveryRequiredError(ProcessingExecutionOrchestrationError):
    """A post-begin failure could not be finalized through the lifecycle service."""

