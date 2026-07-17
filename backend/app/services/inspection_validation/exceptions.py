class InspectionValidationError(Exception):
    """Base class for expected semantic-validation failures."""


class PolicyLoadError(InspectionValidationError, ValueError):
    def __init__(self, finding_code: str, message: str) -> None:
        super().__init__(message)
        self.finding_code = finding_code


class ArtifactResolutionError(InspectionValidationError):
    def __init__(self, finding_code: str) -> None:
        super().__init__(finding_code)
        self.finding_code = finding_code
