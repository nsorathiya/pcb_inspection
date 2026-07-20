from enum import Enum


class ProcessingRunStatus(str, Enum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class ProcessingFinalDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class PersistedPreprocessingOutcome(str, Enum):
    SUCCEEDED = "PREPROCESSING_SUCCEEDED"
    FAILED = "PREPROCESSING_FAILED"
    ERROR = "PREPROCESSING_ERROR"


class PersistedInferenceOutcome(str, Enum):
    SUCCEEDED = "INFERENCE_SUCCEEDED"
    FAILED = "INFERENCE_FAILED"
    ERROR = "INFERENCE_ERROR"


class ProcessingFindingSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class PreprocessingFindingCategory(str, Enum):
    PREREQUISITE = "PREREQUISITE"
    POLICY = "POLICY"
    RGB = "RGB"
    HEIGHT = "HEIGHT"
    REGISTRATION = "REGISTRATION"
    OUTPUT = "OUTPUT"
    INTERNAL = "INTERNAL"


class InferenceFindingCategory(str, Enum):
    PREREQUISITE = "PREREQUISITE"
    POLICY = "POLICY"
    RGB_INPUT = "RGB_INPUT"
    HEIGHT_INPUT = "HEIGHT_INPUT"
    PAIR = "PAIR"
    DECISION = "DECISION"
    INTERNAL = "INTERNAL"
