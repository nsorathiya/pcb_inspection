from __future__ import annotations

import json
from hashlib import sha256

from app.services.inspection_inference.models import (
    InspectionInferenceResult,
    inference_result_to_dict,
)
from app.services.inspection_preprocessing.models import (
    InspectionPreprocessingResult,
    preprocessing_result_to_dict,
)


def _canonical_bytes(document: dict[str, object]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_preprocessing_result_bytes(result: InspectionPreprocessingResult) -> bytes:
    if not isinstance(result, InspectionPreprocessingResult):
        raise TypeError("canonical serialization requires an InspectionPreprocessingResult")
    return _canonical_bytes(preprocessing_result_to_dict(result))


def canonical_preprocessing_result_sha256(result: InspectionPreprocessingResult) -> str:
    return sha256(canonical_preprocessing_result_bytes(result)).hexdigest()


def canonical_inference_result_bytes(result: InspectionInferenceResult) -> bytes:
    if not isinstance(result, InspectionInferenceResult):
        raise TypeError("canonical serialization requires an InspectionInferenceResult")
    return _canonical_bytes(inference_result_to_dict(result))


def canonical_inference_result_sha256(result: InspectionInferenceResult) -> str:
    return sha256(canonical_inference_result_bytes(result)).hexdigest()
