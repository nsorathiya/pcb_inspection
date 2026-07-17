from __future__ import annotations

from pathlib import Path

from app.services.dataset_validation.file_inspection import (
    DecodedRaster,
    FileInspectionError,
    decode_height_values,
    decode_rgb_values,
    sha256_file,
)
from app.services.inspection_preprocessing.exceptions import PreprocessingKnownFailure
from app.services.inspection_preprocessing.models import ValidatedArtifactSource


def _readable_source(source: ValidatedArtifactSource, unavailable_code: str) -> Path:
    path = source.source_path
    if path is None or not path.exists() or not path.is_file() or path.is_symlink():
        raise PreprocessingKnownFailure(unavailable_code)
    try:
        size = path.stat().st_size
        digest = sha256_file(path)
    except OSError as exc:
        raise PreprocessingKnownFailure(unavailable_code) from exc
    if size != source.identity.byte_size or digest != source.identity.sha256:
        raise PreprocessingKnownFailure(
            unavailable_code, details={"reason": "validated_identity_mismatch"}
        )
    return path


def _match_identity(
    decoded: DecodedRaster,
    source: ValidatedArtifactSource,
    *,
    format_code: str,
    channels_code: str,
    bit_depth_code: str | None = None,
    storage_code: str | None = None,
) -> None:
    actual = decoded.metadata
    expected = source.identity
    if actual.detected_format != expected.detected_format:
        raise PreprocessingKnownFailure(format_code, field="detected_format")
    if actual.channels != expected.channels:
        raise PreprocessingKnownFailure(channels_code, field="channels")
    if bit_depth_code and actual.bit_depth != expected.bit_depth:
        raise PreprocessingKnownFailure(bit_depth_code, field="bit_depth")
    if storage_code and actual.storage_data_type != expected.storage_data_type:
        raise PreprocessingKnownFailure(storage_code, field="storage_data_type")
    if (actual.width, actual.height) != (expected.width, expected.height):
        raise PreprocessingKnownFailure(format_code, field="dimensions")


class SyntheticRasterDecoder:
    def decode_rgb(self, source: ValidatedArtifactSource) -> DecodedRaster:
        path = _readable_source(source, "RGB_INPUT_UNAVAILABLE")
        try:
            decoded = decode_rgb_values(path)
        except (FileInspectionError, OSError, ValueError) as exc:
            raise PreprocessingKnownFailure("RGB_PREPROCESSING_UNSUPPORTED", branch="RGB") from exc
        _match_identity(
            decoded,
            source,
            format_code="RGB_FORMAT_INCOMPATIBLE",
            channels_code="RGB_CHANNELS_INCOMPATIBLE",
            bit_depth_code="RGB_BIT_DEPTH_INCOMPATIBLE",
        )
        return decoded

    def decode_height(self, source: ValidatedArtifactSource) -> DecodedRaster:
        path = _readable_source(source, "HEIGHT_INPUT_UNAVAILABLE")
        try:
            decoded = decode_height_values(path)
        except (FileInspectionError, OSError, ValueError) as exc:
            raise PreprocessingKnownFailure("HEIGHT_PREPROCESSING_UNSUPPORTED", branch="HEIGHT") from exc
        _match_identity(
            decoded,
            source,
            format_code="HEIGHT_FORMAT_INCOMPATIBLE",
            channels_code="HEIGHT_CHANNELS_INCOMPATIBLE",
            storage_code="HEIGHT_STORAGE_TYPE_INCOMPATIBLE",
        )
        return decoded
