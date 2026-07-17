from __future__ import annotations

import math
import struct
from hashlib import sha256
from math import prod
from typing import Iterable

from app.services.inspection_preprocessing.exceptions import PreprocessingKnownFailure
from app.services.inspection_preprocessing.models import (
    InternalPreprocessedBuffer,
    OutputDataType,
    OutputLayout,
    PreprocessedBufferDescriptor,
)


def float32_buffer(
    values: Iterable[int | float], descriptor: PreprocessedBufferDescriptor
) -> tuple[InternalPreprocessedBuffer, tuple[float, ...]]:
    converted = tuple(float(value) for value in values)
    data = struct.pack("<" + "f" * len(converted), *converted)
    exact = tuple(struct.unpack("<" + "f" * len(converted), data))
    return InternalPreprocessedBuffer.from_bytes(descriptor, data), exact


def safe_statistics(values: tuple[float, ...]) -> dict[str, int | float | None]:
    finite = tuple(value for value in values if math.isfinite(value))
    nonfinite_count = len(values) - len(finite)
    if finite:
        mean = math.fsum(finite) / len(finite)
        variance = math.fsum((value - mean) ** 2 for value in finite) / len(finite)
        minimum: float | None = min(finite)
        maximum: float | None = max(finite)
        standard_deviation: float | None = math.sqrt(variance)
    else:
        minimum = maximum = mean = standard_deviation = None
    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "finite_value_count": len(finite),
        "nonfinite_value_count": nonfinite_count,
        "element_count": len(values),
    }


def validate_internal_buffer(
    buffer: InternalPreprocessedBuffer,
    *,
    channel_count: int,
    width: int,
    height: int,
) -> None:
    descriptor = buffer.descriptor
    expected_shape = (channel_count, height, width)
    if descriptor.shape != expected_shape or buffer.element_count != prod(expected_shape):
        raise PreprocessingKnownFailure(
            "OUTPUT_SHAPE_INVALID", branch="OUTPUT", field="shape"
        )
    if (
        descriptor.layout is not OutputLayout.CHW
        or not descriptor.contiguous
        or descriptor.byte_order != "LITTLE_ENDIAN"
    ):
        raise PreprocessingKnownFailure(
            "OUTPUT_LAYOUT_INVALID", branch="OUTPUT", field="layout"
        )
    if descriptor.data_type is not OutputDataType.FLOAT32:
        raise PreprocessingKnownFailure(
            "OUTPUT_DATA_TYPE_INVALID", branch="OUTPUT", field="data_type"
        )
    if (
        descriptor.channel_count != channel_count
        or descriptor.width != width
        or descriptor.height != height
    ):
        raise PreprocessingKnownFailure(
            "OUTPUT_SHAPE_INVALID", branch="OUTPUT", field="dimensions"
        )
    if (
        buffer.byte_size != len(buffer.data)
        or buffer.byte_size != buffer.element_count * 4
        or buffer.content_sha256 != sha256(buffer.data).hexdigest()
    ):
        raise PreprocessingKnownFailure(
            "OUTPUT_SHAPE_INVALID", branch="OUTPUT", field="buffer_length"
        )
    values = struct.unpack("<" + "f" * buffer.element_count, buffer.data)
    finite = all(math.isfinite(value) for value in values)
    if descriptor.finite_values_verified != finite or not finite:
        raise PreprocessingKnownFailure(
            "OUTPUT_NONFINITE_VALUES", branch="OUTPUT", field="finite_values"
        )
