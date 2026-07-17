from __future__ import annotations

import math

from app.services.inspection_preprocessing.buffers import float32_buffer, safe_statistics
from app.services.inspection_preprocessing.exceptions import PreprocessingKnownFailure
from app.services.inspection_preprocessing.models import (
    HeightProcessedBranch,
    HeightPreprocessingOutput,
    HeightScalingMode,
    InspectionPreprocessingPolicy,
    InvalidValueHandling,
    OutputDataType,
    OutputLayout,
    PreprocessedBufferDescriptor,
    ROIMode,
    ResizeMode,
    ValidatedInspectionInput,
)
from app.services.inspection_preprocessing.raster_decoding import SyntheticRasterDecoder


class SyntheticHeightPreprocessor:
    def __init__(self, decoder: SyntheticRasterDecoder | None = None) -> None:
        self._decoder = decoder or SyntheticRasterDecoder()

    async def preprocess_height(
        self, inputs: ValidatedInspectionInput, policy: InspectionPreprocessingPolicy
    ) -> HeightProcessedBranch:
        config = policy.height
        identity = inputs.height.identity
        if identity.detected_format not in policy.prerequisites.accepted_height_formats:
            raise PreprocessingKnownFailure("HEIGHT_FORMAT_INCOMPATIBLE", branch="HEIGHT")
        if identity.channels != 1:
            raise PreprocessingKnownFailure("HEIGHT_CHANNELS_INCOMPATIBLE", branch="HEIGHT")
        if identity.storage_data_type not in policy.prerequisites.accepted_height_storage_types:
            raise PreprocessingKnownFailure(
                "HEIGHT_STORAGE_TYPE_INCOMPATIBLE", branch="HEIGHT"
            )
        if config.roi_mode is not ROIMode.FULL_FRAME or config.roi_rectangle is not None:
            raise PreprocessingKnownFailure("HEIGHT_ROI_INVALID", branch="HEIGHT")
        if config.resize_mode is not ResizeMode.NONE:
            raise PreprocessingKnownFailure("HEIGHT_PREPROCESSING_UNSUPPORTED", branch="HEIGHT")
        if (
            config.output_channels != 1
            or config.output_data_type is not OutputDataType.FLOAT32
            or config.output_layout is not OutputLayout.CHW
            or config.scaling_mode is not HeightScalingMode.NONE
        ):
            raise PreprocessingKnownFailure("HEIGHT_PREPROCESSING_UNSUPPORTED", branch="HEIGHT")
        if config.invalid_value_handling is not InvalidValueHandling.REJECT:
            raise PreprocessingKnownFailure(
                "HEIGHT_INVALID_VALUE_POLICY_UNSUPPORTED", branch="HEIGHT"
            )

        decoded = self._decoder.decode_height(inputs.height)
        values = tuple(float(value) for value in decoded.values)
        if any(not math.isfinite(value) for value in values):
            raise PreprocessingKnownFailure(
                "OUTPUT_NONFINITE_VALUES", branch="HEIGHT", field="values"
            )
        descriptor = PreprocessedBufferDescriptor(
            shape=(1, decoded.metadata.height, decoded.metadata.width),
            layout=OutputLayout.CHW,
            data_type=OutputDataType.FLOAT32,
            channel_count=1,
            width=decoded.metadata.width,
            height=decoded.metadata.height,
            byte_order="LITTLE_ENDIAN",
            contiguous=True,
            finite_values_verified=True,
            source_artifact_sha256=identity.sha256,
        )
        buffer, exact_values = float32_buffer(values, descriptor)
        output = HeightPreprocessingOutput(
            descriptor=descriptor,
            roi_mode=ROIMode.FULL_FRAME.value,
            safe_statistics=(
                safe_statistics(exact_values)
                if policy.output.include_safe_summary_statistics
                else None
            ),
            scaling_mode=HeightScalingMode.NONE.value,
            invalid_value_handling=InvalidValueHandling.REJECT.value,
            physical_unit=None,
            physical_scale_applied=False,
        )
        return HeightProcessedBranch(buffer=buffer, output=output)
