from __future__ import annotations

from app.services.inspection_preprocessing.buffers import float32_buffer, safe_statistics
from app.services.inspection_preprocessing.exceptions import PreprocessingKnownFailure
from app.services.inspection_preprocessing.models import (
    InspectionPreprocessingPolicy,
    OutputDataType,
    OutputLayout,
    PreprocessedBufferDescriptor,
    RGBNormalizationMode,
    RGBProcessedBranch,
    RGBPreprocessingOutput,
    ROIMode,
    ResizeMode,
    ValidatedInspectionInput,
)
from app.services.inspection_preprocessing.raster_decoding import SyntheticRasterDecoder


class SyntheticRGBPreprocessor:
    def __init__(self, decoder: SyntheticRasterDecoder | None = None) -> None:
        self._decoder = decoder or SyntheticRasterDecoder()

    async def preprocess_rgb(
        self, inputs: ValidatedInspectionInput, policy: InspectionPreprocessingPolicy
    ) -> RGBProcessedBranch:
        config = policy.rgb
        identity = inputs.rgb.identity
        if identity.detected_format not in policy.prerequisites.accepted_rgb_formats:
            raise PreprocessingKnownFailure("RGB_FORMAT_INCOMPATIBLE", branch="RGB")
        if identity.channels not in policy.prerequisites.accepted_rgb_channels:
            raise PreprocessingKnownFailure("RGB_CHANNELS_INCOMPATIBLE", branch="RGB")
        if identity.bit_depth not in policy.prerequisites.accepted_rgb_bit_depths:
            raise PreprocessingKnownFailure("RGB_BIT_DEPTH_INCOMPATIBLE", branch="RGB")
        if config.roi_mode is not ROIMode.FULL_FRAME or config.roi_rectangle is not None:
            raise PreprocessingKnownFailure("RGB_ROI_INVALID", branch="RGB")
        if config.resize_mode is not ResizeMode.NONE:
            raise PreprocessingKnownFailure("RGB_PREPROCESSING_UNSUPPORTED", branch="RGB")
        if (
            config.output_channels != 3
            or config.output_data_type is not OutputDataType.FLOAT32
            or config.output_layout is not OutputLayout.CHW
            or config.normalization_mode is not RGBNormalizationMode.UNIT_RANGE
        ):
            raise PreprocessingKnownFailure("RGB_PREPROCESSING_UNSUPPORTED", branch="RGB")

        decoded = self._decoder.decode_rgb(inputs.rgb)
        maximum = float((1 << decoded.metadata.bit_depth) - 1)
        interleaved = decoded.values
        chw = tuple(
            float(interleaved[index]) / maximum
            for channel in range(3)
            for index in range(channel, len(interleaved), 3)
        )
        descriptor = PreprocessedBufferDescriptor(
            shape=(3, decoded.metadata.height, decoded.metadata.width),
            layout=OutputLayout.CHW,
            data_type=OutputDataType.FLOAT32,
            channel_count=3,
            width=decoded.metadata.width,
            height=decoded.metadata.height,
            byte_order="LITTLE_ENDIAN",
            contiguous=True,
            finite_values_verified=True,
            source_artifact_sha256=identity.sha256,
        )
        buffer, exact_values = float32_buffer(chw, descriptor)
        output = RGBPreprocessingOutput(
            descriptor=descriptor,
            roi_mode=ROIMode.FULL_FRAME.value,
            safe_statistics=(
                safe_statistics(exact_values)
                if policy.output.include_safe_summary_statistics
                else None
            ),
            normalization_mode=RGBNormalizationMode.UNIT_RANGE.value,
        )
        return RGBProcessedBranch(buffer=buffer, output=output)
