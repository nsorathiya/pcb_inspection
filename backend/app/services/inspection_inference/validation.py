from __future__ import annotations

import math
import re
import struct
from hashlib import sha256
from math import prod

from app.services.inspection_inference.exceptions import InferenceKnownFailure
from app.services.inspection_inference.models import (
    InferenceInputIdentity,
    InspectionInferencePolicy,
    SyntheticInferenceInput,
    ValidatedInferenceInput,
)
from app.services.inspection_preprocessing.models import InternalPreprocessedBuffer

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


class SyntheticInferenceInputValidator:
    """Validate actual immutable bytes and descriptors before mock selection."""

    def validate(
        self,
        inputs: SyntheticInferenceInput,
        policy: InspectionInferencePolicy,
    ) -> ValidatedInferenceInput:
        self._validate_prerequisites(inputs, policy)
        rgb, rgb_identity = self._validate_branch(
            inputs.rgb_buffer,
            branch="RGB",
            expected_channels=3,
            accepted_layouts=policy.prerequisites.accepted_rgb_layouts,
            accepted_data_types=policy.prerequisites.accepted_rgb_data_types,
        )
        height, height_identity = self._validate_branch(
            inputs.height_buffer,
            branch="HEIGHT",
            expected_channels=1,
            accepted_layouts=policy.prerequisites.accepted_height_layouts,
            accepted_data_types=policy.prerequisites.accepted_height_data_types,
        )
        if policy.prerequisites.require_matching_spatial_dimensions and (
            rgb_identity.width,
            rgb_identity.height,
        ) != (height_identity.width, height_identity.height):
            raise InferenceKnownFailure(
                "INPUT_DIMENSION_RELATIONSHIP_INVALID",
                branch="PAIR",
                field="spatial_dimensions",
                details={
                    "rgb_width": rgb_identity.width,
                    "rgb_height": rgb_identity.height,
                    "height_width": height_identity.width,
                    "height_height": height_identity.height,
                },
            )
        return ValidatedInferenceInput(
            source=inputs,
            rgb_buffer=rgb,
            height_buffer=height,
            rgb_identity=rgb_identity,
            height_identity=height_identity,
        )

    @staticmethod
    def _validate_prerequisites(
        inputs: SyntheticInferenceInput,
        policy: InspectionInferencePolicy,
    ) -> None:
        if inputs.preprocessing_outcome is None:
            raise InferenceKnownFailure("PREPROCESSING_RESULT_REQUIRED")
        if inputs.preprocessing_outcome != policy.prerequisites.required_preprocessing_outcome:
            raise InferenceKnownFailure("PREPROCESSING_NOT_SUCCEEDED")
        if not inputs.synthetic_input:
            if policy.prerequisites.require_synthetic_input:
                raise InferenceKnownFailure("SYNTHETIC_INPUT_REQUIRED")
            if not policy.safety.allow_real_input:
                raise InferenceKnownFailure("REAL_INPUT_NOT_ALLOWED")
        if policy.prerequisites.require_mock_preprocessing and not inputs.mock_preprocessing:
            raise InferenceKnownFailure("MOCK_PREPROCESSING_REQUIRED")
        if not policy.safety.allow_mock_engine:
            raise InferenceKnownFailure("MOCK_ENGINE_NOT_ALLOWED")
        if (
            policy.contract_version != "pcb-aoi-inspection-inference-policy/1.0"
            or policy.engine.engine_type != "MOCK"
            or not policy.development_only
            or policy.production_approved
            or policy.safety.allow_real_input
            or policy.safety.allow_model_accuracy_claim
            or policy.safety.allow_production_decision
            or policy.engine.confidence_mode != "NONE"
        ):
            raise InferenceKnownFailure("INFERENCE_POLICY_INVALID")

    @staticmethod
    def _validate_branch(
        buffer: InternalPreprocessedBuffer | None,
        *,
        branch: str,
        expected_channels: int,
        accepted_layouts: tuple[str, ...],
        accepted_data_types: tuple[str, ...],
    ) -> tuple[InternalPreprocessedBuffer, InferenceInputIdentity]:
        prefix = "RGB" if branch == "RGB" else "HEIGHT"
        if buffer is None:
            raise InferenceKnownFailure(f"{prefix}_BUFFER_REQUIRED", branch=branch)
        descriptor = buffer.descriptor
        layout = _enum_value(descriptor.layout)
        data_type = _enum_value(descriptor.data_type)
        if layout not in accepted_layouts or layout != "CHW":
            raise InferenceKnownFailure(
                f"{prefix}_BUFFER_LAYOUT_UNSUPPORTED", branch=branch, field="layout"
            )
        if data_type not in accepted_data_types or data_type != "float32":
            raise InferenceKnownFailure(
                f"{prefix}_BUFFER_DATA_TYPE_UNSUPPORTED",
                branch=branch,
                field="data_type",
            )
        shape = descriptor.shape
        if (
            not isinstance(shape, tuple)
            or len(shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
            or shape[0] != expected_channels
            or descriptor.channel_count != expected_channels
            or descriptor.height != shape[1]
            or descriptor.width != shape[2]
        ):
            raise InferenceKnownFailure(
                f"{prefix}_BUFFER_SHAPE_INVALID", branch=branch, field="shape"
            )
        if (
            descriptor.byte_order != "LITTLE_ENDIAN"
            or descriptor.contiguous is not True
            or descriptor.finite_values_verified is not True
            or not _SHA256.fullmatch(descriptor.source_artifact_sha256)
            or not isinstance(buffer.data, bytes)
        ):
            raise InferenceKnownFailure(
                f"{prefix}_BUFFER_DESCRIPTOR_INVALID",
                branch=branch,
                field="descriptor",
            )
        expected_elements = prod(shape)
        expected_bytes = expected_elements * 4
        if (
            buffer.element_count != expected_elements
            or buffer.byte_size != expected_bytes
            or len(buffer.data) != expected_bytes
        ):
            raise InferenceKnownFailure(
                f"{prefix}_BUFFER_LENGTH_MISMATCH",
                branch=branch,
                field="byte_size",
            )
        actual_hash = sha256(buffer.data).hexdigest()
        if not _SHA256.fullmatch(buffer.content_sha256) or buffer.content_sha256 != actual_hash:
            raise InferenceKnownFailure(
                f"{prefix}_BUFFER_HASH_MISMATCH",
                branch=branch,
                field="buffer_sha256",
            )
        try:
            values = struct.unpack("<" + "f" * expected_elements, buffer.data)
        except struct.error as exc:
            raise InferenceKnownFailure(
                f"{prefix}_BUFFER_LENGTH_MISMATCH", branch=branch, field="byte_size"
            ) from exc
        if any(not math.isfinite(value) for value in values):
            raise InferenceKnownFailure(
                f"{prefix}_BUFFER_DESCRIPTOR_INVALID",
                branch=branch,
                field="finite_values_verified",
            )
        identity = InferenceInputIdentity(
            buffer_sha256=actual_hash,
            shape=shape,
            layout=layout,
            data_type=data_type,
            channel_count=descriptor.channel_count,
            width=descriptor.width,
            height=descriptor.height,
            byte_size=buffer.byte_size,
            source_artifact_sha256=descriptor.source_artifact_sha256,
        )
        return buffer, identity
