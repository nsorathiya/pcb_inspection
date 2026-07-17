from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from app.services.inspection_preprocessing.exceptions import PreprocessingPolicyLoadError
from app.services.inspection_preprocessing.models import (
    HeightPreprocessingPolicy,
    HeightScalingMode,
    InspectionPreprocessingPolicy,
    InterpolationMode,
    InvalidValueHandling,
    OutputDataType,
    OutputLayout,
    PreprocessingOutputPolicy,
    PreprocessingPrerequisites,
    PreprocessingSafetyPolicy,
    RGBNormalizationMode,
    RGBPreprocessingPolicy,
    ROIMode,
    ROIRectangle,
    RegistrationMode,
    RegistrationPolicy,
    ResizeMode,
    validate_preprocessing_policy_document,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCHEMA_PATH = REPOSITORY_ROOT / "contracts" / "inspection_preprocessing_policy.schema.json"
DEFAULT_POLICY_PATH = REPOSITORY_ROOT / "contracts" / "examples" / "inspection_preprocessing_policy.synthetic.json"
SYNTHETIC_POLICY_ID = "synthetic-paired-rgb-height"
SYNTHETIC_POLICY_VERSION = "1.0"
POLICY_CONTRACT_VERSION = "pcb-aoi-inspection-preprocessing-policy/1.0"


def _rectangle(value: Mapping[str, Any] | None) -> ROIRectangle | None:
    return None if value is None else ROIRectangle(**dict(value))


def _to_policy(document: Mapping[str, Any]) -> InspectionPreprocessingPolicy:
    prerequisites = document["prerequisites"]
    rgb = document["rgb"]
    height = document["height"]
    registration = document["registration"]
    output = document["output"]
    safety = document["safety"]
    normalization = {
        key: tuple(values) for key, values in rgb["normalization_parameters"].items()
    }
    return InspectionPreprocessingPolicy(
        contract_version=document["contract_version"],
        policy_id=document["policy_id"],
        policy_version=document["policy_version"],
        name=document["name"],
        description=document["description"],
        development_only=document["development_only"],
        production_approved=document["production_approved"],
        prerequisites=PreprocessingPrerequisites(
            required_inspection_status=prerequisites["required_inspection_status"],
            required_validation_outcome=prerequisites["required_validation_outcome"],
            accepted_rgb_formats=tuple(prerequisites["accepted_rgb_formats"]),
            accepted_height_formats=tuple(prerequisites["accepted_height_formats"]),
            accepted_rgb_channels=tuple(prerequisites["accepted_rgb_channels"]),
            accepted_rgb_bit_depths=tuple(prerequisites["accepted_rgb_bit_depths"]),
            accepted_height_storage_types=tuple(prerequisites["accepted_height_storage_types"]),
        ),
        rgb=RGBPreprocessingPolicy(
            roi_mode=ROIMode(rgb["roi_mode"]),
            roi_rectangle=_rectangle(rgb["roi_rectangle"]),
            resize_mode=ResizeMode(rgb["resize_mode"]),
            target_width=rgb["target_width"],
            target_height=rgb["target_height"],
            output_channels=rgb["output_channels"],
            output_data_type=OutputDataType(rgb["output_data_type"]),
            output_layout=OutputLayout(rgb["output_layout"]),
            normalization_mode=RGBNormalizationMode(rgb["normalization_mode"]),
            normalization_parameters=normalization,
            preserve_aspect_ratio=rgb["preserve_aspect_ratio"],
            interpolation_mode=InterpolationMode(rgb["interpolation_mode"]),
        ),
        height=HeightPreprocessingPolicy(
            roi_mode=ROIMode(height["roi_mode"]),
            roi_rectangle=_rectangle(height["roi_rectangle"]),
            resize_mode=ResizeMode(height["resize_mode"]),
            target_width=height["target_width"],
            target_height=height["target_height"],
            output_channels=height["output_channels"],
            output_data_type=OutputDataType(height["output_data_type"]),
            output_layout=OutputLayout(height["output_layout"]),
            scaling_mode=HeightScalingMode(height["scaling_mode"]),
            scaling_parameters=dict(height["scaling_parameters"]),
            invalid_value_handling=InvalidValueHandling(height["invalid_value_handling"]),
            requires_validity_mask_input=height["requires_validity_mask_input"],
            replacement_value=height["replacement_value"],
            interpolation_mode=InterpolationMode(height["interpolation_mode"]),
        ),
        registration=RegistrationPolicy(
            registration_mode=RegistrationMode(registration["registration_mode"]),
            require_registration_evidence=registration["require_registration_evidence"],
            output_coordinate_reference=registration["output_coordinate_reference"],
            dimension_relationship=registration["dimension_relationship"],
            transform_source=registration["transform_source"],
            allow_synthetic_identity_transform=registration["allow_synthetic_identity_transform"],
        ),
        output=PreprocessingOutputPolicy(**dict(output)),
        safety=PreprocessingSafetyPolicy(**dict(safety)),
    )


class SyntheticPreprocessingPolicyLoader:
    """Load only the repository-owned synthetic policy selected by exact identity."""

    def __init__(
        self,
        *,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
        policy_document: Mapping[str, Any] | None = None,
    ) -> None:
        self._schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self._injected_document = None if policy_document is None else dict(policy_document)

    def load(self, policy_id: str, policy_version: str) -> InspectionPreprocessingPolicy:
        if policy_id != SYNTHETIC_POLICY_ID:
            raise PreprocessingPolicyLoadError("PREPROCESSING_POLICY_NOT_FOUND")
        if policy_version != SYNTHETIC_POLICY_VERSION:
            raise PreprocessingPolicyLoadError("PREPROCESSING_POLICY_VERSION_UNSUPPORTED")
        try:
            document = (
                json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
                if self._injected_document is None
                else dict(self._injected_document)
            )
            if document.get("contract_version") != POLICY_CONTRACT_VERSION:
                raise PreprocessingPolicyLoadError(
                    "PREPROCESSING_POLICY_VERSION_UNSUPPORTED"
                )
            Draft202012Validator(
                self._schema, format_checker=FormatChecker()
            ).validate(document)
            validate_preprocessing_policy_document(document)
            policy = _to_policy(document)
        except PreprocessingPolicyLoadError:
            raise
        except Exception as exc:
            raise PreprocessingPolicyLoadError("PREPROCESSING_POLICY_INVALID") from exc
        if policy.policy_id != policy_id or policy.policy_version != policy_version:
            raise PreprocessingPolicyLoadError("PREPROCESSING_POLICY_INVALID")
        return policy
