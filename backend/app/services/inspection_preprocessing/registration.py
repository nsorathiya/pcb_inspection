from __future__ import annotations

from app.services.inspection_preprocessing.exceptions import PreprocessingKnownFailure
from app.services.inspection_preprocessing.models import (
    HeightProcessedBranch,
    InspectionPreprocessingPolicy,
    RGBProcessedBranch,
    RegistrationMode,
    RegistrationProcessingResult,
    ValidatedInspectionInput,
)

SYNTHETIC_REGISTRATION_WARNING = (
    "Synthetic identity only; equal dimensions do not prove physical registration."
)


class SyntheticIdentityRegistrationProcessor:
    async def coordinate_registration(
        self,
        rgb: RGBProcessedBranch,
        height: HeightProcessedBranch,
        inputs: ValidatedInspectionInput,
        policy: InspectionPreprocessingPolicy,
    ) -> RegistrationProcessingResult:
        config = policy.registration
        if (
            config.registration_mode is not RegistrationMode.SYNTHETIC_IDENTITY_ONLY
            or not config.allow_synthetic_identity_transform
            or not policy.development_only
            or policy.production_approved
            or not inputs.synthetic_input
        ):
            raise PreprocessingKnownFailure(
                "REGISTRATION_MODE_UNSUPPORTED", branch="REGISTRATION"
            )
        rgb_size = (rgb.output.descriptor.width, rgb.output.descriptor.height)
        height_size = (
            height.output.descriptor.width,
            height.output.descriptor.height,
        )
        if policy.output.require_matching_output_dimensions and rgb_size != height_size:
            raise PreprocessingKnownFailure(
                "OUTPUT_DIMENSION_RELATIONSHIP_INVALID",
                branch="REGISTRATION",
                field="output_dimensions",
                details={
                    "rgb_width": rgb_size[0],
                    "rgb_height": rgb_size[1],
                    "height_width": height_size[0],
                    "height_height": height_size[1],
                },
            )
        return RegistrationProcessingResult(
            registration_mode=RegistrationMode.SYNTHETIC_IDENTITY_ONLY.value,
            registration_status="SYNTHETIC_IDENTITY",
            transform_applied=False,
            transform_reference=None,
            synthetic_identity=True,
            output_coordinate_reference=config.output_coordinate_reference,
            registration_warning=SYNTHETIC_REGISTRATION_WARNING,
        )
