from __future__ import annotations

import json
from dataclasses import replace

from app.testing.synthetic_aoi.models import (
    GENERATOR_ID,
    GENERATOR_VERSION,
    SYNTHETIC_STATEMENT,
    ArtifactPlan,
    ScenarioPlan,
)
from app.testing.synthetic_aoi.raster_generation import (
    encode_classic_tiff,
    encode_npy_float32,
    encode_png,
    float32_height_values,
    height_uint16_values,
    rgb_pattern,
    uint16_big_endian,
    uint16_little_endian,
)

SCENARIO_IDS = (
    "valid_rgb_png_height_tiff",
    "valid_rgb_tiff_height_png16",
    "valid_rgb_png_height_npy_float32",
    "valid_different_dimensions",
    "valid_with_mask_and_calibration_evidence",
    "missing_rgb",
    "missing_height",
    "corrupt_rgb",
    "corrupt_height",
    "truncated_rgb_png",
    "truncated_height_tiff",
    "height_png_uint8",
    "height_png_rgb",
    "height_png_rgba",
    "height_colorized_preview",
    "unsupported_rgb_extension",
    "unsupported_height_extension",
    "hash_mismatch_rgb",
    "hash_mismatch_height",
    "byte_size_mismatch_rgb",
    "byte_size_mismatch_height",
    "dimension_mismatch_without_registration",
    "required_mask_missing",
    "required_calibration_missing",
    "required_registration_missing",
    "duplicate_rgb_reference",
    "duplicate_height_reference",
    "unsafe_relative_path_reference",
)

DEFAULT_POLICY_ID = "development-native-rgb-height"
DEFAULT_POLICY_VERSION = "1.0"
RGB_WIDTH = 16
RGB_HEIGHT = 12


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _rgb_png(seed: int, scenario_id: str, width: int = RGB_WIDTH, height: int = RGB_HEIGHT) -> ArtifactPlan:
    content = encode_png(
        width,
        height,
        bit_depth=8,
        color_type=2,
        pixel_bytes=rgb_pattern(width, height, seed, scenario_id),
    )
    return ArtifactPlan("rgb", ("rgb.png",), "rgb.png", content, "image/png")


def _rgb_tiff(seed: int, scenario_id: str) -> ArtifactPlan:
    content = encode_classic_tiff(
        RGB_WIDTH,
        RGB_HEIGHT,
        samples=3,
        bits=8,
        photometric=2,
        pixel_bytes=rgb_pattern(RGB_WIDTH, RGB_HEIGHT, seed, scenario_id),
    )
    return ArtifactPlan("rgb", ("rgb.tiff",), "rgb.tiff", content, "image/tiff")


def _height_tiff(
    seed: int,
    scenario_id: str,
    width: int = RGB_WIDTH,
    height: int = RGB_HEIGHT,
) -> ArtifactPlan:
    values = height_uint16_values(width, height, seed, scenario_id)
    content = encode_classic_tiff(
        width,
        height,
        samples=1,
        bits=16,
        photometric=1,
        pixel_bytes=uint16_little_endian(values),
    )
    return ArtifactPlan(
        "height",
        ("height.tiff",),
        "height.tiff",
        content,
        "image/tiff",
    )


def _height_png(
    seed: int,
    scenario_id: str,
    *,
    bit_depth: int = 16,
    color_type: int = 0,
) -> ArtifactPlan:
    if color_type == 0 and bit_depth == 16:
        pixels = uint16_big_endian(
            height_uint16_values(RGB_WIDTH, RGB_HEIGHT, seed, scenario_id)
        )
    elif color_type == 0:
        pixels = bytes(
            value & 0xFF
            for value in height_uint16_values(
                RGB_WIDTH,
                RGB_HEIGHT,
                seed,
                scenario_id,
            )
        )
    else:
        channels = {2: 3, 6: 4}[color_type]
        rgb = rgb_pattern(RGB_WIDTH, RGB_HEIGHT, seed, scenario_id)
        pixels = bytearray()
        for index in range(0, len(rgb), 3):
            pixels.extend(rgb[index : index + 3])
            if channels == 4:
                pixels.append(255)
        pixels = bytes(pixels)
    content = encode_png(
        RGB_WIDTH,
        RGB_HEIGHT,
        bit_depth=bit_depth,
        color_type=color_type,
        pixel_bytes=pixels,
    )
    return ArtifactPlan(
        "height",
        ("height.png",),
        "height.png",
        content,
        "image/png",
    )


def _height_npy(seed: int, scenario_id: str) -> ArtifactPlan:
    content = encode_npy_float32(
        RGB_WIDTH,
        RGB_HEIGHT,
        float32_height_values(RGB_WIDTH, RGB_HEIGHT, seed, scenario_id),
    )
    return ArtifactPlan(
        "height",
        ("height.npy",),
        "height.npy",
        content,
        "application/octet-stream",
    )


def _validity_mask() -> ArtifactPlan:
    pixels = bytes(
        0 if x == RGB_WIDTH - 1 and y == RGB_HEIGHT - 1 else 255
        for y in range(RGB_HEIGHT)
        for x in range(RGB_WIDTH)
    )
    content = encode_png(
        RGB_WIDTH,
        RGB_HEIGHT,
        bit_depth=8,
        color_type=0,
        pixel_bytes=pixels,
    )
    return ArtifactPlan(
        "mask",
        ("validity_mask.png",),
        "validity_mask.png",
        content,
        "image/png",
    )


def _calibration_evidence(seed: int, scenario_id: str) -> ArtifactPlan:
    content = _json_bytes(
        {
            "fixture_statement": SYNTHETIC_STATEMENT,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "model_accuracy_evidence": False,
            "note": "Synthetic calibration-shaped evidence; values are not physically meaningful.",
            "production_approved": False,
            "scenario_id": scenario_id,
            "seed": seed,
            "synthetic": True,
            "training_approved": False,
        }
    )
    return ArtifactPlan(
        "calibration",
        ("calibration.json",),
        "calibration.json",
        content,
        "application/json",
    )


def _plan(
    scenario_id: str,
    description: str,
    artifacts: tuple[ArtifactPlan, ...],
    *,
    intake: str = "ACCEPTED",
    validation: str = "VALIDATION_PASSED",
    findings: tuple[str, ...] = (),
    policy_id: str = DEFAULT_POLICY_ID,
    notes: tuple[str, ...] = (),
) -> ScenarioPlan:
    return ScenarioPlan(
        scenario_id=scenario_id,
        description=description,
        expected_intake_outcome=intake,
        expected_technical_validation_outcome=validation,
        expected_finding_codes=findings,
        policy_id=policy_id,
        policy_version=DEFAULT_POLICY_VERSION,
        artifacts=artifacts,
        notes=notes
        or ("Software fixture only; no physical acquisition behavior is represented.",),
    )


def build_scenario(scenario_id: str, seed: int) -> ScenarioPlan:
    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"Unknown synthetic scenario {scenario_id!r}")
    rgb = _rgb_png(seed, scenario_id)
    height = _height_tiff(seed, scenario_id)

    if scenario_id == "valid_rgb_png_height_tiff":
        return _plan(scenario_id, "Readable RGB PNG and scalar uint16 TIFF with equal dimensions.", (rgb, height))
    if scenario_id == "valid_rgb_tiff_height_png16":
        return _plan(
            scenario_id,
            "Readable RGB TIFF and scalar 16-bit grayscale PNG with equal dimensions.",
            (_rgb_tiff(seed, scenario_id), _height_png(seed, scenario_id)),
        )
    if scenario_id == "valid_rgb_png_height_npy_float32":
        return _plan(
            scenario_id,
            "Readable RGB PNG and two-dimensional float32 NPY with equal dimensions.",
            (rgb, _height_npy(seed, scenario_id)),
        )
    if scenario_id == "valid_different_dimensions":
        different_height = _height_tiff(seed, scenario_id, width=8, height=6)
        return _plan(
            scenario_id,
            "Both files are readable but use deliberately different dimensions.",
            (rgb, different_height),
            validation="VALIDATION_FAILED",
            findings=("DIMENSION_RELATIONSHIP_UNSUPPORTED",),
            notes=(
                "An appropriate policy would require reviewed registration-transform evidence; none is implied here.",
            ),
        )
    if scenario_id == "valid_with_mask_and_calibration_evidence":
        return _plan(
            scenario_id,
            "Readable equal-dimension pair with a validity mask and synthetic calibration-shaped evidence.",
            (rgb, height, _validity_mask(), _calibration_evidence(seed, scenario_id)),
            notes=("Synthetic calibration values are not physically meaningful.",),
        )
    if scenario_id == "missing_rgb":
        return _plan(
            scenario_id,
            "RGB artifact reference is deliberately absent.",
            (height,),
            intake="REJECTED",
            validation="VALIDATION_FAILED",
            findings=("RGB_RAW_MISSING", "INCOMPLETE_RAW_PAIR"),
        )
    if scenario_id == "missing_height":
        return _plan(
            scenario_id,
            "Height artifact reference is deliberately absent.",
            (rgb,),
            intake="REJECTED",
            validation="VALIDATION_FAILED",
            findings=("HEIGHT_RAW_MISSING", "INCOMPLETE_RAW_PAIR"),
        )
    if scenario_id == "corrupt_rgb":
        corrupt = replace(rgb, content=b"SYNTHETIC-CORRUPT-RGB-CONTENT\n")
        return _plan(scenario_id, "RGB bytes have no supported image structure.", (corrupt, height), validation="VALIDATION_FAILED", findings=("FILE_UNREADABLE",))
    if scenario_id == "corrupt_height":
        corrupt = replace(height, content=b"SYNTHETIC-CORRUPT-HEIGHT-CONTENT\n")
        return _plan(scenario_id, "Height bytes have no supported native raster structure.", (rgb, corrupt), validation="VALIDATION_FAILED", findings=("FILE_UNREADABLE",))
    if scenario_id == "truncated_rgb_png":
        truncated = replace(rgb, content=rgb.content[:-7])
        return _plan(scenario_id, "RGB PNG is deliberately truncated.", (truncated, height), validation="VALIDATION_FAILED", findings=("FILE_UNREADABLE",))
    if scenario_id == "truncated_height_tiff":
        truncated = replace(height, content=height.content[:-3])
        return _plan(scenario_id, "Height TIFF strip is deliberately truncated.", (rgb, truncated), validation="VALIDATION_FAILED", findings=("FILE_UNREADABLE",))
    if scenario_id == "height_png_uint8":
        invalid = _height_png(seed, scenario_id, bit_depth=8)
        return _plan(scenario_id, "Height PNG is grayscale but only 8-bit.", (rgb, invalid), validation="VALIDATION_FAILED", findings=("HEIGHT_BIT_DEPTH_TOO_LOW",))
    if scenario_id == "height_png_rgb":
        invalid = _height_png(seed, scenario_id, bit_depth=8, color_type=2)
        return _plan(scenario_id, "Height reference contains an RGB PNG.", (rgb, invalid), validation="VALIDATION_FAILED", findings=("HEIGHT_NOT_SINGLE_CHANNEL",))
    if scenario_id == "height_png_rgba":
        invalid = _height_png(seed, scenario_id, bit_depth=8, color_type=6)
        return _plan(scenario_id, "Height reference contains an RGBA PNG.", (rgb, invalid), validation="VALIDATION_FAILED", findings=("HEIGHT_NOT_SINGLE_CHANNEL",))
    if scenario_id == "height_colorized_preview":
        invalid = replace(
            _height_png(seed, scenario_id, bit_depth=8, color_type=2),
            references=("height_colorized_preview.png",),
            generated_file="height_colorized_preview.png",
        )
        return _plan(scenario_id, "Height reference is a colorized preview PNG.", (rgb, invalid), validation="VALIDATION_FAILED", findings=("HEIGHT_COLORIZED_PREVIEW_REJECTED",))
    if scenario_id == "unsupported_rgb_extension":
        invalid = replace(rgb, references=("rgb.unsupported",), generated_file="rgb.unsupported")
        return _plan(scenario_id, "Valid PNG bytes use an unsupported RGB intake extension.", (invalid, height), intake="REJECTED", validation="VALIDATION_FAILED", findings=("EXTENSION_CONTENT_MISMATCH",))
    if scenario_id == "unsupported_height_extension":
        invalid = replace(height, references=("height.unsupported",), generated_file="height.unsupported")
        return _plan(scenario_id, "Valid TIFF bytes use an unsupported height intake extension.", (rgb, invalid), intake="REJECTED", validation="VALIDATION_FAILED", findings=("EXTENSION_CONTENT_MISMATCH",))
    if scenario_id == "hash_mismatch_rgb":
        mismatch = replace(rgb, declared_sha256="0" * 64)
        return _plan(scenario_id, "Declared RGB hash deliberately differs from the exact bytes.", (mismatch, height), intake="REJECTED", validation="VALIDATION_FAILED", findings=("ARTIFACT_SHA256_MISMATCH",))
    if scenario_id == "hash_mismatch_height":
        mismatch = replace(height, declared_sha256="0" * 64)
        return _plan(scenario_id, "Declared height hash deliberately differs from the exact bytes.", (rgb, mismatch), intake="REJECTED", validation="VALIDATION_FAILED", findings=("ARTIFACT_SHA256_MISMATCH",))
    if scenario_id == "byte_size_mismatch_rgb":
        mismatch = replace(rgb, declared_byte_size=len(rgb.content) + 1)
        return _plan(scenario_id, "Declared RGB size deliberately differs from the exact bytes.", (mismatch, height), intake="REJECTED", validation="VALIDATION_FAILED", findings=("ARTIFACT_SIZE_MISMATCH",))
    if scenario_id == "byte_size_mismatch_height":
        mismatch = replace(height, declared_byte_size=len(height.content) + 1)
        return _plan(scenario_id, "Declared height size deliberately differs from the exact bytes.", (rgb, mismatch), intake="REJECTED", validation="VALIDATION_FAILED", findings=("ARTIFACT_SIZE_MISMATCH",))
    if scenario_id == "dimension_mismatch_without_registration":
        return _plan(
            scenario_id,
            "Readable files have different dimensions and no registration evidence.",
            (rgb, _height_tiff(seed, scenario_id, width=8, height=6)),
            validation="VALIDATION_FAILED",
            findings=("DIMENSION_RELATIONSHIP_UNSUPPORTED", "REGISTRATION_EVIDENCE_MISSING"),
        )
    if scenario_id == "required_mask_missing":
        return _plan(scenario_id, "Selected synthetic policy requires a missing validity mask.", (rgb, height), validation="VALIDATION_FAILED", findings=("VALIDITY_MASK_MISSING",), policy_id="synthetic-requires-validity-mask")
    if scenario_id == "required_calibration_missing":
        return _plan(scenario_id, "Selected synthetic policy requires missing calibration evidence.", (rgb, height), validation="VALIDATION_FAILED", findings=("CALIBRATION_EVIDENCE_MISSING",), policy_id="synthetic-requires-calibration")
    if scenario_id == "required_registration_missing":
        return _plan(scenario_id, "Selected synthetic policy requires missing registration evidence.", (rgb, height), validation="VALIDATION_FAILED", findings=("REGISTRATION_EVIDENCE_MISSING",), policy_id="synthetic-requires-registration")
    if scenario_id == "duplicate_rgb_reference":
        duplicate = replace(rgb, references=("rgb.png", "rgb.png"))
        return _plan(scenario_id, "RGB artifact reference is deliberately duplicated.", (duplicate, height), intake="REJECTED", validation="VALIDATION_FAILED", findings=("DUPLICATE_RGB_RAW",))
    if scenario_id == "duplicate_height_reference":
        duplicate = replace(height, references=("height.tiff", "height.tiff"))
        return _plan(scenario_id, "Height artifact reference is deliberately duplicated.", (rgb, duplicate), intake="REJECTED", validation="VALIDATION_FAILED", findings=("DUPLICATE_HEIGHT_RAW",))
    if scenario_id == "unsafe_relative_path_reference":
        unsafe = replace(
            rgb,
            references=("../outside.png",),
            reference_expected_to_resolve=False,
        )
        return _plan(scenario_id, "RGB metadata contains a deliberate parent-directory traversal reference.", (unsafe, height), intake="REJECTED", validation="VALIDATION_FAILED", findings=("ARTIFACT_PATH_UNSAFE",))
    raise AssertionError("Scenario catalogue and builder are out of sync")
