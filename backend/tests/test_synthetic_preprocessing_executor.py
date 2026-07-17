import asyncio
import hashlib
import json
import math
import struct
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from app.services.dataset_validation.file_inspection import inspect_height, inspect_rgb
from app.services.inspection_preprocessing import (
    ArtifactInputIdentity,
    SyntheticInspectionPreprocessingService,
    SyntheticPreprocessingPolicyLoader,
    ValidatedArtifactSource,
    ValidatedInspectionInput,
    preprocessing_result_json,
    validate_preprocessing_result_document,
)
from app.services.inspection_preprocessing.exceptions import PreprocessingPolicyLoadError
from app.services.inspection_preprocessing.models import OutputDataType, OutputLayout
from app.testing.synthetic_aoi import DEFAULT_SEED, generate_fixtures
from app.testing.synthetic_aoi.raster_generation import (
    encode_npy_float32,
    float32_height_values,
    height_uint16_values,
    rgb_pattern,
)

ROOT = Path(__file__).resolve().parents[2]
FIXED_TIME = datetime(2026, 7, 17, 14, 0, tzinfo=timezone.utc)
FIXED_PREPROCESSING_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
FIXED_VALIDATION_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(role: str, path: Path) -> ArtifactInputIdentity:
    metadata = inspect_rgb(path) if role == "rgb" else inspect_height(path)
    return ArtifactInputIdentity(
        artifact_type="RGB_RAW" if role == "rgb" else "HEIGHT_RAW",
        sha256=_sha256(path),
        byte_size=path.stat().st_size,
        detected_format=metadata.detected_format,
        width=metadata.width,
        height=metadata.height,
        channels=metadata.channels,
        bit_depth=metadata.bit_depth,
        storage_data_type=metadata.storage_data_type,
    )


def _source(role: str, path: Path) -> ValidatedArtifactSource:
    return ValidatedArtifactSource(_identity(role, path), path)


def _generated_input(tmp_path: Path, scenario_id: str) -> tuple[dict, Path, ValidatedInspectionInput]:
    root = tmp_path / "generated"
    generate_fixtures(root, scenario_ids=(scenario_id,))
    scenario_root = root / "scenarios" / scenario_id
    record = _json(scenario_root / "scenario.json")
    rgb_path = scenario_root / record["artifacts"]["rgb"]["generated_file"]
    height_path = scenario_root / record["artifacts"]["height"]["generated_file"]
    inputs = ValidatedInspectionInput(
        inspection_id=record["scenario_uuid"],
        validation_id=FIXED_VALIDATION_ID,
        inspection_status="READY",
        validation_outcome="VALIDATION_PASSED",
        synthetic_input=True,
        rgb=_source("rgb", rgb_path),
        height=_source("height", height_path),
    )
    return record, scenario_root, inputs


def _service(**overrides) -> SyntheticInspectionPreprocessingService:
    return SyntheticInspectionPreprocessingService(
        clock=lambda: FIXED_TIME,
        preprocessing_id_generator=lambda: FIXED_PREPROCESSING_ID,
        implementation_id="synthetic-mock-preprocessor-test",
        implementation_version="1.0.0-test",
        **overrides,
    )


def _execute(inputs, policy=None, service=None):
    selected = policy or SyntheticPreprocessingPolicyLoader().load(
        "synthetic-paired-rgb-height", "1.0"
    )
    return asyncio.run((service or _service()).preprocess_inspection(inputs, selected))


@pytest.fixture(scope="module")
def result_validator():
    catalogue = _json(ROOT / "contracts" / "inspection_preprocessing_findings.json")
    schema = _json(ROOT / "contracts" / "inspection_preprocessing_result.schema.json")
    registry = Registry().with_resource(catalogue["$id"], Resource.from_contents(catalogue))
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker()), catalogue


@pytest.mark.parametrize(
    "scenario_id",
    [
        "valid_rgb_png_height_tiff",
        "valid_rgb_tiff_height_png16",
        "valid_rgb_png_height_npy_float32",
    ],
)
def test_supported_synthetic_pairs_succeed_and_validate_schema(
    tmp_path, scenario_id, result_validator
):
    _, _, inputs = _generated_input(tmp_path, scenario_id)
    execution = _execute(inputs)
    document = execution.result.to_dict()
    validator, catalogue = result_validator

    assert execution.result.outcome.value == "PREPROCESSING_SUCCEEDED"
    assert execution.rgb_buffer is not None and execution.height_buffer is not None
    assert [item.code for item in execution.result.findings] == [
        "SYNTHETIC_IDENTITY_REGISTRATION_USED"
    ]
    assert execution.result.findings[0].blocking is False
    assert execution.result.registration.registration_status == "SYNTHETIC_IDENTITY"
    assert execution.result.registration.transform_applied is False
    assert execution.result.registration.synthetic_identity is True
    assert execution.result.registration.transform_reference is None
    validator.validate(document)
    validate_preprocessing_result_document(document, catalogue)


def test_rgb_is_float32_chw_unit_range_in_explicit_rgb_plane_order(tmp_path):
    scenario_id = "valid_rgb_png_height_tiff"
    _, _, inputs = _generated_input(tmp_path, scenario_id)
    execution = _execute(inputs)
    descriptor = execution.rgb_buffer.descriptor
    values = struct.unpack("<" + "f" * execution.rgb_buffer.element_count, execution.rgb_buffer.data)
    source = rgb_pattern(16, 12, DEFAULT_SEED, scenario_id)
    expected = tuple(
        source[index] / 255.0
        for channel in range(3)
        for index in range(channel, len(source), 3)
    )

    assert descriptor.shape == (3, 12, 16)
    assert descriptor.layout is OutputLayout.CHW
    assert descriptor.data_type is OutputDataType.FLOAT32
    assert descriptor.byte_order == "LITTLE_ENDIAN"
    assert values == pytest.approx(expected)
    assert min(values) >= 0.0 and max(values) <= 1.0


@pytest.mark.parametrize(
    "scenario_id",
    ["valid_rgb_png_height_tiff", "valid_rgb_tiff_height_png16"],
)
def test_uint16_height_values_are_preserved_as_float32_without_normalization(
    tmp_path, scenario_id
):
    _, _, inputs = _generated_input(tmp_path, scenario_id)
    execution = _execute(inputs)
    output = execution.result.height_output
    values = struct.unpack("<" + "f" * execution.height_buffer.element_count, execution.height_buffer.data)
    expected = height_uint16_values(16, 12, DEFAULT_SEED, scenario_id)

    assert execution.height_buffer.descriptor.shape == (1, 12, 16)
    assert values == pytest.approx(expected)
    assert max(values) > 255
    assert output.scaling_mode == "NONE"
    assert output.physical_unit is None
    assert output.physical_scale_applied is False


def test_float32_npy_values_are_preserved_and_statistics_are_derived(tmp_path):
    scenario_id = "valid_rgb_png_height_npy_float32"
    _, _, inputs = _generated_input(tmp_path, scenario_id)
    execution = _execute(inputs)
    values = struct.unpack("<" + "f" * execution.height_buffer.element_count, execution.height_buffer.data)
    expected = float32_height_values(16, 12, DEFAULT_SEED, scenario_id)
    statistics = execution.result.height_output.safe_statistics

    assert values == pytest.approx(expected)
    assert statistics["minimum"] == min(values)
    assert statistics["maximum"] == max(values)
    assert statistics["finite_value_count"] == len(values)
    assert statistics["nonfinite_value_count"] == 0
    assert statistics["element_count"] == len(values)


def test_fixed_providers_produce_identical_results_buffers_and_hashes(tmp_path):
    _, _, inputs = _generated_input(tmp_path, "valid_rgb_png_height_tiff")
    first = _execute(inputs)
    second = _execute(inputs)

    assert preprocessing_result_json(first.result) == preprocessing_result_json(second.result)
    assert first.rgb_buffer.data == second.rgb_buffer.data
    assert first.height_buffer.data == second.height_buffer.data
    assert first.rgb_buffer.content_sha256 == second.rgb_buffer.content_sha256
    assert first.height_buffer.content_sha256 == second.height_buffer.content_sha256


def test_sources_are_immutable_and_executor_creates_no_files(tmp_path):
    _, scenario_root, inputs = _generated_input(tmp_path, "valid_rgb_png_height_tiff")
    before_tree = {
        path.relative_to(scenario_root).as_posix(): (
            _sha256(path), path.stat().st_size, path.stat().st_mtime_ns
        )
        for path in scenario_root.rglob("*")
        if path.is_file()
    }
    before_status = inputs.inspection_status

    _execute(inputs)

    after_tree = {
        path.relative_to(scenario_root).as_posix(): (
            _sha256(path), path.stat().st_size, path.stat().st_mtime_ns
        )
        for path in scenario_root.rglob("*")
        if path.is_file()
    }
    assert after_tree == before_tree
    assert inputs.inspection_status == before_status == "READY"
    assert not list(tmp_path.rglob("*.sqlite3"))


def test_public_result_has_no_paths_or_buffer_bytes(tmp_path):
    _, _, inputs = _generated_input(tmp_path, "valid_rgb_png_height_tiff")
    document = _execute(inputs).result.to_dict()
    serialized = json.dumps(document)
    forbidden = {"source_path", "path", "filename", "data", "buffer", "tensor", "confidence", "classification", "model_id"}

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert forbidden.isdisjoint(keys(document))
    assert str(tmp_path) not in serialized
    assert not any(
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
        for value in document.values()
        if isinstance(value, str)
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"inspection_status": "RECEIVED"}, "INSPECTION_NOT_READY"),
        ({"validation_outcome": None}, "VALIDATION_RESULT_REQUIRED"),
        ({"validation_outcome": "VALIDATION_FAILED"}, "VALIDATION_NOT_PASSED"),
        ({"synthetic_input": False}, "PREPROCESSING_POLICY_INVALID"),
    ],
)
def test_prerequisite_failures_are_safe(tmp_path, change, code):
    _, _, inputs = _generated_input(tmp_path, "valid_rgb_png_height_tiff")
    execution = _execute(replace(inputs, **change))
    assert execution.result.outcome.value == "PREPROCESSING_FAILED"
    assert [item.code for item in execution.result.findings] == [code]
    assert execution.rgb_buffer is None and execution.height_buffer is None


def test_production_approved_policy_cannot_use_mock_executor(tmp_path):
    _, _, inputs = _generated_input(tmp_path, "valid_rgb_png_height_tiff")
    policy = SyntheticPreprocessingPolicyLoader().load("synthetic-paired-rgb-height", "1.0")
    policy = replace(policy, development_only=False, production_approved=True)
    execution = _execute(inputs, policy=policy)
    assert execution.result.outcome.value == "PREPROCESSING_FAILED"
    assert execution.result.findings[0].code == "PREPROCESSING_POLICY_INVALID"


def test_policy_loader_rejects_unknown_version_and_malformed_documents():
    loader = SyntheticPreprocessingPolicyLoader()
    with pytest.raises(PreprocessingPolicyLoadError) as missing:
        loader.load("unknown", "1.0")
    assert missing.value.finding_code == "PREPROCESSING_POLICY_NOT_FOUND"
    with pytest.raises(PreprocessingPolicyLoadError) as version:
        loader.load("synthetic-paired-rgb-height", "999")
    assert version.value.finding_code == "PREPROCESSING_POLICY_VERSION_UNSUPPORTED"
    with pytest.raises(PreprocessingPolicyLoadError) as malformed:
        SyntheticPreprocessingPolicyLoader(policy_document={}).load(
            "synthetic-paired-rgb-height", "1.0"
        )
    assert malformed.value.finding_code == "PREPROCESSING_POLICY_VERSION_UNSUPPORTED"


def test_malformed_supported_contract_policy_maps_to_invalid():
    document = _json(ROOT / "contracts" / "examples" / "inspection_preprocessing_policy.synthetic.json")
    document.pop("rgb")
    with pytest.raises(PreprocessingPolicyLoadError) as error:
        SyntheticPreprocessingPolicyLoader(policy_document=document).load(
            "synthetic-paired-rgb-height", "1.0"
        )
    assert error.value.finding_code == "PREPROCESSING_POLICY_INVALID"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("detected_format", "JPEG", "RGB_FORMAT_INCOMPATIBLE"),
        ("channels", 1, "RGB_CHANNELS_INCOMPATIBLE"),
        ("bit_depth", 16, "RGB_BIT_DEPTH_INCOMPATIBLE"),
    ],
)
def test_rgb_identity_incompatibilities_fail_before_decoding(tmp_path, field, value, code):
    _, _, inputs = _generated_input(tmp_path, "valid_rgb_png_height_tiff")
    rgb = replace(inputs.rgb, identity=replace(inputs.rgb.identity, **{field: value}))
    execution = _execute(replace(inputs, rgb=rgb))
    assert execution.result.findings[0].code == code


def test_missing_and_corrupt_rgb_fail_safely(tmp_path):
    _, _, inputs = _generated_input(tmp_path, "valid_rgb_png_height_tiff")
    missing = _execute(replace(inputs, rgb=replace(inputs.rgb, source_path=None)))
    assert missing.result.findings[0].code == "RGB_INPUT_UNAVAILABLE"

    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"synthetic corrupt rgb")
    corrupt_identity = replace(
        inputs.rgb.identity,
        sha256=_sha256(corrupt_path),
        byte_size=corrupt_path.stat().st_size,
    )
    corrupt = _execute(
        replace(inputs, rgb=ValidatedArtifactSource(corrupt_identity, corrupt_path))
    )
    assert corrupt.result.findings[0].code == "RGB_PREPROCESSING_UNSUPPORTED"
    assert str(corrupt_path) not in preprocessing_result_json(corrupt.result)


def _invalid_height_input(tmp_path, scenario_id):
    root = tmp_path / scenario_id
    generate_fixtures(root, scenario_ids=(scenario_id,))
    scenario_root = root / "scenarios" / scenario_id
    record = _json(scenario_root / "scenario.json")
    height_path = scenario_root / record["artifacts"]["height"]["generated_file"]
    metadata = inspect_rgb(height_path)
    identity = ArtifactInputIdentity(
        "HEIGHT_RAW", _sha256(height_path), height_path.stat().st_size,
        metadata.detected_format, metadata.width, metadata.height,
        metadata.channels, metadata.bit_depth, metadata.storage_data_type,
    )
    return ValidatedArtifactSource(identity, height_path)


@pytest.mark.parametrize(
    ("scenario_id", "code"),
    [
        ("height_png_uint8", "HEIGHT_STORAGE_TYPE_INCOMPATIBLE"),
        ("height_png_rgb", "HEIGHT_CHANNELS_INCOMPATIBLE"),
    ],
)
def test_invalid_height_png_inputs_fail_safely(tmp_path, scenario_id, code):
    _, _, inputs = _generated_input(tmp_path / "valid", "valid_rgb_png_height_tiff")
    execution = _execute(replace(inputs, height=_invalid_height_input(tmp_path, scenario_id)))
    assert execution.result.findings[0].code == code


def test_missing_unsupported_and_corrupt_height_fail_safely(tmp_path):
    _, _, inputs = _generated_input(tmp_path / "base", "valid_rgb_png_height_tiff")
    missing = _execute(replace(inputs, height=replace(inputs.height, source_path=None)))
    assert missing.result.findings[0].code == "HEIGHT_INPUT_UNAVAILABLE"

    unsupported = replace(
        inputs.height,
        identity=replace(inputs.height.identity, storage_data_type="float64"),
    )
    assert _execute(replace(inputs, height=unsupported)).result.findings[0].code == "HEIGHT_STORAGE_TYPE_INCOMPATIBLE"

    path = tmp_path / "corrupt-height.tiff"
    path.write_bytes(b"synthetic corrupt height")
    identity = replace(inputs.height.identity, sha256=_sha256(path), byte_size=path.stat().st_size)
    corrupt = _execute(replace(inputs, height=ValidatedArtifactSource(identity, path)))
    assert corrupt.result.findings[0].code == "HEIGHT_PREPROCESSING_UNSUPPORTED"


def test_nonfinite_npy_height_is_rejected(tmp_path):
    _, _, inputs = _generated_input(tmp_path / "base", "valid_rgb_png_height_npy_float32")
    values = list(float32_height_values(16, 12, DEFAULT_SEED, "nonfinite"))
    values[4] = math.nan
    path = tmp_path / "nonfinite.npy"
    path.write_bytes(encode_npy_float32(16, 12, tuple(values)))
    height = _source("height", path)
    execution = _execute(replace(inputs, height=height))
    assert execution.result.findings[0].code == "OUTPUT_NONFINITE_VALUES"


def test_different_dimensions_fail_without_resize(tmp_path):
    _, _, inputs = _generated_input(tmp_path, "valid_different_dimensions")
    execution = _execute(inputs)
    assert execution.result.outcome.value == "PREPROCESSING_FAILED"
    assert execution.result.findings[0].code == "OUTPUT_DIMENSION_RELATIONSHIP_INVALID"
    assert execution.rgb_buffer.descriptor.shape == (3, 12, 16)
    assert execution.height_buffer.descriptor.shape == (1, 6, 8)


class MutatingRGBProcessor:
    def __init__(self, mutation):
        from app.services.inspection_preprocessing.rgb_processor import SyntheticRGBPreprocessor
        self._delegate = SyntheticRGBPreprocessor()
        self._mutation = mutation

    async def preprocess_rgb(self, inputs, policy):
        branch = await self._delegate.preprocess_rgb(inputs, policy)
        descriptor = replace(branch.buffer.descriptor, **self._mutation.get("descriptor", {}))
        buffer = replace(branch.buffer, descriptor=descriptor, **self._mutation.get("buffer", {}))
        return replace(branch, buffer=buffer)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"descriptor": {"shape": (3, 1, 1)}}, "OUTPUT_SHAPE_INVALID"),
        ({"descriptor": {"layout": OutputLayout.HWC}}, "OUTPUT_LAYOUT_INVALID"),
        ({"descriptor": {"data_type": OutputDataType.UINT8}}, "OUTPUT_DATA_TYPE_INVALID"),
        ({"buffer": {"data": b"", "byte_size": 0}}, "OUTPUT_SHAPE_INVALID"),
    ],
)
def test_invalid_internal_rgb_outputs_are_detected(tmp_path, mutation, code):
    _, _, inputs = _generated_input(tmp_path, "valid_rgb_png_height_tiff")
    execution = _execute(inputs, service=_service(rgb_processor=MutatingRGBProcessor(mutation)))
    assert execution.result.outcome.value == "PREPROCESSING_FAILED"
    assert execution.result.findings[0].code == code


class FailingRGBProcessor:
    async def preprocess_rgb(self, *_args):
        raise RuntimeError(r"failure at C:\private\synthetic.png with traceback")


def test_unexpected_processor_failure_is_safe_preprocessing_error(tmp_path, result_validator):
    _, _, inputs = _generated_input(tmp_path, "valid_rgb_png_height_tiff")
    execution = _execute(inputs, service=_service(rgb_processor=FailingRGBProcessor()))
    serialized = preprocessing_result_json(execution.result)
    assert execution.result.outcome.value == "PREPROCESSING_ERROR"
    assert [item.code for item in execution.result.findings] == ["PREPROCESSING_INTERNAL_ERROR"]
    assert "private" not in serialized.lower()
    assert "traceback" not in serialized.lower()
    validator, catalogue = result_validator
    validator.validate(execution.result.to_dict())
    validate_preprocessing_result_document(execution.result.to_dict(), catalogue)


def test_executor_package_has_no_database_fastapi_inference_or_model_dependency():
    package = ROOT / "backend" / "app" / "services" / "inspection_preprocessing"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "app.db" not in source
    assert "fastapi" not in source.lower()
    assert "inspection_validation" not in source
    assert "app.services.inference" not in source
    assert "torch" not in source.lower()
    assert "onnx" not in source.lower()
    assert "sqlite" not in source.lower()
    assert "inspection_preprocessing" not in (ROOT / "backend" / "app" / "main.py").read_text(encoding="utf-8")
