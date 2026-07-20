import asyncio
import hashlib
import json
import struct
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from app.services.dataset_validation.file_inspection import inspect_height, inspect_rgb
from app.services.inspection_inference import (
    DeterministicMockInferenceEngine,
    SyntheticInferenceInput,
    SyntheticMockInferencePolicyLoader,
    SyntheticMockInferenceService,
    canonical_decision_bytes,
    inference_result_json,
    validate_inference_result_document,
)
from app.services.inspection_inference.exceptions import InferencePolicyLoadError
from app.services.inspection_preprocessing import (
    ArtifactInputIdentity,
    InternalPreprocessedBuffer,
    SyntheticInspectionPreprocessingService,
    SyntheticPreprocessingPolicyLoader,
    ValidatedArtifactSource,
    ValidatedInspectionInput,
)
from app.services.inspection_preprocessing.models import OutputDataType, OutputLayout
from app.testing.synthetic_aoi import generate_fixtures

ROOT = Path(__file__).resolve().parents[2]
FIXED_TIME = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)
FIXED_VALIDATION_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
FIXED_PREPROCESSING_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
FIXED_INFERENCE_ID = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"


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


def _preprocessed_fixture(tmp_path: Path, scenario_id: str):
    root = tmp_path / scenario_id
    generate_fixtures(root, scenario_ids=(scenario_id,))
    scenario_root = root / "scenarios" / scenario_id
    record = _json(scenario_root / "scenario.json")
    rgb_path = scenario_root / record["artifacts"]["rgb"]["generated_file"]
    height_path = scenario_root / record["artifacts"]["height"]["generated_file"]
    validated = ValidatedInspectionInput(
        inspection_id=record["scenario_uuid"],
        validation_id=FIXED_VALIDATION_ID,
        inspection_status="READY",
        validation_outcome="VALIDATION_PASSED",
        synthetic_input=True,
        rgb=ValidatedArtifactSource(_identity("rgb", rgb_path), rgb_path),
        height=ValidatedArtifactSource(_identity("height", height_path), height_path),
    )
    preprocessing_policy = SyntheticPreprocessingPolicyLoader().load(
        "synthetic-paired-rgb-height", "1.0"
    )
    preprocessing_service = SyntheticInspectionPreprocessingService(
        clock=lambda: FIXED_TIME,
        preprocessing_id_generator=lambda: FIXED_PREPROCESSING_ID,
        implementation_id="synthetic-mock-preprocessor-inference-test",
        implementation_version="1.0.0-test",
    )
    execution = asyncio.run(
        preprocessing_service.preprocess_inspection(validated, preprocessing_policy)
    )
    assert execution.result.outcome.value == "PREPROCESSING_SUCCEEDED"
    return scenario_root, execution


def _inference_input(execution) -> SyntheticInferenceInput:
    result = execution.result
    return SyntheticInferenceInput(
        inspection_id=result.inspection_id,
        validation_id=result.validation_id,
        preprocessing_id=result.preprocessing_id,
        preprocessing_outcome=result.outcome.value,
        synthetic_input=result.synthetic_input,
        mock_preprocessing=result.mock_implementation,
        rgb_buffer=execution.rgb_buffer,
        height_buffer=execution.height_buffer,
    )


def _policy():
    return SyntheticMockInferencePolicyLoader().load(
        "synthetic-deterministic-mock-inference", "1.0"
    )


def _service(*, engine=None, clock=None, inference_id=FIXED_INFERENCE_ID):
    return SyntheticMockInferenceService(
        engine=engine,
        clock=clock or (lambda: FIXED_TIME),
        inference_id_generator=lambda: inference_id,
    )


def _run(inputs, policy=None, service=None):
    return asyncio.run(
        (service or _service()).run_inference(inputs, policy or _policy())
    )


@pytest.fixture(scope="module")
def result_validation():
    catalogue = _json(ROOT / "contracts" / "inspection_inference_findings.json")
    taxonomy = _json(ROOT / "contracts" / "defect_taxonomy.json")
    schema = _json(ROOT / "contracts" / "inspection_inference_result.schema.json")
    registry = Registry().with_resources(
        [
            (catalogue["$id"], Resource.from_contents(catalogue)),
            (taxonomy["$id"], Resource.from_contents(taxonomy)),
        ]
    )
    validator = Draft202012Validator(
        schema, registry=registry, format_checker=FormatChecker()
    )
    return validator, catalogue, taxonomy


@pytest.mark.parametrize(
    "scenario_id",
    [
        "valid_rgb_png_height_tiff",
        "valid_rgb_tiff_height_png16",
        "valid_rgb_png_height_npy_float32",
    ],
)
def test_supported_preprocessing_outputs_produce_schema_valid_mock_result(
    tmp_path, scenario_id, result_validation
):
    _, preprocessing = _preprocessed_fixture(tmp_path, scenario_id)
    result = _run(_inference_input(preprocessing))
    validator, catalogue, taxonomy = result_validation

    assert result.execution_outcome.value == "INFERENCE_SUCCEEDED"
    assert result.decision is not None
    assert result.confidence is None
    assert result.rgb_input.shape == (3, 12, 16)
    assert result.height_input.shape == (1, 12, 16)
    assert result.rgb_input.layout == result.height_input.layout == "CHW"
    assert result.rgb_input.data_type == result.height_input.data_type == "float32"
    validator.validate(result.to_dict())
    validate_inference_result_document(result.to_dict(), catalogue, taxonomy)


def _input_for_decision(base, decision):
    for index in range(256):
        preprocessing_id = str(uuid5(NAMESPACE_URL, f"mock-{decision}-{index}"))
        candidate = replace(base, preprocessing_id=preprocessing_id)
        result = _run(candidate)
        if result.decision is not None and result.decision.value == decision:
            return candidate, result
    raise AssertionError(f"unable to select deterministic {decision} test input")


@pytest.mark.parametrize("decision", ["PASS", "FAIL", "UNCERTAIN"])
def test_deterministic_decision_semantics_and_required_findings(
    tmp_path, decision, result_validation
):
    _, preprocessing = _preprocessed_fixture(
        tmp_path / decision, "valid_rgb_png_height_tiff"
    )
    _, result = _input_for_decision(_inference_input(preprocessing), decision)
    codes = [item.code for item in result.findings]
    taxonomy = result_validation[2]["$defs"]["supported_defect_type"]["enum"]

    assert result.execution_outcome.value == "INFERENCE_SUCCEEDED"
    assert result.decision.value == decision
    assert result.confidence is None
    assert len(result.decision_digest) == 64
    assert result.decision_digest == result.decision_digest.lower()
    assert {"MOCK_INFERENCE_USED", "MOCK_DECISION_GENERATED", "CONFIDENCE_UNAVAILABLE"}.issubset(codes)
    if decision == "FAIL":
        assert result.defect_type in taxonomy
        assert result.defect_type != "no_defect"
        assert "MOCK_FAIL_DEFECT_ASSIGNED" in codes
    else:
        assert result.defect_type is None
        assert "MOCK_FAIL_DEFECT_ASSIGNED" not in codes


def test_digest_uses_exact_canonical_json_and_documented_prefix_algorithm(tmp_path):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    inputs = _inference_input(preprocessing)
    policy = _policy()
    service = _service()
    result = _run(inputs, policy, service)
    validated = service._validator.validate(inputs, policy)
    expected = hashlib.sha256(
        canonical_decision_bytes(
            validated,
            policy,
            engine_id=service._engine.engine_id,
            engine_version=service._engine.engine_version,
        )
    ).hexdigest()
    bucket = int(expected[:16], 16) % policy.engine.decision_bucket_count

    assert result.decision_digest == expected
    selected = {
        "PASS": policy.engine.pass_buckets,
        "FAIL": policy.engine.fail_buckets,
        "UNCERTAIN": policy.engine.uncertain_buckets,
    }[result.decision.value]
    assert bucket in selected


def test_fixed_providers_produce_identical_result_digest_decision_and_findings(tmp_path):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    inputs = _inference_input(preprocessing)
    first = _run(inputs)
    second = _run(inputs)

    assert inference_result_json(first) == inference_result_json(second)
    assert first.decision_digest == second.decision_digest
    assert first.decision == second.decision
    assert first.defect_type == second.defect_type
    assert first.findings == second.findings


def test_timestamps_and_inference_id_do_not_change_decision_digest(tmp_path):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    inputs = _inference_input(preprocessing)
    first = _run(inputs)
    later = FIXED_TIME + timedelta(days=1)
    second = _run(
        inputs,
        service=_service(
            clock=lambda: later,
            inference_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        ),
    )
    assert first.inference_id != second.inference_id
    assert first.started_at != second.started_at
    assert first.decision_digest == second.decision_digest
    assert first.decision == second.decision


def _changed_valid_buffer(buffer: InternalPreprocessedBuffer) -> InternalPreprocessedBuffer:
    first = struct.unpack("<f", buffer.data[:4])[0]
    data = struct.pack("<f", first + 0.25) + buffer.data[4:]
    return InternalPreprocessedBuffer.from_bytes(buffer.descriptor, data)


def test_buffer_hash_policy_version_and_engine_version_change_digest(tmp_path):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    inputs = _inference_input(preprocessing)
    baseline = _run(inputs)
    changed_buffer = _run(
        replace(inputs, rgb_buffer=_changed_valid_buffer(inputs.rgb_buffer))
    )
    changed_policy = _run(inputs, policy=replace(_policy(), policy_version="1.0-test"))
    changed_engine = _run(
        inputs,
        service=_service(
            engine=DeterministicMockInferenceEngine(engine_version="1.0.1")
        ),
    )
    assert len(
        {
            baseline.decision_digest,
            changed_buffer.decision_digest,
            changed_policy.decision_digest,
            changed_engine.decision_digest,
        }
    ) == 4


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"preprocessing_outcome": None}, "PREPROCESSING_RESULT_REQUIRED"),
        ({"preprocessing_outcome": "PREPROCESSING_FAILED"}, "PREPROCESSING_NOT_SUCCEEDED"),
        ({"preprocessing_outcome": "PREPROCESSING_ERROR"}, "PREPROCESSING_NOT_SUCCEEDED"),
        ({"synthetic_input": False}, "SYNTHETIC_INPUT_REQUIRED"),
        ({"mock_preprocessing": False}, "MOCK_PREPROCESSING_REQUIRED"),
    ],
)
def test_prerequisite_failures_return_no_decision(tmp_path, change, code):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    result = _run(replace(_inference_input(preprocessing), **change))
    assert result.execution_outcome.value == "INFERENCE_FAILED"
    assert [item.code for item in result.findings] == [code]
    assert result.decision is result.defect_type is result.confidence is None
    assert result.decision_basis is result.decision_digest is None


def test_mock_engine_disabled_and_unsafe_policy_fail_safely(tmp_path):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    inputs = _inference_input(preprocessing)
    policy = _policy()
    disabled = replace(
        policy,
        safety=replace(policy.safety, allow_mock_engine=False),
    )
    assert _run(inputs, disabled).findings[0].code == "MOCK_ENGINE_NOT_ALLOWED"
    unsafe = replace(
        policy,
        safety=replace(policy.safety, allow_model_accuracy_claim=True),
    )
    assert _run(inputs, unsafe).findings[0].code == "INFERENCE_POLICY_INVALID"


def test_policy_loader_rejects_unknown_version_and_malformed_documents():
    loader = SyntheticMockInferencePolicyLoader()
    with pytest.raises(InferencePolicyLoadError) as missing:
        loader.load("unknown", "1.0")
    assert missing.value.finding_code == "INFERENCE_POLICY_NOT_FOUND"
    with pytest.raises(InferencePolicyLoadError) as version:
        loader.load("synthetic-deterministic-mock-inference", "999")
    assert version.value.finding_code == "INFERENCE_POLICY_VERSION_UNSUPPORTED"
    with pytest.raises(InferencePolicyLoadError) as unsupported:
        SyntheticMockInferencePolicyLoader(policy_document={}).load(
            "synthetic-deterministic-mock-inference", "1.0"
        )
    assert unsupported.value.finding_code == "INFERENCE_POLICY_VERSION_UNSUPPORTED"

    malformed = _json(
        ROOT / "contracts" / "examples" / "inspection_inference_policy.mock.json"
    )
    malformed.pop("engine")
    with pytest.raises(InferencePolicyLoadError) as invalid:
        SyntheticMockInferencePolicyLoader(policy_document=malformed).load(
            "synthetic-deterministic-mock-inference", "1.0"
        )
    assert invalid.value.finding_code == "INFERENCE_POLICY_INVALID"


def _mutate_descriptor(buffer, **changes):
    return replace(buffer, descriptor=replace(buffer.descriptor, **changes))


@pytest.mark.parametrize(
    ("branch", "mutation", "code"),
    [
        ("rgb", lambda value: None, "RGB_BUFFER_REQUIRED"),
        ("rgb", lambda value: replace(value, content_sha256="0" * 64), "RGB_BUFFER_HASH_MISMATCH"),
        ("rgb", lambda value: replace(value, data=value.data[:-4]), "RGB_BUFFER_LENGTH_MISMATCH"),
        ("rgb", lambda value: _mutate_descriptor(value, shape=(3, 1, 1)), "RGB_BUFFER_SHAPE_INVALID"),
        ("rgb", lambda value: _mutate_descriptor(value, channel_count=2), "RGB_BUFFER_SHAPE_INVALID"),
        ("rgb", lambda value: _mutate_descriptor(value, layout=OutputLayout.HWC), "RGB_BUFFER_LAYOUT_UNSUPPORTED"),
        ("rgb", lambda value: _mutate_descriptor(value, data_type=OutputDataType.UINT16), "RGB_BUFFER_DATA_TYPE_UNSUPPORTED"),
        ("height", lambda value: None, "HEIGHT_BUFFER_REQUIRED"),
        ("height", lambda value: replace(value, content_sha256="0" * 64), "HEIGHT_BUFFER_HASH_MISMATCH"),
        ("height", lambda value: replace(value, data=value.data[:-4]), "HEIGHT_BUFFER_LENGTH_MISMATCH"),
        ("height", lambda value: _mutate_descriptor(value, shape=(1, 1, 1)), "HEIGHT_BUFFER_SHAPE_INVALID"),
        ("height", lambda value: _mutate_descriptor(value, channel_count=2), "HEIGHT_BUFFER_SHAPE_INVALID"),
        ("height", lambda value: _mutate_descriptor(value, layout=OutputLayout.HWC), "HEIGHT_BUFFER_LAYOUT_UNSUPPORTED"),
        ("height", lambda value: _mutate_descriptor(value, data_type=OutputDataType.UINT16), "HEIGHT_BUFFER_DATA_TYPE_UNSUPPORTED"),
    ],
)
def test_buffer_incompatibilities_fail_without_decision(
    tmp_path, branch, mutation, code
):
    _, preprocessing = _preprocessed_fixture(
        tmp_path / f"{branch}-{code}", "valid_rgb_png_height_tiff"
    )
    inputs = _inference_input(preprocessing)
    field = f"{branch}_buffer"
    result = _run(replace(inputs, **{field: mutation(getattr(inputs, field))}))
    assert result.execution_outcome.value == "INFERENCE_FAILED"
    assert [item.code for item in result.findings] == [code]
    assert result.decision is None


def test_invalid_descriptor_and_nonfinite_bytes_are_rejected(tmp_path):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    inputs = _inference_input(preprocessing)
    contiguous = _run(
        replace(
            inputs,
            rgb_buffer=_mutate_descriptor(inputs.rgb_buffer, contiguous=False),
        )
    )
    assert contiguous.findings[0].code == "RGB_BUFFER_DESCRIPTOR_INVALID"

    data = struct.pack("<f", float("nan")) + inputs.height_buffer.data[4:]
    nonfinite = replace(
        inputs.height_buffer,
        data=data,
        content_sha256=hashlib.sha256(data).hexdigest(),
    )
    result = _run(replace(inputs, height_buffer=nonfinite))
    assert result.findings[0].code == "HEIGHT_BUFFER_DESCRIPTOR_INVALID"


def test_mismatched_spatial_dimensions_fail(tmp_path):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    inputs = _inference_input(preprocessing)
    source = inputs.height_buffer
    descriptor = replace(source.descriptor, shape=(1, 11, 16), height=11)
    height = InternalPreprocessedBuffer.from_bytes(descriptor, source.data[: 11 * 16 * 4])
    result = _run(replace(inputs, height_buffer=height))
    assert result.execution_outcome.value == "INFERENCE_FAILED"
    assert result.findings[0].code == "INPUT_DIMENSION_RELATIONSHIP_INVALID"


class FailingEngine:
    engine_id = "synthetic-failing-mock-engine"
    engine_version = "1.0.0-test"

    async def infer(self, *_args):
        raise RuntimeError(r"failure at C:\private\model.onnx with traceback details")


def test_unexpected_engine_failure_is_safe_inference_error(
    tmp_path, result_validation
):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    result = _run(
        _inference_input(preprocessing),
        service=_service(engine=FailingEngine()),
    )
    serialized = inference_result_json(result)
    assert result.execution_outcome.value == "INFERENCE_ERROR"
    assert [item.code for item in result.findings] == ["INFERENCE_INTERNAL_ERROR"]
    assert result.decision is result.decision_digest is None
    assert "private" not in serialized.lower()
    assert "traceback" not in serialized.lower()
    validator, catalogue, taxonomy = result_validation
    validator.validate(result.to_dict())
    validate_inference_result_document(result.to_dict(), catalogue, taxonomy)


def test_input_buffers_descriptors_and_preprocessing_result_are_immutable(tmp_path):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    inputs = _inference_input(preprocessing)
    before = (
        inputs.rgb_buffer.data,
        inputs.height_buffer.data,
        inputs.rgb_buffer.content_sha256,
        inputs.height_buffer.content_sha256,
        inputs.rgb_buffer.descriptor,
        inputs.height_buffer.descriptor,
        preprocessing.result.to_dict(),
    )
    _run(inputs)
    after = (
        inputs.rgb_buffer.data,
        inputs.height_buffer.data,
        inputs.rgb_buffer.content_sha256,
        inputs.height_buffer.content_sha256,
        inputs.rgb_buffer.descriptor,
        inputs.height_buffer.descriptor,
        preprocessing.result.to_dict(),
    )
    assert after == before


def test_public_result_is_path_free_and_contains_no_buffer_data_or_model_metrics(tmp_path):
    _, preprocessing = _preprocessed_fixture(tmp_path, "valid_rgb_png_height_tiff")
    document = _run(_inference_input(preprocessing)).to_dict()
    serialized = json.dumps(document)

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    forbidden = {
        "data",
        "bytes",
        "tensor",
        "path",
        "filename",
        "model_path",
        "probability",
        "accuracy",
    }
    assert forbidden.isdisjoint(key.lower() for key in keys(document))
    assert str(tmp_path) not in serialized
    assert document["confidence"] is None


def test_inference_reads_no_artifact_file_and_generates_no_output(
    tmp_path, monkeypatch
):
    scenario_root, preprocessing = _preprocessed_fixture(
        tmp_path, "valid_rgb_png_height_tiff"
    )
    inputs = _inference_input(preprocessing)
    policy = _policy()
    service = _service()
    before = {
        path.relative_to(scenario_root).as_posix(): (
            _sha256(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in scenario_root.rglob("*")
        if path.is_file()
    }

    def reject_file_read(*_args, **_kwargs):
        raise AssertionError("inference attempted a filesystem content read")

    monkeypatch.setattr(Path, "read_bytes", reject_file_read)
    monkeypatch.setattr(Path, "read_text", reject_file_read)
    result = _run(inputs, policy, service)
    monkeypatch.undo()
    after = {
        path.relative_to(scenario_root).as_posix(): (
            _sha256(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in scenario_root.rglob("*")
        if path.is_file()
    }
    assert result.execution_outcome.value == "INFERENCE_SUCCEEDED"
    assert after == before
    assert not list(tmp_path.rglob("*.sqlite3"))


def test_inference_package_has_no_database_api_model_or_framework_dependency():
    package = ROOT / "backend" / "app" / "services" / "inspection_inference"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    main = (ROOT / "backend" / "app" / "main.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "app.db" not in source
    assert "fastapi" not in lowered
    assert "sqlite" not in lowered
    assert "torch" not in lowered
    assert "onnx" not in lowered
    assert "inspection_validation" not in source
    assert "inspection_inference" not in main
    assert "SyntheticMockInferenceService" not in main
