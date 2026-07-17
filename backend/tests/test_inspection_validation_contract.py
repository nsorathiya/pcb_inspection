import copy
import json
from pathlib import Path
from uuid import UUID

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from app.services.inspection_validation import (
    FilesystemIntegrityInspector,
    InspectionPairValidator,
    InspectionValidationStatusTransition,
    NativeFormatInspector,
    ValidationArtifactRetriever,
    ValidationOutcome,
    ValidationPolicyEvaluator,
    ValidationResultPersistence,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_ROOT = REPOSITORY_ROOT / "contracts"
EXAMPLES_ROOT = CONTRACTS_ROOT / "examples"
RESULT_SCHEMA_PATH = CONTRACTS_ROOT / "inspection_validation_result.schema.json"
POLICY_SCHEMA_PATH = CONTRACTS_ROOT / "inspection_validation_policy.schema.json"
FINDINGS_PATH = CONTRACTS_ROOT / "inspection_validation_findings.json"
POLICY_EXAMPLE_PATH = (
    EXAMPLES_ROOT / "inspection_validation_policy.development.json"
)
RESULT_EXAMPLE_PATHS = (
    EXAMPLES_ROOT / "inspection_validation_result.passed.json",
    EXAMPLES_ROOT / "inspection_validation_result.failed.json",
    EXAMPLES_ROOT / "inspection_validation_result.error.json",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def findings_catalogue() -> dict:
    return _json(FINDINGS_PATH)


@pytest.fixture(scope="module")
def result_validator(findings_catalogue) -> Draft202012Validator:
    schema = _json(RESULT_SCHEMA_PATH)
    registry = Registry().with_resource(
        findings_catalogue["$id"],
        Resource.from_contents(findings_catalogue),
    )
    return Draft202012Validator(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )


@pytest.fixture(scope="module")
def policy_validator() -> Draft202012Validator:
    return Draft202012Validator(
        _json(POLICY_SCHEMA_PATH),
        format_checker=FormatChecker(),
    )


def _assert_invalid(validator: Draft202012Validator, value: dict) -> None:
    assert list(validator.iter_errors(value))


def test_contract_documents_are_valid_json_schemas() -> None:
    for path in (RESULT_SCHEMA_PATH, POLICY_SCHEMA_PATH, FINDINGS_PATH):
        Draft202012Validator.check_schema(_json(path))


@pytest.mark.parametrize("example_path", RESULT_EXAMPLE_PATHS)
def test_validation_result_examples_satisfy_schema(
    result_validator,
    example_path,
) -> None:
    result_validator.validate(_json(example_path))


def test_default_development_policy_satisfies_schema(policy_validator) -> None:
    policy_validator.validate(_json(POLICY_EXAMPLE_PATH))


def test_contract_versions_are_explicit(findings_catalogue) -> None:
    result_schema = _json(RESULT_SCHEMA_PATH)
    policy_schema = _json(POLICY_SCHEMA_PATH)
    assert result_schema["properties"]["contract_version"]["const"] == (
        "pcb-aoi-inspection-validation/1.0"
    )
    assert policy_schema["properties"]["contract_version"]["const"] == (
        "pcb-aoi-inspection-validation-policy/1.0"
    )
    assert findings_catalogue["catalog_version"] == (
        "pcb-aoi-inspection-validation-findings/1.0"
    )
    for path in (*RESULT_EXAMPLE_PATHS, POLICY_EXAMPLE_PATH):
        assert _json(path)["contract_version"]


@pytest.mark.parametrize("example_path", RESULT_EXAMPLE_PATHS)
def test_validation_and_inspection_ids_are_canonical_uuids(example_path) -> None:
    example = _json(example_path)
    for field in ("validation_id", "inspection_id"):
        assert str(UUID(example[field])) == example[field]


@pytest.mark.parametrize("field", ["validation_id", "inspection_id"])
def test_noncanonical_uuid_is_rejected(result_validator, field) -> None:
    example = _json(RESULT_EXAMPLE_PATHS[0])
    example[field] = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    _assert_invalid(result_validator, example)


@pytest.mark.parametrize("field", ["started_at", "completed_at"])
def test_timestamps_require_timezone_information(result_validator, field) -> None:
    example = _json(RESULT_EXAMPLE_PATHS[0])
    example[field] = "2026-07-17T12:00:00"
    _assert_invalid(result_validator, example)


def test_finding_codes_and_order_values_are_unique(findings_catalogue) -> None:
    entries = findings_catalogue["findings"]
    codes = [entry["code"] for entry in entries]
    orders = [entry["order"] for entry in entries]
    assert len(codes) == len(set(codes))
    assert len(orders) == len(set(orders))
    assert codes == findings_catalogue["$defs"]["finding_code"]["enum"]
    assert orders == sorted(orders)


def test_catalogue_severities_categories_and_blocking_defaults_are_valid(
    findings_catalogue,
) -> None:
    severities = set(findings_catalogue["$defs"]["severity"]["enum"])
    categories = set(findings_catalogue["$defs"]["category"]["enum"])
    assert severities == {"INFO", "WARNING", "ERROR"}
    assert categories == {
        "PAIR",
        "FILE_INTEGRITY",
        "FORMAT",
        "IMAGE_PROPERTIES",
        "HEIGHT_PROPERTIES",
        "REGISTRATION_EVIDENCE",
        "CALIBRATION_EVIDENCE",
        "POLICY",
        "INTERNAL",
    }
    for entry in findings_catalogue["findings"]:
        assert entry["default_severity"] in severities
        assert entry["category"] in categories
        assert isinstance(entry["default_blocking"], bool)
        if entry["default_severity"] == "ERROR":
            assert entry["default_blocking"] is True


def test_examples_use_known_findings_with_consistent_defaults(
    findings_catalogue,
) -> None:
    by_code = {entry["code"]: entry for entry in findings_catalogue["findings"]}
    for path in RESULT_EXAMPLE_PATHS:
        for finding in _json(path)["findings"]:
            authoritative = by_code[finding["code"]]
            assert finding["severity"] == authoritative["default_severity"]
            assert finding["category"] == authoritative["category"]
            assert finding["blocking"] == authoritative["default_blocking"]


def test_unknown_finding_code_fails_result_schema(result_validator) -> None:
    example = _json(RESULT_EXAMPLE_PATHS[0])
    example["findings"][0]["code"] = "UNVERSIONED_UNKNOWN_CODE"
    _assert_invalid(result_validator, example)


def test_example_findings_are_deterministically_ordered(findings_catalogue) -> None:
    order_by_code = {
        entry["code"]: entry["order"] for entry in findings_catalogue["findings"]
    }
    artifact_order = {
        None: 0,
        "RGB_RAW": 1,
        "HEIGHT_RAW": 2,
        "VALIDITY_MASK": 3,
        "CALIBRATION": 4,
    }
    for path in RESULT_EXAMPLE_PATHS:
        findings = _json(path)["findings"]
        keys = [
            (
                order_by_code[finding["code"]],
                artifact_order[finding.get("artifact_type")],
                finding.get("field", ""),
                finding["message"],
            )
            for finding in findings
        ]
        assert keys == sorted(keys)


def test_passed_result_cannot_contain_blocking_error(result_validator) -> None:
    example = _json(RESULT_EXAMPLE_PATHS[0])
    example["findings"].append(
        {
            "code": "RGB_FORMAT_UNSUPPORTED",
            "severity": "ERROR",
            "category": "FORMAT",
            "message": "Detected RGB format is unsupported.",
            "artifact_type": "RGB_RAW",
            "blocking": True,
        }
    )
    _assert_invalid(result_validator, example)


def test_passed_result_cannot_contain_policy_blocking_warning(
    result_validator,
) -> None:
    example = _json(RESULT_EXAMPLE_PATHS[0])
    example["findings"][0]["blocking"] = True
    _assert_invalid(result_validator, example)


def test_failed_result_requires_a_blocking_finding(result_validator) -> None:
    example = _json(RESULT_EXAMPLE_PATHS[1])
    for finding in example["findings"]:
        finding["blocking"] = False
    _assert_invalid(result_validator, example)


def test_error_result_requires_internal_error_finding(result_validator) -> None:
    example = _json(RESULT_EXAMPLE_PATHS[2])
    example["findings"] = []
    _assert_invalid(result_validator, example)


@pytest.mark.parametrize(
    ("artifact_field", "finding_code", "artifact_type"),
    [
        ("rgb_artifact", "RGB_RAW_MISSING", "RGB_RAW"),
        ("height_artifact", "HEIGHT_RAW_MISSING", "HEIGHT_RAW"),
    ],
)
def test_missing_raw_artifact_cannot_produce_validation_passed(
    result_validator,
    artifact_field,
    finding_code,
    artifact_type,
) -> None:
    example = _json(RESULT_EXAMPLE_PATHS[0])
    missing = copy.deepcopy(_json(RESULT_EXAMPLE_PATHS[1])["height_artifact"])
    missing["artifact_type"] = artifact_type
    example[artifact_field] = missing
    example["findings"].append(
        {
            "code": finding_code,
            "severity": "ERROR",
            "category": "PAIR",
            "message": f"The required {artifact_type} artifact is missing.",
            "artifact_type": artifact_type,
            "blocking": True,
        }
    )
    _assert_invalid(result_validator, example)


def test_example_summary_counts_match_findings() -> None:
    for path in RESULT_EXAMPLE_PATHS:
        example = _json(path)
        findings = example["findings"]
        summary = example["summary"]
        assert summary["finding_count"] == len(findings)
        assert summary["info_count"] == sum(
            finding["severity"] == "INFO" for finding in findings
        )
        assert summary["warning_count"] == sum(
            finding["severity"] == "WARNING" for finding in findings
        )
        assert summary["error_count"] == sum(
            finding["severity"] == "ERROR" for finding in findings
        )
        assert summary["blocking_count"] == sum(
            finding["blocking"] for finding in findings
        )


def test_default_policy_rejects_8_bit_and_requires_scalar_height() -> None:
    policy = _json(POLICY_EXAMPLE_PATH)
    assert policy["minimum_height_bit_depth"] >= 16
    assert "uint8" not in policy["allowed_height_storage_types"]
    assert policy["require_single_channel_height"] is True
    assert policy["require_explicit_height_invalid_value_policy"] is True


@pytest.mark.parametrize(
    ("field", "unsupported_value"),
    [
        ("dimension_relationship", "SCALE_TO_FIT"),
        ("allowed_height_formats", ["POINT_CLOUD"]),
        ("allowed_height_storage_types", ["rgb8"]),
        ("minimum_height_bit_depth", 8),
    ],
)
def test_unsupported_policy_values_fail_schema(
    policy_validator,
    field,
    unsupported_value,
) -> None:
    policy = _json(POLICY_EXAMPLE_PATH)
    policy[field] = unsupported_value
    _assert_invalid(policy_validator, policy)


def test_point_cloud_and_mesh_formats_are_absent_from_contract_1_0() -> None:
    documents = (
        _json(POLICY_SCHEMA_PATH),
        _json(RESULT_SCHEMA_PATH),
        _json(POLICY_EXAMPLE_PATH),
    )
    serialized = json.dumps(documents).upper()
    assert "POINT_CLOUD" not in serialized
    assert "POINTCLOUD" not in serialized
    assert '"MESH"' not in serialized


def test_synthetic_examples_are_explicit_and_make_no_production_claim() -> None:
    for path in RESULT_EXAMPLE_PATHS:
        example = _json(path)
        assert example["summary"]["synthetic_example"] is True
    policy = _json(POLICY_EXAMPLE_PATH)
    assert policy["development_only"] is True
    assert "not approved for production" in policy["description"].lower()


def test_result_contract_has_no_path_or_binary_fields() -> None:
    schema_text = RESULT_SCHEMA_PATH.read_text(encoding="utf-8").lower()
    for forbidden in (
        '"absolute_path"',
        '"relative_path"',
        '"filename"',
        '"binary"',
    ):
        assert forbidden not in schema_text


def test_future_service_concerns_remain_separate_protocols() -> None:
    protocols = (
        ValidationArtifactRetriever,
        FilesystemIntegrityInspector,
        NativeFormatInspector,
        ValidationPolicyEvaluator,
        ValidationResultPersistence,
        InspectionValidationStatusTransition,
        InspectionPairValidator,
    )
    assert all(getattr(protocol, "_is_protocol", False) for protocol in protocols)
    assert {outcome.value for outcome in ValidationOutcome} == {
        "VALIDATION_PASSED",
        "VALIDATION_FAILED",
        "VALIDATION_ERROR",
    }
