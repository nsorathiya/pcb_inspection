import copy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from referencing import Registry, Resource

from app.services.inspection_preprocessing import (
    PreprocessedBufferDescriptor,
    validate_preprocessing_policy_document,
    validate_preprocessing_result_document,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts"
EXAMPLES = CONTRACTS / "examples"
POLICY_SCHEMA_PATH = CONTRACTS / "inspection_preprocessing_policy.schema.json"
RESULT_SCHEMA_PATH = CONTRACTS / "inspection_preprocessing_result.schema.json"
CATALOGUE_PATH = CONTRACTS / "inspection_preprocessing_findings.json"
POLICY_PATH = EXAMPLES / "inspection_preprocessing_policy.synthetic.json"
RESULT_PATHS = {
    "PREPROCESSING_SUCCEEDED": EXAMPLES / "inspection_preprocessing_result.succeeded.json",
    "PREPROCESSING_FAILED": EXAMPLES / "inspection_preprocessing_result.failed.json",
    "PREPROCESSING_ERROR": EXAMPLES / "inspection_preprocessing_result.error.json",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def policy_schema():
    return _json(POLICY_SCHEMA_PATH)


@pytest.fixture(scope="module")
def result_schema():
    return _json(RESULT_SCHEMA_PATH)


@pytest.fixture(scope="module")
def catalogue():
    return _json(CATALOGUE_PATH)


@pytest.fixture(scope="module")
def policy_validator(policy_schema):
    return Draft202012Validator(policy_schema, format_checker=FormatChecker())


@pytest.fixture(scope="module")
def result_validator(result_schema, catalogue):
    registry = Registry().with_resource(catalogue["$id"], Resource.from_contents(catalogue))
    return Draft202012Validator(result_schema, registry=registry, format_checker=FormatChecker())


def _valid_policy(policy_validator) -> dict:
    policy = _json(POLICY_PATH)
    policy_validator.validate(policy)
    validate_preprocessing_policy_document(policy)
    return policy


def _valid_result(outcome, result_validator, catalogue) -> dict:
    result = _json(RESULT_PATHS[outcome])
    result_validator.validate(result)
    validate_preprocessing_result_document(result, catalogue)
    return result


def _assert_policy_semantic_failure(policy_validator, document, match):
    policy_validator.validate(document)
    with pytest.raises(ValueError, match=match):
        validate_preprocessing_policy_document(document)


def test_schemas_are_draft_2020_12_and_well_formed(policy_schema, result_schema, catalogue):
    for schema in (policy_schema, result_schema, catalogue):
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)


def test_contract_versions_are_explicit(policy_schema, result_schema, catalogue):
    assert policy_schema["properties"]["contract_version"]["const"] == "pcb-aoi-inspection-preprocessing-policy/1.0"
    assert result_schema["properties"]["contract_version"]["const"] == "pcb-aoi-inspection-preprocessing/1.0"
    assert catalogue["catalog_version"] == "pcb-aoi-inspection-preprocessing-findings/1.0"


def test_finding_codes_are_unique_and_catalogue_order_is_stable(catalogue):
    entries = catalogue["findings"]
    codes = [item["code"] for item in entries]
    orders = [item["order"] for item in entries]
    assert len(codes) == len(set(codes))
    assert orders == sorted(set(orders))
    assert codes == catalogue["$defs"]["finding_code"]["enum"]


def test_example_policy_validates(policy_validator):
    policy = _valid_policy(policy_validator)
    assert policy["development_only"] is True
    assert policy["production_approved"] is False
    assert policy["safety"] == {
        "allow_mock_implementation": True,
        "allow_synthetic_input": True,
        "allow_real_input": False,
        "allow_uncalibrated_height": True,
    }


@pytest.mark.parametrize("outcome", tuple(RESULT_PATHS))
def test_example_results_validate(outcome, result_validator, catalogue):
    result = _valid_result(outcome, result_validator, catalogue)
    assert result["outcome"] == outcome
    assert result["synthetic_input"] is True
    assert result["production_approved"] is False


@pytest.mark.parametrize("field", ["preprocessing_id", "inspection_id", "validation_id"])
def test_canonical_uuid_is_required(field, result_validator):
    result = _json(RESULT_PATHS["PREPROCESSING_SUCCEEDED"])
    result[field] = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    with pytest.raises(ValidationError):
        result_validator.validate(result)


@pytest.mark.parametrize("field", ["started_at", "completed_at"])
def test_timezone_aware_timestamps_are_required(field, result_validator):
    result = _json(RESULT_PATHS["PREPROCESSING_SUCCEEDED"])
    result[field] = "2026-07-17T12:00:00"
    with pytest.raises(ValidationError):
        result_validator.validate(result)


def test_unknown_finding_code_severity_and_category_are_rejected(result_validator):
    for field, value in (("code", "UNKNOWN"), ("severity", "FATAL"), ("category", "CLASSIFICATION")):
        result = _json(RESULT_PATHS["PREPROCESSING_FAILED"])
        result["findings"][0][field] = value
        with pytest.raises(ValidationError):
            result_validator.validate(result)


def test_success_cannot_contain_blocking_finding(result_validator):
    result = _json(RESULT_PATHS["PREPROCESSING_SUCCEEDED"])
    result["findings"][0]["blocking"] = True
    with pytest.raises(ValidationError):
        result_validator.validate(result)


def test_failed_requires_blocking_finding(result_validator):
    result = _json(RESULT_PATHS["PREPROCESSING_FAILED"])
    result["findings"][0]["blocking"] = False
    with pytest.raises(ValidationError):
        result_validator.validate(result)


def test_error_requires_internal_error(result_validator):
    result = _json(RESULT_PATHS["PREPROCESSING_ERROR"])
    result["findings"][0].update({"code": "OUTPUT_SHAPE_INVALID", "category": "OUTPUT"})
    with pytest.raises(ValidationError):
        result_validator.validate(result)


def test_summary_counts_must_match_findings(result_validator, catalogue):
    result = _json(RESULT_PATHS["PREPROCESSING_SUCCEEDED"])
    result["summary"]["warnings"] = 0
    result_validator.validate(result)
    with pytest.raises(ValueError, match="summary counts"):
        validate_preprocessing_result_document(result, catalogue)


def test_catalogue_severity_category_and_order_are_semantically_enforced(result_validator, catalogue):
    result = _json(RESULT_PATHS["PREPROCESSING_SUCCEEDED"])
    result["findings"].insert(0, {
        "code": "OUTPUT_SHAPE_INVALID", "severity": "ERROR", "category": "OUTPUT",
        "message": "A preprocessing output shape is invalid.", "blocking": False,
    })
    result["summary"] = {"total_findings": 2, "blocking_findings": 0, "warnings": 1, "errors": 1}
    result_validator.validate(result)
    with pytest.raises(ValueError, match="catalogue order"):
        validate_preprocessing_result_document(result, catalogue)


def test_development_policy_cannot_claim_production_approval(policy_validator):
    policy = _valid_policy(policy_validator)
    policy["production_approved"] = True
    with pytest.raises(ValidationError):
        policy_validator.validate(policy)


def test_production_policy_cannot_enable_synthetic_identity(policy_validator):
    policy = _valid_policy(policy_validator)
    policy.update({"development_only": False, "production_approved": True})
    with pytest.raises(ValidationError):
        policy_validator.validate(policy)


def test_synthetic_identity_requires_development_policy(policy_validator):
    policy = _valid_policy(policy_validator)
    policy["development_only"] = False
    with pytest.raises(ValidationError):
        policy_validator.validate(policy)


def test_mean_std_requires_means_and_positive_standard_deviations(policy_validator):
    policy = _valid_policy(policy_validator)
    policy["rgb"]["normalization_mode"] = "MEAN_STD"
    _assert_policy_semantic_failure(policy_validator, policy, "one mean")
    policy["rgb"]["normalization_parameters"] = {"means": [0.1, 0.2, 0.3], "standard_deviations": [0.1, 0.0, 0.2]}
    with pytest.raises(ValidationError):
        policy_validator.validate(policy)


@pytest.mark.parametrize("mode", ["RESIZE", "LETTERBOX", "CENTER_CROP"])
def test_resize_modes_require_target_dimensions(mode, policy_validator):
    policy = _valid_policy(policy_validator)
    policy["rgb"]["resize_mode"] = mode
    _assert_policy_semantic_failure(policy_validator, policy, "target dimensions")


def test_none_resize_uses_null_target_dimensions(policy_validator):
    policy = _valid_policy(policy_validator)
    assert policy["rgb"]["resize_mode"] == "NONE"
    assert policy["rgb"]["target_width"] is None and policy["rgb"]["target_height"] is None
    policy["rgb"]["target_width"] = 16
    _assert_policy_semantic_failure(policy_validator, policy, "null target")


def test_mask_handling_requires_mask_input(policy_validator):
    policy = _valid_policy(policy_validator)
    policy["height"]["invalid_value_handling"] = "MASK"
    _assert_policy_semantic_failure(policy_validator, policy, "validity-mask")


def test_replacement_handling_requires_value(policy_validator):
    policy = _valid_policy(policy_validator)
    policy["height"]["invalid_value_handling"] = "REPLACE_WITH_CONSTANT"
    _assert_policy_semantic_failure(policy_validator, policy, "replacement value")


def test_preserve_nan_requires_floating_output(policy_validator):
    policy = _valid_policy(policy_validator)
    policy["height"].update({"invalid_value_handling": "PRESERVE_NAN", "output_data_type": "uint16"})
    _assert_policy_semantic_failure(policy_validator, policy, "floating-point")


def test_declared_physical_scaling_requires_sources(policy_validator):
    policy = _valid_policy(policy_validator)
    policy["height"]["scaling_mode"] = "DECLARED_PHYSICAL_SCALE"
    _assert_policy_semantic_failure(policy_validator, policy, "unit, scale source, and offset source")


def _all_keys(value):
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(child) for child in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value), set())
    return set()


def _all_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)
    elif isinstance(value, str):
        yield value


def test_point_clouds_meshes_bytes_and_paths_are_not_public_contract_fields(result_schema, result_validator):
    forbidden = {"point_cloud", "pointcloud", "mesh", "tensor", "bytes", "buffer", "path", "absolute_path", "relative_path", "filename"}
    assert forbidden.isdisjoint(key.lower() for key in _all_keys(result_schema))
    assert set(PreprocessedBufferDescriptor.__dataclass_fields__).isdisjoint({"bytes", "buffer", "tensor", "path"})
    for forbidden_field in ("point_cloud", "mesh", "tensor_bytes", "filesystem_path"):
        result = _json(RESULT_PATHS["PREPROCESSING_SUCCEEDED"])
        result[forbidden_field] = "not accepted"
        with pytest.raises(ValidationError):
            result_validator.validate(result)
    for path in RESULT_PATHS.values():
        document = _json(path)
        assert forbidden.isdisjoint(key.lower() for key in _all_keys(document))
        for value in _all_strings(document):
            assert not PurePosixPath(value).is_absolute()
            assert not PureWindowsPath(value).is_absolute()


def test_preprocessing_outcomes_never_use_pcb_decisions(result_schema):
    outcomes = result_schema["properties"]["outcome"]["enum"]
    assert set(outcomes).isdisjoint({"PASS", "FAIL", "UNCERTAIN"})
    assert outcomes == ["PREPROCESSING_SUCCEEDED", "PREPROCESSING_FAILED", "PREPROCESSING_ERROR"]
