import copy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from referencing import Registry, Resource

from app.services.inspection_inference import (
    validate_inference_policy_document,
    validate_inference_result_document,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts"
EXAMPLES = CONTRACTS / "examples"
POLICY_SCHEMA_PATH = CONTRACTS / "inspection_inference_policy.schema.json"
RESULT_SCHEMA_PATH = CONTRACTS / "inspection_inference_result.schema.json"
CATALOGUE_PATH = CONTRACTS / "inspection_inference_findings.json"
TAXONOMY_PATH = CONTRACTS / "defect_taxonomy.json"
POLICY_PATH = EXAMPLES / "inspection_inference_policy.mock.json"
RESULT_PATHS = {
    "PASS": EXAMPLES / "inspection_inference_result.pass.json",
    "FAIL": EXAMPLES / "inspection_inference_result.fail.json",
    "UNCERTAIN": EXAMPLES / "inspection_inference_result.uncertain.json",
    "ERROR": EXAMPLES / "inspection_inference_result.error.json",
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
def taxonomy():
    return _json(TAXONOMY_PATH)


@pytest.fixture(scope="module")
def policy_validator(policy_schema):
    return Draft202012Validator(policy_schema, format_checker=FormatChecker())


@pytest.fixture(scope="module")
def result_validator(result_schema, catalogue, taxonomy):
    registry = Registry().with_resources(
        [
            (catalogue["$id"], Resource.from_contents(catalogue)),
            (taxonomy["$id"], Resource.from_contents(taxonomy)),
        ]
    )
    return Draft202012Validator(
        result_schema,
        registry=registry,
        format_checker=FormatChecker(),
    )


def _valid_policy(policy_validator) -> dict:
    policy = _json(POLICY_PATH)
    policy_validator.validate(policy)
    validate_inference_policy_document(policy)
    return policy


def _valid_result(name, result_validator, catalogue, taxonomy) -> dict:
    result = _json(RESULT_PATHS[name])
    result_validator.validate(result)
    validate_inference_result_document(result, catalogue, taxonomy)
    return result


def test_schemas_are_draft_2020_12_and_well_formed(
    policy_schema, result_schema, catalogue
):
    for schema in (policy_schema, result_schema, catalogue):
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)


def test_contract_versions_are_explicit(policy_schema, result_schema, catalogue):
    assert policy_schema["properties"]["contract_version"]["const"] == (
        "pcb-aoi-inspection-inference-policy/1.0"
    )
    assert result_schema["properties"]["contract_version"]["const"] == (
        "pcb-aoi-inspection-inference/1.0"
    )
    assert catalogue["catalog_version"] == (
        "pcb-aoi-inspection-inference-findings/1.0"
    )


def test_finding_codes_are_unique_and_catalogue_order_is_stable(catalogue):
    entries = catalogue["findings"]
    codes = [item["code"] for item in entries]
    orders = [item["order"] for item in entries]
    assert len(codes) == len(set(codes))
    assert orders == sorted(set(orders))
    assert codes == catalogue["$defs"]["finding_code"]["enum"]


def test_mock_policy_validates_and_has_complete_nonoverlapping_buckets(
    policy_validator,
):
    policy = _valid_policy(policy_validator)
    engine = policy["engine"]
    groups = (
        engine["pass_buckets"],
        engine["fail_buckets"],
        engine["uncertain_buckets"],
    )
    combined = [bucket for group in groups for bucket in group]
    assert all(groups)
    assert len(combined) == len(set(combined))
    assert set(combined) == set(range(engine["decision_bucket_count"]))
    assert policy["development_only"] is True
    assert policy["production_approved"] is False
    assert policy["safety"]["allow_real_input"] is False
    assert policy["engine"]["confidence_mode"] == "NONE"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"fail_buckets": [3, 4, 5, 6, 7]}, "overlap"),
        ({"uncertain_buckets": [8, 9, 10]}, "complete"),
        ({"uncertain_buckets": [8, 9, 10, 12]}, "outside"),
        ({"pass_buckets": []}, "at least one"),
    ],
)
def test_invalid_bucket_assignments_fail_semantic_validation(
    policy_validator, mutation, message
):
    policy = _json(POLICY_PATH)
    policy["engine"].update(mutation)
    if mutation == {"pass_buckets": []}:
        with pytest.raises(ValidationError):
            policy_validator.validate(policy)
    else:
        policy_validator.validate(policy)
        with pytest.raises(ValueError, match=message):
            validate_inference_policy_document(policy)


@pytest.mark.parametrize("name", tuple(RESULT_PATHS))
def test_synthetic_result_examples_validate(
    name, result_validator, catalogue, taxonomy
):
    result = _valid_result(name, result_validator, catalogue, taxonomy)
    assert result["mock_inference"] is True
    assert result["production_approved"] is False
    assert result["confidence"] is None


@pytest.mark.parametrize("name", ["PASS", "FAIL", "UNCERTAIN"])
def test_success_examples_match_documented_digest_bucket_and_taxonomy_algorithm(
    name, policy_validator, taxonomy
):
    policy = _valid_policy(policy_validator)
    result = _json(RESULT_PATHS[name])
    decision_input = {
        "engine_id": result["engine_id"],
        "engine_version": result["engine_version"],
        "height": {
            "buffer_sha256": result["height_input"]["buffer_sha256"],
            "data_type": result["height_input"]["data_type"],
            "layout": result["height_input"]["layout"],
            "shape": result["height_input"]["shape"],
        },
        "inspection_id": result["inspection_id"],
        "policy_id": result["policy_id"],
        "policy_version": result["policy_version"],
        "preprocessing_id": result["preprocessing_id"],
        "rgb": {
            "buffer_sha256": result["rgb_input"]["buffer_sha256"],
            "data_type": result["rgb_input"]["data_type"],
            "layout": result["rgb_input"]["layout"],
            "shape": result["rgb_input"]["shape"],
        },
        "strategy": result["decision_basis"],
        "validation_id": result["validation_id"],
    }
    digest = hashlib.sha256(
        json.dumps(decision_input, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    bucket = int(digest[:16], 16) % policy["engine"]["decision_bucket_count"]
    buckets = policy["engine"][f"{name.lower()}_buckets"]
    assert digest == result["decision_digest"]
    assert bucket in buckets
    if name == "FAIL":
        defect_digest = hashlib.sha256(
            f"{digest}:{taxonomy['taxonomy_version']}".encode("ascii")
        ).hexdigest()
        defects = taxonomy["$defs"]["supported_defect_type"]["enum"]
        assert result["defect_type"] == defects[int(defect_digest[:16], 16) % len(defects)]


def test_success_requires_a_decision(result_validator):
    result = _json(RESULT_PATHS["PASS"])
    result["decision"] = None
    with pytest.raises(ValidationError):
        result_validator.validate(result)


@pytest.mark.parametrize("outcome", ["INFERENCE_FAILED", "INFERENCE_ERROR"])
def test_failed_and_error_outcomes_reject_decisions(outcome, result_validator):
    result = _json(RESULT_PATHS["ERROR"])
    result["execution_outcome"] = outcome
    result.update(
        decision="PASS",
        decision_basis="DETERMINISTIC_HASH_BUCKET",
        decision_digest="a" * 64,
    )
    with pytest.raises(ValidationError):
        result_validator.validate(result)


def test_pass_rejects_defect_type(result_validator):
    result = _json(RESULT_PATHS["PASS"])
    result["defect_type"] = "dispense_error"
    with pytest.raises(ValidationError):
        result_validator.validate(result)


@pytest.mark.parametrize("defect_type", [None, "no_defect", "invented_defect"])
def test_fail_requires_authoritative_taxonomy_defect(defect_type, result_validator):
    result = _json(RESULT_PATHS["FAIL"])
    result["defect_type"] = defect_type
    with pytest.raises(ValidationError):
        result_validator.validate(result)


def test_uncertain_rejects_defect_type(result_validator):
    result = _json(RESULT_PATHS["UNCERTAIN"])
    result["defect_type"] = "misalignment"
    with pytest.raises(ValidationError):
        result_validator.validate(result)


def test_summary_catalogue_and_order_are_semantically_enforced(
    result_validator, catalogue, taxonomy
):
    result = _json(RESULT_PATHS["PASS"])
    result["summary"]["warnings"] = 0
    result_validator.validate(result)
    with pytest.raises(ValueError, match="summary counts"):
        validate_inference_result_document(result, catalogue, taxonomy)

    result = _json(RESULT_PATHS["PASS"])
    result["findings"].reverse()
    with pytest.raises(ValueError, match="catalogue order"):
        validate_inference_result_document(result, catalogue, taxonomy)


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


def test_public_schema_and_examples_have_no_bytes_paths_or_model_metrics(
    result_schema, result_validator
):
    forbidden = {
        "bytes",
        "data",
        "tensor",
        "path",
        "filename",
        "model_path",
        "model_weights",
        "probability",
        "accuracy",
    }
    assert forbidden.isdisjoint(key.lower() for key in _all_keys(result_schema))
    for path in RESULT_PATHS.values():
        document = _json(path)
        assert forbidden.isdisjoint(key.lower() for key in _all_keys(document))
        for value in _all_strings(document):
            assert not PurePosixPath(value).is_absolute()
            assert not PureWindowsPath(value).is_absolute()
    result = _json(RESULT_PATHS["PASS"])
    result["tensor_bytes"] = "not accepted"
    with pytest.raises(ValidationError):
        result_validator.validate(result)


def test_mock_policy_cannot_enable_production_real_input_or_confidence(
    policy_validator,
):
    mutations = [
        ("production_approved", True),
        ("development_only", False),
    ]
    for field, value in mutations:
        policy = copy.deepcopy(_json(POLICY_PATH))
        policy[field] = value
        with pytest.raises(ValidationError):
            policy_validator.validate(policy)
    for section, field, value in (
        ("safety", "allow_real_input", True),
        ("safety", "allow_model_accuracy_claim", True),
        ("safety", "allow_production_decision", True),
        ("engine", "confidence_mode", "DIGEST_NUMBER"),
    ):
        policy = copy.deepcopy(_json(POLICY_PATH))
        policy[section][field] = value
        with pytest.raises(ValidationError):
            policy_validator.validate(policy)


def test_execution_outcomes_are_separate_from_mock_decisions(result_schema):
    outcomes = result_schema["properties"]["execution_outcome"]["enum"]
    assert outcomes == ["INFERENCE_SUCCEEDED", "INFERENCE_FAILED", "INFERENCE_ERROR"]
    assert set(outcomes).isdisjoint({"PASS", "FAIL", "UNCERTAIN"})
