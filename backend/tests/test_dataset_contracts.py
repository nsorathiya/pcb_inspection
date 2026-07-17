import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_ROOT = REPOSITORY_ROOT / "contracts"
EXAMPLES_ROOT = CONTRACTS_ROOT / "examples"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


SAMPLE_SCHEMA = _load_json(CONTRACTS_ROOT / "pcb_aoi_sample.schema.json")
SPLIT_SCHEMA = _load_json(CONTRACTS_ROOT / "dataset_split_manifest.schema.json")
DATASET_SCHEMA = _load_json(CONTRACTS_ROOT / "dataset_manifest.schema.json")
TAXONOMY = _load_json(CONTRACTS_ROOT / "defect_taxonomy.json")
SAMPLE_OK = _load_json(EXAMPLES_ROOT / "sample_ok.metadata.json")
SAMPLE_NOK = _load_json(EXAMPLES_ROOT / "sample_nok.metadata.json")
SPLIT_MANIFEST = _load_json(EXAMPLES_ROOT / "split_manifest.example.json")
DATASET_MANIFEST = _load_json(EXAMPLES_ROOT / "dataset_manifest.example.json")

FORMAT_CHECKER = FormatChecker()
REGISTRY = Registry().with_resource(
    TAXONOMY["$id"],
    Resource.from_contents(TAXONOMY),
)
SAMPLE_VALIDATOR = Draft202012Validator(
    SAMPLE_SCHEMA,
    registry=REGISTRY,
    format_checker=FORMAT_CHECKER,
)
SPLIT_VALIDATOR = Draft202012Validator(
    SPLIT_SCHEMA,
    format_checker=FORMAT_CHECKER,
)
DATASET_VALIDATOR = Draft202012Validator(
    DATASET_SCHEMA,
    format_checker=FORMAT_CHECKER,
)


class SplitManifestSemanticError(ValueError):
    pass


def _validate_split_manifest(instance: dict) -> None:
    SPLIT_VALIDATOR.validate(instance)
    sample_ids: set[str] = set()
    protected_group_splits: dict[str, str] = {}
    protected_splits = {"train", "validation", "test", "holdout"}

    for assignment in instance["assignments"]:
        sample_id = assignment["sample_id"]
        if sample_id in sample_ids:
            raise SplitManifestSemanticError(
                f"sample_id {sample_id!r} has duplicate split assignments"
            )
        sample_ids.add(sample_id)

        split = assignment["split"]
        if split not in protected_splits:
            continue
        grouping_key = assignment["grouping_key_value"]
        previous_split = protected_group_splits.setdefault(grouping_key, split)
        if previous_split != split:
            raise SplitManifestSemanticError(
                f"grouping key {grouping_key!r} crosses protected splits "
                f"{previous_split!r} and {split!r}"
            )


def test_schemas_are_valid_draft_2020_12() -> None:
    for schema in (SAMPLE_SCHEMA, SPLIT_SCHEMA, DATASET_SCHEMA, TAXONOMY):
        Draft202012Validator.check_schema(schema)


def test_provided_examples_validate_against_their_schemas() -> None:
    SAMPLE_VALIDATOR.validate(SAMPLE_OK)
    SAMPLE_VALIDATOR.validate(SAMPLE_NOK)
    DATASET_VALIDATOR.validate(DATASET_MANIFEST)
    _validate_split_manifest(SPLIT_MANIFEST)


@pytest.mark.parametrize("field", ["sample_id", "board_id", "recipe_id"])
def test_missing_required_identity_fields_fail(field: str) -> None:
    invalid = copy.deepcopy(SAMPLE_OK)
    del invalid[field]

    with pytest.raises(ValidationError):
        SAMPLE_VALIDATOR.validate(invalid)


def test_nok_without_defect_type_fails() -> None:
    invalid = copy.deepcopy(SAMPLE_NOK)
    del invalid["ground_truth"]["defect_type"]

    with pytest.raises(ValidationError):
        SAMPLE_VALIDATOR.validate(invalid)


def test_ok_with_incompatible_defect_type_fails() -> None:
    invalid = copy.deepcopy(SAMPLE_OK)
    invalid["ground_truth"]["defect_type"] = "misalignment"

    with pytest.raises(ValidationError):
        SAMPLE_VALIDATOR.validate(invalid)


def test_nok_with_unrecognized_defect_type_fails() -> None:
    invalid = copy.deepcopy(SAMPLE_NOK)
    invalid["ground_truth"]["defect_type"] = "unsupported_synthetic_defect"

    with pytest.raises(ValidationError):
        SAMPLE_VALIDATOR.validate(invalid)


def test_invalid_sha256_fails() -> None:
    invalid = copy.deepcopy(SAMPLE_OK)
    invalid["integrity"]["rgb_sha256"] = "not-a-sha256"

    with pytest.raises(ValidationError):
        SAMPLE_VALIDATOR.validate(invalid)


@pytest.mark.parametrize("timestamp_path", ["capture", "provenance"])
def test_timezone_less_timestamps_fail(timestamp_path: str) -> None:
    invalid = copy.deepcopy(SAMPLE_OK)
    if timestamp_path == "capture":
        invalid["capture"]["captured_at"] = "2026-07-17T09:45:00"
    else:
        invalid["provenance"]["imported_at"] = "2026-07-17T10:30:00"

    with pytest.raises(ValidationError):
        SAMPLE_VALIDATOR.validate(invalid)


def test_unsupported_3d_storage_type_fails() -> None:
    invalid = copy.deepcopy(SAMPLE_OK)
    invalid["height_3d"]["storage_data_type"] = "uint8"

    with pytest.raises(ValidationError):
        SAMPLE_VALIDATOR.validate(invalid)


def test_duplicate_split_assignments_fail() -> None:
    invalid = copy.deepcopy(SPLIT_MANIFEST)
    duplicate = copy.deepcopy(invalid["assignments"][0])
    duplicate["split"] = "train"
    duplicate["grouping_key_value"] = "SYNTH-BOARD-DIFFERENT"
    invalid["assignments"].append(duplicate)

    with pytest.raises(SplitManifestSemanticError, match="duplicate"):
        _validate_split_manifest(invalid)


def test_same_grouping_key_cannot_cross_protected_splits() -> None:
    invalid = copy.deepcopy(SPLIT_MANIFEST)
    invalid["assignments"][1]["grouping_key_value"] = invalid["assignments"][0][
        "grouping_key_value"
    ]

    with pytest.raises(SplitManifestSemanticError, match="crosses protected"):
        _validate_split_manifest(invalid)


def test_excluded_sample_requires_exclusion_reason() -> None:
    invalid = copy.deepcopy(SPLIT_MANIFEST)
    invalid["assignments"][0]["split"] = "excluded"

    with pytest.raises(ValidationError):
        _validate_split_manifest(invalid)


def test_split_values_are_restricted() -> None:
    invalid = copy.deepcopy(SPLIT_MANIFEST)
    invalid["assignments"][0]["split"] = "development"

    with pytest.raises(ValidationError):
        _validate_split_manifest(invalid)


def test_contract_schema_and_taxonomy_versions_are_present() -> None:
    assert SAMPLE_SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert SAMPLE_SCHEMA["properties"]["contract_version"]["const"] == (
        "pcb-aoi-dataset/1.0"
    )
    assert SPLIT_SCHEMA["properties"]["split_manifest_version"]["const"] == (
        "pcb-aoi-split-manifest/1.0"
    )
    assert DATASET_SCHEMA["properties"]["dataset_manifest_version"]["const"] == (
        "pcb-aoi-dataset-manifest/1.0"
    )
    assert TAXONOMY["taxonomy_version"] == "pcb-aoi-defects/1.0"
    semantic_rule_ids = {
        rule["id"] for rule in SPLIT_SCHEMA["x-semantic_rules"]
    }
    assert semantic_rule_ids == {
        "unique_sample_assignment",
        "group_partition_exclusivity",
    }
