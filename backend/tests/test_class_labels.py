import json
from pathlib import Path

import pytest

from app.core.class_labels import (
    ClassLabelContractError,
    ModelOutputIndexError,
    load_class_label_contract,
)
from app.core.model_compatibility import (
    MODEL_METADATA_SCHEMA_VERSION,
    ModelLabelCompatibilityError,
    metadata_path_for,
    sha256_file,
    validate_model_label_compatibility,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPOSITORY_ROOT / "backend" / "dataset"
EXISTING_MODEL_PATH = REPOSITORY_ROOT / "backend" / "saved_model" / "best_model.pth"


def test_contract_class_indices_are_unique() -> None:
    contract = load_class_label_contract()
    indices = [label.index for label in contract.classes]

    assert len(indices) == len(set(indices))


def test_contract_class_names_are_unique() -> None:
    contract = load_class_label_contract()
    names = [label.name for label in contract.classes]

    assert len(names) == len(set(names))


def test_contract_indices_are_contiguous_from_zero() -> None:
    contract = load_class_label_contract()
    indices = [label.index for label in contract.classes]

    assert indices == list(range(len(indices)))


def test_imagefolder_training_mapping_matches_contract() -> None:
    contract = load_class_label_contract()
    dataset_classes = sorted(
        path.name for path in DATASET_ROOT.iterdir() if path.is_dir()
    )
    imagefolder_mapping = {
        class_name: index for index, class_name in enumerate(dataset_classes)
    }

    contract.validate_class_to_idx(
        imagefolder_mapping,
        source="Test ImageFolder mapping",
    )
    assert imagefolder_mapping == contract.class_to_idx


def test_inference_index_to_name_conversion_uses_contract() -> None:
    contract = load_class_label_contract()

    for label in contract.classes:
        assert contract.name_for_index(label.index) == label.name


def test_mismatched_mapping_raises_clear_error() -> None:
    contract = load_class_label_contract()
    legacy_inference_mapping = {
        "missing_part": 0,
        "dispense_error": 1,
        "misalignment": 2,
        "no_defect": 3,
    }

    with pytest.raises(
        ClassLabelContractError,
        match="does not match class-label contract",
    ):
        contract.validate_class_to_idx(
            legacy_inference_mapping,
            source="Legacy inference mapping",
        )


@pytest.mark.parametrize("invalid_index", [-1, 4])
def test_out_of_range_model_output_index_raises_clear_error(
    invalid_index: int,
) -> None:
    contract = load_class_label_contract()

    with pytest.raises(ModelOutputIndexError, match="outside class-label contract"):
        contract.name_for_index(invalid_index)


def test_existing_model_is_hash_bound_and_blocked_as_unverified() -> None:
    contract = load_class_label_contract()

    with pytest.raises(
        ModelLabelCompatibilityError,
        match="class mapping is unverified",
    ):
        validate_model_label_compatibility(EXISTING_MODEL_PATH, contract)


def test_verified_model_with_mismatched_mapping_is_blocked(tmp_path) -> None:
    contract = load_class_label_contract()
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"test model bytes")
    metadata = {
        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "status": "verified",
        "class_labels_schema_version": contract.schema_version,
        "class_to_idx": {
            "missing_part": 0,
            "dispense_error": 1,
            "misalignment": 2,
            "no_defect": 3,
        },
        "reason": "Test fixture with an intentionally mismatched mapping.",
    }
    metadata_path_for(model_path).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ModelLabelCompatibilityError,
        match="mapping is incompatible",
    ):
        validate_model_label_compatibility(model_path, contract)
