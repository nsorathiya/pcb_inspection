import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLASS_LABELS_PATH = REPOSITORY_ROOT / "contracts" / "class_labels.json"


class ClassLabelContractError(ValueError):
    """Raised when a class-label contract or mapping is invalid."""


class ModelOutputIndexError(IndexError):
    """Raised when a model output index is not defined by the contract."""


@dataclass(frozen=True)
class ClassLabel:
    index: int
    name: str


@dataclass(frozen=True)
class ClassLabelContract:
    schema_version: str
    classes: tuple[ClassLabel, ...]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ClassLabelContractError("Class-label schema_version must not be empty")
        if not self.classes:
            raise ClassLabelContractError("Class-label contract must define classes")

        indices = [label.index for label in self.classes]
        if len(indices) != len(set(indices)):
            raise ClassLabelContractError("Class-label indices must be unique")

        expected_indices = list(range(len(self.classes)))
        if indices != expected_indices:
            raise ClassLabelContractError(
                "Class-label indices must be contiguous and ordered from zero; "
                f"expected {expected_indices}, got {indices}"
            )

        names = [label.name for label in self.classes]
        if len(names) != len(set(names)):
            raise ClassLabelContractError("Class-label names must be unique")

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {label.name: label.index for label in self.classes}

    @property
    def class_count(self) -> int:
        return len(self.classes)

    def name_for_index(self, index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ModelOutputIndexError(
                f"Model output index must be an integer, got {index!r}"
            )
        if index < 0 or index >= self.class_count:
            raise ModelOutputIndexError(
                f"Model output index {index} is outside class-label contract "
                f"{self.schema_version}; valid range is 0..{self.class_count - 1}"
            )
        return self.classes[index].name

    def validate_class_to_idx(
        self,
        actual_mapping: Mapping[str, int],
        *,
        source: str,
    ) -> None:
        actual = dict(actual_mapping)
        expected = self.class_to_idx
        if actual != expected:
            raise ClassLabelContractError(
                f"{source} class_to_idx does not match class-label contract "
                f"{self.schema_version}; expected {expected}, got {actual}"
            )


def load_class_label_contract(
    contract_path: Path = DEFAULT_CLASS_LABELS_PATH,
) -> ClassLabelContract:
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClassLabelContractError(
            f"Unable to load class-label contract {contract_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ClassLabelContractError("Class-label contract root must be an object")

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ClassLabelContractError(
            "Class-label contract schema_version must be a non-empty string"
        )

    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, list):
        raise ClassLabelContractError("Class-label contract classes must be a list")

    classes: list[ClassLabel] = []
    for position, raw_label in enumerate(raw_classes):
        if not isinstance(raw_label, dict):
            raise ClassLabelContractError(
                f"Class-label entry {position} must be an object"
            )
        index = raw_label.get("index")
        name = raw_label.get("name")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ClassLabelContractError(
                f"Class-label entry {position} index must be an integer"
            )
        if not isinstance(name, str) or not name:
            raise ClassLabelContractError(
                f"Class-label entry {position} name must be a non-empty string"
            )
        classes.append(ClassLabel(index=index, name=name))

    return ClassLabelContract(
        schema_version=schema_version,
        classes=tuple(classes),
    )
