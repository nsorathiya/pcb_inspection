import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.core.class_labels import (
    ClassLabelContract,
    ClassLabelContractError,
)

MODEL_METADATA_SCHEMA_VERSION = "1.0"


class ModelLabelCompatibilityError(RuntimeError):
    """Raised when a model cannot be safely mapped to named classes."""


@dataclass(frozen=True)
class ModelLabelMetadata:
    schema_version: str
    model_file: str
    model_sha256: str
    status: str
    class_labels_schema_version: str | None
    class_to_idx: dict[str, int] | None
    reason: str


def metadata_path_for(model_path: Path) -> Path:
    return model_path.with_suffix(".metadata.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_label_metadata(metadata_path: Path) -> ModelLabelMetadata:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLabelCompatibilityError(
            f"Unable to load model label metadata {metadata_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ModelLabelCompatibilityError("Model label metadata must be an object")

    schema_version = payload.get("schema_version")
    model_file = payload.get("model_file")
    model_sha256 = payload.get("model_sha256")
    status = payload.get("status")
    labels_version = payload.get("class_labels_schema_version")
    raw_mapping = payload.get("class_to_idx")
    reason = payload.get("reason")

    if schema_version != MODEL_METADATA_SCHEMA_VERSION:
        raise ModelLabelCompatibilityError(
            "Unsupported model label metadata schema_version "
            f"{schema_version!r}; expected {MODEL_METADATA_SCHEMA_VERSION!r}"
        )
    if not isinstance(model_file, str) or not model_file:
        raise ModelLabelCompatibilityError("Model metadata model_file is invalid")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise ModelLabelCompatibilityError("Model metadata model_sha256 is invalid")
    if status not in {"verified", "unverified"}:
        raise ModelLabelCompatibilityError(
            "Model metadata status must be 'verified' or 'unverified'"
        )
    if labels_version is not None and not isinstance(labels_version, str):
        raise ModelLabelCompatibilityError(
            "Model metadata class_labels_schema_version is invalid"
        )
    if raw_mapping is not None:
        if not isinstance(raw_mapping, dict) or not all(
            isinstance(name, str)
            and isinstance(index, int)
            and not isinstance(index, bool)
            for name, index in raw_mapping.items()
        ):
            raise ModelLabelCompatibilityError(
                "Model metadata class_to_idx must map strings to integers"
            )
        class_to_idx = dict(raw_mapping)
    else:
        class_to_idx = None
    if not isinstance(reason, str) or not reason:
        raise ModelLabelCompatibilityError("Model metadata reason is invalid")

    return ModelLabelMetadata(
        schema_version=schema_version,
        model_file=model_file,
        model_sha256=model_sha256.lower(),
        status=status,
        class_labels_schema_version=labels_version,
        class_to_idx=class_to_idx,
        reason=reason,
    )


def validate_model_label_compatibility(
    model_path: Path,
    contract: ClassLabelContract,
    metadata_path: Path | None = None,
) -> ModelLabelMetadata:
    model_path = model_path.resolve()
    metadata_path = metadata_path or metadata_path_for(model_path)
    metadata = load_model_label_metadata(metadata_path)

    if metadata.model_file != model_path.name:
        raise ModelLabelCompatibilityError(
            f"Model metadata names {metadata.model_file!r}, not {model_path.name!r}"
        )

    try:
        actual_sha256 = sha256_file(model_path)
    except OSError as exc:
        raise ModelLabelCompatibilityError(
            f"Unable to hash model file {model_path}: {exc}"
        ) from exc
    if actual_sha256 != metadata.model_sha256:
        raise ModelLabelCompatibilityError(
            f"Model SHA-256 does not match metadata for {model_path.name}"
        )

    if metadata.status != "verified":
        raise ModelLabelCompatibilityError(
            f"Model {model_path.name} class mapping is unverified: {metadata.reason}"
        )
    if metadata.class_labels_schema_version != contract.schema_version:
        raise ModelLabelCompatibilityError(
            f"Model {model_path.name} expects class-label contract "
            f"{metadata.class_labels_schema_version!r}, configured contract is "
            f"{contract.schema_version!r}"
        )
    if metadata.class_to_idx is None:
        raise ModelLabelCompatibilityError(
            f"Verified model {model_path.name} has no class_to_idx metadata"
        )
    try:
        contract.validate_class_to_idx(
            metadata.class_to_idx,
            source=f"Model {model_path.name}",
        )
    except ClassLabelContractError as exc:
        raise ModelLabelCompatibilityError(
            f"Model {model_path.name} mapping is incompatible: {exc}"
        ) from exc

    return metadata


def write_verified_model_label_metadata(
    model_path: Path,
    contract: ClassLabelContract,
    class_to_idx: Mapping[str, int],
) -> Path:
    contract.validate_class_to_idx(class_to_idx, source="Training dataset")
    resolved_model_path = model_path.resolve()
    payload = {
        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "model_file": resolved_model_path.name,
        "model_sha256": sha256_file(resolved_model_path),
        "status": "verified",
        "class_labels_schema_version": contract.schema_version,
        "class_to_idx": contract.class_to_idx,
        "reason": (
            "Generated after training validated ImageFolder class_to_idx against "
            "the authoritative class-label contract."
        ),
    }
    metadata_path = metadata_path_for(resolved_model_path)
    metadata_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path
