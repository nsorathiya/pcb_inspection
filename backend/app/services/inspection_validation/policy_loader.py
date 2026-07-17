from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from app.services.inspection_validation.exceptions import PolicyLoadError
from app.services.inspection_validation.interfaces import (
    DimensionRelationship,
    InspectionValidationPolicy,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCHEMA_PATH = REPOSITORY_ROOT / "contracts" / "inspection_validation_policy.schema.json"
DEFAULT_DEVELOPMENT_POLICY_PATH = REPOSITORY_ROOT / "contracts" / "examples" / "inspection_validation_policy.development.json"
DEVELOPMENT_POLICY_ID = "development-native-rgb-height"
DEVELOPMENT_POLICY_VERSION = "1.0"


def _resolve(schema: Mapping[str, Any], rule: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = rule.get("$ref")
    if not reference:
        return rule
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        raise ValueError("unsupported policy schema reference")
    return schema["$defs"][reference.rsplit("/", 1)[-1]]


def _validate_value(value: Any, rule: Mapping[str, Any], schema: Mapping[str, Any], field: str) -> None:
    rule = _resolve(schema, rule)
    if "const" in rule and value != rule["const"]:
        raise ValueError(f"{field} has an unsupported value")
    expected = rule.get("type")
    type_ok = {
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "array": isinstance(value, list),
    }.get(expected, True)
    if not type_ok:
        raise ValueError(f"{field} has an invalid type")
    if "enum" in rule and value not in rule["enum"]:
        raise ValueError(f"{field} has an unsupported value")
    if isinstance(value, str):
        if len(value) < rule.get("minLength", 0) or len(value) > rule.get("maxLength", 10**9):
            raise ValueError(f"{field} has an invalid length")
        if "pattern" in rule and re.fullmatch(rule["pattern"], value) is None:
            raise ValueError(f"{field} has an invalid format")
    if isinstance(value, int) and not isinstance(value, bool):
        if value < rule.get("minimum", value) or value > rule.get("maximum", value):
            raise ValueError(f"{field} is outside supported bounds")
    if isinstance(value, list):
        if len(value) < rule.get("minItems", 0):
            raise ValueError(f"{field} does not contain enough values")
        if rule.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            raise ValueError(f"{field} contains duplicate values")
        for index, item in enumerate(value):
            _validate_value(item, rule.get("items", {}), schema, f"{field}[{index}]")


def validate_policy_document(document: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    if not isinstance(document, dict):
        raise ValueError("policy document must be an object")
    required = set(schema["required"])
    properties = schema["properties"]
    if required.difference(document):
        raise ValueError("policy document is missing required values")
    if schema.get("additionalProperties") is False and set(document).difference(properties):
        raise ValueError("policy document contains unknown values")
    for field, value in document.items():
        _validate_value(value, properties[field], schema, field)
    if document["maximum_width"] < document["minimum_width"]:
        raise ValueError("maximum_width must not be smaller than minimum_width")
    if document["maximum_height"] < document["minimum_height"]:
        raise ValueError("maximum_height must not be smaller than minimum_height")
    if "NAN" in document["allowed_height_invalid_value_policies"] and not {
        "float32", "float64"
    }.intersection(document["allowed_height_storage_types"]):
        raise ValueError("NAN policy requires floating height storage")
    if document["dimension_relationship"] == "REGISTERED_TRANSFORM_REQUIRED" and not document["require_registration_evidence"]:
        raise ValueError("registered-transform policy must require registration evidence")


def _to_policy(document: Mapping[str, Any]) -> InspectionValidationPolicy:
    return InspectionValidationPolicy(
        contract_version=document["contract_version"],
        policy_id=document["policy_id"],
        policy_version=document["policy_version"],
        display_name=document["display_name"],
        description=document["description"],
        development_only=document["development_only"],
        allowed_rgb_formats=tuple(document["allowed_rgb_formats"]),
        allowed_height_formats=tuple(document["allowed_height_formats"]),
        allowed_rgb_channels=tuple(document["allowed_rgb_channels"]),
        allowed_rgb_bit_depths=tuple(document["allowed_rgb_bit_depths"]),
        allowed_height_storage_types=tuple(document["allowed_height_storage_types"]),
        allowed_height_invalid_value_policies=tuple(document["allowed_height_invalid_value_policies"]),
        minimum_height_bit_depth=document["minimum_height_bit_depth"],
        require_single_channel_height=document["require_single_channel_height"],
        require_explicit_height_invalid_value_policy=document["require_explicit_height_invalid_value_policy"],
        minimum_width=document["minimum_width"],
        minimum_height=document["minimum_height"],
        maximum_width=document["maximum_width"],
        maximum_height=document["maximum_height"],
        dimension_relationship=DimensionRelationship(document["dimension_relationship"]),
        require_calibration_artifact=document["require_calibration_artifact"],
        require_validity_mask=document["require_validity_mask"],
        require_registration_evidence=document["require_registration_evidence"],
        warning_as_blocking=document["warning_as_blocking"],
    )


class ValidationPolicyLoader:
    def __init__(self, schema_path: Path = DEFAULT_SCHEMA_PATH, *, registry: Mapping[tuple[str, str], Path] | None = None) -> None:
        self._schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self._registry = dict(registry or {
            (DEVELOPMENT_POLICY_ID, DEVELOPMENT_POLICY_VERSION): DEFAULT_DEVELOPMENT_POLICY_PATH,
        })

    def supports_policy_id(self, policy_id: str) -> bool:
        return any(key[0] == policy_id for key in self._registry)

    def supports(self, policy_id: str, policy_version: str) -> bool:
        return (policy_id, policy_version) in self._registry

    def load(self, policy_id: str, policy_version: str) -> InspectionValidationPolicy:
        path = self._registry.get((policy_id, policy_version))
        if path is None:
            known_id = any(key[0] == policy_id for key in self._registry)
            code = "POLICY_VERSION_UNSUPPORTED" if known_id else "POLICY_NOT_FOUND"
            raise PolicyLoadError(code, "selected validation policy is unavailable")
        return self.load_path(path, expected_policy_id=policy_id, expected_policy_version=policy_version)

    def load_path(self, path: Path, *, expected_policy_id: str | None = None, expected_policy_version: str | None = None) -> InspectionValidationPolicy:
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            validate_policy_document(document, self._schema)
            policy = _to_policy(document)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PolicyLoadError("POLICY_VERSION_UNSUPPORTED", "validation policy is malformed or unsupported") from exc
        if expected_policy_id is not None and policy.policy_id != expected_policy_id:
            raise PolicyLoadError("POLICY_NOT_FOUND", "validation policy identity does not match selection")
        if expected_policy_version is not None and policy.policy_version != expected_policy_version:
            raise PolicyLoadError("POLICY_VERSION_UNSUPPORTED", "validation policy version does not match selection")
        return policy
