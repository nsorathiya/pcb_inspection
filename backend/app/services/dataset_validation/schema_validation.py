from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from app.services.dataset_validation.models import Finding

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONTRACTS_ROOT = REPOSITORY_ROOT / "contracts"


@dataclass(frozen=True)
class SchemaBundle:
    sample_schema: dict[str, Any]
    split_schema: dict[str, Any]
    dataset_schema: dict[str, Any]
    taxonomy: dict[str, Any]
    sample_validator: Draft202012Validator
    split_validator: Draft202012Validator
    dataset_validator: Draft202012Validator


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def load_schema_bundle() -> SchemaBundle:
    sample_schema = _read_json(CONTRACTS_ROOT / "pcb_aoi_sample.schema.json")
    split_schema = _read_json(
        CONTRACTS_ROOT / "dataset_split_manifest.schema.json"
    )
    dataset_schema = _read_json(CONTRACTS_ROOT / "dataset_manifest.schema.json")
    taxonomy = _read_json(CONTRACTS_ROOT / "defect_taxonomy.json")
    for schema in (sample_schema, split_schema, dataset_schema, taxonomy):
        Draft202012Validator.check_schema(schema)
    registry = Registry().with_resource(
        taxonomy["$id"],
        Resource.from_contents(taxonomy),
    )
    checker = FormatChecker()
    return SchemaBundle(
        sample_schema=sample_schema,
        split_schema=split_schema,
        dataset_schema=dataset_schema,
        taxonomy=taxonomy,
        sample_validator=Draft202012Validator(
            sample_schema,
            registry=registry,
            format_checker=checker,
        ),
        split_validator=Draft202012Validator(
            split_schema,
            format_checker=checker,
        ),
        dataset_validator=Draft202012Validator(
            dataset_schema,
            format_checker=checker,
        ),
    )


def schema_findings(
    validator: Draft202012Validator,
    instance: Any,
    *,
    scope: str,
    path: str,
    sample_id: str | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.validator or "",
            error.message,
        ),
    )
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        findings.append(
            Finding(
                code=f"schema.{scope}",
                category="schema",
                message=f"{location}: {error.message}",
                sample_id=sample_id,
                path=path,
            )
        )
    return findings
