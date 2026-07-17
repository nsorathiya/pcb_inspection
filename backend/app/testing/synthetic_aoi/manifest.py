from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from uuid import UUID, uuid5

from app.testing.synthetic_aoi.models import (
    FIXTURE_TIMESTAMP,
    GENERATOR_ID,
    GENERATOR_VERSION,
    SCENARIO_CONTRACT_VERSION,
    SYNTHETIC_STATEMENT,
    ScenarioPlan,
)

SYNTHETIC_UUID_NAMESPACE = UUID("63ed0bb8-1a60-5f85-932d-b3bb7c1d4867")


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def deterministic_uuid(seed: int, scenario_id: str) -> str:
    name = f"{GENERATOR_ID}:{GENERATOR_VERSION}:{seed}:{scenario_id}"
    return str(uuid5(SYNTHETIC_UUID_NAMESPACE, name))


def _safe_generated_name(value: str) -> str:
    pure = PurePosixPath(value)
    if pure.is_absolute() or len(pure.parts) != 1 or ".." in pure.parts:
        raise ValueError(f"Unsafe generated fixture filename: {value!r}")
    return value


def write_scenario(root: Path, plan: ScenarioPlan, seed: int) -> tuple[Path, ...]:
    scenario_root = root / "scenarios" / plan.scenario_id
    scenario_root.mkdir(parents=True, exist_ok=False)
    created_paths: list[Path] = []
    artifact_records: dict[str, dict[str, Any]] = {}
    for artifact in plan.artifacts:
        filename = _safe_generated_name(artifact.generated_file)
        path = scenario_root / filename
        path.write_bytes(artifact.content)
        created_paths.append(path)
        actual_sha256 = sha256_bytes(artifact.content)
        actual_byte_size = len(artifact.content)
        artifact_records[artifact.role] = {
            "actual_byte_size": actual_byte_size,
            "actual_sha256": actual_sha256,
            "declared_byte_size": (
                actual_byte_size
                if artifact.declared_byte_size is None
                else artifact.declared_byte_size
            ),
            "declared_sha256": (
                actual_sha256
                if artifact.declared_sha256 is None
                else artifact.declared_sha256
            ),
            "generated_file": filename,
            "media_type": artifact.media_type,
            "reference_expected_to_resolve": artifact.reference_expected_to_resolve,
            "references": list(artifact.references),
            "role": artifact.role,
        }
    record = {
        "artifacts": artifact_records,
        "contract_version": SCENARIO_CONTRACT_VERSION,
        "description": plan.description,
        "expected_finding_codes": list(plan.expected_finding_codes),
        "expected_intake_outcome": plan.expected_intake_outcome,
        "expected_technical_validation_outcome": (
            plan.expected_technical_validation_outcome
        ),
        "fixture_statement": SYNTHETIC_STATEMENT,
        "fixture_timestamp": FIXTURE_TIMESTAMP,
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "model_accuracy_evidence": False,
        "notes": list(plan.notes),
        "policy": {
            "policy_id": plan.policy_id,
            "policy_version": plan.policy_version,
        },
        "production_approved": False,
        "scenario_id": plan.scenario_id,
        "scenario_uuid": deterministic_uuid(seed, plan.scenario_id),
        "seed": seed,
        "synthetic": True,
        "training_approved": False,
    }
    path = scenario_root / "scenario.json"
    path.write_bytes(json_bytes(record))
    created_paths.append(path)
    return tuple(created_paths)


def file_inventory(root: Path, relative_paths: Iterable[Path]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(relative_paths, key=lambda item: item.as_posix()):
        content = (root / path).read_bytes()
        inventory.append(
            {
                "byte_size": len(content),
                "path": path.as_posix(),
                "sha256": sha256_bytes(content),
            }
        )
    return inventory


def output_tree_sha256(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item["byte_size"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
