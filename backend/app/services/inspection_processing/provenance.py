from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from app.db.models import ArtifactType
from app.services.inspection_processing.exceptions import (
    SyntheticProvenanceMismatchError,
    SyntheticProvenanceUnavailableError,
)
from app.services.inspection_processing.input_builder import ProcessingInputSnapshot
from app.testing.synthetic_aoi.manifest import output_tree_sha256, sha256_bytes
from app.testing.synthetic_aoi.models import (
    GENERATION_MANIFEST_FILENAME,
    GENERATION_MANIFEST_VERSION,
    GENERATOR_ID,
    GENERATOR_VERSION,
    MARKER_FILENAME,
    MARKER_VERSION,
    SCENARIO_CONTRACT_VERSION,
    SYNTHETIC_STATEMENT,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SCENARIO_SCHEMA_PATH = REPOSITORY_ROOT / "contracts" / "synthetic_inspection_scenario.schema.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCENARIO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

_MARKER_FIELDS = {
    "contract_version", "fixture_statement", "generator_id", "generator_version",
    "generation_manifest_sha256", "model_accuracy_evidence", "production_approved",
    "scenario_id", "seed", "synthetic", "training_approved",
}
_MANIFEST_FIELDS = {
    "contract_version", "fixture_statement", "fixture_timestamp", "files",
    "generator_id", "generator_version", "generation_uuid", "integrity_scope",
    "model_accuracy_evidence", "output_tree_sha256", "production_approved",
    "scenario_count", "scenario_id", "scenario_ids", "seed", "synthetic",
    "training_approved",
}
_ROLE_BY_TYPE = {
    ArtifactType.RGB_RAW: "rgb",
    ArtifactType.HEIGHT_RAW: "height",
    ArtifactType.VALIDITY_MASK: "mask",
    ArtifactType.CALIBRATION: "calibration",
}


@dataclass(frozen=True)
class VerifiedSyntheticScenario:
    scenario_id: str
    generator_id: str
    generator_version: str


def _is_redirect(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _synthetic_flags(value: Mapping[str, Any]) -> bool:
    return (
        value.get("synthetic") is True
        and value.get("training_approved") is False
        and value.get("production_approved") is False
        and value.get("model_accuracy_evidence") is False
        and value.get("fixture_statement") == SYNTHETIC_STATEMENT
    )


class SyntheticFixtureProvenanceVerifier:
    """Verify one injected generator-owned root without caller paths or scanning."""

    def __init__(
        self,
        fixture_root: Path,
        *,
        scenario_schema_path: Path = SCENARIO_SCHEMA_PATH,
    ) -> None:
        self._root = Path(os.path.abspath(os.fspath(Path(fixture_root).expanduser())))
        try:
            schema = json.loads(Path(scenario_schema_path).read_text(encoding="utf-8"))
            self._scenario_validator = Draft202012Validator(
                schema, format_checker=FormatChecker()
            )
        except Exception as exc:
            raise SyntheticProvenanceUnavailableError(
                "synthetic provenance contract is unavailable"
            ) from exc

    def verify(self, snapshot: ProcessingInputSnapshot) -> VerifiedSyntheticScenario:
        try:
            marker_path = self._owned_file(PurePosixPath(MARKER_FILENAME))
            manifest_path = self._owned_file(PurePosixPath(GENERATION_MANIFEST_FILENAME))
            marker_bytes = marker_path.read_bytes()
            manifest_bytes = manifest_path.read_bytes()
            marker = self._json_object(marker_bytes)
            manifest = self._json_object(manifest_bytes)
        except SyntheticProvenanceUnavailableError:
            raise
        except Exception as exc:
            raise SyntheticProvenanceUnavailableError(
                "trusted synthetic provenance is unavailable"
            ) from exc

        try:
            self._validate_marker(marker, manifest_bytes)
            inventory = self._validate_manifest(manifest, marker)
            scenarios = self._read_scenarios(manifest, inventory)
            matches = [item for item in scenarios if self._matches(item, snapshot)]
        except SyntheticProvenanceUnavailableError:
            raise
        except Exception as exc:
            raise SyntheticProvenanceMismatchError(
                "trusted synthetic provenance did not verify"
            ) from exc
        if len(matches) != 1:
            raise SyntheticProvenanceMismatchError(
                "registered artifacts do not identify one trusted synthetic scenario"
            )
        selected = matches[0]
        return VerifiedSyntheticScenario(
            scenario_id=selected["scenario_id"],
            generator_id=selected["generator_id"],
            generator_version=selected["generator_version"],
        )

    def _owned_file(self, relative: PurePosixPath) -> Path:
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any("\\" in part or ":" in part for part in relative.parts)
        ):
            raise SyntheticProvenanceMismatchError(
                "synthetic provenance inventory is unsafe"
            )
        if not self._root.is_dir() or _is_redirect(self._root):
            raise SyntheticProvenanceUnavailableError(
                "trusted synthetic provenance is unavailable"
            )
        current = self._root
        for part in relative.parts:
            current /= part
            if os.path.lexists(current) and _is_redirect(current):
                raise SyntheticProvenanceMismatchError(
                    "synthetic provenance contains a redirected entry"
                )
        try:
            current.resolve(strict=False).relative_to(self._root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise SyntheticProvenanceMismatchError(
                "synthetic provenance inventory is unsafe"
            ) from exc
        if not current.is_file():
            raise SyntheticProvenanceUnavailableError(
                "trusted synthetic provenance is unavailable"
            )
        return current

    @staticmethod
    def _json_object(content: bytes) -> dict[str, Any]:
        value = json.loads(content.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        return value

    @staticmethod
    def _validate_marker(marker: Mapping[str, Any], manifest_bytes: bytes) -> None:
        if (
            set(marker) != _MARKER_FIELDS
            or marker.get("contract_version") != MARKER_VERSION
            or marker.get("generator_id") != GENERATOR_ID
            or marker.get("generator_version") != GENERATOR_VERSION
            or marker.get("scenario_id") != "generator_owned_output"
            or not _synthetic_flags(marker)
            or marker.get("generation_manifest_sha256") != sha256_bytes(manifest_bytes)
        ):
            raise ValueError("invalid ownership marker")

    def _validate_manifest(
        self,
        manifest: Mapping[str, Any],
        marker: Mapping[str, Any],
    ) -> dict[str, Mapping[str, Any]]:
        if (
            set(manifest) != _MANIFEST_FIELDS
            or manifest.get("contract_version") != GENERATION_MANIFEST_VERSION
            or manifest.get("generator_id") != GENERATOR_ID
            or manifest.get("generator_version") != GENERATOR_VERSION
            or manifest.get("scenario_id") != "generation_manifest"
            or manifest.get("seed") != marker.get("seed")
            or not _synthetic_flags(manifest)
            or not isinstance(manifest.get("files"), list)
            or not isinstance(manifest.get("scenario_ids"), list)
        ):
            raise ValueError("invalid generation manifest")
        scenario_ids = manifest["scenario_ids"]
        if (
            manifest.get("scenario_count") != len(scenario_ids)
            or len(scenario_ids) != len(set(scenario_ids))
            or any(not isinstance(value, str) or not _SCENARIO_ID.fullmatch(value)
                   for value in scenario_ids)
        ):
            raise ValueError("invalid scenario inventory")

        inventory: dict[str, Mapping[str, Any]] = {}
        for item in manifest["files"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"byte_size", "path", "sha256"}
                or not isinstance(item.get("path"), str)
                or isinstance(item.get("byte_size"), bool)
                or not isinstance(item.get("byte_size"), int)
                or item["byte_size"] < 0
                or not isinstance(item.get("sha256"), str)
                or not _SHA256.fullmatch(item["sha256"])
            ):
                raise ValueError("invalid file inventory")
            pure = PurePosixPath(item["path"])
            if (
                pure.is_absolute()
                or len(pure.parts) < 3
                or pure.parts[0] != "scenarios"
                or ".." in pure.parts
                or "\\" in item["path"]
                or ":" in item["path"]
                or item["path"] in inventory
            ):
                raise ValueError("unsafe file inventory")
            path = self._owned_file(pure)
            content = path.read_bytes()
            if len(content) != item["byte_size"] or sha256_bytes(content) != item["sha256"]:
                raise ValueError("file inventory mismatch")
            inventory[item["path"]] = item
        ordered = [inventory[key] for key in sorted(inventory)]
        if manifest.get("output_tree_sha256") != output_tree_sha256(ordered):
            raise ValueError("output tree digest mismatch")
        return inventory

    def _read_scenarios(
        self,
        manifest: Mapping[str, Any],
        inventory: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        result = []
        for scenario_id in manifest["scenario_ids"]:
            relative = f"scenarios/{scenario_id}/scenario.json"
            if relative not in inventory:
                raise ValueError("scenario record is absent from inventory")
            scenario = self._json_object(self._owned_file(PurePosixPath(relative)).read_bytes())
            self._scenario_validator.validate(scenario)
            if (
                scenario.get("contract_version") != SCENARIO_CONTRACT_VERSION
                or scenario.get("scenario_id") != scenario_id
                or scenario.get("generator_id") != GENERATOR_ID
                or scenario.get("generator_version") != GENERATOR_VERSION
                or scenario.get("seed") != manifest.get("seed")
                or not _synthetic_flags(scenario)
            ):
                raise ValueError("scenario provenance mismatch")
            self._validate_scenario_artifacts(scenario, inventory)
            result.append(scenario)
        return tuple(result)

    def _validate_scenario_artifacts(
        self,
        scenario: Mapping[str, Any],
        inventory: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for role, artifact in scenario["artifacts"].items():
            filename = artifact["generated_file"]
            pure_name = PurePosixPath(filename)
            if pure_name.is_absolute() or len(pure_name.parts) != 1 or ".." in pure_name.parts:
                raise ValueError("unsafe scenario artifact name")
            relative = f"scenarios/{scenario['scenario_id']}/{filename}"
            item = inventory.get(relative)
            if item is None:
                raise ValueError("scenario artifact is absent from inventory")
            content = self._owned_file(PurePosixPath(relative)).read_bytes()
            if (
                artifact.get("role") != role
                or artifact.get("actual_sha256") != sha256_bytes(content)
                or artifact.get("actual_byte_size") != len(content)
                or item.get("sha256") != artifact.get("actual_sha256")
                or item.get("byte_size") != artifact.get("actual_byte_size")
            ):
                raise ValueError("scenario artifact identity mismatch")

    @staticmethod
    def _matches(
        scenario: Mapping[str, Any], snapshot: ProcessingInputSnapshot
    ) -> bool:
        artifacts = scenario.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return False
        registered = {
            _ROLE_BY_TYPE[item.artifact_type]: (item.sha256, item.byte_size)
            for item in snapshot.artifacts
        }
        if set(artifacts) != set(registered):
            return False
        for role, identity in registered.items():
            value = artifacts.get(role)
            if not isinstance(value, Mapping):
                return False
            if (value.get("actual_sha256"), value.get("actual_byte_size")) != identity:
                return False
        return True
