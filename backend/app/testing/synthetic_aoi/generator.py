from __future__ import annotations

import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from app.testing.synthetic_aoi.manifest import (
    deterministic_uuid,
    file_inventory,
    json_bytes,
    output_tree_sha256,
    sha256_bytes,
    write_scenario,
)
from app.testing.synthetic_aoi.models import (
    DEFAULT_SEED,
    FIXTURE_TIMESTAMP,
    GENERATION_MANIFEST_VERSION,
    GENERATION_MANIFEST_FILENAME,
    GENERATOR_ID,
    GENERATOR_VERSION,
    MARKER_FILENAME,
    MARKER_VERSION,
    SYNTHETIC_STATEMENT,
    GenerationResult,
    SyntheticFixtureError,
)
from app.testing.synthetic_aoi.scenarios import SCENARIO_IDS, build_scenario

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_ROOT = REPOSITORY_ROOT / "runtime"
FORBIDDEN_REPOSITORY_DIRECTORIES = (
    REPOSITORY_ROOT / "backend" / "dataset",
    REPOSITORY_ROOT / "backend" / "dataset_raw",
    REPOSITORY_ROOT / "backend" / "test",
    REPOSITORY_ROOT / "backend" / "tests",
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _absolute_unresolved(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_link_components(path: Path) -> None:
    components = (path, *path.parents)
    for component in components:
        if component.exists() and _is_link_or_reparse(component):
            raise SyntheticFixtureError(
                "Output path and its existing parents must not be symbolic links or reparse points"
            )


def _validate_output_path(
    output_root: Path,
    *,
    allow_runtime_for_tests: bool,
) -> Path:
    output = _absolute_unresolved(output_root)
    _reject_link_components(output)
    resolved = output.resolve(strict=False)
    filesystem_root = Path(resolved.anchor).resolve(strict=False)
    home = Path.home().resolve(strict=False)
    repository = REPOSITORY_ROOT.resolve(strict=True)
    runtime = RUNTIME_ROOT.resolve(strict=False)

    if resolved == filesystem_root:
        raise SyntheticFixtureError("Filesystem or drive roots are unsafe outputs")
    if resolved == home or _is_within(home, resolved):
        raise SyntheticFixtureError("Home directories and their ancestors are unsafe outputs")
    if resolved == repository or _is_within(repository, resolved):
        raise SyntheticFixtureError("Repository roots and their ancestors are unsafe outputs")
    for forbidden in FORBIDDEN_REPOSITORY_DIRECTORIES:
        forbidden_resolved = forbidden.resolve(strict=False)
        if resolved == forbidden_resolved or _is_within(resolved, forbidden_resolved):
            raise SyntheticFixtureError("Dataset and backend test directories are forbidden outputs")
    if _is_within(resolved, repository):
        runtime_allowed = allow_runtime_for_tests and _is_within(resolved, runtime)
        if not runtime_allowed:
            raise SyntheticFixtureError(
                "Generated fixtures must stay outside the repository source tree"
            )
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SyntheticFixtureError("Generator ownership metadata is unreadable") from exc
    if not isinstance(value, dict):
        raise SyntheticFixtureError("Generator ownership metadata is invalid")
    return value


def _safe_owned_path(value: str) -> PurePosixPath:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in value
        or ":" in value
        or not pure.parts
        or pure.parts[0] != "scenarios"
    ):
        raise SyntheticFixtureError("Generated file inventory contains an unsafe path")
    return pure


def _validate_owned_output(root: Path) -> tuple[Path, ...]:
    if _is_link_or_reparse(root) or not root.is_dir():
        raise SyntheticFixtureError("Existing output is not a regular generator-owned directory")
    marker_path = root / MARKER_FILENAME
    manifest_path = root / GENERATION_MANIFEST_FILENAME
    if not marker_path.is_file() or not manifest_path.is_file():
        raise SyntheticFixtureError(
            "Existing output has no valid synthetic-fixture ownership marker"
        )
    if _is_link_or_reparse(marker_path) or _is_link_or_reparse(manifest_path):
        raise SyntheticFixtureError("Generator ownership files must not be links")
    marker = _read_json(marker_path)
    manifest = _read_json(manifest_path)
    marker_keys = {
        "contract_version",
        "fixture_statement",
        "generator_id",
        "generator_version",
        "generation_manifest_sha256",
        "model_accuracy_evidence",
        "production_approved",
        "scenario_id",
        "seed",
        "synthetic",
        "training_approved",
    }
    manifest_keys = {
        "contract_version",
        "fixture_statement",
        "fixture_timestamp",
        "files",
        "generator_id",
        "generator_version",
        "generation_uuid",
        "integrity_scope",
        "model_accuracy_evidence",
        "output_tree_sha256",
        "production_approved",
        "scenario_count",
        "scenario_id",
        "scenario_ids",
        "seed",
        "synthetic",
        "training_approved",
    }
    if set(marker) != marker_keys or set(manifest) != manifest_keys:
        raise SyntheticFixtureError("Generator ownership metadata has an unknown shape")
    if (
        marker.get("contract_version") != MARKER_VERSION
        or marker.get("generator_id") != GENERATOR_ID
        or marker.get("generator_version") != GENERATOR_VERSION
        or marker.get("synthetic") is not True
        or marker.get("training_approved") is not False
        or marker.get("production_approved") is not False
        or marker.get("model_accuracy_evidence") is not False
    ):
        raise SyntheticFixtureError("Existing output marker is not owned by this generator version")
    manifest_bytes = manifest_path.read_bytes()
    if marker.get("generation_manifest_sha256") != sha256_bytes(manifest_bytes):
        raise SyntheticFixtureError("Generation manifest no longer matches its ownership marker")
    if (
        manifest.get("contract_version") != GENERATION_MANIFEST_VERSION
        or manifest.get("generator_id") != GENERATOR_ID
        or manifest.get("generator_version") != GENERATOR_VERSION
        or not isinstance(manifest.get("files"), list)
    ):
        raise SyntheticFixtureError("Existing generation manifest is incompatible")

    inventory_by_path: dict[str, dict[str, Any]] = {}
    for item in manifest["files"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"byte_size", "path", "sha256"}
            or not isinstance(item.get("path"), str)
        ):
            raise SyntheticFixtureError("Existing generation manifest inventory is invalid")
        pure = _safe_owned_path(item["path"])
        relative = pure.as_posix()
        if relative in inventory_by_path:
            raise SyntheticFixtureError("Existing generation manifest repeats a file path")
        inventory_by_path[relative] = item

    expected_files = set(inventory_by_path) | {
        MARKER_FILENAME,
        GENERATION_MANIFEST_FILENAME,
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _is_link_or_reparse(path):
            raise SyntheticFixtureError("Existing generated output contains a link")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise SyntheticFixtureError("Existing generated output contains an unknown entry")
    if actual_files != expected_files:
        raise SyntheticFixtureError(
            "Existing generated output has missing, extra, or unknown files"
        )
    expected_directories = {"scenarios"}
    for relative in inventory_by_path:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() not in {".", ""}:
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_directories != expected_directories:
        raise SyntheticFixtureError(
            "Existing generated output has missing, extra, or unknown directories"
        )
    for relative, item in inventory_by_path.items():
        content = (root / Path(*PurePosixPath(relative).parts)).read_bytes()
        if item.get("byte_size") != len(content) or item.get("sha256") != sha256_bytes(content):
            raise SyntheticFixtureError(
                "Existing generated file bytes no longer match the generation manifest"
            )
    ordered_inventory = [inventory_by_path[key] for key in sorted(inventory_by_path)]
    if manifest.get("output_tree_sha256") != output_tree_sha256(ordered_inventory):
        raise SyntheticFixtureError("Existing output-tree digest is invalid")
    return tuple(
        root / Path(*PurePosixPath(relative).parts)
        for relative in sorted(actual_files)
    )


def _remove_verified_output(root: Path, files: tuple[Path, ...]) -> None:
    for path in sorted(files, key=lambda item: len(item.parts), reverse=True):
        path.unlink()
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in directories:
        directory.rmdir()
    root.rmdir()


def _cleanup_staging(root: Path, created_files: set[Path]) -> None:
    if not root.exists() or _is_link_or_reparse(root):
        return
    for path in sorted(created_files, key=lambda item: len(item.parts), reverse=True):
        if path.is_file() and not _is_link_or_reparse(path):
            path.unlink()
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not _is_link_or_reparse(path):
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        root.rmdir()
    except OSError:
        pass


def _selected_scenarios(scenario_ids: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if scenario_ids is None:
        return SCENARIO_IDS
    requested = tuple(scenario_ids)
    if not requested:
        raise SyntheticFixtureError("At least one scenario must be selected")
    if len(requested) != len(set(requested)):
        raise SyntheticFixtureError("Scenario selections must be unique")
    unknown = sorted(set(requested).difference(SCENARIO_IDS))
    if unknown:
        raise SyntheticFixtureError(f"Unknown synthetic scenario: {unknown[0]}")
    requested_set = set(requested)
    return tuple(value for value in SCENARIO_IDS if value in requested_set)


def generate_fixtures(
    output_root: Path,
    *,
    seed: int = DEFAULT_SEED,
    scenario_ids: tuple[str, ...] | list[str] | None = None,
    overwrite_generated: bool = False,
    allow_runtime_for_tests: bool = False,
) -> GenerationResult:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise SyntheticFixtureError("Seed must be an integer from 0 through 2^63-1")
    selected = _selected_scenarios(scenario_ids)
    output = _validate_output_path(
        Path(output_root),
        allow_runtime_for_tests=allow_runtime_for_tests,
    )
    existing_files: tuple[Path, ...] | None = None
    if output.exists():
        if not overwrite_generated:
            raise SyntheticFixtureError(
                "Output already exists; explicit overwrite-generated permission is required"
            )
        existing_files = _validate_owned_output(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_components(output.parent)
    staging = output.with_name(f".{output.name}.synthetic-aoi-staging")
    if staging.exists() or staging.is_symlink():
        raise SyntheticFixtureError("Generator staging path already exists")
    staging.mkdir()
    created_files: set[Path] = set()
    try:
        for scenario_id in selected:
            created_files.update(
                write_scenario(staging, build_scenario(scenario_id, seed), seed)
            )
        scenario_files = [
            path.relative_to(staging)
            for path in (staging / "scenarios").rglob("*")
            if path.is_file()
        ]
        inventory = file_inventory(staging, scenario_files)
        tree_hash = output_tree_sha256(inventory)
        generation_manifest = {
            "contract_version": GENERATION_MANIFEST_VERSION,
            "fixture_statement": SYNTHETIC_STATEMENT,
            "fixture_timestamp": FIXTURE_TIMESTAMP,
            "files": inventory,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "generation_uuid": deterministic_uuid(seed, "generation_manifest"),
            "integrity_scope": "All files below scenarios/. Top-level ownership metadata is excluded to avoid self-referential hashes.",
            "model_accuracy_evidence": False,
            "output_tree_sha256": tree_hash,
            "production_approved": False,
            "scenario_count": len(selected),
            "scenario_id": "generation_manifest",
            "scenario_ids": list(selected),
            "seed": seed,
            "synthetic": True,
            "training_approved": False,
        }
        manifest_bytes = json_bytes(generation_manifest)
        manifest_path = staging / GENERATION_MANIFEST_FILENAME
        manifest_path.write_bytes(manifest_bytes)
        created_files.add(manifest_path)
        marker = {
            "contract_version": MARKER_VERSION,
            "fixture_statement": SYNTHETIC_STATEMENT,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "generation_manifest_sha256": sha256_bytes(manifest_bytes),
            "model_accuracy_evidence": False,
            "production_approved": False,
            "scenario_id": "generator_owned_output",
            "seed": seed,
            "synthetic": True,
            "training_approved": False,
        }
        marker_path = staging / MARKER_FILENAME
        marker_path.write_bytes(json_bytes(marker))
        created_files.add(marker_path)
        if existing_files is not None:
            _remove_verified_output(output, existing_files)
        staging.rename(output)
    except Exception:
        _cleanup_staging(staging, created_files)
        raise
    return GenerationResult(
        output_root=output,
        seed=seed,
        scenario_ids=selected,
        output_tree_sha256=tree_hash,
    )


def validate_generated_fixtures(
    output_root: Path,
    *,
    required_scenario_ids: tuple[str, ...] | list[str] | None = None,
    allow_runtime_for_tests: bool = False,
) -> GenerationResult:
    """Validate an existing generator-owned tree without modifying it."""
    output = _validate_output_path(
        Path(output_root),
        allow_runtime_for_tests=allow_runtime_for_tests,
    )
    _validate_owned_output(output)
    manifest = _read_json(output / GENERATION_MANIFEST_FILENAME)
    try:
        seed = manifest["seed"]
        scenario_ids = tuple(manifest["scenario_ids"])
        tree_hash = manifest["output_tree_sha256"]
    except (KeyError, TypeError) as exc:
        raise SyntheticFixtureError(
            "Existing generation manifest is incomplete"
        ) from exc
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**63 - 1
        or not scenario_ids
        or len(scenario_ids) != len(set(scenario_ids))
        or any(value not in SCENARIO_IDS for value in scenario_ids)
        or not isinstance(tree_hash, str)
    ):
        raise SyntheticFixtureError("Existing generation manifest is invalid")
    required = (
        ()
        if required_scenario_ids is None
        else _selected_scenarios(required_scenario_ids)
    )
    missing = tuple(value for value in required if value not in scenario_ids)
    if missing:
        raise SyntheticFixtureError(
            f"Existing generated output is missing required scenario: {missing[0]}"
        )
    return GenerationResult(
        output_root=output,
        seed=seed,
        scenario_ids=scenario_ids,
        output_tree_sha256=tree_hash,
    )
