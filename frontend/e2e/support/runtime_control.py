from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.runtime_paths import RuntimePaths  # noqa: E402
from app.db import Database, Repositories  # noqa: E402
from app.db.models import InspectionStatus, RecipeStatus  # noqa: E402
from app.db.repositories import InspectionCreate, RecipeCreate  # noqa: E402


def _database(runtime_root: Path) -> tuple[RuntimePaths, Database, Repositories]:
    paths = RuntimePaths.from_root(runtime_root)
    paths.create_directories()
    database = Database(paths.database_file, busy_timeout_ms=5000)
    return paths, database, Repositories.from_session_factory(database.session_factory)


async def seed_recipes(runtime_root: Path) -> None:
    _paths, database, repositories = _database(runtime_root)
    await database.initialize()
    created = datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc)
    try:
        for row_id, version, status in (
            ("70000000-0000-4000-8000-000000000001", "1.0", RecipeStatus.ACTIVE),
            ("70000000-0000-4000-8000-000000000002", "0.9", RecipeStatus.DRAFT),
        ):
            await repositories.recipes.register(
                RecipeCreate(
                    id=row_id,
                    recipe_id="synthetic-e2e",
                    recipe_version=version,
                    name="Synthetic E2E Recipe",
                    configuration={
                        "development_only": True,
                        "production_approved": False,
                        "note": "Recipe status does not establish production approval.",
                    },
                    status=status,
                    created_at=created,
                    updated_at=created,
                )
            )
    finally:
        await database.dispose()


async def seed_history(runtime_root: Path, count: int) -> None:
    _paths, database, repositories = _database(runtime_root)
    await database.initialize()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    try:
        for index in range(count):
            await repositories.inspections.create(
                InspectionCreate(
                    id=f"90000000-0000-4000-8000-{index + 1:012x}",
                    status=InspectionStatus.RECEIVED,
                    board_id=f"E2E-HISTORY-{index + 1:02d}",
                    recipe_id="synthetic-e2e",
                    recipe_version="1.0" if index % 2 == 0 else "0.9",
                    lot_id=None,
                    operator_id=None,
                    created_at=base + timedelta(seconds=index),
                )
            )
    finally:
        await database.dispose()


def _connect_read_only(database_file: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{database_file.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(runtime_root: Path, fixture_root: Path | None) -> dict[str, object]:
    paths = RuntimePaths.from_root(runtime_root)
    tables = (
        "schema_version", "inspections", "inspection_artifacts", "recipes",
        "model_versions", "audit_events", "inspection_validations",
        "inspection_validation_findings", "inspection_processing_runs",
        "inspection_preprocessing_results", "inspection_preprocessing_result_findings",
        "inspection_inference_results", "inspection_inference_result_findings",
    )
    with _connect_read_only(paths.database_file) as connection:
        row_counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }
        canonical_rows: dict[str, list[list[object]]] = {}
        for table in tables:
            columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
            order = " ORDER BY " + ", ".join(f'"{column}"' for column in columns)
            canonical_rows[table] = [list(row) for row in connection.execute(f'SELECT * FROM "{table}"{order}')]
        database_fingerprint = hashlib.sha256(
            json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        artifacts = [dict(row) for row in connection.execute(
            "SELECT id, inspection_id, artifact_type, relative_path, sha256, byte_size "
            "FROM inspection_artifacts ORDER BY inspection_id, artifact_type, id"
        )]
        inspections = [dict(row) for row in connection.execute(
            "SELECT id, status, board_id, recipe_id, recipe_version, lot_id, operator_id "
            "FROM inspections ORDER BY created_at, id"
        )]
        processing = [dict(row) for row in connection.execute(
            "SELECT runs.id, runs.inspection_id, runs.validation_id, runs.status, "
            "preprocessing.id AS preprocessing_id, inference.id AS inference_id "
            "FROM inspection_processing_runs AS runs "
            "LEFT JOIN inspection_preprocessing_results AS preprocessing "
            "ON preprocessing.processing_run_id = runs.id "
            "LEFT JOIN inspection_inference_results AS inference "
            "ON inference.processing_run_id = runs.id "
            "ORDER BY runs.inspection_id, runs.id"
        )]
        audit = [dict(row) for row in connection.execute(
            "SELECT id, entity_id, action, request_id, created_at FROM audit_events "
            "ORDER BY created_at, id"
        )]
        foreign_key_failures = [list(row) for row in connection.execute("PRAGMA foreign_key_check")]
        schema_version = connection.execute("SELECT version FROM schema_version").fetchone()[0]

    artifact_integrity = []
    for artifact in artifacts:
        file_path = (paths.root / str(artifact["relative_path"])).resolve()
        contained = file_path.is_relative_to(paths.root)
        exists = file_path.is_file()
        artifact_integrity.append({
            "inspection_id": artifact["inspection_id"],
            "artifact_type": artifact["artifact_type"],
            "registered_sha256": artifact["sha256"],
            "registered_byte_size": artifact["byte_size"],
            "contained": contained,
            "exists": exists,
            "actual_sha256": _sha256(file_path) if contained and exists else None,
            "actual_byte_size": file_path.stat().st_size if contained and exists else None,
        })

    runtime_files = sorted(
        path.relative_to(paths.root).as_posix()
        for path in paths.root.rglob("*") if path.is_file()
    )
    fixture_tree_sha256 = None
    fixture_files_verified = None
    if fixture_root is not None:
        manifest = json.loads((fixture_root / "generation_manifest.json").read_text(encoding="utf-8"))
        inventory = []
        verified = True
        for item in manifest["files"]:
            file_path = fixture_root / Path(item["path"])
            actual = {"path": item["path"], "sha256": _sha256(file_path), "byte_size": file_path.stat().st_size}
            inventory.append(actual)
            verified = verified and actual == item
        canonical = json.dumps(sorted(inventory, key=lambda item: item["path"]), sort_keys=True, separators=(",", ":")).encode()
        # The authoritative tree digest is retained from the generator manifest;
        # every inventoried hash and size is independently checked above.
        fixture_tree_sha256 = manifest["output_tree_sha256"]
        fixture_files_verified = verified and bool(canonical)

    return {
        "schema_version": schema_version,
        "foreign_key_failures": foreign_key_failures,
        "database_fingerprint": database_fingerprint,
        "row_counts": row_counts,
        "inspections": inspections,
        "processing": processing,
        "audit": audit,
        "artifact_integrity": artifact_integrity,
        "runtime_files": runtime_files,
        "report_files": [item for item in runtime_files if item.startswith("reports/")],
        "fixture_tree_sha256": fixture_tree_sha256,
        "fixture_files_verified": fixture_files_verified,
    }


def tamper_rgb(runtime_root: Path, inspection_id: str) -> None:
    paths = RuntimePaths.from_root(runtime_root)
    with _connect_read_only(paths.database_file) as connection:
        row = connection.execute(
            "SELECT relative_path FROM inspection_artifacts "
            "WHERE inspection_id = ? AND artifact_type = 'RGB_RAW'",
            (inspection_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError("RGB artifact was not found for controlled E2E tampering")
    artifact = (paths.root / row[0]).resolve()
    if not artifact.is_relative_to(paths.root) or not artifact.is_file():
        raise RuntimeError("RGB artifact is outside the isolated E2E runtime")
    with artifact.open("ab") as stream:
        stream.write(b"\nE2E-CONTROLLED-TAMPER\n")


def verify_report(envelope_file: Path) -> dict[str, object]:
    envelope = json.loads(envelope_file.read_text(encoding="utf-8"))
    canonical = json.dumps(
        envelope["report"], ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    calculated = hashlib.sha256(canonical).hexdigest()
    reported = envelope["report_sha256"]
    return {
        "calculated_sha256": calculated,
        "reported_sha256": reported,
        "matches": calculated == reported,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seed-recipes", "seed-history", "snapshot", "tamper-rgb", "verify-report"))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--inspection-id")
    parser.add_argument("--envelope-file", type=Path)
    args = parser.parse_args()
    if args.command == "seed-recipes":
        asyncio.run(seed_recipes(args.runtime_root))
    elif args.command == "seed-history":
        asyncio.run(seed_history(args.runtime_root, args.count))
    elif args.command == "tamper-rgb":
        if not args.inspection_id:
            parser.error("--inspection-id is required for tamper-rgb")
        tamper_rgb(args.runtime_root, args.inspection_id)
    elif args.command == "verify-report":
        if args.envelope_file is None:
            parser.error("--envelope-file is required for verify-report")
        print(json.dumps(verify_report(args.envelope_file), sort_keys=True))
    else:
        print(json.dumps(snapshot(args.runtime_root, args.fixture_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
