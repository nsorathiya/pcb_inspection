# SQLite Inspection Database and Audit Foundation

Date: 2026-07-17

## Scope

The foundation stores inspection metadata, artifact references, recipe/model
registrations, and append-only audit events. `POST /api/v1/inspections` now uses
this layer for paired intake, but no GET inspection endpoint, image processing,
inference, or model registration side effect exists. In particular, the legacy
`best_model.pth` is not registered.

Database metadata does not replace immutable raw file storage. Artifact rows
contain relative references and integrity metadata only. The separately
testable storage and registration coordinator is documented in
`docs/artifact_storage.md`.

## Location and configuration

The database is always below the centralized runtime root:

```text
runtime/
  database/
    pcb_aoi.sqlite3
```

Supported settings:

| Variable | Default | Rule |
| --- | --- | --- |
| `PCB_AOI_RUNTIME_ROOT` | Repository-local `runtime` | Central runtime root. |
| `PCB_AOI_DATABASE_FILENAME` | `pcb_aoi.sqlite3` | Filename only; absolute and escaping paths are rejected. |
| `PCB_AOI_SQLITE_BUSY_TIMEOUT_MS` | `5000` | Integer from 1 through 60000. |
| `PCB_AOI_DATABASE_ECHO` | `false` | Enables SQLAlchemy SQL logging for local diagnostics. |

The database directory and schema are created idempotently during FastAPI
lifespan startup. Importing the application or database modules does not create
directories or a database file.

## Connection behavior

Every new SQLite connection executes:

```sql
PRAGMA foreign_keys=ON;
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=<configured milliseconds>;
```

Startup creates/checks the schema and performs `SELECT 1` before the application
logs successful startup or can answer health requests. Initialization errors
are logged at ERROR level and fail startup without exposing paths, SQL URLs,
table names, or exception details through the public health response.

WAL improves reader/writer coexistence; it does not provide multiple concurrent
writers. SQLite still serializes writes. Repository methods keep transactions
small, and callers must handle busy/constraint failures explicitly.

## Tables

### `inspections`

Stores the inspection lifecycle and traceability context. Allowed statuses:

- `RECEIVED`: metadata record accepted for validation.
- `VALIDATION_FAILED`: input validation failed safely.
- `READY`: validated and ready for later processing.
- `PROCESSING`: future processing is in progress.
- `PASS`: completed acceptable decision.
- `FAIL`: completed unacceptable decision.
- `UNCERTAIN`: completed decision requiring later review/workflow.
- `ERROR`: processing failed.

`REVIEW` is intentionally not a persisted final status. PASS, FAIL, and
UNCERTAIN require `completed_at`. Confidence is constrained to 0-1 and
processing time is non-negative.

### `inspection_artifacts`

Stores only safe relative paths, SHA-256, size, type, media type, and timestamp.
Allowed types are raw RGB/height, validity mask, calibration, RGB/height
preview, result overlay, and report. Foreign keys require a real inspection.
Absolute paths and `..` traversal are blocked.

### `recipes`

Stores versioned JSON recipe configuration. Statuses are `DRAFT`, `ACTIVE`, and
`RETIRED`. `(recipe_id, recipe_version)` is unique.

### `model_versions`

Stores model registration metadata only. Compatibility is `VERIFIED`,
`UNVERIFIED`, or `INCOMPATIBLE`; lifecycle status is `REGISTERED`, `ACTIVE`,
`RETIRED`, or `BLOCKED`. `(model_id, model_version)` is unique. Optional model
artifact references remain relative and hash constrained.

### `audit_events`

Stores entity/action, optional actor/request IDs, deterministic JSON details,
and timestamp. The repository exposes append and read operations only. Update
or delete methods and APIs are intentionally absent; audit events are
append-only by repository convention.

### `schema_version`

Contains one row identifying schema version `1`. Alembic is intentionally
deferred while there is one local initial schema and no deployed upgrade path.
Startup rejects an unknown version instead of silently changing it.

Before the first incompatible schema change, add Alembic (or an equivalently
reviewed ordered migration mechanism), create a migration from version 1,
exercise upgrades against a database copy, and update the version only after a
successful transaction. `create_all` is initialization, not an upgrade engine.

## Backup and recovery note

A live WAL database may have three coordinated files:

```text
pcb_aoi.sqlite3
pcb_aoi.sqlite3-wal
pcb_aoi.sqlite3-shm
```

Do not copy only the main file while the application is writing. Use SQLite's
online backup API or stop the application cleanly and copy the database, WAL,
and SHM consistently. Backup/restore automation remains a later stabilization
task.

## Tests

From the repository root in Windows PowerShell:

```powershell
python -m pytest .\backend\tests\test_database.py -q
python -m pytest .\backend\tests\test_health.py .\backend\tests\test_runtime_paths.py -q
python -m pytest .\backend\tests -q
```

Tests use temporary runtime roots and never use the default live database.
