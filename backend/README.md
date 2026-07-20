# Backend Foundation, Paired Intake, and Technical Validation

The application provides a model-independent health endpoint, paired RGB plus
height/depth intake with immutable raw storage, and explicit-policy technical
validation/result APIs. Technical validation reads native metadata but does not
preprocess data, run inference, classify a PCB, or import the existing PyTorch
prototype.

Run all commands below in Windows PowerShell from the repository root.

## Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install foundation and test dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\backend\requirements-dev.txt
```

The existing 2D prototype has additional ML dependencies. Install them only when working with the legacy prototype:

```powershell
python -m pip install -r .\backend\requirements.txt
```

## Start the foundation backend

```powershell
python -m uvicorn app.main:app --app-dir .\backend --reload
```

Verify the endpoint from another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

The multipart intake endpoint is `POST /api/v1/inspections`. See
`docs/inspection_intake_api.md` for fields, a PowerShell example, response and
error contracts, and the precise meaning of `RECEIVED`.

The read-only details endpoint is
`GET /api/v1/inspections/{inspection_id}`. It returns persisted lifecycle and
artifact integrity metadata without storage paths, artifact bytes, semantic
validation, or classification. See `docs/inspection_details_api.md` for its
response, error, ordering, and request-ID contracts.

The paired RGB/height semantic-validation domain has versioned result, finding,
and policy contracts plus a reusable read-only engine. The backend exposes
`POST /api/v1/inspections/{inspection_id}/validate` for explicit-policy
technical execution and `GET /api/v1/inspections/{inspection_id}/validation`
for the latest persisted result. Neither endpoint preprocesses data, runs AI
inference, or classifies the PCB. Exact POST retries are system-idempotent and
revalidation is unsupported. See `docs/inspection_validation_api.md` for the
request, response, error, request-ID, status, and concurrency contracts. See
`docs/inspection_semantic_validation_contract.md` for the contract and
`docs/inspection_semantic_validation_engine.md` for policy loading, artifact
reads, integrity and native-format checks, deterministic execution, and current
registration-evidence limitations. See
`docs/inspection_validation_persistence.md` for validation migration,
canonical hashes, validation keys, transactions, and idempotent replay.
See `docs/inspection_validation_lifecycle.md` for the guarded `RECEIVED`
transitions, lifecycle audit, standalone-result adoption, concurrency, and
rollback behavior. `READY` means technical readiness, not PCB PASS.

The next technical boundary is defined by versioned preprocessing policy,
result, and finding contracts plus replaceable interfaces. A development-only
executor implements deterministic in-memory RGB and height preprocessing for
the generated fixture subset. It keeps branches separate, emits little-endian
float32 CHW buffers, uses synthetic identity registration, and exposes no
buffer bytes or paths in its result. It is not wired into FastAPI and does not
persist results, change status, run inference, or support real input. See
`docs/inspection_preprocessing_contract.md` and
`docs/synthetic_preprocessing_executor.md`.

The following replaceable boundary is a versioned, deterministic mock
inference engine for those separate in-memory buffers. It validates the actual
buffer bytes and descriptors, then uses a documented SHA-256 bucket solely to
select mock `PASS`, `FAIL`, or `UNCERTAIN` workflow values. Mock `FAIL` labels
come from the authoritative defect taxonomy, and confidence is always null.
This engine performs no image-content analysis, reads no source artifacts, is
not wired into FastAPI, and makes no production or model-accuracy claim. See
`docs/mock_inference_engine.md`.

Schema version 3 adds a separate processing persistence and guarded lifecycle
foundation. It atomically adopts already-completed typed preprocessing and
mock-inference results, preserving ordered findings and canonical hashes while
moving READY to PROCESSING and PROCESSING to PASS, FAIL, UNCERTAIN, or technical
ERROR. It does not execute either service and is not exposed by an HTTP route.
Stored mock outcomes are development workflow values, not production PCB
decisions. See `docs/inspection_processing_lifecycle.md`.

## Run tests

Run the focused logging, startup, health, and request-ID tests:

```powershell
python -m pytest .\backend\tests\test_health.py -q
```

Run the focused runtime-storage tests:

```powershell
python -m pytest .\backend\tests\test_runtime_paths.py -q
```

Run the focused SQLite database, repository, and audit-foundation tests:

```powershell
python -m pytest .\backend\tests\test_database.py -q
```

Run the focused immutable artifact-storage tests:

```powershell
python -m pytest .\backend\tests\test_artifact_storage.py -q
```

Run the focused paired inspection-intake API tests:

```powershell
python -m pytest .\backend\tests\test_inspection_intake.py -q
```

Run the focused read-only inspection-details API tests:

```powershell
python -m pytest .\backend\tests\test_inspection_details.py -q
```

Run the focused inspection semantic-validation contract tests:

```powershell
python -m pytest .\backend\tests\test_inspection_validation_contract.py -q
```

Run the focused read-only semantic-validation engine tests:

```powershell
python -m pytest .\backend\tests\test_inspection_semantic_validation_engine.py -q
```

Run the focused validation-result persistence and schema-migration tests:

```powershell
python -m pytest .\backend\tests\test_inspection_validation_persistence.py -q
```

Run the focused atomic validation-lifecycle tests:

```powershell
python -m pytest .\backend\tests\test_inspection_validation_lifecycle.py -q
```

Run the focused validation execution/result API tests:

```powershell
python -m pytest .\backend\tests\test_inspection_validation_api.py -q
```

Run the focused preprocessing contract and interface tests:

```powershell
python -m pytest .\backend\tests\test_inspection_preprocessing_contract.py -q
```

Run the focused deterministic synthetic preprocessing executor tests:

```powershell
python -m pytest .\backend\tests\test_synthetic_preprocessing_executor.py -q
```

Run the focused mock inference contract and deterministic engine tests:

```powershell
python -m pytest `
  .\backend\tests\test_inspection_inference_contract.py `
  .\backend\tests\test_deterministic_mock_inference.py -q
```

Run the focused schema-v3 processing persistence, lifecycle, rollback, and
concurrency tests:

```powershell
python -m pytest .\backend\tests\test_inspection_processing_lifecycle.py -q
```

Run the complete backend foundation test suite:

```powershell
python -m pytest .\backend\tests
```

## Configuration

Copy `.env.example` to `.env` to change local settings, or set variables in the current PowerShell session:

```powershell
$env:PCB_AOI_APPLICATION_NAME = "pcb-aoi-api"
$env:PCB_AOI_APPLICATION_VERSION = "0.1.0"
$env:PCB_AOI_ENVIRONMENT = "development"
$env:PCB_AOI_API_PREFIX = "/api/v1"
$env:PCB_AOI_DEBUG = "false"
$env:PCB_AOI_LOG_LEVEL = "INFO"
$env:PCB_AOI_LOG_FORMAT = "plain"
$env:PCB_AOI_RUNTIME_ROOT = "runtime"
$env:PCB_AOI_DATABASE_FILENAME = "pcb_aoi.sqlite3"
$env:PCB_AOI_SQLITE_BUSY_TIMEOUT_MS = "5000"
$env:PCB_AOI_DATABASE_ECHO = "false"
$env:PCB_AOI_MAX_RGB_BYTES = "52428800"
$env:PCB_AOI_MAX_HEIGHT_BYTES = "268435456"
$env:PCB_AOI_MAX_MASK_BYTES = "67108864"
$env:PCB_AOI_MAX_CALIBRATION_BYTES = "5242880"
$env:PCB_AOI_MAX_GENERATED_ARTIFACT_BYTES = "52428800"
```

The supported variables are:

| Variable | Default |
| --- | --- |
| `PCB_AOI_APPLICATION_NAME` | `pcb-aoi-api` |
| `PCB_AOI_APPLICATION_VERSION` | `0.1.0` |
| `PCB_AOI_ENVIRONMENT` | `development` |
| `PCB_AOI_API_PREFIX` | `/api/v1` |
| `PCB_AOI_DEBUG` | `false` |
| `PCB_AOI_LOG_LEVEL` | `INFO` |
| `PCB_AOI_LOG_FORMAT` | `plain` |
| `PCB_AOI_RUNTIME_ROOT` | Repository-local `runtime` directory |
| `PCB_AOI_DATABASE_FILENAME` | `pcb_aoi.sqlite3` |
| `PCB_AOI_SQLITE_BUSY_TIMEOUT_MS` | `5000` |
| `PCB_AOI_DATABASE_ECHO` | `false` |
| `PCB_AOI_MAX_RGB_BYTES` | `52428800` (50 MiB) |
| `PCB_AOI_MAX_HEIGHT_BYTES` | `268435456` (256 MiB) |
| `PCB_AOI_MAX_MASK_BYTES` | `67108864` (64 MiB) |
| `PCB_AOI_MAX_CALIBRATION_BYTES` | `5242880` (5 MiB) |
| `PCB_AOI_MAX_GENERATED_ARTIFACT_BYTES` | `52428800` (50 MiB) |

`PCB_AOI_LOG_LEVEL` accepts `DEBUG`, `INFO`, `WARNING`, `ERROR`, or
`CRITICAL`. The current `plain` format is readable development output with
timestamps, severity, logger name, service name, and request ID. Format
selection is centralized so a JSON formatter can be introduced later without
changing application call sites; JSON output is not implemented in this phase.

## Runtime storage

`PCB_AOI_RUNTIME_ROOT` defines the local root for future runtime-generated
files. The default is the repository's `runtime` directory, resolved from the
installed backend source location rather than the process working directory.
Relative overrides are also resolved from the repository root, so changing the
directory from which Uvicorn is launched does not change the storage location.

The application creates this directory tree idempotently during startup:

| Directory | Intended purpose |
| --- | --- |
| `runtime/raw_uploads` | Immutable RGB, height, mask, and calibration bytes |
| `runtime/previews` | Immutable generated preview artifacts |
| `runtime/results` | Immutable generated result overlays |
| `runtime/reports` | Immutable generated report artifacts |
| `runtime/tmp` | Temporary runtime files |
| `runtime/database` | SQLite metadata database and WAL/SHM files |

To select a different location for the current PowerShell session:

```powershell
$env:PCB_AOI_RUNTIME_ROOT = "C:\pcb-aoi-runtime"
python -m uvicorn app.main:app --app-dir .\backend --reload
```

If any required directory cannot be created, application startup logs an error
and fails. The public health response does not include the local runtime path.

The SQLite file is `runtime/database/pcb_aoi.sqlite3` by default. Startup
enables foreign keys, WAL mode, and the configured busy timeout, applies the
ordered schema migrations to version 3 idempotently, and proves the database can
answer a query before the health endpoint is available. WAL improves
reader/writer coexistence but SQLite
still permits only one writer at a time.

Database rows store metadata and safe relative artifact references; they do not
replace immutable artifact bytes. The storage service calculates SHA-256 and
size while streaming, enforces per-category limits, and atomically finalizes
internally generated filenames without overwriting different content. It does
not inspect image semantics. The paired intake endpoint uses this service but
does not decode or validate image pixels. See `docs/artifact_storage.md` for
the layout, extension policy, rollback behavior, and Windows filesystem
limitations.

During live backup, the main database,
`-wal`, and `-shm` files must be handled consistently. See
`docs/database_foundation.md` for table/status meanings and migration scope.

## Request IDs

Every HTTP response includes an `X-Request-ID` header. If a caller supplies
`X-Request-ID`, the application preserves it unchanged. Otherwise, the
application generates a UUID request ID. During request handling the same value
is available as `request.state.request_id` and is added to application log
records through request-local context.

For an inspection-details GET, this header identifies the current read
request. The body field `intake_request_id` is the separately persisted ID from
the original upload and is not overwritten by the current request ID.
