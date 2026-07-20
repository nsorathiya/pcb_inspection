# Development-Only Synthetic Inspection Processing API

## Safety boundary

These routes exist only to exercise the trusted synthetic processing workflow
during development. They do not run a real AI model, analyze PCB defects, or
produce production PCB dispositions. `PASS`, `FAIL`, and `UNCERTAIN` are
deterministic mock workflow decisions. Confidence does not exist in this API.

POST delegates exclusively to the existing `InspectionProcessingOrchestrator`.
The client selects policy identifiers and versions but cannot supply a fixture
root, scenario, path, buffer, implementation, engine, decision, or defect type.
GET reads persisted database evidence only.

The separate read-only `GET /api/v1/inspections` history route includes a
compact latest-processing summary and explicit synthetic/mock/nonproduction
flags. It omits findings and never executes preprocessing or inference; use the
inspection-specific processing GET for detailed persisted evidence. See
`docs/inspection_history_api.md`.

## Configuration

POST is disabled by default. Set both application-owned variables before
starting the backend with a reviewed, generator-owned fixture tree:

```powershell
$env:PCB_AOI_ENABLE_SYNTHETIC_PROCESSING_API = "true"
$env:PCB_AOI_SYNTHETIC_FIXTURE_ROOT = "C:\pcb-aoi-trusted-fixtures"
python -m uvicorn app.main:app --app-dir .\backend --reload
```

`PCB_AOI_SYNTHETIC_FIXTURE_ROOT` is never accepted from HTTP. Relative values
are resolved from the repository root. The application neither creates a
fixture tree nor reads its manifest during import or startup. If the feature is
disabled, or enabled without a configured root, POST returns HTTP 503. GET does
not require synthetic execution to remain enabled because it only restores
already-persisted evidence.

## Routes and explicit policy selection

- `POST /api/v1/inspections/{inspection_id}/process` executes or exactly
  replays trusted synthetic processing.
- `GET /api/v1/inspections/{inspection_id}/processing` returns the deterministic
  latest persisted processing lifecycle.

POST accepts exactly four required string fields:

```json
{
  "preprocessing_policy_id": "synthetic-paired-rgb-height",
  "preprocessing_policy_version": "1.0",
  "inference_policy_id": "synthetic-deterministic-mock-inference",
  "inference_policy_version": "1.0"
}
```

Unknown fields and missing fields are rejected. Policy identifiers and versions
must match the existing reviewed contracts exactly; labels, paths, or execution
settings cannot be overridden by the request.

## POST lifecycle and exact retries

A new inspection must already be `READY` with persisted passed validation. The
orchestrator verifies the trusted synthetic provenance and current immutable
artifact bytes before it begins the guarded lifecycle. It then runs the existing
synthetic preprocessing and deterministic mock inference services and completes
the processing run as a mock final status or a persisted technical `ERROR`.
Both are completed results and return HTTP 200.

An exact retry for the same inspection, validation, policies, fixture identity,
and service identities returns the same persisted processing, preprocessing,
and inference identities without rereading files or rerunning either service.
The response reports `lifecycle_idempotent_existing=true` and
`execution_started_now=false`. Reprocessing with different selections or after
a final result is unsupported.

The response exposes typed identities, lifecycle statuses, mock decision,
optional authoritative mock defect type, safe summaries, and ordered findings.
It exposes no filesystem path, buffer, digest-bucket internals, confidence, or
production claim. `production_approved` is always false for this workflow.

## GET persisted retrieval

GET validates the inspection ID, loads the existing inspection and repository's
deterministic latest processing run, and restores persisted preprocessing and
inference evidence through the same consistency-checking mapper used by POST
completion and replay. It does not execute preprocessing or inference, read an
artifact or manifest file, write database state, or append an audit event.

GET returns 404 separately for an unknown inspection and for an inspection with
no processing result. Its body `request_id` and response `X-Request-ID` identify
the current GET, not the request ID persisted in the earlier POST audit.

## Error contract

Errors use the common safe body with top-level `code`, `message`, and
`request_id` fields; internal exception text, paths, SQL, and buffers are never
returned.

| HTTP | Meaning |
| --- | --- |
| 400 | Malformed inspection UUID or invalid policy identifier/version text |
| 404 | Inspection, policy/version, or persisted processing result not found |
| 409 | Lifecycle/provenance/integrity conflict, processing in progress, unsupported reprocessing, or optional evidence |
| 422 | Missing/invalid JSON fields or extra request fields |
| 500 | Persisted evidence is inconsistent, recovery is required, or completion cannot be trusted |
| 503 | Synthetic POST execution is disabled or lacks a configured fixture root |

Optional mask or calibration evidence is currently rejected before lifecycle
begin with HTTP 409 and code
`OPTIONAL_EVIDENCE_PROCESSING_UNSUPPORTED`. Schema version 3 cannot persist the
required authoritative optional-evidence summaries, so the API does not ignore
them or fabricate metadata.

## Request IDs

Every response includes `X-Request-ID`. A supplied header is preserved; when it
is absent middleware generates a UUID. POST passes that same value into the
trusted orchestrator for lifecycle audit. GET returns its current request ID but
does not mutate audit history.

## PowerShell sequence

The intake command below uses PowerShell 7 multipart support. Paths must refer
to a trusted generated pair whose fixture manifest is beneath the configured
fixture root.

```powershell
$base = "http://127.0.0.1:8000/api/v1"
$intake = Invoke-RestMethod -Method Post -Uri "$base/inspections" -Form @{
  board_id = "synthetic-board"
  recipe_id = "development-recipe"
  recipe_version = "1.0"
  rgb_image = Get-Item "C:\pcb-aoi-trusted-fixtures\case\rgb.png"
  height_map = Get-Item "C:\pcb-aoi-trusted-fixtures\case\height.tiff"
}

$validation = Invoke-RestMethod -Method Post `
  -Uri "$base/inspections/$($intake.inspection_id)/validate" `
  -ContentType "application/json" `
  -Body (@{ policy_id = "development-native-rgb-height"; policy_version = "1.0" } | ConvertTo-Json)

$selection = @{
  preprocessing_policy_id = "synthetic-paired-rgb-height"
  preprocessing_policy_version = "1.0"
  inference_policy_id = "synthetic-deterministic-mock-inference"
  inference_policy_version = "1.0"
} | ConvertTo-Json

$processed = Invoke-RestMethod -Method Post `
  -Uri "$base/inspections/$($intake.inspection_id)/process" `
  -ContentType "application/json" -Headers @{ "X-Request-ID" = "dev-process-001" } `
  -Body $selection

$persisted = Invoke-RestMethod -Method Get `
  -Uri "$base/inspections/$($intake.inspection_id)/processing"
```

## Tests

```powershell
python -m pytest .\backend\tests\test_inspection_processing_api.py -q
python -m pytest .\backend\tests\test_synthetic_processing_orchestrator.py -q
python -m pytest .\backend\tests -q
```
