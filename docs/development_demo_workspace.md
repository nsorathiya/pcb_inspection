# Development demo workspace

The persistent demo workspace gives the vision team a populated, explicitly
synthetic application before verified production data is available. It is
disabled by default and never represents model accuracy or production PCB
disposition.

## Enable locally

Choose a dedicated fixture directory outside the repository and start the
backend from the repository root:

```powershell
$env:PCB_AOI_ENABLE_DEMO_WORKSPACE = "true"
$env:PCB_AOI_SYNTHETIC_FIXTURE_ROOT = "C:\pcb-aoi-demo\synthetic-fixtures"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir .\backend --reload
```

The History page then shows **Load Demo Workspace**. Loading occurs only after
an explicit click. `PCB_AOI_ENABLE_SYNTHETIC_PROCESSING_API` is a separate gate
for manual per-inspection processing and is not required by the demo loader.

## API

- `GET /api/v1/development/demo-workspace` returns path-free availability and
  persisted state without writing data or reading fixture files.
- `POST /api/v1/development/demo-workspace/load` generates or validates the
  repository-owned fixture tree and loads missing demo evidence.

The loader creates recipe `synthetic-e2e` version `1.0` as `ACTIVE` and version
`0.9` as `DRAFT`. It creates one each of deterministic mock PASS, FAIL, and
UNCERTAIN; one controlled technical preprocessing ERROR; and one technical
validation failure. These are software workflow examples, not real predictions.

## Safety and retry behaviour

- Existing inspections are never deleted or updated by identity.
- Reserved UUIDs and database uniqueness prevent duplicate demo records.
- An application-level lock serializes simultaneous load clicks.
- Exact retries return the already loaded workspace without rerunning stages.
- Conflicting data under a reserved identity fails with HTTP 409.
- Existing generator-owned fixtures are integrity-validated without rewriting
  them. Unknown, modified, missing, linked, or unsafe fixture trees fail closed.
- All RGB and height bytes enter through the existing immutable paired-intake
  service. Validation and processing use the existing guarded lifecycle
  services; decisions are never inserted directly.
- API responses contain no runtime or fixture paths.
- No database schema change is required.

The concurrency guarantee applies to requests handled by one application
process, which is the supported local development deployment. Reserved IDs and
database constraints remain a second duplicate barrier if multiple processes
are accidentally used.

## Focused verification

```powershell
.\.venv\Scripts\python.exe -m pytest .\backend\tests\test_demo_workspace_api.py -q

Set-Location .\frontend
npm run test:run -- src/pages/HistoryPage.test.tsx
```
