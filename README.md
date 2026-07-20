# PCB AOI Development MVP

This repository contains a file-based PCB automated optical inspection (AOI)
development workflow. The current supported application pairs an RGB image with
a native height/depth file, persists inspection evidence, performs technical
validation, and can run deterministic synthetic processing for trusted fixtures.

> **Development warning:** synthetic `MOCK PASS`, `MOCK FAIL`, and
> `MOCK UNCERTAIN` results are workflow-development values. They are not real AI
> predictions and are not approved for production PCB disposition.

The original single-image 2D prototype remains in `backend/main.py`,
`backend/api.py`, and the unused legacy JavaScript frontend sources. It is not
part of the supported operator workflow and has not been modified.

## Supported application

- Backend: FastAPI application under `backend/app`
- Operator UI: React, Vite, and strict TypeScript under `frontend`
- Runtime: repository-local by default and excluded from source control
- Database: SQLite schema version 3
- Workflow: paired intake, technical validation, and optional synthetic
  preprocessing/mock inference

The operator UI provides:

- inspection history with backend cursor pagination and exact filters;
- explicit recipe/version selection from the read-only catalogue;
- paired RGB and height/depth file intake;
- lifecycle-gated validation and synthetic processing actions;
- persisted validation findings and synthetic processing evidence; and
- structured error messages with request IDs.

See [docs/operator_frontend.md](docs/operator_frontend.md) for the frontend
contract and [backend/README.md](backend/README.md) for backend setup and API
details.

## Local development

From the repository root, prepare and start the backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\backend\requirements-dev.txt
python -m uvicorn app.main:app --app-dir .\backend --reload
```

In another PowerShell window, install the locked frontend dependencies and run
the Vite development server:

```powershell
Set-Location .\frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The development server proxies same-origin
`/api` requests to `http://127.0.0.1:8000`. For a separately hosted backend,
set `VITE_API_BASE_URL` in `frontend/.env.local`; do not place secrets there.

Synthetic processing is disabled by default. Enabling it is only appropriate
for trusted, generator-owned development fixtures; see
[docs/synthetic_processing_orchestrator.md](docs/synthetic_processing_orchestrator.md).

## Verification

```powershell
Set-Location .\frontend
npm run lint
npm run typecheck
npm run test:run
npm run build

Set-Location ..
python -m pytest .\backend\tests
```

The UI does not add authentication, recipe mutation, model management,
reprocessing, continuous polling, reporting, image previews, real inference, or
production approval.
