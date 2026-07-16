# Proposed Phase 1 Changes

Date: 2026-07-16

Status: proposal only; not approved or implemented.

## Objective

Create a deterministic, testable project foundation without changing the current model, dataset, training code, or `POST /predict` behavior. Phase 1 should make backend startup and health verification independent of the saved model and establish safe locations and configuration for later phases.

## Scope

In scope:

- Root repository hygiene and environment documentation.
- A Python package entry point for the backend.
- Typed settings for paths, environment, CORS, and logging.
- Structured application logging.
- `GET /api/v1/health` with a stable JSON contract.
- Baseline backend API tests.
- Frontend environment variable for the API base URL.
- Verification of frontend startup/build after an approved dependency install.
- Empty untracked runtime directory convention.

Out of scope:

- 2D/3D upload or validation.
- Database tables or inspection persistence.
- Mock or real inference changes.
- Model training, evaluation, conversion, or activation.
- Dataset restructuring.
- Removal of already tracked datasets, uploads, outputs, activations, or model files.
- UI redesign.
- Camera, PLC, conveyor, MES/ERP, or authentication.

## Proposed file changes

Exact names should be confirmed during the Phase 1 implementation-plan review.

### Create

| File | Purpose |
| --- | --- |
| `.gitignore` | Ignore Python/Node caches, local environments, secrets, databases, logs, runtime storage, generated previews, archives, and future model artifacts. Existing tracked files remain tracked until separately approved. |
| `.env.example` | Document non-secret backend/frontend local settings. |
| `backend/app/__init__.py` | Define the backend package. |
| `backend/app/main.py` | Application factory and canonical FastAPI entry point. |
| `backend/app/api/__init__.py` | API package marker. |
| `backend/app/api/health.py` | Versioned health router. |
| `backend/app/core/__init__.py` | Core package marker. |
| `backend/app/core/config.py` | Typed settings and path resolution. |
| `backend/app/core/logging.py` | Central logging configuration. |
| `backend/tests/__init__.py` | Test package marker. |
| `backend/tests/test_health.py` | Startup and health contract tests. |
| `backend/pytest.ini` or root `pyproject.toml` | Deterministic pytest discovery and configuration; choose one during planning. |
| `runtime/.gitkeep` | Documents the local runtime root while its contents remain ignored. |

### Modify

| File | Proposed change |
| --- | --- |
| `backend/requirements.txt` | Add explicit runtime/test dependencies and supported version bounds or split runtime/dev requirements. Do not add ML dependencies to the health-test import path. |
| `frontend/src/App.jsx` | Read API base URL from `VITE_API_BASE_URL` while preserving current behavior. |
| `README.md` | Correct paths, supported tool versions, environment setup, startup commands, health check, and current limitations. |

### Preserve unchanged

- `backend/main.py`
- `backend/api.py`
- `backend/predictor.py`
- `backend/model.py`
- `backend/trainer.py`
- `backend/data_loader.py`
- `backend/visualize_neurons.py`
- All current datasets, test images, uploads, outputs, activations, and model files
- Existing `POST /predict` behavior

If backward-compatible routing from the new application entry point to the legacy endpoint is desired, that decision should be made explicitly in the detailed Phase 1 plan.

## Proposed health contract

`GET /api/v1/health`

Expected HTTP 200 response:

```json
{
  "status": "ok",
  "service": "pcb-aoi-api",
  "api_version": "v1"
}
```

The liveness endpoint must not require loading PyTorch, OpenCV, a checkpoint, a database, or dataset files. Readiness checks for those dependencies should be separate in later phases.

## Dependency strategy

Phase 1 should separate lightweight platform dependencies from future ML/runtime dependencies so health tests can run without loading the 51 MB checkpoint.

Minimum backend foundation dependencies:

- FastAPI
- Uvicorn
- Pydantic Settings
- HTTPX for FastAPI test client support
- Pytest

Existing OpenCV, PyTorch, Torchvision, Pillow, and multipart dependencies remain necessary for the legacy prototype, but should not be imported by the new health path.

No package should be installed until the detailed plan is approved. Supported Python must also be resolved: the project instructions prefer Python 3.12+, while the currently active interpreter is Python 3.10.0.

## Verification plan

After approval and implementation:

1. Create/activate a documented virtual environment.
2. Install approved backend dependencies.
3. Run `python -m pytest` from the backend or repository root, according to the selected test configuration.
4. Start the canonical backend entry point and verify `/api/v1/health` returns the exact contract.
5. Install frontend dependencies with the lockfile using `npm ci`.
6. Run `npm run lint`.
7. Run `npm run build`.
8. Verify the existing prototype endpoint remains unchanged if it is mounted in the canonical app.
9. Verify runtime output paths resolve under the configured runtime root and remain ignored.
10. Review `git status` to confirm no runtime file, secret, database, cache, model output, or archive is newly tracked.

## Acceptance criteria

- One documented canonical backend startup command works from a known directory.
- `GET /api/v1/health` responds without loading the model or requiring data files.
- Health response is covered by an automated API test.
- Configuration is typed, environment-driven, and has no committed secret values.
- Logging initializes consistently and is covered by at least a startup smoke test.
- Frontend API base URL is environment-driven.
- Frontend lint and production build pass after approved dependency installation.
- New runtime files are ignored and kept outside tracked source paths.
- No current dataset, model, upload, output, or activation file is deleted or untracked.
- No 2D/3D feature, database, model, or UI redesign is introduced.

## Risks and controls

| Risk | Control |
| --- | --- |
| New and legacy FastAPI entry points diverge. | Define one canonical application factory and explicitly document whether legacy `/predict` is mounted or launched separately. |
| Root ignore rules hide a file that should be versioned. | Test ignore patterns with representative paths before staging. Do not use broad image-extension rules. |
| Dependency changes break the prototype. | Keep legacy ML imports out of the health path; install in a clean environment and record versions. |
| Python 3.10 vs preferred 3.12 creates inconsistent behavior. | Select and document one supported version before generating a lock/constraints file. |
| Existing tracked runtime/data files remain confusing. | Document them as legacy; handle untracking in a separately approved, reversible change. |
| Frontend environment change breaks local calls. | Preserve `http://127.0.0.1:8000` as a documented development default and test the production build. |

## Rollback approach

Phase 1 should be committed as one scoped change after tests pass. Rollback is the reversal of that commit. Because existing source and data files are preserved, rollback does not require data migration or model restoration.

## Recommended first implementation task

Implement only the lightweight backend application factory, typed settings, and model-independent `GET /api/v1/health` endpoint with one API test. Defer frontend configuration, logging refinements, and broader Git hygiene to subsequent small commits within Phase 1 after the health foundation is reviewed.
