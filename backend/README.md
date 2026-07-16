# Backend Foundation

The foundation application provides a model-independent health endpoint. It does not import or load the existing PyTorch prototype.

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

## Run tests

Run the focused logging, startup, health, and request-ID tests:

```powershell
python -m pytest .\backend\tests\test_health.py -q
```

Run the focused runtime-storage tests:

```powershell
python -m pytest .\backend\tests\test_runtime_paths.py -q
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
| `runtime/raw_uploads` | Future immutable raw uploads |
| `runtime/previews` | Future generated previews |
| `runtime/results` | Future processing results |
| `runtime/reports` | Future generated reports |
| `runtime/tmp` | Temporary runtime files |

To select a different location for the current PowerShell session:

```powershell
$env:PCB_AOI_RUNTIME_ROOT = "C:\pcb-aoi-runtime"
python -m uvicorn app.main:app --app-dir .\backend --reload
```

If any required directory cannot be created, application startup logs an error
and fails. The public health response does not include the local runtime path.

## Request IDs

Every HTTP response includes an `X-Request-ID` header. If a caller supplies
`X-Request-ID`, the application preserves it unchanged. Otherwise, the
application generates a UUID request ID. During request handling the same value
is available as `request.state.request_id` and is added to application log
records through request-local context.
