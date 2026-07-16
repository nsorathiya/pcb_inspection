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
```

The supported variables are:

| Variable | Default |
| --- | --- |
| `PCB_AOI_APPLICATION_NAME` | `pcb-aoi-api` |
| `PCB_AOI_APPLICATION_VERSION` | `0.1.0` |
| `PCB_AOI_ENVIRONMENT` | `development` |
| `PCB_AOI_API_PREFIX` | `/api/v1` |
| `PCB_AOI_DEBUG` | `false` |
