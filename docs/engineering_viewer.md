# Read-only synthetic engineering viewer

The engineering viewer is a disabled-by-default development feature for inspecting
the exact RGB and native-height pair already registered to an inspection. It is not
a production AOI decision tool and does not establish physical calibration or
registration.

## Configuration

Set the flag before starting the backend:

```powershell
$env:PCB_AOI_ENABLE_ENGINEERING_VIEWER = "true"
```

The default is `false`. When disabled, viewer routes return the safe
`ENGINEERING_VIEWER_DISABLED` response. The viewer uses the normal configured
`PCB_AOI_RUNTIME_ROOT`; callers cannot select a path or artifact ID.

## Endpoints

- `GET /api/v1/inspections/{id}/engineering-view`
- `GET /api/v1/inspections/{id}/engineering-view/rgb-preview`
- `GET /api/v1/inspections/{id}/engineering-view/height-preview`
- `GET /api/v1/inspections/{id}/engineering-view/sample?rgb_x=0&rgb_y=0&height_x=0&height_y=0`

The metadata response includes registered integrity identity, native raster metadata,
finite native-height min/max, valid/invalid counts, a fixed 64-bin histogram,
calibration and registration status, and any already-persisted validation and
processing evidence. It never returns storage paths.

Matching a supported file format does not prove synthetic provenance. The response
reports `synthetic_input_verified: true` only when existing persisted processing
evidence verified trusted synthetic input; otherwise it returns `false` with an
explicit provenance-not-verified warning.

Supported development inputs are RGB PNG/TIFF with three RGB channels and native
height TIFF `uint16`, PNG16, or two-dimensional NPY `float32`. Inputs outside this
narrow synthetic subset fail closed.

## Preview and sampling safety

Both previews are generated in memory as browser-compatible PNG responses. They are
never stored or registered. The height preview is explicitly marked as a derived
native-min/max grayscale visualization; it is not raw height evidence. Preview
responses use `Cache-Control: no-store` and safe headers identifying the derivation.

RGB and height sample coordinates are independent because physical registration is
not established. Sampling returns native RGB samples and the native height value.
Non-finite height samples return `value: null` and `valid: false`. Physical units are
always `null` until reviewed calibration evidence provides an authoritative unit.

## Read-only and integrity guarantees

Every request selects only the registered artifacts owned by the requested inspection.
The existing managed-path resolver enforces runtime confinement and rejects traversal,
symbolic links, reparse points, non-regular files, and category/inspection mismatches.
SHA-256 and byte size are streamed and compared with database registration before any
decode.

Viewer requests do not write files or database rows, append audit events, execute
validation, run preprocessing, or invoke inference. The optional height ROI endpoint
is intentionally deferred; the current surface provides only strictly bounded point
sampling and whole-image 64-bin statistics.

## Tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest .\backend\tests\test_engineering_viewer_api.py -q
.\.venv\Scripts\python.exe -m pytest .\backend\tests\test_health.py `
  .\backend\tests\test_inspection_details.py `
  .\backend\tests\test_inspection_validation_api.py `
  .\backend\tests\test_inspection_processing_api.py -q
```
