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
- `GET /api/v1/inspections/{id}/engineering-view/height-roi?x=0&y=0&width=8&height=8`

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
validation, run preprocessing, or invoke inference. The bounded height ROI endpoint
accepts a positive in-bounds rectangle of at most 1,048,576 pixels and returns native
finite min/max/mean and valid/invalid counts. It does not convert values to physical
units or generate a persistent measurement record.

## Vision engineering workspace

The frontend route `/inspections/{id}/engineering-view` presents the read-only API as
the **PCB 2D/3D Vision Engineering Workspace**. Open it from an inspection detail
page with **Open Engineering Workspace**. The page provides RGB, height, side-by-side,
alpha-overlay, and split-comparison modes; synchronized normalized zoom and pan;
independent RGB/height coordinates; an explicit native-value sample action; and an
SVG 64-bin height histogram.

### Direct interaction and session tools

Pointer and Sample clicks are converted from browser CSS coordinates to zero-based
native raster coordinates after canvas zoom and pan. Height clicks also invert the
session-only affine display transform. A click outside the native raster is ignored.
RGB and height selections remain independent: selecting one never overwrites the
other. Sample performs one read-only GET for the newly selected coordinate pair,
cancels an older in-flight sample request, and shows the backend request ID if the
latest request fails. Manual coordinate inputs remain available for exact entry.

Selected RGB and height pixels have separate labelled crosshairs. They remain visible
while other tools are active and can be cleared together. Arrow keys adjust the active
selection by one native pixel; Shift+Arrow adjusts it by ten, always clamped to the
corresponding raster.

Exactly one canvas tool is active:

- `V` Pointer
- `H` Pan
- `S` Sample
- `C` Correspondence
- `R` Rectangle
- `L` Line

View shortcuts are `+`/`-` for zoom, `F` for fit, and `0` for actual pixels. Escape
cancels an incomplete interaction. Ctrl/Cmd+Z undoes session actions and
Ctrl/Cmd+Shift+Z or Ctrl/Cmd+Y redoes them. The bounded in-memory history retains the
latest 50 session states and is cleared on reload or route change.

The RGB and height inspectors report their own selected coordinate, native values,
dimensions, and integrity/calibration/registration evidence. Before an explicit
sample, values say **Not sampled**. Physical values and units remain **Unavailable**.
A persistent status bar reports inspection identity, raster dimensions, zoom, active
tool, selections, pair count, registration status, and unavailable units.

A non-blocking quick-start guide opens on entry, can be dismissed or reopened, and is
session-only. Toolbar buttons expose pressed state and keyboard shortcuts to assistive
technology. The interface never maps a viewer action to a production PASS/FAIL action.

The layout keeps the evidence navigator on the left, the vision canvas in the centre,
metadata and pixel inspection on the right, and persisted pipeline evidence below.
It adapts to desktop, laptop, and tablet widths without changing coordinate semantics.
The canvas never treats the derived preview as native height data, never invents a
physical unit, and never calls a validation or processing endpoint.

### Session-only alignment and measurements

The workspace provides development-only translation X/Y, rotation, scale X/Y,
overlay opacity, and a visible 3x3 affine matrix. Alignment is applied only to the
rendered height layer. It is React browser state: reload, route change, or closing
the page clears it. The frontend does not send alignment to the backend and does not
claim automatic or production registration.

An operator can pair explicit RGB and height correspondence pixels. Per-pair, mean,
and maximum residuals are calculated in pixels. An optional translation suggestion
is displayed and must be applied explicitly; it is never automatic registration.
Rectangle and line tools retain separate RGB/height coordinate spaces. Rectangle
width, height, and area and line distance are pixel measurements. Height
rectangles may read bounded native-value statistics from the read-only ROI endpoint.
No millimetre or micrometre conversion is performed.

**Export alignment JSON** downloads the versioned
`pcb-aoi-development-alignment/1.0` contract using the deterministic filename
`inspection-{id}-development-alignment.json`. It is generated entirely in the
browser and contains no paths, request ID, confidence, or fabricated physical units.
The contract always records `development_only=true` and
`production_approved=false`.

## Tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest .\backend\tests\test_engineering_viewer_api.py -q
.\.venv\Scripts\python.exe -m pytest .\backend\tests\test_health.py `
  .\backend\tests\test_inspection_details.py `
  .\backend\tests\test_inspection_validation_api.py `
  .\backend\tests\test_inspection_processing_api.py -q
Set-Location .\frontend
npm run test:run -- src/pages/EngineeringViewPage.test.tsx
npm run test:run -- src/utils/engineeringSession.test.ts
npm run test:run
npm run build
npm run test:e2e
```
