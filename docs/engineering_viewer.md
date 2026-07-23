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
- `GET /api/v1/inspections/{id}/engineering-view/height-preview?palette=grayscale&display_min=0&display_max=65535&show_invalid=false`
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
visualization; it is not raw height evidence. With no query parameters it remains the
deterministic native-min/max grayscale rendering. `palette` accepts the stable,
project-owned names `grayscale`, `blue-yellow`, `viridis-like`, and `high-contrast`.
The latter two names describe this project's deterministic colour tables and do not
claim compatibility with a third-party palette library.

`display_min` and `display_max` must be finite, supplied together, and satisfy
`display_min < display_max`. The service accepts bounds outside the native range:
each finite native sample is clipped to the nearest display endpoint before colour
mapping; the requested bounds themselves are not silently changed. `show_invalid`
defaults to `false`. When `true`, values invalid according to the current synthetic
decoder are rendered magenta; when `false`, they use the low-end display colour.
Invalid samples remain excluded from the native histogram and valid count either
way. These options never change sampling or ROI statistics.

Height preview response headers expose the selected palette, native min/max, display
min/max, invalid-visibility state, and the warning
`DISPLAY_RANGE_CHANGES_DERIVED_COLOUR_VIEW_ONLY_NATIVE_VALUES_UNCHANGED`.
Preview responses use `Cache-Control: no-store`, expose no paths, and verify source
ownership, SHA-256, and byte size before decoding.

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
tool, selections, palette, display interval, invalid visibility, pair count,
registration status, and unavailable units.

### Height visualization and native measurement UX

The derived-height controls select one of the four stable palettes, apply or reset a
display interval within the metadata-reported native limits, and show or hide invalid
pixels. Palette, interval, and invalid visibility are browser-session values in the
same bounded undo/redo history as the other engineering controls. They return to
grayscale, native min/max, and hidden invalid pixels on reload or full session reset.
The interface always states:

> Display range changes only the derived colour view. Native height values remain unchanged.

The legend labels the selected palette and applied display endpoints. Invalid pixels
have a separate magenta key and the exact explanation **Invalid according to the
current synthetic decoder.** This has no vendor, defect, tolerance, or production
meaning.

The 64-bin histogram remains a histogram of finite native samples. It shows the
current display endpoints, the latest valid sampled height, and the selected height
ROI's native min/max interval. Each bin is keyboard-focusable and reports its native
range and count on focus or hover. Clicking or pressing Enter/Space merely selects a
bin; the operator must then explicitly choose **Use selected bin as display range**.
Valid and excluded-invalid totals remain visible.

A non-blocking quick-start guide opens on entry, can be dismissed or reopened, and is
session-only. Toolbar buttons expose pressed state and keyboard shortcuts to assistive
technology. The interface never maps a viewer action to a production PASS/FAIL action.

The layout keeps the evidence navigator on the left, the vision canvas in the centre,
metadata and pixel inspection on the right, and persisted pipeline evidence below.
It adapts to desktop, laptop, and tablet widths without changing coordinate semantics.
The canvas never treats the derived preview as native height data, never invents a
physical unit, and never calls a validation or processing endpoint.

### Session-only alignment and measurements

The workspace groups controls as **View**, **Transform**, **Correspondence**, and
**Results**. Translation X/Y is bounded to pixels, rotation to degrees, and scale is
bounded and unitless. The Results group labels the 3x3 affine display matrix as a
height-display-to-RGB-display transform around the RGB display centre. It is never
presented as a camera, calibration, or production matrix.

The default **Original** view leaves the height preview untransformed. **Apply
transform to view only** switches to a prominent **Development-aligned** state;
**Return to original** restores original rendering, while identity reset remains in
the same bounded undo/redo history. Alignment is applied only to the browser-rendered
height layer. Reload, route change, or closing the page clears it. The frontend does
not send alignment to the backend and does not claim automatic or production
registration.

Correspondence is an optional guided workflow: select RGB, select height, explicitly
choose **Add Pair**, then repeat or review. Add Pair remains disabled until both
points are valid. Pairs receive stable numbers displayed with distinct shapes and
text on both rasters, remain visible through zoom and pan, and can be selected from
either raster or the list. Removal is per pair; clearing all requires confirmation.

Each pair draws an optional development-residual line from the transformed height
point to its RGB target, with a pixel magnitude. The largest residual is highlighted.
For mismatched raster dimensions, height coordinates are normalized into the RGB
display-pixel comparison space before the display transform is evaluated. The
summary reports count, mean, maximum, minimum, median, highest pair number, and an
optional translation suggestion. These are explicitly *development residuals* with
no threshold, pass/fail state, or registration-quality claim. The translation
suggestion must be applied explicitly; it is never automatic registration.

Manual flicker alternates Original and Development-aligned rendering at 2.5 changes
per second. It starts and stops only by operator action, stops when the comparison
mode/page is left or the tab is hidden, and is unavailable when reduced motion is
requested. Flicker is a visual aid, not proof of alignment.

Rectangle and line tools retain separate RGB/height coordinate spaces. Rectangle
X/Y, width, height, and area are pixel measurements. Height rectangles read bounded
native min/max/mean/range and valid/invalid counts from the existing read-only ROI
endpoint. A newer ROI request aborts the older browser request, and structured errors
retain the request ID. The 1,048,576-pixel backend maximum is unchanged. Selecting a
height ROI marks its native min/max on the existing histogram; no matrix or new
histogram data is requested.

Lines report coordinate space, start/end X/Y, dx/dy, Euclidean distance in pixels,
and the screen-coordinate direction from `atan2(dy, dx)` in degrees. There is no line
profile, calibration, or persistence.

**Reset Engineering Session** requires confirmation and resets the active tool,
crosshairs and coordinates, view mode, zoom/pan, palette/range/invalid state,
alignment, correspondences, ROI/line measurements, and undo/redo history. It does
not reload backend evidence or mutate persisted state. It is distinct from
**Reset view** and **Reset transform to identity**.

**Export alignment JSON** downloads the versioned
`pcb-aoi-development-alignment/1.0` contract using the deterministic filename
`inspection-{id}-development-alignment.json`. It is generated entirely in the
browser and contains numbered pairs, the active Original/Development view, complete
residual summary, display-coordinate labels, and stable limitation warnings. It has
no paths, request ID, confidence, or fabricated physical units. The contract always
records `development_only=true`, `production_approved=false`, and browser-view-only
application.

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
