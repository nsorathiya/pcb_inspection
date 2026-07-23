# Engineering workspace usability verification

## Scope and result boundary

Task 30D verifies the synthetic, read-only engineering workspace across
responsive layouts, keyboard operation, accessibility semantics, temporary
large evidence, request behaviour, browser memory diagnostics, persisted-state
integrity, and cleanup.

This work does **not** establish real-data accuracy, production readiness,
calibration, registration quality, physical measurement, or production
inspection capability. No real Vision Team evidence was available.

## Test architecture

The verification uses three layers:

1. Pytest creates temporary deterministic native rasters and exercises the
   engineering metadata, preview, sample, and bounded ROI APIs.
2. Vitest verifies session state, keyboard-equivalent measurements,
   AbortController cleanup, flicker cleanup, and object-URL revocation.
3. Playwright starts an isolated SQLite runtime, backend, Vite frontend, and
   Chromium browser. It loads the demo workspace, generates a temporary Full HD
   pair outside the repository, records read-only snapshots, runs the viewer
   journeys, and removes the entire runtime.

Normal generator fixtures and temporary large evidence are never modified in
place. Large binaries are generated below the E2E temporary root and are not
committed.

## Responsive verification

| Profile | Viewport |
| --- | --- |
| Desktop | 1440 × 900 |
| Laptop | 1280 × 720 |
| Compact laptop | 1024 × 768 |
| Tablet portrait | 768 × 1024 |

At each viewport the suite verifies the header, navigation, persistent
synthetic warning, evidence navigator, canvas, inspector, pipeline evidence,
histogram, session alignment controls, and status bar. It rejects page-wide
horizontal overflow. Toolbar overflow is accepted only with local
`overflow-x: auto` or `scroll`. At 768 px, principal viewer toolbar and tool
buttons are at least 44 px high.

The only responsive product change is a 44 px control minimum at widths up to
800 px. No layout framework was added.

## Keyboard journey

Playwright completes this path without pointer activation:

- focus the MOCK PASS history link and press Enter;
- focus **Open Engineering Workspace** and press Enter;
- select view modes with focus and Enter;
- enter separate RGB and height coordinates;
- select the active space and use Arrow keys to move a pixel;
- sample values;
- create and add a correspondence pair;
- edit transforms and undo/redo;
- create a line and height rectangle through coordinate controls;
- open keyboard help, cancel with Escape, and reset the session.

Canvas dragging remains available. Its equivalent keyboard measurement path is:

1. choose active coordinate space and enter X/Y;
2. choose Line or Rectangle;
3. select **Set selected coordinate as measurement start**;
4. change X/Y;
5. select **Complete line/rectangle at selected coordinate**.

Global shortcuts intentionally do not run while focus is inside an input. Move
focus to a non-input control before Arrow-key pixel adjustment. Escape clears
incomplete correspondence or measurement state and closes keyboard help.

## Accessibility checks

Automated checks verify one `main` landmark, heading hierarchy, no duplicate
IDs, accessible names for every interactive element, `aria-pressed` view/tool
state, warning note semantics, code/message/request-ID errors, 64 named and
focusable histogram bins, a named live status bar, semantic correspondence and
measurement lists, non-colour-only status text, visible focus, and
reduced-motion flicker protection.

No accessibility dependency was added. Contrast continues to use the existing
design tokens.

## Large synthetic image matrix

| RGB | Height | Relation | Coverage |
| --- | --- | --- | --- |
| 1920 × 1080 PNG RGB8 | 1920 × 1080 TIFF uint16 | Matching | Pytest APIs and real Chromium |
| 2560 × 1440 TIFF RGB8 | 1920 × 1080 PNG16 | Differing | Pytest APIs |
| Existing PNG/TIFF fixtures | Existing TIFF/PNG16/NPY float32 fixtures | Matching and differing | Regression |
| 3840 × 2160 | — | — | Not fully materialized; limitation below |

Tests verify decoded dimensions, histogram totals, RGB `[17, 34, 51]`, native
height `117` at X 17/Y 23, a bounded 64 × 64 ROI, browser PNG previews,
zoom/pan, source SHA-256, byte size, and modification time.

The ceiling remains 16,777,216 pixels per raster and 1,048,576 pixels per ROI.
A 3840 × 2160 raster is below the ceiling, but the current pure-Python decoder
materializes large tuples. A full 4K multi-endpoint pass was not added because
it would create disproportionate CI memory pressure. Full HD browser and QHD
service coverage are the practical hardening boundary.

## Diagnostic timings

Timings are observations, not pass/fail budgets. Local focused Pytest results
on 2026-07-23:

| Operation | 1920 × 1080 PNG/TIFF | 2560 × 1440 TIFF + 1920 × 1080 PNG16 |
| --- | ---: | ---: |
| Intake | 0.156 s | 0.167 s |
| Metadata | 2.945 s | 2.460 s |
| RGB preview | 4.185 s | 5.560 s |
| Height preview | 5.652 s | 5.018 s |
| Sample | 1.619 s | 1.061 s |
| ROI | 1.543 s | 1.081 s |

Playwright records initial render, view-mode change, Full HD preview load,
sample, and ROI timings in a diagnostic attachment and prints compact values
to the test log. Attachments are retained only on failure.

The final complete local Playwright run observed:

| Browser operation | Time |
| --- | ---: |
| Initial demo workspace render | 1,197 ms |
| View-mode change | 339 ms |
| Sample request and rendered evidence | 314 ms |
| ROI request and rendered statistics | 328 ms |
| Temporary Full HD workspace and both previews | 34,892 ms |

Local environment: Windows, Python 3.10.0, Node 22.12.0, npm 10.9.0,
Playwright 1.61.1 Chromium. The repository and CI pin Node 24.14.0; the local
Node mismatch is a verification limitation, not a dependency change.

## Browser memory observations

Where Chromium DevTools Protocol metrics are available, Playwright records JS
heap used/total before the workspace, after previews, after repeated view
changes, after sampling/alignment/ROI/zoom/pan, and after leaving. Unsupported
metrics are diagnostic omissions. Available values reject an obvious spread of
256 MiB or more across the bounded journey; this is not a production budget.

Unit tests separately prove export object-URL revocation, stale sample and ROI
abort, event-listener cleanup, and flicker interval cleanup.

The final Chromium run reported JS heap used at approximately 5.58 MB before
the workspace, 8.62 MB after previews, 14.10 MB after repeated view changes,
16.09 MB after sample/alignment/ROI/zoom/pan, and 14.90 MB after leaving.
V8 retained allocated heap capacity, but used heap fell after navigation and
the observed spread was far below the 256 MiB obvious-growth guard.

## Network observations

The suite proves pointer movement creates no requests; four explicit samples
create four sample GETs; one height rectangle creates one ROI GET; metadata is
bounded to the initial Vite/React StrictMode mount replay (at most two GETs);
previews are bounded; and transforms, undo/redo, correspondence, and reset
produce no writes. History N+1 query coverage remains in the complete synthetic
release journey.

## Read-only database and source-integrity proof

The post-demo snapshot contains schema version, foreign-key check, a canonical
database fingerprint, table counts, audit/processing relationships, runtime
and report inventories, and artifact integrity. After viewer journeys the
fingerprint, counts, artifacts, fixture inventory, and runtime inventory must
be identical and report files must remain empty.

SHA-256, byte size, and nanosecond modification time (where reliable) are
compared for persisted RGB/height artifacts, every trusted fixture,
`SYNTHETIC_FIXTURES_MARKER.json`, `generation_manifest.json`, and the temporary
Full HD RGB/height/manifest. The Full HD inspection is created before its
read-only baseline.

## Error and cleanup behaviour

Journeys cover viewer disabled, missing inspection, controlled unsupported
evidence, backend integrity regressions, and backend unavailable. Server
failures expose code, message, and request ID.

Teardown stops owned processes and removes the runtime, database/WAL/SHM,
fixtures, large evidence, and downloads. Diagnostics are now removed only
after Playwright reports the **entire** suite passed. Later failures can no
longer be hidden by an earlier success marker. Failure screenshots, traces,
network summaries, logs, and cleanup evidence are retained; successful runs
leave no test residue.

## Commands

```powershell
cd frontend
npm run test:e2e:engineering
```

Complete verification remains `python -m pytest backend/tests` followed by
`npm ci`, lint, typecheck, unit tests, build, and `npm run test:e2e`.

## Remaining limitations

- No real Vision Team RGB/height pair was available.
- No real accuracy, repeatability, calibration, registration, physical
  measurement, or production decision was evaluated.
- The largest real-browser case is Full HD; QHD is service-level and full 4K
  browser materialization is deferred for the memory reason above.
- Memory and timing values are environment-specific diagnostics, not service
  objectives.
- Vite development mode observes React StrictMode's bounded effect replay.
