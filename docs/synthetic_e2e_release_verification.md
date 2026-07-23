# Synthetic browser E2E release verification

## Scope and safety boundary

This runbook verifies the supported development-only operator journey in a real
Chromium browser. It exercises the actual FastAPI application, React/Vite
frontend, SQLite schema version 3, repository-owned fixture generator, guarded
technical validation, deterministic synthetic processing, persisted reads,
audit projection, and development report. It does not exercise the legacy 2D
prototype.

> **Development only:** Mock decisions are not real predictions.

Completion does not establish HALCON equivalence or production AOI approval.
The suite does not add or verify real AI, calibration, model loading,
confidence, production disposition, authentication, hardware, revalidation, or
reprocessing.

## Prerequisites and supported versions

- Python 3.12 or newer is the project/CI target, with
  `backend/requirements-dev.txt` installed. The final local regression also
  verified compatibility with the existing repository `.venv` on Python
  3.10.0; that compatibility result does not replace the 3.12 CI target.
- Node 24.14.0, declared in `.node-version` and constrained to Node 24 by
  `frontend/package.json`.
- npm 10.9.0, declared by `packageManager`.
- Playwright 1.61.1 and its pinned Chromium browser. Install the browser with
  `npx playwright install chromium`; CI uses `--with-deps chromium`.

The E2E launcher uses explicit Python and Node executables and normalizes the
Windows `Path`/`PATH` environment. Vite development serving was selected because
it already provides SPA deep-link fallback and the repository API proxy without
adding a production hosting package. `VITE_DEV_PROXY_TARGET` is set by the test
launcher to the isolated backend port; normal development still defaults to
`http://127.0.0.1:8000`.

## Isolation and orchestration

`npm run test:e2e` performs these steps:

1. Creates a unique directory under the operating-system temporary directory
   and reserves independent available loopback ports.
2. Invokes `scripts/generate_synthetic_inspection_fixtures.py` with fixed seed
   `20260717` for only `valid_rgb_png_height_tiff` and
   `valid_different_dimensions`.
   It also creates a deterministic Full HD pair below the same temporary root
   for engineering-viewer hardening; those binaries never enter the repository.
3. Applies the normal migrations and seeds two versions of recipe
   `synthetic-e2e`: `1.0` ACTIVE and `0.9` DRAFT. Catalogue status is not
   production approval.
4. Starts the actual `create_app()` FastAPI application with the temporary
   runtime/database, trusted generated fixture root, synthetic API enabled, and
   warning-level logs. It waits for `/api/v1/health`.
5. Starts the actual Vite development server with its proxy pointed at that
   backend and waits for the frontend document.
6. Runs one serial, stateful release journey in pinned Playwright Chromium.
7. Stops only the owned frontend/backend processes, waits for exit, removes the
   temporary runtime, fixtures, database sidecars, and downloads, and verifies
   that the temporary root is absent.

Application startup does not read fixture manifests or run processing. Fixture
ownership is evaluated only by the existing orchestrator during explicit
processing. Deterministic ID providers and query instrumentation are injected
through test-only seams after application construction; production behavior and
database schema are unchanged.

## Covered workflow and scenarios

The complete PASS journey is:

`History -> New inspection -> exact recipe/version selection -> paired RGB and
height upload -> technical validation -> synthetic processing -> persisted
detail reload -> audit timeline -> development report -> JSON download -> print
media`

| Scenario | Browser expectation |
| --- | --- |
| Deterministic PASS | `VALIDATION_PASSED`, READY, `MOCK PASS`, no defect label, no confidence |
| Deterministic FAIL | `MOCK FAIL` with authoritative mock label `misalignment`, no confidence |
| Deterministic UNCERTAIN | `MOCK UNCERTAIN`, no defect label, no confidence |
| Technical preprocessing error | `PREPROCESSING_ERROR` and technical ERROR, never PCB FAIL |
| Dimension validation failure | completed `VALIDATION_FAILED`, visible finding, no processing action |

The PASS path additionally proves that optional lot/operator values can remain
null, separate recipe versions are not collapsed, actions issue one POST,
reloading preserves the processing/preprocessing/inference identities, and no
reprocessing action appears.

History coverage verifies newest-first results, compact lifecycle/mock labels,
recipe identity, null display, exact board filtering, 25-row cursor pagination,
absence of a total count, and absence of per-row detail requests. The suite
seeds old RECEIVED rows only to cross the page boundary.

## Audit, report, download, and print

Audit assertions verify chronological intake, validation, processing-start,
and mock-completion actions; historical request IDs remain historical. Database
snapshots prove that reading audit/report/history/recipes and reloading pages do
not create audit events, change workflow state, or execute processing.

The report test verifies the displayed SHA-256 by independently applying the
documented canonical JSON rules to the raw response. Raw response text is used
because parsing through JavaScript removes the lexical distinction between
JSON numbers such as `1567.0` and `1567`. The real browser download event is
captured; the safe deterministic filename, parsed object equality, inspection
ID, contract version, path omission, confidence omission, and download deletion
are checked. The current retrieval request ID remains outside the report;
historical audit request IDs remain part of persisted audit evidence by the
existing report contract.

Print verification uses browser print-media emulation, not an operating-system
dialog. It retains the report title, inspection ID, SHA-256, evidence,
limitations, and nonproduction warning while hiding navigation and interactive
buttons. It makes no PDF or certificate claim.

## Responsive and accessibility checks

The suite uses 1440x900 desktop, 1280x720 laptop, and 768x1024 tablet viewports.
It verifies usable navigation, scroll-safe history, reachable intake controls,
nonoverlapping evidence panels, readable audit/report content, and a visible
development warning. Semantic checks cover main/navigation landmarks, headings,
labels, standard keyboard-operable controls, accessible names, live feedback,
table headers, the ordered audit timeline, warning text, text/icon result
meaning, and duplicate IDs. No heavyweight accessibility scanner was added.

## Integrity and bounded-query checks

Read-only snapshots and profiling verify:

- schema version 3 and a clean SQLite foreign-key check;
- registered RGB/height hashes and byte sizes against actual isolated runtime
  files and generated manifest identities;
- all artifact paths remain inside the temporary runtime;
- the source fixture inventory remains byte-for-byte unchanged;
- report GET creates no report file and read routes execute no writes;
- persisted reload creates no new processing/preprocessing/inference rows;
- a populated history page uses three SELECTs, recipes use one, audit uses two,
  and report remains within its documented bounded multi-evidence query range.

Endpoint elapsed times are recorded for diagnostics but are not acceptance
thresholds. Query counts detect obvious N+1 regressions without adding
production benchmarking code.

## Errors and request correlation

The browser verifies malformed UUID and real unknown-inspection 404 handling,
plus backend-unavailable behavior. Controlled browser responses exercise the
existing frontend presentation of structured 503, 409, and 422 responses; the
backend's real status/error contracts remain covered by backend API tests. Each
action sends a distinct `X-Request-ID`, and visible structured errors retain
their safe code, message, and request ID. Request IDs are correlation values,
not authentication or processing idempotency keys.

The suite does not deliberately corrupt the SQLite database because existing
backend consistency tests cover safe 500 projection and database corruption is
not required for the supported browser workflow.

## Commands

From `frontend`:

```powershell
npm ci
npx playwright install chromium
npm run test:e2e
npm run test:e2e:engineering
npm run test:e2e:headed
npm run test:e2e:report
```

`test:e2e:headed` is for interactive diagnosis. The release sequence from the
repository root is:

```powershell
.\.venv\Scripts\python.exe -m pytest .\backend\tests

Set-Location .\frontend
npm ci
npm run lint
npm run typecheck
npm run test:run
npm run build
npm run test:e2e
```

## Failure diagnostics and cleanup

Playwright keeps failure screenshots, HTML context, and traces under ignored
`frontend/test-results/`; its HTML report is under ignored
`frontend/playwright-report/`. Backend/frontend logs, query profiles, and a
cleanup record are retained there only for failed runs. Inspect the HTML report
with `npm run test:e2e:report`. Diagnostics contain metadata and request
summaries, not uploaded artifact bytes or secrets.

Successful runs remove normal diagnostics, generated fixtures, database/WAL/SHM
files, runtime artifacts, downloads, and the unique temporary root only after
the reporter confirms the complete suite passed. Failed runs retain diagnostics
even when an earlier test succeeded. Setup and teardown handle partial startup;
process-tree fallback is scoped to recorded owned PIDs and never scans or
terminates unrelated developer services.

## CI behavior

`.github/workflows/verification.yml` runs on pushes and pull requests. It uses
Python 3.12, the project Node version, `npm ci`, pinned Chromium installation,
the complete backend suite, frontend lint/typecheck/unit/build, and browser E2E.
E2E diagnostics and the Playwright report are uploaded only after failure. CI
contains no deployment, publishing, production secrets, or permanent runtime
configuration.

## Known limitations

- Outcomes are deterministic workflow fixtures, not accuracy or defect-detection
  evidence.
- Synthetic identity registration is not physical 2D/3D registration or
  calibration.
- The browser-level 503/409/422 presentation cases are controlled responses;
  real backend contract behavior is verified separately by backend tests.
- The suite proves supported Chromium behavior. It is not a cross-browser or
  mobile-phone compatibility matrix.
- It does not approve real datasets, model weights, recipes, production hosting,
  operational security, HALCON equivalence, or production AOI disposition.
