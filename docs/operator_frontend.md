# Operator frontend

## Purpose and safety boundary

The operator frontend is the React/Vite client for the existing file-based AOI
development workflow. It lets an operator browse persisted inspections, choose
an exact recipe identity, register paired RGB and native height/depth files, run
technical validation, and—when the backend is explicitly configured—run the
trusted synthetic processing orchestrator.

The application permanently displays this warning:

> Development mode: processing uses deterministic synthetic mock inference.
> Results are not production PCB decisions.

Every processing result repeats a stronger nonproduction warning. `MOCK PASS`,
`MOCK FAIL`, and `MOCK UNCERTAIN` are deterministic digest-bucket workflow
values. They are not image-content analysis, real AI predictions, accuracy
evidence, or production disposition. The UI does not display or invent a
confidence value.

## Frontend routes

| Route | Purpose |
| --- | --- |
| `/` | Newest-first inspection history and supported exact filters |
| `/inspections/new` | Recipe selection and paired-file intake |
| `/inspections/:inspectionId` | Authoritative inspection state, validation, processing, and workflow actions |

Browser routing includes an unknown-route page. The detail route rejects a
malformed inspection UUID before requesting the backend, while a well-formed but
unknown UUID is handled through the backend's structured 404 response. Vite's
history fallback supports local deep links; a production web server must also
route unknown frontend paths to `index.html`.

## Backend API dependencies

The frontend calls only these existing endpoints:

- `GET /api/v1/health`
- `GET /api/v1/recipes`
- `POST /api/v1/inspections`
- `GET /api/v1/inspections`
- `GET /api/v1/inspections/{inspection_id}`
- `POST /api/v1/inspections/{inspection_id}/validate`
- `GET /api/v1/inspections/{inspection_id}/validation`
- `POST /api/v1/inspections/{inspection_id}/process`
- `GET /api/v1/inspections/{inspection_id}/processing`

Typed service modules under `frontend/src/api` isolate routing, multipart
construction, query serialization, policy requests, response fields, and the
structured error contract from page components. Server responses remain
authoritative; the browser does not implement lifecycle transitions.

## Environment configuration

`VITE_API_BASE_URL` is the only frontend API setting. An empty value uses
same-origin relative requests. During local development, Vite proxies `/api` to
`http://127.0.0.1:8000`, so backend CORS changes are unnecessary.

For a different backend origin, create `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Vite environment variables are public browser configuration. Do not put
secrets, runtime roots, fixture roots, or other backend filesystem settings in
them.

## Local setup

From the repository root, start the backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\backend\requirements-dev.txt
python -m uvicorn app.main:app --app-dir .\backend --reload
```

Start the frontend in another PowerShell window:

```powershell
Set-Location .\frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`.

## Inspection history

The dashboard renders the compact validation and processing summaries already
included by `GET /api/v1/inspections`; it does not make per-row detail calls.
Filters cover inspection status, board ID, recipe ID, lot ID, validation
outcome, processing status, mock decision, validation/processing presence, and
UTC creation range. Filters use API-supported exact values rather than free
text search.

Pagination preserves the applied filter state and keeps a local stack only for
Previous Page navigation. The opaque `next_cursor` is returned unchanged to the
backend. The UI neither decodes cursors nor invents a total count. Loading,
empty database, empty filtered result, backend error, backend-unavailable, and
next-page states are explicit. Synthetic result labels always retain `MOCK` or
`Synthetic` wording.

## Recipe selection and paired intake

The read-only selector loads cursor-paginated recipes and supports exact recipe
ID, opaque version, name, and status filters. Every version remains a separate
radio option; no version is selected automatically and no semantic “latest”
version is inferred. `ACTIVE` is shown as catalogue status, not production
approval.

The intake form requires `board_id`, one explicitly selected recipe, one RGB
image, and one height/depth file. `lot_id`, `operator_id`, and `station_id` are
optional. Selected files remain separate and show browser-local filename, byte
size, and reported type/extension. Clear and replacement controls do not read
file contents, compute hashes, or claim persisted evidence. Input `accept`
values are selection guidance; backend validation remains authoritative.

The multipart request uses the exact fields `board_id`, `recipe_id`,
`recipe_version`, `rgb_image`, and `height_map`. Blank optional values are
omitted, and recipe ID/version strings are preserved unchanged. No synthetic
flag, validation ID, processing ID, decision, integrity expectation, or upload
percentage is supplied. The submit button is disabled during intake. After the
201 response, the client navigates to the detail route and reports the returned
inspection ID and `RECEIVED` state; it does not automatically validate.

## Detail and validation workflow

The detail page shows safe persisted inspection metadata, lifecycle state,
technical error evidence, and registered artifact integrity summaries without
storage paths. A manual Refresh reloads the detail record and both persisted
child-result endpoints.

For `RECEIVED`, the available action sends only this explicit validation policy:

```json
{
  "policy_id": "development-native-rgb-height",
  "policy_version": "1.0"
}
```

`VALIDATION_PASSED`, `VALIDATION_FAILED`, and `VALIDATION_ERROR` are completed
technical results. The UI displays policy and validator identity, counts,
ordered findings, RGB evidence, and height evidence. A missing persisted result
is the neutral state “Not validated yet.” The explanation is explicit:
`VALIDATION_PASSED` means technically ready for preprocessing; it is not PCB
PASS.

## Synthetic processing workflow

For authoritative state `READY`, the UI calls only the processing orchestrator
endpoint and sends exactly four fields:

```json
{
  "preprocessing_policy_id": "synthetic-paired-rgb-height",
  "preprocessing_policy_version": "1.0",
  "inference_policy_id": "synthetic-deterministic-mock-inference",
  "inference_policy_version": "1.0"
}
```

The result view includes run and policy identities, preprocessing and inference
execution outcomes, ordered findings, timestamps, and accessible `MOCK PASS`,
`MOCK FAIL`, `MOCK UNCERTAIN`, or `TECHNICAL ERROR` states. Only `MOCK FAIL`
may show its mock taxonomy defect label. An exact idempotent replay is labelled
as persisted replay evidence. HTTP 409 lifecycle conflicts and HTTP 503 disabled
processing are shown through the common structured-error component.

Actions follow only the current detail status: `RECEIVED` can validate,
`READY` can process, and `PROCESSING` requires manual refresh. Validation
failure, final synthetic states, and technical error do not expose processing or
reprocessing actions. Action requests are protected against duplicate clicks
and followed by authoritative backend retrieval. There is no continuous poll.

## Errors and request correlation

Each unrelated API call creates a fresh UUID with `crypto.randomUUID()` and
sends it as `X-Request-ID`. The client prefers the backend response header and,
for structured errors, preserves `code`, safe `message`, and `request_id`.
Visible errors include those three safe fields and a retry action where the
operation is safe. Stack traces, SQL, filesystem paths, raw exception objects,
and raw HTML are not rendered.

Network failure, 30-second client timeout, caller abort, generic HTTP failures,
and structured 400/404/409/422/500/503 responses have distinct client behavior.
Request IDs provide correlation only; they are not authentication or
idempotency keys.

## Accessibility and responsive behavior

The application uses a semantic header, navigation, main region, forms,
fieldsets, labels, description lists, table headers, ordered findings, linked
form-error summary, and live regions for asynchronous feedback. Keyboard focus
is visible and actions are standard buttons or links. Status and decision
states use text plus a shape/icon and never rely on colour alone.

The desktop-first industrial layout supports operator-station, laptop, and
tablet widths. History remains horizontally scrollable at narrower widths;
forms and evidence grids collapse without hiding actions or safety warnings.

## Tests and build

Install exactly from the npm lock file, then run verification:

```powershell
Set-Location .\frontend
npm ci
npm run lint
npm run typecheck
npm run test:run
npm run build
npm run test:e2e
```

Tests cover API base and request IDs, structured/network/timeout errors,
multipart and policy payload safety, history filters/cursors/states, recipe
versions and file intake, duplicate action protection, lifecycle-gated actions,
validation evidence, synthetic decisions, technical failure, replay evidence,
request IDs, and the permanent development warning.

The Playwright suite runs the actual frontend and backend in a real pinned
Chromium browser using a fresh temporary database, trusted generated fixtures,
and two explicit recipe versions. It covers the complete PASS journey plus
shorter FAIL, UNCERTAIN, technical ERROR, and validation-failure paths; history
pagination, report download, print media, responsive/accessibility assertions,
read-only database checks, and owned-process cleanup are included. See
`docs/synthetic_e2e_release_verification.md` for commands and limitations.

## Audit timeline and development report

The inspection detail embeds a semantic ordered Audit Timeline backed only by
persisted events. It supports cursor-based “Load more,” empty/loading/error
states, historical actor/request IDs, safe detail rows, and explicit redaction
indicators. The detail page also links to the dedicated Development Report.

The report route displays inspection, artifact, validation, processing,
findings, audit, limitation, and SHA-256 evidence with a persistent
development/nonproduction warning. JSON download uses the exact backend report
object. Print uses browser print CSS and retains identity, hash, warning,
evidence, and limitations while hiding navigation and buttons. Neither action
changes backend state or creates a PDF/report file.

## Known limitations

- Synthetic processing POST is disabled unless the backend is explicitly
  configured with trusted generated fixtures.
- The UI has no authentication or authorization and is development-only.
- Recipe catalogue is read-only; recipe creation and activation are absent.
- There is no reprocessing, background polling, backend PDF/report-file
  generation, preview, overlay, camera, PLC, MES, model administration, or real
  inference.
- File upload progress is an indeterminate busy state because the API supplies
  no byte-progress contract.
- Production hosting must configure SPA fallback and an appropriate API routing
  or reverse-proxy policy.
