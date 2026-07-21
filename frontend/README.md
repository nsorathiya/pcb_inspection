# PCB AOI Operator Frontend

React + Vite + strict TypeScript interface for the repository's existing
file-based AOI development APIs.

The UI supports inspection history, read-only recipe selection, paired-file
intake, technical validation, persisted evidence retrieval, an audit timeline,
and a deterministic development report with JSON download and browser print.
Mock results are not real AI predictions or production PCB decisions.

## Run locally

Start the FastAPI application on `http://127.0.0.1:8000`, then:

```powershell
npm ci
npm run dev
```

Vite serves the UI at `http://127.0.0.1:5173` and proxies `/api` to the local
backend. The default API base is same-origin. To use another API origin, create
`frontend/.env.local` containing, for example:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

`VITE_*` values are shipped to the browser. Never store secrets or backend
filesystem configuration in them.

When the backend has the development demo workspace configured, History shows
an explicit **Load Demo Workspace** action. It is hidden when the feature is
disabled, disables duplicate clicks while loading, reports structured errors,
and refreshes history after a successful idempotent load. See
`../docs/development_demo_workspace.md`.

## Verify

Use Node 24.14.0 and npm 10.9.0. Install the pinned Playwright Chromium browser
once with `npx playwright install chromium`.

```powershell
npm ci
npm run lint
npm run typecheck
npm run test:run
npm run build
npm run test:e2e
```

For interactive diagnosis use `npm run test:e2e:headed`; after a failed run,
open retained ignored diagnostics with `npm run test:e2e:report`. The suite uses
only unique temporary runtime data and cleans successful-run diagnostics. See
`../docs/synthetic_e2e_release_verification.md` for the complete release
verification contract.

See `../docs/operator_frontend.md` for routes, API behavior, lifecycle rules,
accessibility, and limitations.
