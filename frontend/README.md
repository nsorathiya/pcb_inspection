# PCB AOI Operator Frontend

React + Vite + strict TypeScript interface for the repository's existing
file-based AOI development APIs.

The UI supports inspection history, read-only recipe selection, paired-file
intake, technical validation, persisted evidence retrieval, an audit timeline,
and a deterministic development report with JSON download and browser print.
Mock results are not real AI predictions or production PCB decisions.

When the backend engineering-viewer flag is enabled, inspection detail links to the
read-only **PCB 2D/3D Vision Engineering Workspace**. It supports direct independent
RGB/height pixel selection, native sampling, labelled crosshairs, one active canvas
tool, keyboard shortcuts, 50-state session undo/redo, client-only alignment and pixel
measurements, onboarding, and a persistent status bar. The optional guided alignment
workflow stages RGB and height points before an explicit **Add Pair**, shows numbered
landmarks and development residual vectors, summarizes pixel residuals, and supports
Original/Development-aligned and reduced-motion-safe manual flicker comparison.
Controls and the 3x3 matrix are explicitly display-only: no viewer interaction
persists alignment, changes an artifact, or creates a production decision. See
`../docs/engineering_viewer.md`.

The height workspace also provides four deterministic derived-preview palettes,
session-only native display-range controls, an explicit invalid-pixel view, an
accessible 64-bin native histogram with display/sample/ROI markers, bounded native
height ROI statistics, complete pixel-only line geometry, and a confirmed
**Reset Engineering Session** action. Display changes participate in undo/redo and
never alter native sampling, source artifacts, database state, or audit history.

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

Focused engineering-workspace verification:

```powershell
npm run test:run -- src/utils/engineeringSession.test.ts src/pages/EngineeringViewPage.test.tsx
npx playwright test e2e/specs/zz-engineering-workspace.spec.ts
```

For interactive diagnosis use `npm run test:e2e:headed`; after a failed run,
open retained ignored diagnostics with `npm run test:e2e:report`. The suite uses
only unique temporary runtime data and cleans successful-run diagnostics. See
`../docs/synthetic_e2e_release_verification.md` for the complete release
verification contract.

See `../docs/operator_frontend.md` for routes, API behavior, lifecycle rules,
accessibility, and limitations.
