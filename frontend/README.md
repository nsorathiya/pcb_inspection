# PCB AOI Operator Frontend

React + Vite + strict TypeScript interface for the repository's existing
file-based AOI development APIs.

The UI supports inspection history, read-only recipe selection, paired-file
intake, technical validation, persisted evidence retrieval, and explicitly
synthetic processing. Mock results are not real AI predictions or production
PCB decisions.

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

## Verify

```powershell
npm run lint
npm run typecheck
npm run test:run
npm run build
```

See `../docs/operator_frontend.md` for routes, API behavior, lifecycle rules,
accessibility, and limitations.
