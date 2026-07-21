# Inspection audit timeline API

`GET /api/v1/inspections/{inspection_id}/audit` returns a read-only, persisted
timeline for one canonical inspection UUID. It performs no lifecycle action,
does not inspect files, and does not append an event merely because the timeline
was viewed.

## Ordering and pagination

Rows are scoped to `entity_type=inspection` and the requested `entity_id`, then
ordered by `created_at ASC, id ASC`. `limit` defaults to 50 and accepts 1–200.
The query fetches one extra row to determine `has_more`; no total count or SQL
offset is used.

`next_cursor` is an opaque URL-safe base64 encoding of canonical compact JSON.
It binds cursor version 1, the last UTC timestamp, audit-event UUID, and
inspection UUID. Malformed base64/JSON, missing fields, noncanonical UUIDs,
timezone-less timestamps, unsupported versions, and cross-inspection reuse are
rejected with structured HTTP 400 errors.

```json
{
  "items": [],
  "page": {"limit": 50, "has_more": false, "next_cursor": null},
  "request_id": "current-read-request"
}
```

## Safe detail projection

The service uses centralized action identifiers and action-specific allowlists.
Known intake, validation, processing-start, mock-result, and processing-error
events expose only established scalar identities, versions, outcome/status
values, hashes, counts, and explicit mock/nonproduction flags. Unknown keys are
removed. Unknown actions remain visible by code and timestamp but expose no raw
details. Path-shaped keys/values, arbitrary exception text, SQL, stack traces,
binary content, and unsafe nested values are removed and set
`details_redacted=true`.

Historical `request_id` values belong to their original events. The response
body and `X-Request-ID` header contain the current retrieval request ID.

## PowerShell and tests

```powershell
$base = "http://127.0.0.1:8000/api/v1"
$page = Invoke-RestMethod "$base/inspections/$inspectionId/audit?limit=50"
if ($page.page.has_more) {
  $next = [uri]::EscapeDataString($page.page.next_cursor)
  Invoke-RestMethod "$base/inspections/$inspectionId/audit?limit=50&cursor=$next"
}

Set-Location .\backend
..\.venv\Scripts\python.exe -m pytest tests\test_inspection_audit_api.py -q
```
