# Read-Only Inspection History API

Date: 2026-07-20

## Route and safety boundary

`GET /api/v1/inspections` returns a compact projection of existing schema-version
3 database records. It performs no inserts, updates, deletes, or audit writes. It
does not open artifacts, read fixture manifests, validate inputs, preprocess
images, run mock inference, generate previews, or create reports. The current
status always comes from `inspections.status`; inconsistent child evidence fails
the whole request safely instead of rewriting that status.

Detailed findings remain available from
`GET /api/v1/inspections/{inspection_id}/validation` and
`GET /api/v1/inspections/{inspection_id}/processing`. History retrieval never
reruns either workflow.

## Response

```json
{
  "items": [
    {
      "inspection_id": "36a7a10f-3c74-4af8-b0e2-718ad1065e45",
      "status": "FAIL",
      "board_id": "BOARD-17",
      "recipe": {"recipe_id": "pcb-default", "recipe_version": "1.0"},
      "lot_id": "LOT-9",
      "operator_id": "operator-4",
      "created_at": "2026-07-20T10:00:00Z",
      "started_at": null,
      "completed_at": "2026-07-20T10:00:03Z",
      "technical_error_code": null,
      "validation": {
        "validation_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        "outcome": "VALIDATION_PASSED",
        "policy_id": "development-native-rgb-height",
        "policy_version": "1.0",
        "validator_version": "1.0.0",
        "completed_at": "2026-07-20T10:00:01Z",
        "total_findings": 0,
        "blocking_findings": 0,
        "warnings": 0,
        "errors": 0
      },
      "processing": {
        "processing_run_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        "processing_status": "COMPLETED",
        "preprocessing_id": "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
        "preprocessing_outcome": "PREPROCESSING_SUCCEEDED",
        "inference_id": "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        "inference_execution_outcome": "INFERENCE_SUCCEEDED",
        "mock_decision": "FAIL",
        "defect_type": "misalignment",
        "started_at": "2026-07-20T10:00:01Z",
        "completed_at": "2026-07-20T10:00:03Z",
        "synthetic_input": true,
        "mock_preprocessing": true,
        "mock_inference": true,
        "production_approved": false
      }
    }
  ],
  "page": {"limit": 25, "has_more": false, "next_cursor": null},
  "applied_filters": {},
  "request_id": "history-request-123"
}
```

Validation and processing are `null` when no corresponding persisted result
exists. The processing flags deliberately identify the current workflow as
synthetic, mock, and nonproduction. History never returns findings, confidence,
artifact paths, filenames, raw result JSON, fixture locations, buffers, model
paths, or historical audit request IDs.

`lot_id` and `operator_id` reflect their nullable inspection columns and are
returned as `null` when omitted or empty at intake. `station_id` remains
successful-intake audit metadata only and is neither returned nor searchable in
history.

## Ordering and pagination

Rows use immutable newest-first ordering:

1. `inspections.created_at DESC`
2. `inspections.id DESC`

`limit` defaults to 25 and permits 1 through 100. The service fetches one extra
row to calculate `has_more`; it does not calculate `total_count`. A total would
add expense while implying stability that concurrent intake cannot provide.

`next_cursor` is compact canonical JSON encoded with URL-safe base64. It contains
the cursor contract version (`pcb-aoi-inspection-history-cursor/1.0`), the last
UTC `created_at`, the canonical inspection UUID, and a SHA-256 digest of the
normalized filters. The token contains no offset, path, or secret and is not an
authentication token. Malformed, noncanonical, timezone-less, unsupported, or
filter-mismatched cursors return HTTP 400. `limit` is intentionally excluded
from the filter digest, so it may change between pages.

Traversal moves strictly below the `(created_at, inspection_id)` boundary. New
inspections created after page one can sort before that first boundary, but do
not duplicate or corrupt traversal of older records. There is no long-lived
database snapshot across requests.

## Filters

All supplied filters are combined with AND and identity metadata uses exact
matching:

- `status`
- `board_id`
- `recipe_id`
- `recipe_version`
- `lot_id`
- `operator_id`
- `created_from` (inclusive)
- `created_to` (exclusive)
- `validation_outcome`
- `processing_status`
- `mock_decision`
- `defect_type`
- `has_validation`
- `has_processing`

`station_id` is not supported because schema version 3 has no station column on
`Inspection`. Timestamps must include an offset, normalize to UTC, and satisfy
`created_from < created_to`. Enums and defect types are validated against the
existing contracts. Blank, control-character, and oversized metadata filters
are rejected. `applied_filters` returns normalized safe values. Query-parameter
ordering and equivalent UTC offsets produce the same filter digest.

Latest validation selection uses `completed_at DESC`, `created_at DESC`, then
`id ASC`. Latest processing selection uses `started_at DESC`, `created_at DESC`,
then `id ASC`. Findings are never joined. A nonempty page uses three bounded
SELECTs independent of page size: the inspection page, batch latest validation,
and batch latest processing with preprocessing/inference. An empty page uses one.

## Errors and request IDs

Errors use `{code, message, request_id}`. HTTP 400 covers invalid filters and
cursors; HTTP 422 covers FastAPI query-shape failures such as an out-of-range
limit; HTTP 500 covers safe retrieval or persisted-consistency failures. SQL,
paths, tracebacks, and internal exception text are never returned.

The response body `request_id` equals the current `X-Request-ID` response header.
An incoming ID is preserved by existing middleware; otherwise one is generated.
Viewing history creates no audit event.

## PowerShell examples

```powershell
$headers = @{ "X-Request-ID" = "history-request-123" }
$page = Invoke-RestMethod `
  -Headers $headers `
  -Uri "http://127.0.0.1:8000/api/v1/inspections?limit=25&status=FAIL"

$cursor = [System.Uri]::EscapeDataString($page.page.next_cursor)
Invoke-RestMethod `
  -Headers $headers `
  -Uri "http://127.0.0.1:8000/api/v1/inspections?limit=50&status=FAIL&cursor=$cursor"
```

A future frontend can request the first page with UI filters, retain
`applied_filters`, and request subsequent pages with `next_cursor`. It should
link an item to the inspection-specific details, validation, and processing
routes rather than expecting detailed evidence in this compact list.

## Focused tests

```powershell
.\.venv\Scripts\python.exe -m pytest .\backend\tests\test_inspection_history_api.py -q
```
