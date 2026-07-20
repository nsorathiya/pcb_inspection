# Read-Only Inspection Details API

Date: 2026-07-17

## Scope

`GET /api/v1/inspections/{inspection_id}` returns persisted inspection lifecycle
metadata and registered artifact integrity metadata for one canonical inspection
UUID. It is a read-only metadata endpoint. It does not read or return artifact
bytes, expose storage locations, decode or semantically validate images, align
RGB and height data, run inference, or classify the board.

The separate `GET /api/v1/inspections` collection route provides compact,
cursor-paginated history without artifact details. See
`docs/inspection_history_api.md`.

## PowerShell example

```powershell
$inspectionId = "36a7a10f-3c74-4af8-b0e2-718ad1065e45"
$headers = @{ "X-Request-ID" = "details-request-456" }
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/inspections/$inspectionId" `
  -Headers $headers
```

The path value must be the lowercase, hyphenated canonical UUID form. A
malformed or non-canonical value is rejected before a database query.

## Success response

HTTP status: `200 OK`

```json
{
  "inspection_id": "36a7a10f-3c74-4af8-b0e2-718ad1065e45",
  "status": "RECEIVED",
  "board_id": "PCB_A",
  "recipe_id": "PCB_A",
  "recipe_version": "1.0",
  "lot_id": null,
  "intake_request_id": "intake-request-123",
  "created_at": "2026-07-17T12:00:00Z",
  "started_at": null,
  "completed_at": null,
  "error": null,
  "artifacts": [
    {
      "artifact_type": "RGB_RAW",
      "sha256": "<lowercase-sha256>",
      "byte_size": 1234,
      "media_type": "image/png",
      "created_at": "2026-07-17T12:00:00Z"
    },
    {
      "artifact_type": "HEIGHT_RAW",
      "sha256": "<lowercase-sha256>",
      "byte_size": 4321,
      "media_type": "image/tiff",
      "created_at": "2026-07-17T12:00:01Z"
    }
  ]
}
```

The endpoint reports every status already persisted by the lifecycle schema:
`RECEIVED`, `VALIDATION_FAILED`, `READY`, `PROCESSING`, `PASS`, `FAIL`,
`UNCERTAIN`, or `ERROR`. It never derives or changes a status. An inspection
with no registered artifacts returns an empty `artifacts` array.

For `ERROR` or `VALIDATION_FAILED`, `error` contains a stable public code and
safe message. The established intake-failure code/message is returned when it
matches the persisted safe contract. Unknown persisted exception text is
replaced with a generic status-appropriate error, so paths, SQL, and exception
details are not exposed.

## Request IDs

`intake_request_id` is the request ID persisted when the original paired upload
was received. It can be `null` for records created without intake request
context.

`X-Request-ID` in the GET response header identifies the current details
request. A caller-supplied header is preserved; otherwise the middleware
generates a UUID. The current GET request ID never replaces
`intake_request_id` in the body.

## Artifact metadata meaning and order

Each artifact exposes only:

- `artifact_type`
- `sha256`
- `byte_size`
- `media_type`
- `created_at`

Artifacts follow the authoritative `ArtifactType` enum order: `RGB_RAW`,
`HEIGHT_RAW`, `VALIDITY_MASK`, `CALIBRATION`, `RGB_PREVIEW`, `HEIGHT_PREVIEW`,
`RESULT_OVERLAY`, then `REPORT`. Timestamp and internal row ID provide stable
tie-breaking for duplicate types, but internal IDs are never returned.

SHA-256, byte size, type, and media type describe stored registration metadata.
They do not prove image semantics, spatial alignment, physical 3D scaling,
board acceptability, or a PASS result.

The response deliberately excludes relative and absolute paths, filenames,
binary content, internal artifact database IDs, operator/audit details, model
identity or artifact paths, confidence, and inferred classification.

## Error responses

Errors use the shared structured contract and the current GET request ID:

```json
{
  "code": "INSPECTION_NOT_FOUND",
  "message": "Inspection was not found.",
  "request_id": "details-request-456"
}
```

| HTTP status | Code | Meaning |
| ---: | --- | --- |
| 400 | `INVALID_INSPECTION_ID` | The path value is not a canonical UUID. No inspection query is performed. |
| 404 | `INSPECTION_NOT_FOUND` | The canonical UUID has no persisted inspection. |
| 500 | `INSPECTION_READ_FAILED` | An unexpected repository/database read failed. The response omits exception, path, and SQL details. |

The GET operation performs repository reads only. It does not transition the
inspection, register artifacts, or append an audit event.

## Tests

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  .\backend\tests\test_inspection_details.py -q
.\.venv\Scripts\python.exe -m pytest .\backend\tests -q
```

Tests use temporary runtime roots and real temporary SQLite databases.
