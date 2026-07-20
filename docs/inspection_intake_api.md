# Paired 2D/3D Inspection-Intake API

Date: 2026-07-17

## Scope and meaning of RECEIVED

`POST /api/v1/inspections` accepts one RGB image and one native height/depth
file for the same physical inspection instance. A successful request creates a
single `RECEIVED` inspection and immutably stores and registers both raw files.

`RECEIVED` means only that the pair and its intake metadata were accepted and
stored. It does not mean that either file has been decoded, that dimensions or
bit depth are valid, that 2D and 3D are aligned, that the pair is ready for
training, that production acceptance has occurred, or that the inspection is
PASS, FAIL, or UNCERTAIN. Preprocessing, preview generation, alignment,
inference, and classification are not part of this endpoint.

The caller is responsible for submitting files from the same physical
inspection instance. This intake gate does not prove that relationship.

## Multipart request

Content type: `multipart/form-data`

FastAPI uses the declared `python-multipart` dependency to stream and parse the
form. No image-decoding or ML package is required by this endpoint.

Required file fields:

| Field | Intake extensions |
| --- | --- |
| `rgb_image` | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` |
| `height_map` | `.png`, `.tif`, `.tiff`, `.npy` |

Required text fields:

- `board_id`
- `recipe_id`
- `recipe_version`

Persisted recipe identities and display names can be listed with the read-only,
cursor-paginated `GET /api/v1/recipes` catalogue. Clients should submit the
selected `recipe_id` and `recipe_version` unchanged. Intake continues to store
the submitted identity without resolving the recipe table or checking recipe
status. Catalogue listing therefore does not prove model compatibility,
calibration validity, production approval, or suitability for the board. See
`docs/recipe_catalogue_api.md`.

Optional text fields:

- `lot_id`
- `operator_id`
- `station_id`
- `rgb_sha256`
- `height_sha256`
- `rgb_byte_size`
- `height_byte_size`

Identifiers are trimmed, limited to 128 characters, and reject control
characters. Expected hashes must be 64 lowercase hexadecimal characters.
Expected sizes must be non-negative decimal integers. Known multipart fields
must occur at most once. The server generates the inspection UUID; clients
cannot provide one.

The original filenames are informational. They never provide an inspection
directory or stored filename stem. Filename extension and media-type checks are
a conservative intake gate only; they do not prove the bytes use the declared
format. `station_id` is retained as safe intake-audit metadata because the
current inspection schema has no station column.

## PowerShell example

```powershell
$uri = "http://127.0.0.1:8000/api/v1/inspections"
$form = @{
  board_id = "PCB_A"
  recipe_id = "PCB_A"
  recipe_version = "1.0"
  lot_id = "LOT_2026_001"
  operator_id = "operator-1"
  station_id = "station-1"
  rgb_image = Get-Item "C:\intake\rgb.png"
  height_map = Get-Item "C:\intake\height.npy"
}
Invoke-RestMethod -Method Post -Uri $uri -Form $form
```

PowerShell 7 or later supports `Invoke-RestMethod -Form`. Hash and size fields
may be added when the caller has authoritative values for the exact uploaded
bytes.

## Success response

HTTP status: `201 Created`

```json
{
  "inspection_id": "36a7a10f-3c74-4af8-b0e2-718ad1065e45",
  "status": "RECEIVED",
  "board_id": "PCB_A",
  "recipe_id": "PCB_A",
  "recipe_version": "1.0",
  "lot_id": "LOT_2026_001",
  "request_id": "caller-or-generated-request-id",
  "created_at": "2026-07-17T12:00:00Z",
  "artifacts": [
    {
      "artifact_type": "RGB_RAW",
      "sha256": "<lowercase-sha256>",
      "byte_size": 12345,
      "media_type": "image/png"
    },
    {
      "artifact_type": "HEIGHT_RAW",
      "sha256": "<lowercase-sha256>",
      "byte_size": 54321,
      "media_type": "application/octet-stream"
    }
  ]
}
```

The response intentionally omits storage paths, model metadata, confidence,
classification, preview URLs, and internal artifact row IDs. The existing
`X-Request-ID` response header is returned and matches `request_id`.

Persisted intake and artifact metadata can later be retrieved with
`GET /api/v1/inspections/{inspection_id}`. That read-only contract, including
the distinction between the original intake request ID and the current GET
request ID, is documented in `docs/inspection_details_api.md`.

Persisted inspections can also be browsed newest-first through the read-only,
cursor-paginated `GET /api/v1/inspections` history route. It does not reopen
uploads or rerun intake, validation, or processing. See
`docs/inspection_history_api.md`.

A `RECEIVED` inspection can be technically validated with
`POST /api/v1/inspections/{inspection_id}/validate` using the explicit
development policy. Validation is a separate lifecycle operation: intake does
not run it, and technical readiness never means PCB PASS. See
`docs/inspection_validation_api.md`.

## Size and integrity enforcement

The existing storage limits apply while each upload stream is written:

- `PCB_AOI_MAX_RGB_BYTES`
- `PCB_AOI_MAX_HEIGHT_BYTES`
- `PCB_AOI_MAX_MASK_BYTES`
- `PCB_AOI_MAX_CALIBRATION_BYTES`
- `PCB_AOI_MAX_GENERATED_ARTIFACT_BYTES`

Only the RGB and height limits are used by this endpoint. Upload files are
streamed through a worker-thread filesystem boundary; the endpoint does not
read the entire upload into application memory. SHA-256 and size are calculated
over exact bytes. Supplied expectations must match before intake succeeds.
Upload streams are closed after both success and handled failure.

## Error response

Errors contain only a stable code, safe message, and request ID:

```json
{
  "code": "ARTIFACT_INTEGRITY_MISMATCH",
  "message": "Artifact integrity check failed.",
  "request_id": "caller-or-generated-request-id"
}
```

Status mapping:

| Status | Examples |
| ---: | --- |
| 400 | Empty/malformed identifier, hash or size; duplicate field; unsupported extension or media type; expected integrity mismatch |
| 409 | Immutable artifact conflict |
| 413 | Configured RGB or height size limit exceeded |
| 422 | Missing required form field or incomplete file pair |
| 500 | Unexpected storage, inspection creation, audit, or database registration failure |

Responses do not include absolute paths, SQL, exception text, tracebacks, or
uploaded content.

## Pair compensation and audit behavior

The coordinator creates the inspection first, stores/registers RGB, then
stores/registers height. A successful operation appends
`INSPECTION_RECEIVED`, including artifact types and sizes plus optional station
metadata.

If a handled failure occurs after inspection creation:

1. Exact artifact rows created by the current operation are removed in reverse
   order.
2. A file is removed only when the operation created it and its current hash
   and size still match.
3. Pre-existing idempotent artifacts are never deleted.
4. The inspection transitions only from `RECEIVED` to `ERROR`, records
   `completed_at`, and receives a generic error code/message.
5. `INSPECTION_INTAKE_FAILED` is appended with a generic failure category and
   compensation status.
6. The request never returns HTTP 201 for that inspection.

SQLite and the filesystem cannot share one atomic transaction. A process,
machine, storage-device, or database failure in the narrow intervals between
file finalization, row registration, status transition, and audit insertion can
still leave an orphan or incomplete compensation record. The endpoint fails
closed, but future reconciliation tooling is required for crash recovery.

## Retry and idempotency limitation

There is no idempotency-key contract yet. Retrying a request generates a new
inspection UUID and may create a new inspection. The endpoint does not
deduplicate by image hash because two real physical inspections can
legitimately produce identical bytes.

## Tests

From the repository root in Windows PowerShell:

```powershell
python -m pytest .\backend\tests\test_inspection_intake.py -q
python -m pytest .\backend\tests\test_health.py `
  .\backend\tests\test_database.py `
  .\backend\tests\test_artifact_storage.py -q
python -m pytest .\backend\tests -q
```

Tests use temporary runtime roots and real temporary SQLite databases.
