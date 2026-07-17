# Inspection Technical-Validation API

Date: 2026-07-17

## Scope

Two endpoints expose the existing paired RGB/height technical validator:

- `POST /api/v1/inspections/{inspection_id}/validate`
- `GET /api/v1/inspections/{inspection_id}/validation`

They do not preprocess images, run AI inference, detect defects, or decide PCB
quality. `VALIDATION_PASSED` means only that the registered raw pair is
technically ready for a future preprocessing stage under the selected policy.
It never means PCB PASS.

## Explicit development policy

POST requires this JSON body:

```json
{
  "policy_id": "development-native-rgb-height",
  "policy_version": "1.0"
}
```

This is the one explicitly registered repository policy. It is marked
`development_only=true` and is not approved for production acceptance. The
server validates it against the authoritative policy schema during application
assembly. Policy selection is never inferred from board ID, recipe ID,
filename, environment, or a client filesystem path.

Policy identifiers reject blank values, control characters, unsupported
characters, and excessive lengths. The request cannot supply validation IDs,
validation keys, outcomes, findings, actors, or policy documents.

## POST example

```powershell
$inspectionId = "36a7a10f-3c74-4af8-b0e2-718ad1065e45"
$headers = @{ "X-Request-ID" = "technical-validation-123" }
$body = @{
  policy_id = "development-native-rgb-height"
  policy_version = "1.0"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/inspections/$inspectionId/validate" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

Only an inspection in `RECEIVED` may begin a new lifecycle. Revalidation is not
supported.

## POST execution and system idempotency

Before file bytes are read, the application:

1. validates the canonical inspection UUID and explicit policy selection;
2. reads the inspection and registered artifact identities;
3. calculates the authoritative deterministic validation key; and
4. looks up an existing result for that inspection and key.

The key includes inspection ID; RGB and height SHA-256 and size; relevant mask
and calibration evidence identities; validation contract version; policy ID
and version; and validator version. It excludes paths, request/actor identity,
timestamps, and generated validation ID.

An exact retry returns the existing validation ID, key, result, and current
inspection status with `idempotent_existing=true`. It does not rerun the
semantic engine, duplicate findings, or add an audit event. Standalone
persisted evidence on a `RECEIVED` inspection is adopted atomically without
rerunning. A changed artifact identity, policy, or validator version is a
different validation identity and is rejected after the inspection leaves
`RECEIVED`.

When no result exists, the read-only semantic engine inspects the stored bytes.
`ValidationCommitService` then atomically persists the result and ordered
findings, applies the guarded status transition, and appends one lifecycle
audit event.

## Success response

All completed technical outcomes return HTTP 200. The response contains the
safe typed result fields plus persisted inspection state:

```json
{
  "inspection_id": "36a7a10f-3c74-4af8-b0e2-718ad1065e45",
  "validation_id": "57c54044-5dd8-442f-8e32-9af29eeef420",
  "validation_key": "<lowercase-sha256>",
  "validation_outcome": "VALIDATION_PASSED",
  "inspection_status": "READY",
  "policy": {
    "policy_id": "development-native-rgb-height",
    "policy_version": "1.0"
  },
  "validator_version": "1.0.0",
  "started_at": "2026-07-17T12:00:00Z",
  "completed_at": "2026-07-17T12:00:01Z",
  "summary": {
    "finding_count": 0,
    "info_count": 0,
    "warning_count": 0,
    "error_count": 0,
    "blocking_count": 0,
    "technically_ready": true,
    "synthetic_example": false
  },
  "artifacts": {
    "rgb": {
      "artifact_type": "RGB_RAW",
      "sha256": "<lowercase-sha256>",
      "byte_size": 1234,
      "declared_media_type": "image/png",
      "detected_format": "PNG",
      "width": 128,
      "height": 128,
      "channels": 3,
      "bit_depth": 8,
      "storage_data_type": null,
      "readability_status": "READABLE"
    },
    "height": {
      "artifact_type": "HEIGHT_RAW",
      "sha256": "<lowercase-sha256>",
      "byte_size": 5678,
      "declared_media_type": "image/tiff",
      "detected_format": "TIFF",
      "width": 128,
      "height": 128,
      "channels": 1,
      "bit_depth": 16,
      "storage_data_type": "uint16",
      "readability_status": "READABLE"
    }
  },
  "findings": [],
  "idempotent_existing": false,
  "request_id": "technical-validation-123"
}
```

Responses never include storage paths, filenames, binary content, internal
artifact IDs, SQL, model identity, confidence, model output, or PCB
classification.

## Completed failed and error outcomes

`VALIDATION_FAILED` is a completed technical result, returns HTTP 200, contains
one or more blocking input/policy findings, and transitions the inspection to
`VALIDATION_FAILED`.

```json
{
  "validation_outcome": "VALIDATION_FAILED",
  "inspection_status": "VALIDATION_FAILED",
  "summary": {"blocking_count": 1, "technically_ready": false}
}
```

`VALIDATION_ERROR` is also a completed typed result and returns HTTP 200. It
contains the safe `VALIDATOR_INTERNAL_ERROR` finding and transitions the
inspection to `ERROR`. Parser exception strings and stack traces are not
returned.

```json
{
  "validation_outcome": "VALIDATION_ERROR",
  "inspection_status": "ERROR",
  "findings": [{"code": "VALIDATOR_INTERNAL_ERROR", "blocking": true}]
}
```

The abbreviated examples above show outcome semantics; actual responses use
the complete typed schema shown in the success example.

## GET latest result

```powershell
$inspectionId = "36a7a10f-3c74-4af8-b0e2-718ad1065e45"
$headers = @{ "X-Request-ID" = "technical-validation-get-456" }
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/inspections/$inspectionId/validation" `
  -Headers $headers
```

GET returns the latest persisted result, current inspection status, and
findings in stored ordinal order. It uses the same safe response mapper as
POST. It does not execute validation, read artifact files, verify current file
availability, update status, or append audit. Because revalidation is not
supported, an inspection normally has at most one lifecycle result; latest
ordering remains deterministic for standalone historical data.

## Status transitions

| Technical outcome | Inspection transition | Meaning |
| --- | --- | --- |
| `VALIDATION_PASSED` | `RECEIVED` to `READY` | Technically ready for future preprocessing only. |
| `VALIDATION_FAILED` | `RECEIVED` to `VALIDATION_FAILED` | Input or policy is not technically ready. |
| `VALIDATION_ERROR` | `RECEIVED` to `ERROR` | The validator could not complete reliably. |

No endpoint here produces `PROCESSING`, `PASS`, `FAIL`, or `UNCERTAIN`.

## Errors

Errors use the shared safe structure:

```json
{
  "code": "STABLE_ERROR_CODE",
  "message": "Safe message.",
  "request_id": "current-request-id"
}
```

| HTTP | Representative codes and meaning |
| ---: | --- |
| 400 | `INVALID_INSPECTION_ID`, `INVALID_VALIDATION_POLICY_SELECTION` |
| 404 | `INSPECTION_NOT_FOUND`, `VALIDATION_POLICY_NOT_FOUND`, `VALIDATION_POLICY_VERSION_UNSUPPORTED`, `INSPECTION_VALIDATION_NOT_FOUND` |
| 409 | `INSPECTION_NOT_ELIGIBLE_FOR_VALIDATION`, `VALIDATION_LIFECYCLE_CONFLICT` |
| 422 | `INVALID_VALIDATION_REQUEST` for missing/invalid JSON body shape |
| 500 | `VALIDATION_ORCHESTRATION_FAILED` when no authoritative lifecycle result could be completed/read reliably |

A completed `VALIDATION_FAILED` or `VALIDATION_ERROR` is not an HTTP error.

## Request IDs

The POST audit uses the current middleware `request_id`; authentication and
actor identity do not exist yet, so `actor_id` remains null. The response body
and `X-Request-ID` header use the current request ID. A GET uses its own current
request ID and never rewrites the request ID already stored in lifecycle audit
evidence.

## Current limitations

- Revalidation and result supersession are not defined.
- The development policy is not production-approved.
- A `RECEIVED` row with missing or ambiguous registered raw identities cannot
  form the required deterministic key and fails with a safe consistency
  conflict before execution.
- SQLite `BEGIN IMMEDIATE` is the final writer/concurrency authority. Concurrent
  exact work can consume duplicate read-only validation effort before both
  callers converge on one committed lifecycle.
- A process crash during read-only file inspection leaves no lifecycle result;
  a crash during the database transaction is rolled back by SQLite.

## Tests

```powershell
python -m pytest .\backend\tests\test_inspection_validation_api.py -q
python -m pytest .\backend\tests\test_inspection_semantic_validation_engine.py `
  .\backend\tests\test_inspection_validation_persistence.py `
  .\backend\tests\test_inspection_validation_lifecycle.py -q
python -m pytest .\backend\tests -q
```
