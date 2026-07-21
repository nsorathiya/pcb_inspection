# Deterministic inspection development report

`GET /api/v1/inspections/{inspection_id}/report` returns contract
`pcb-aoi-inspection-development-report/1.0`. Its JSON Schema is
`contracts/inspection_development_report.schema.json`; a synthetic partial
example is under `contracts/examples/`.

This report is explicitly development-only. It is not a production inspection
certificate, production quality certificate, calibration certificate,
model-validation report, accuracy report, or legal disposition record.

## Persisted evidence only

The service loads one database snapshot containing inspection identity,
registered artifact metadata, latest validation and findings, latest processing
run and child results/findings, and inspection-owned audit events. It does not
open artifact files, read fixture manifests, recalculate source hashes, create
previews/overlays, execute validation/preprocessing/inference, add audit events,
or persist a report file. The `runtime/reports` directory remains unused by this
endpoint.

Artifact entries include type, SHA-256, byte size, media type, and registration
timestamp. They exclude paths and storage roots. Audit details use the same
safety projection as the audit endpoint. Mock inference evidence omits the
persisted null confidence field and states that no confidence was produced.

## Partial lifecycles and consistency

Reports support RECEIVED, READY, VALIDATION_FAILED, validation ERROR,
PROCESSING, mock PASS/FAIL/UNCERTAIN, and processing ERROR. Unavailable stages
are `null`; they are never fabricated. Before returning, the service checks
parent/child ownership, processing-to-validation identity, preprocessing and
inference ownership, contiguous finding ordinals, canonical result hashes,
summary counts, persisted contract semantics, null mock confidence, defect-type
rules, decisions, timestamps, and inspection-status compatibility. Inconsistent
evidence returns a safe structured 500 response.

## Canonical hashing

The serializer uses sorted keys, compact separators, UTF-8, UTC `Z` timestamps,
deterministic persisted ordering, and `allow_nan=false`. There is no generation
timestamp or random report ID. The current request ID is outside the report and
therefore outside the hash. The SHA-256 proves response reproducibility, not
production certification.

## Operator frontend

The detail page embeds the paginated Audit Timeline and links to
`/inspections/{inspectionId}/report`. “Download JSON” creates a browser Blob
from the exact received `report` object using deterministic filename
`inspection-<uuid>-development-report.json`, then revokes the object URL.
“Print” uses `window.print()` and print CSS; no PDF is generated.

Browser release verification captures the real download, checks semantic
equality with the chosen backend `report` object, and independently recalculates
`report_sha256` from the raw response using these canonicalization rules. Raw
JSON is required for this check because JavaScript number parsing does not
retain lexical forms such as `1567.0`. See
`docs/synthetic_e2e_release_verification.md`.

## PowerShell and tests

```powershell
$base = "http://127.0.0.1:8000/api/v1"
$result = Invoke-RestMethod "$base/inspections/$inspectionId/report" `
  -Headers @{ "X-Request-ID" = "development-report-read-001" }
$result.report_sha256

Set-Location .\backend
..\.venv\Scripts\python.exe -m pytest tests\test_inspection_report_api.py -q
```
