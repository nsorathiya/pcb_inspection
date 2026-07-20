# Inspection Processing Persistence and Lifecycle

## Scope and safety status

Schema version 3 adds persistence and lifecycle coordination for already
completed `InspectionPreprocessingResult` and `InspectionInferenceResult`
objects. The coordinator never executes preprocessing or inference, reads no
artifact files or buffer bytes, and is not exposed by an HTTP endpoint.

The supported engine type is `MOCK`. Stored mock PASS, FAIL, and UNCERTAIN
values exercise workflow semantics only. They are not production PCB decisions
and make no model-accuracy or confidence claim.

## Schema version 3 tables

- `inspection_processing_runs` owns one guarded attempt per inspection and
  deterministic processing key. It records policy, implementation, and engine
  identities plus STARTED/COMPLETED/ERROR lifecycle fields.
- `inspection_preprocessing_results` stores one immutable canonical result per
  run; `inspection_preprocessing_result_findings` stores findings by ordinal.
- `inspection_inference_results` stores at most one immutable canonical result
  per run and references its preprocessing result;
  `inspection_inference_result_findings` stores ordered findings.

There are no public update or delete repository methods for these result or
finding tables. Only the lifecycle service can conditionally update a STARTED
run and its PROCESSING inspection.

## Deterministic processing identity

`generate_processing_key(...)` serializes sorted, compact UTF-8 JSON and hashes
the exact bytes with lowercase SHA-256. Inputs include inspection and
validation IDs, validation result hash, RGB/height raw SHA-256 and size, sorted
optional evidence identities, preprocessing policy and implementation
identity, and inference policy and engine identity/type.

Timestamps, actor/request IDs, run/result IDs, paths, buffers, and bytes are
excluded. Changing raw identity or any policy, implementation, or engine
version changes the key.

## Canonical result hashes

The result serializers use the typed contracts, UTC `Z` timestamps, validated
catalogue finding order, sorted keys, compact separators, UTF-8, and
`allow_nan=False`. Result SHA-256 values are calculated from those exact bytes
before SQLite storage. SQLite JSON rendering is never the hash authority.
Schema semantics, summary counts, identities, taxonomy values, confidence-null
behavior, and path-free finding details are validated without repair.

## Atomic begin transaction

`ProcessingLifecycleService.begin_processing(...)` uses one `AsyncSession`,
`BEGIN IMMEDIATE`, and this authoritative conditional update:

```sql
UPDATE inspections
SET status = 'PROCESSING'
WHERE id = ? AND status = 'READY';
```

It verifies the inspection, passed validation, ownership, and validation hash;
inserts one STARTED run; applies exactly one transition; appends
`INSPECTION_PROCESSING_STARTED`; and commits together. Exact-key replay returns
the existing lifecycle without another audit. A competing key cannot leave a
losing run row.

## Atomic completion and status mapping

Completion accepts already-produced typed results only. One transaction stores
results/findings, conditionally completes the STARTED run, conditionally moves
the inspection from PROCESSING, and appends one final audit.

| Technical result | Run | Inspection | Audit action |
|---|---|---|---|
| Inference succeeded + PASS | COMPLETED | PASS | `INSPECTION_MOCK_RESULT_PASS` |
| Inference succeeded + FAIL | COMPLETED | FAIL | `INSPECTION_MOCK_RESULT_FAIL` |
| Inference succeeded + UNCERTAIN | COMPLETED | UNCERTAIN | `INSPECTION_MOCK_RESULT_UNCERTAIN` |
| Preprocessing failed/error | ERROR | ERROR | `INSPECTION_PROCESSING_ERROR` |
| Inference failed/error | ERROR | ERROR | `INSPECTION_PROCESSING_ERROR` |

Preprocessing alone never yields a PCB decision. Successful preprocessing
requires an inference result to complete. Failed/error preprocessing rejects an
inference result. Technical failures use stable codes and generic messages;
finding messages and exception text are never copied into inspection errors.
Mock confidence remains null.

## Idempotency, concurrency, and rollback

SQLite writer serialization plus conditional updates provide cross-session and
cross-process guards; no Python lock is authoritative. Exact result-hash replay
returns idempotently only when run and inspection states are consistent. A
different hash conflicts and immutable rows are not overwritten. Reprocessing
and multiple attempts are unsupported.

Any result, finding, run update, inspection transition, or audit failure rolls
back the entire transaction. Artifact records and files are not modified. Back
up the SQLite database, WAL, and SHM consistently before applying migration 3
in a deployed runtime.

## Trusted internal execution integration

`InspectionProcessingOrchestrator` now provides the separate trusted execution
boundary that this lifecycle intentionally does not implement. It verifies an
injected generator-owned synthetic manifest, performs exact-key replay before
file reads, uses `begin_processing()` and `complete_processing()` as the only
mutation paths, and invokes the existing synthetic preprocessing and mock
inference services only for the winning new run. The lifecycle's transaction,
status, audit, rollback, and idempotency rules are unchanged. See
`docs/synthetic_processing_orchestrator.md`.

## Current transport boundary

No API route, startup execution hook, report, preview, model load, or frontend
integration is added. A later task may adapt a reviewed transport to the
internal orchestrator while preserving this typed lifecycle boundary.

## PowerShell tests

```powershell
python -m pytest .\backend\tests\test_inspection_processing_lifecycle.py -q
python -m pytest .\backend\tests\test_synthetic_processing_orchestrator.py -q
python -m pytest .\backend\tests -q
```
