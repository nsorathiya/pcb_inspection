# Atomic Inspection-Validation Lifecycle

Date: 2026-07-17

## Purpose and boundary

`ValidationCommitService` accepts an already completed typed
`InspectionValidationResult` and its system-generated deterministic validation
key. It does not execute semantic validation, read artifact files, inspect image
content, preprocess data, run inference, or classify a PCB.

The coordinator is wired through `InspectionValidationOrchestrator` to
`POST /api/v1/inspections/{inspection_id}/validate`. Application assembly
constructs reusable services and validates the registered development policy;
neither import nor startup executes an inspection validation. The orchestrator
performs the key lookup before engine execution and passes completed typed
results to this coordinator.

## Atomic database effects

One SQLite transaction coordinates all four lifecycle effects:

1. immutable validation-result persistence;
2. every deterministically ordered finding;
3. one guarded inspection-status transition; and
4. one safe lifecycle audit event.

The coordinator starts with `BEGIN IMMEDIATE`, which acquires SQLite's writer
reservation before lifecycle reads. This serializes competing writers across
connections and processes. It then uses a conditional update whose predicate
requires the inspection ID and `status = RECEIVED`; exactly one updated row is
required. No in-process lock is used as the lifecycle guard.

Any insertion, finding, transition, audit, or commit failure rolls the
transaction back. Artifact rows and filesystem files are outside the mutation
surface and remain unchanged.

## Allowed transitions and fields

Only these transitions exist:

| Validation outcome | Transition | Completion and error fields |
| --- | --- | --- |
| `VALIDATION_PASSED` | `RECEIVED` to `READY` | `completed_at`, `error_code`, and `error_message` remain null. |
| `VALIDATION_FAILED` | `RECEIVED` to `VALIDATION_FAILED` | Validation completion timestamp, `INPUT_VALIDATION_FAILED`, and a generic safe message. |
| `VALIDATION_ERROR` | `RECEIVED` to `ERROR` | Validation completion timestamp, `VALIDATOR_INTERNAL_ERROR`, and a generic safe message. |

The coordinator cannot produce `PROCESSING`, `PASS`, `FAIL`, or `UNCERTAIN`.
It does not alter board, recipe, model, confidence, processing-time, or artifact
metadata. Finding messages and internal exception text are never copied to the
inspection error message.

`READY` means the raw pair is technically ready to continue under the selected
validation policy. It is not PCB PASS and contains no product-quality decision.

## Audit actions and safe details

Exactly one new lifecycle coordination appends one of:

- `INSPECTION_VALIDATION_PASSED`;
- `INSPECTION_VALIDATION_FAILED`; or
- `INSPECTION_VALIDATION_ERROR`.

The event targets entity type `inspection`. Optional actor and request IDs are
preserved after bounded safe-text validation. Details contain only validation
ID and key, outcome, resulting status, policy identity, validator version,
result SHA-256, and total, blocking, and warning finding counts. They exclude
artifact paths, filenames, finding details, binaries, SQL, stack traces, and
exception representations.

## Idempotency and standalone adoption

Task-17 standalone persistence remains available for immutable technical
evidence that must not affect inspection lifecycle. Application code that must
coordinate status and audit should use `ValidationCommitService`.

The coordinator handles the validation key as follows:

- No existing key: insert result/findings, transition `RECEIVED`, and audit.
- Same key and result hash while `RECEIVED`: adopt the standalone evidence,
  transition, and audit without duplicating result or findings.
- Same key and result hash at the expected target: return full idempotent
  success without another transition or audit.
- Same key with a different result hash: raise
  `ValidationPersistenceConflictError` and change nothing.
- Same immutable result in an incompatible status: raise
  `ValidationLifecycleConsistencyError` and change nothing.
- A different key after the inspection leaves `RECEIVED`: reject and roll back
  the new evidence.

Revalidation is unsupported because there is no approved policy for superseding
immutable validation evidence, reopening terminal lifecycle state, or relating
multiple attempts. A future version must define that policy explicitly.

## Typed result and errors

`ValidationCommitResult` reports validation/inspection identities, key, result
hash, validation outcome, resulting inspection status, persistence and lifecycle
idempotency flags, nullable audit action, and commit timestamp. It contains no
ORM objects, paths, SQL, connection details, or internal exceptions.

Public lifecycle errors include:

- `InspectionNotFoundError`;
- `ValidationCommitConflictError`;
- `InvalidInspectionTransitionError`;
- `ValidationLifecycleConsistencyError`; and
- the existing `ValidationPersistenceConflictError` for key/hash disagreement.

Other transaction failures are converted to the safe base
`ValidationCommitError` after rollback.

## API integration

The validation API is documented in `docs/inspection_validation_api.md`.
POST explicitly selects the registered development policy, replays/adopts
existing immutable evidence before reading image bytes, and invokes this
service for all lifecycle mutations. GET reads the latest persisted result and
ordered findings without executing validation or writing the database.
Authentication remains absent; actor ID is null and request ID comes from the
existing correlation middleware.

## Tests

From the repository root in Windows PowerShell:

```powershell
python -m pytest .\backend\tests\test_inspection_validation_lifecycle.py -q
python -m pytest .\backend\tests\test_inspection_validation_persistence.py -q
python -m pytest .\backend\tests\test_inspection_semantic_validation_engine.py -q
python -m pytest .\backend\tests -q
```
