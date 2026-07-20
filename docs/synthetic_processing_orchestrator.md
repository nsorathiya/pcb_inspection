# Trusted Synthetic Processing Orchestrator

## Purpose and boundary

`InspectionProcessingOrchestrator` is an internal, FastAPI-independent boundary
for executing the existing development-only synthetic preprocessing and
deterministic mock-inference services. It accepts only an inspection UUID and
exact preprocessing/inference policy identities. Callers cannot supply status,
validation outcome, a synthetic flag, a processing key, run/result IDs,
decisions, confidence, buffers, or filesystem paths.

This boundary does not implement preprocessing, inference, persistence, status
transitions, or audit logic. It coordinates the existing services and delegates
all mutations to `ProcessingLifecycleService`.

Mock `PASS`, `FAIL`, and `UNCERTAIN` values are deterministic software-workflow
outputs. They are synthetic, mock, nonproduction, confidence-free, and are not
production PCB inspection decisions or model predictions.

## Architecture

Responsibilities remain split across focused modules:

- `input_builder.py` reads and cross-checks persisted inspection, latest passed
  validation, and registered artifact identities. It also performs the narrow
  execution-time managed-path and SHA-256/size preflight.
- `provenance.py` verifies one explicitly injected generator-owned fixture root.
- Existing policy loaders validate exact repository-owned policy identities.
- Existing `generate_processing_key()` creates the canonical technical key.
- `assembly.py` builds existing typed inputs, creates schema-shaped internal
  error results after a begin, validates persisted replay evidence, and maps a
  path-free result.
- `orchestrator.py` sequences these components and invokes the existing
  preprocessing, inference, and lifecycle services.

No in-process lock is authoritative. SQLite writer serialization, the unique
inspection/key constraint, and guarded lifecycle transitions select the winner.

## Trusted synthetic provenance

The fixture root is an application-owned constructor dependency. It is never a
method argument and is not supplied by an HTTP or client field. The verifier
does not scan arbitrary directories. It reads only the fixed ownership files
and the paths declared by their inventory.

Verification requires:

- the exact `SYNTHETIC_FIXTURES_MARKER.json` shape, supported generator identity
  and version, and nonapproval flags;
- the marker SHA-256 over the exact `generation_manifest.json` bytes;
- the exact generation-manifest shape, supported contract, generator and seed;
- a safe, unique inventory whose files match recorded SHA-256 and byte size;
- the canonical output-tree digest;
- every declared `scenario.json` validating against the versioned scenario
  schema and agreeing with generator, seed, synthetic, training, production,
  and model-accuracy-evidence claims;
- every scenario artifact's actual identity agreeing with both its bytes and
  the generation inventory; and
- exactly one scenario whose RGB, height, and registered optional-evidence
  SHA-256/size identities match the database records.

Filename, inspection/board/recipe ID, directory name, media type, or a caller
boolean cannot establish provenance. Public results and errors contain no
fixture or source path.

## Prerequisites and policy selection

The reader requires a canonical existing inspection, a supported persisted
`VALIDATION_PASSED` result belonging to it, exactly one `RGB_RAW`, exactly one
`HEIGHT_RAW`, and deterministic optional-evidence registration. It does not
rerun semantic validation.

Only exact caller selections accepted by the existing loaders are allowed:

- preprocessing `synthetic-paired-rgb-height` version `1.0`;
- inference `synthetic-deterministic-mock-inference` version `1.0`.

Unknown, unsupported, malformed, or unsafe policy identities fail before any
processing mutation. No recipe, board, filename, or filesystem policy path is
used for selection.

The current schema-v3 processing key includes the persisted validation result
hash; registered RGB, height, and evidence identities; loaded policy
identities; preprocessing implementation identity; and mock-engine identity.
It excludes paths, scenario names, timestamps, actors, request IDs, generated
run/result IDs, buffers, and buffer bytes.

## Replay and concurrency

The exact processing key is looked up before provenance or source-file access.
A completed or technical-error run is reconstructed from canonical persisted
result JSON plus its ordered finding rows. Contract, hashes, scalar columns,
child presence, safety flags, final state, and inspection status must agree.
Replay performs no lifecycle call, audit, manifest verification, file preflight,
preprocessing, or inference.

A matching `STARTED` run with a `PROCESSING` inspection raises the typed
already-in-progress conflict. Missing or inconsistent child evidence raises a
typed consistency error and is never overwritten. A concurrent different key
finds the inspection already `PROCESSING` or loses the guarded begin; no losing
run is persisted.

## New execution lifecycle

For a new trusted READY input, the orchestrator calls only
`ProcessingLifecycleService.begin_processing()`. After that guarded transaction
creates the run, changes READY to PROCESSING, and appends the start audit, the
orchestrator:

1. resolves managed artifact paths beneath their expected runtime category;
2. rejects traversal, absolute/drive paths, links, reparse points, missing or
   nonregular files;
3. rechecks current SHA-256 and byte size without copying or converting bytes;
4. constructs the existing `ValidatedInspectionInput` from persisted technical
   summaries and verified provenance;
5. invokes `SyntheticInspectionPreprocessingService` exactly once;
6. retains its RGB/height buffers only in memory;
7. when preprocessing succeeds, constructs `SyntheticInferenceInput` without
   rereading artifacts and invokes `SyntheticMockInferenceService` exactly once;
8. passes only completed typed results to
   `ProcessingLifecycleService.complete_processing()`.

Preprocessing failed/error skips inference. Inference failed/error and all
preprocessing technical failures finish through the lifecycle as inspection and
run `ERROR`. Successful mock decisions finish as PASS, FAIL, or UNCERTAIN under
the existing lifecycle rules.

## Unexpected failures and operational recovery

Before begin, safe typed errors leave the inspection READY and create no run or
audit. After begin, file-preflight, input-assembly, or unexpected executor
failures are converted—when identities are sufficient—into schema-valid
`PREPROCESSING_ERROR` or `INFERENCE_ERROR` results containing only the existing
generic internal finding. The lifecycle service remains authoritative for the
ERROR transition and audit.

If lifecycle error completion itself fails, the orchestrator raises
`ProcessingExecutionRecoveryRequiredError`. The inspection may remain
PROCESSING and requires operational recovery. There is deliberately no second
ad hoc update path.

## Source immutability and current limitations

Execution creates no preview, report, normalized file, adjacent temporary file,
or persisted preprocessing buffer. Registered artifacts, validation evidence,
and fixture manifests are read-only.

The current validation persistence contract stores full technical summaries for
RGB and height only. Although optional mask/calibration identities are included
in provenance and key construction, the orchestrator rejects such a new run
before begin rather than fabricate the technical `ArtifactInputIdentity` fields
required by the preprocessing result contract. A future contract task may add
authoritative persisted evidence summaries before optional evidence is executed.

The development-only processing POST endpoint is now a thin application-owned
adapter over this orchestrator. Its trusted fixture root comes only from
application configuration, never from the request. No startup execution,
background job, model/checkpoint load, real-data path, confidence, preview,
report, schema change, or migration is included. See
`docs/inspection_processing_api.md` for the transport contract.

## Internal Python usage

```python
orchestrator = InspectionProcessingOrchestrator(
    repositories,
    runtime_paths,
    trusted_fixture_root,  # application-owned dependency, never client input
    ProcessingLifecycleService(
        database.session_factory,
        repository=repositories.processing,
    ),
)

result = await orchestrator.execute_processing(
    inspection_id,
    "synthetic-paired-rgb-height",
    "1.0",
    "synthetic-deterministic-mock-inference",
    "1.0",
    actor_id=actor_id,
    request_id=request_id,
)
```

Future API integration should adapt authenticated/authorized transport input to
this narrow method, inject the manifest root from application-owned
configuration, map typed errors safely, and leave every trust decision and
lifecycle mutation inside the existing internal boundaries.

## Tests

```powershell
python -m pytest .\backend\tests\test_synthetic_processing_orchestrator.py -q
python -m pytest .\backend\tests -q
```
