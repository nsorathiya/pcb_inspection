# Versioned Deterministic Mock Inference Engine

## Purpose and development-only status

The `inspection_inference` package defines the replaceable boundary after
successful RGB/height preprocessing and implements its first engine for
software-flow testing. `SyntheticMockInferenceService` accepts the separate
immutable buffers produced by `SyntheticInspectionPreprocessingService`,
validates their bytes and descriptors, and returns a path-free
`pcb-aoi-inspection-inference/1.0` result.

This implementation is deterministic, synthetic-only, mock-only, in-memory,
and development-only. No trained model analyzes the buffers. Its `PASS`,
`FAIL`, and `UNCERTAIN` values are mock UI/workflow selections, not PCB
quality decisions, defect detection, model predictions, calibrated
confidence, performance evidence, or production acceptance.

## Versioned contracts

- Policy: `pcb-aoi-inspection-inference-policy/1.0`
- Result: `pcb-aoi-inspection-inference/1.0`
- Finding catalogue: `pcb-aoi-inspection-inference-findings/1.0`
- Policy identity: `synthetic-deterministic-mock-inference` version `1.0`
- Engine identity: `synthetic-deterministic-mock-engine` version `1.0.0`

The policy is loaded only from the repository-owned example selected by exact
ID and version. It is validated against JSON Schema and cross-field rules. It
cannot permit real input, production decisions, model-accuracy claims, or a
confidence mode.

## Technical outcome versus mock decision

`execution_outcome` describes whether the inference software ran reliably:

- `INFERENCE_SUCCEEDED`: both buffers and prerequisites were accepted and a
  deterministic mock decision was generated.
- `INFERENCE_FAILED`: a known prerequisite, policy, buffer, or pair
  incompatibility blocked execution.
- `INFERENCE_ERROR`: an unexpected internal failure prevented reliable
  completion.

Only `INFERENCE_SUCCEEDED` has a decision. The separate decision is `PASS`,
`FAIL`, or `UNCERTAIN`. Failed and error results have null decision fields.
Preprocessing success does not imply inference success, and mock inference
success does not imply PCB PASS.

## Replaceable interfaces

The package exposes protocols for:

- `InferencePolicyLoader`;
- `InferenceInputValidator`;
- `InferenceEngine`;
- `InferenceOrchestrator`; and
- the future `InferenceResultSink` boundary.

Callers depend on the orchestrator and engine protocols rather than a model
framework. A later reviewed `OnnxInferenceEngine` can implement the engine
protocol without changing orchestrator callers. No ONNX engine, model loader,
registry, checkpoint handling, or result sink exists in this task.

## Input buffer validation

Before selection, the validator independently checks the actual RGB and height
buffers instead of trusting their descriptors. It requires:

- `PREPROCESSING_SUCCEEDED`, explicitly synthetic input, and mock
  preprocessing;
- immutable byte buffers with matching stored and calculated SHA-256;
- descriptor shape, channel count, width, height, element count, and byte
  length agreement;
- contiguous little-endian finite float32 `CHW` data;
- RGB shape `[3,H,W]` and height shape `[1,H,W]`;
- a lowercase source-artifact SHA-256 identity; and
- matching spatial dimensions under the mock policy.

The engine reads neither original artifact nor preprocessing source paths.
Public results expose only hashes, shapes, layout, type, dimensions, channel
count, byte size, and source-artifact hash. They never expose buffer bytes.

## Canonical decision digest

The engine builds compact UTF-8 JSON with sorted keys from only:

- inspection, validation, and preprocessing IDs;
- RGB/height buffer SHA-256, shape, layout, and data type;
- policy ID and version;
- engine ID and version; and
- the decision strategy.

It serializes with separators `(',', ':')`, calculates SHA-256 over those
bytes, and stores the lowercase hexadecimal result as `decision_digest`.
Timestamps, inference ID, request ID, actor identity, paths, and mutable
runtime state are excluded.

## Exact bucket algorithm

Policy version `1.0` defines 12 complete, non-overlapping buckets:

- buckets `0`, `1`, `2`, `3`: mock `PASS`;
- buckets `4`, `5`, `6`, `7`: mock `FAIL`;
- buckets `8`, `9`, `10`, `11`: mock `UNCERTAIN`.

The selector interprets the first 16 hexadecimal characters (64 bits) of
`decision_digest` as an unsigned integer and calculates:

```text
bucket = integer(decision_digest[0:16], base=16) % decision_bucket_count
```

Digest bucketing does not inspect pixels, height values, defects, components,
or PCB quality. It is only a reproducible demonstration selector.

## Mock FAIL taxonomy assignment

For mock `FAIL`, the engine reads the supported NOK defect types from the
authoritative `contracts/defect_taxonomy.json`; it does not maintain a second
class list. It calculates:

```text
defect_digest = SHA-256(decision_digest + ":" + taxonomy_version)
defect_index = integer(defect_digest[0:16], base=16) % defect_count
```

The selected taxonomy value is a deterministic UI/demo label only. It is not
image-content detection. `no_defect` is not a defect type; mock `PASS` is
represented by `decision="PASS"` and `defect_type=null`.

## No-confidence policy

Contract version `1.0` uses `confidence_mode="NONE"`. Every result has
`confidence=null`, and successful results include `CONFIDENCE_UNAVAILABLE`.
No digest fragment is converted into a probability, score, percentage, or
confidence estimate.

## Findings

Findings come only from `inspection_inference_findings.json` and are ordered
by catalogue order plus stable branch, field, code, message, and safe-detail
ties. Every successful execution includes:

- `MOCK_INFERENCE_USED`;
- `MOCK_DECISION_GENERATED`; and
- `CONFIDENCE_UNAVAILABLE`.

Mock `FAIL` additionally includes `MOCK_FAIL_DEFECT_ASSIGNED`. Known failures
are blocking and return `INFERENCE_FAILED`. Unexpected exceptions return only
the safe blocking `INFERENCE_INTERNAL_ERROR`; paths, exception text, and
tracebacks are not returned.

## Determinism and input immutability

Clock, inference-ID generation, engine ID, and engine version are injectable.
Fixed providers and identical input identities produce identical public JSON,
decision digest, decision, mock FAIL assignment, and finding order. Changing a
buffer hash, policy version, or engine version changes the digest, although a
new digest may map to the same decision bucket.

The service never mutates RGB bytes, height bytes, descriptors, or the public
preprocessing result. Inputs use frozen dataclasses and immutable `bytes`.
Inference creates no converted files, cache, preview, report, or persisted
buffer.

## Supported synthetic flow

Tests exercise inference after successful preprocessing for:

- RGB PNG plus uint16 TIFF height;
- RGB TIFF plus 16-bit grayscale PNG height; and
- RGB PNG plus two-dimensional float32 NPY height.

Inference does not rerun semantic validation or preprocessing and does not
read the source artifacts.

## Example Python usage

```python
from app.services.inspection_inference import (
    SyntheticInferenceInput,
    SyntheticMockInferencePolicyLoader,
    SyntheticMockInferenceService,
)

policy = SyntheticMockInferencePolicyLoader().load(
    "synthetic-deterministic-mock-inference",
    "1.0",
)
inference_input = SyntheticInferenceInput(
    inspection_id=preprocessing.result.inspection_id,
    validation_id=preprocessing.result.validation_id,
    preprocessing_id=preprocessing.result.preprocessing_id,
    preprocessing_outcome=preprocessing.result.outcome.value,
    synthetic_input=preprocessing.result.synthetic_input,
    mock_preprocessing=preprocessing.result.mock_implementation,
    rgb_buffer=preprocessing.rgb_buffer,
    height_buffer=preprocessing.height_buffer,
)
result = await SyntheticMockInferenceService().run_inference(
    inference_input,
    policy,
)
public_document = result.to_dict()
```

Only `result.to_dict()` belongs at a future public boundary. The internal
preprocessing buffers must remain inside trusted execution code.

## Unsupported integration and real-data replacement points

The engine itself still has no database, HTTP, FastAPI startup, report,
preview, model-file, training, or frontend integration. Schema version 3 now
provides a separate result-only persistence/lifecycle coordinator for an
already completed typed mock result; it never invokes this engine or reads its
input buffers. See `docs/inspection_processing_lifecycle.md`. A persisted mock
result must not be promoted or relabelled as a production inspection decision.

Real-data work must separately define model input compatibility, model and
taxonomy version evidence, output semantics, thresholds, calibration,
confidence interpretation, uncertainty rules, runtime failure handling,
latency limits, model provenance, validation data, and acceptance criteria.
Only after those decisions and representative verified data exist should a
real engine and policy be proposed.
