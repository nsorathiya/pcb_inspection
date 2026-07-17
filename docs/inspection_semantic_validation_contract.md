# Paired RGB/Height Semantic-Validation Contract

Date: 2026-07-17

## Purpose

Contract version `pcb-aoi-inspection-validation/1.0` defines how a future
validator will describe the technical suitability of one persisted `RGB_RAW`
plus `HEIGHT_RAW` pair for preprocessing. It answers whether the pair exists,
the registered bytes are intact and readable, native content is supported, the
height data is genuinely scalar, and a selected recipe-specific policy is
satisfied.

This task defines contracts only. No stored inspection is read or validated,
no result is persisted, no API is added, and no inspection status changes.

## Non-goals and quality terminology

Technical validation never determines whether a PCB is good, defective, or
acceptable. It does not detect solder defects, align images, normalize data,
generate previews, preprocess inputs, run inference, assign confidence, or
produce a PCB quality classification.

The validation outcomes deliberately avoid plain PASS or FAIL:

| Outcome | Meaning |
| --- | --- |
| `VALIDATION_PASSED` | The raw pair is technically suitable to continue to future preprocessing under the selected policy. It does not mean the PCB passed inspection. |
| `VALIDATION_FAILED` | Validation completed and at least one policy or input finding blocks technical readiness. |
| `VALIDATION_ERROR` | The validator could not complete reliably because of an unexpected internal failure. It is not silently converted to an input failure. |

## Result contract

`contracts/inspection_validation_result.schema.json` is the JSON Schema for a
single validation result. It requires:

- contract, validation, inspection, policy, and validator identities;
- timezone-aware start and completion timestamps;
- one safe technical summary each for `RGB_RAW` and `HEIGHT_RAW`;
- deterministically ordered findings; and
- summary counts, technical-readiness state, and an explicit synthetic-example
  marker.

Artifact summaries contain only artifact type, registered SHA-256 and byte
size, declared media type, detected format, width, height, channels, bit depth,
storage data type, and readability status. They contain no storage path,
filename, binary content, database artifact ID, exception text, or user
identity.

Readability is one of `READABLE`, `MISSING`, `UNREADABLE`,
`INTEGRITY_FAILED`, or `UNINSPECTED`. A successful outcome requires both raw
artifact summaries to be readable and the height summary to be single-channel.

## Finding structure and deterministic order

The authoritative catalogue is
`contracts/inspection_validation_findings.json`, version
`pcb-aoi-inspection-validation-findings/1.0`. Each finding contains:

- stable `code`;
- `severity`: `INFO`, `WARNING`, or `ERROR`;
- one controlled category;
- safe message;
- optional artifact type and field;
- explicit `blocking` boolean; and
- optional safe primitive technical details.

Finding details explicitly exclude path and filename keys. Findings are sorted
by catalogue order, then artifact order (`RGB_RAW`, `HEIGHT_RAW`,
`VALIDITY_MASK`, `CALIBRATION`), field, and message. Catalogue order numbers
are unique and versioned.

Allowed categories are `PAIR`, `FILE_INTEGRITY`, `FORMAT`,
`IMAGE_PROPERTIES`, `HEIGHT_PROPERTIES`, `REGISTRATION_EVIDENCE`,
`CALIBRATION_EVIDENCE`, `POLICY`, and `INTERNAL`.

## Finding-code catalogue

| Category | Codes |
| --- | --- |
| Pair | `RGB_RAW_MISSING`, `HEIGHT_RAW_MISSING`, `DUPLICATE_RGB_RAW`, `DUPLICATE_HEIGHT_RAW`, `INCOMPLETE_RAW_PAIR`, `VALIDITY_MASK_MISSING`, `DIMENSION_RELATIONSHIP_UNSUPPORTED` |
| File integrity | `ARTIFACT_FILE_MISSING`, `ARTIFACT_SIZE_MISMATCH`, `ARTIFACT_SHA256_MISMATCH`, `ARTIFACT_PATH_UNSAFE`, `ARTIFACT_NOT_REGULAR_FILE`, `ARTIFACT_SYMLINK_REJECTED` |
| Format | `RGB_FORMAT_UNSUPPORTED`, `HEIGHT_FORMAT_UNSUPPORTED`, `EXTENSION_CONTENT_MISMATCH`, `MEDIA_TYPE_CONTENT_MISMATCH`, `FILE_UNREADABLE` |
| RGB properties | `RGB_DIMENSIONS_INVALID`, `RGB_CHANNELS_UNSUPPORTED`, `RGB_BIT_DEPTH_UNSUPPORTED`, `RGB_COLOR_MODE_UNSUPPORTED` |
| Height properties | `HEIGHT_DIMENSIONS_INVALID`, `HEIGHT_NOT_SINGLE_CHANNEL`, `HEIGHT_STORAGE_TYPE_UNSUPPORTED`, `HEIGHT_BIT_DEPTH_TOO_LOW`, `HEIGHT_COLORIZED_PREVIEW_REJECTED`, `HEIGHT_NAN_POLICY_INCOMPATIBLE`, `HEIGHT_INVALID_VALUE_POLICY_MISSING` |
| Registration | `REGISTRATION_EVIDENCE_MISSING` |
| Calibration | `CALIBRATION_EVIDENCE_MISSING` |
| Policy | `POLICY_NOT_FOUND`, `POLICY_VERSION_UNSUPPORTED` |
| Internal | `VALIDATOR_INTERNAL_ERROR` |

These are input-validation findings only. Solder-defect, component-defect, and
PCB-quality codes belong to later classification/taxonomy contracts and must
not be added here.

## Outcome and blocking rules

- Any blocking non-internal `ERROR` produces `VALIDATION_FAILED`.
- A failed result contains at least one blocking finding.
- A nonblocking `WARNING` may coexist with `VALIDATION_PASSED`.
- When `warning_as_blocking=true`, policy evaluation promotes applicable
  warnings to blocking before selecting the outcome.
- An unexpected validator exception produces `VALIDATION_ERROR` with
  `VALIDATOR_INTERNAL_ERROR`.
- `VALIDATION_ERROR` is operationally distinct from an unsuitable input pair.
- Missing, unreadable, size-mismatched, or hash-mismatched raw artifacts can
  never produce technical readiness.

## Validation policy

Policy schema version `pcb-aoi-inspection-validation-policy/1.0` is defined in
`contracts/inspection_validation_policy.schema.json`. Recipe selection will
identify a policy by `policy_id` and `policy_version`; recipe-specific values
are data, not Python constants.

Policies control:

- allowed RGB and height containers;
- RGB channels and bit depths;
- height scalar storage types and minimum bit depth;
- explicit invalid/no-data policies;
- single-channel height enforcement;
- minimum and maximum raster dimensions;
- same-dimension, different-dimension, or registered-transform pair rules;
- calibration, validity-mask, and registration evidence requirements; and
- whether warnings block readiness.

Schema validation checks document shape and enumerated values. Cross-field
rules such as maximum dimensions not being smaller than minimum dimensions,
NaN requiring floating storage, and registered-transform mode requiring
registration evidence are listed as semantic rules for the future policy
loader/evaluator.

## Default development policy

`contracts/examples/inspection_validation_policy.development.json` is clearly
marked `development_only=true` and is not production-approved. It permits RGB
PNG, JPEG, BMP, and TIFF and height PNG, TIFF, and NPY. It requires explicit
no-data policy, scalar single-channel height, and at least 16-bit height
storage. Allowed height scalar types are `uint16`, `int16`, `uint32`,
`float32`, and `float64`.

The example rejects 8-bit height, `uint8` height storage, multichannel height,
unsupported containers, and an incomplete raw pair. Calibration, validity
mask, and registration evidence are visible but non-required in this
development example. A reviewed recipe can make them mandatory through a
different policy document.

## Native RGB rules

RGB content must be readable, content-detected rather than trusted by extension
or media type, within policy dimensions, and use an allowed channel count,
color mode, and bit depth. Extension/content mismatch is blocking. A declared
media-type mismatch defaults to a nonblocking warning but can be promoted by
policy.

Validation does not color-convert, demosaic, resize, normalize, or otherwise
change RGB bytes.

## Native height/depth rules

A screenshot, heatmap, colorized preview, or ordinary RGB image is not raw 3D
data. Default development validation blocks 8-bit height and any multichannel
height raster. Contract 1.0 allows policy-controlled scalar `uint16`, `int16`,
`uint32`, `float32`, and `float64` storage only.

Storage type alone does not prove physical accuracy. Physical scale, offset,
units, invalid/no-data values, validity masks, calibration, and registration
are separate evidence. NaN is meaningful only for floating storage. Sentinel
and validity-mask policies require their corresponding evidence.

No normalization, type conversion, clipping, rescaling, interpolation, or
invalid-value replacement occurs during validation. Point clouds and meshes
are unsupported in contract 1.0 and are absent from both result and policy
format enumerations.

## Pair, calibration, and registration rules

Exactly one `RGB_RAW` and one `HEIGHT_RAW` are required. Duplicates or a
missing member block readiness. Equal dimensions do not prove spatial
registration. The policy chooses one of:

- `SAME_DIMENSIONS_REQUIRED`;
- `DIFFERENT_DIMENSIONS_ALLOWED`; or
- `REGISTERED_TRANSFORM_REQUIRED`.

Calibration and registration evidence are evaluated independently. A policy
may require a calibration artifact, validity mask, registration evidence, or a
registered transform. Validation verifies evidence availability and technical
consistency; it does not recalibrate sensors or apply a transform.

## Existing parser reuse and known gaps

The future native-format adapter should reuse
`app.services.dataset_validation.file_inspection` rather than duplicate binary
parsers. Existing reusable behavior is:

- content-based PNG, BMP, and JPEG RGB inspection;
- classic strip-based scalar TIFF with uncompressed/Deflate data;
- two-dimensional scalar NPY inspection;
- streaming SHA-256, safe relative-path checks, regular-file checks, and
  symlink rejection.

Contract 1.0 intentionally does not change those parsers. Before the example
policy can be executed in full, focused parser work with representative files
is required for two gaps:

- RGB TIFF is policy-representable but `inspect_rgb` currently supports only
  PNG, BMP, and JPEG.
- Scalar 16-bit height PNG is policy-representable but `inspect_height`
  currently rejects every PNG as a preview.

BigTIFF, tiled TIFF, LZW TIFF, HDF5, EXR, point clouds, and meshes remain
unsupported. Existing parser support for `int32` NPY/TIFF does not make it
policy-approved; policy contract 1.0 does not allow `int32`.

## Future service boundaries

`app.services.inspection_validation.interfaces` defines protocols only:

1. `ValidationArtifactRetriever` retrieves registered artifact references.
2. `FilesystemIntegrityInspector` resolves managed storage and checks file,
   size, and SHA-256 integrity.
3. `NativeFormatInspector` adapts the existing binary inspectors.
4. `ValidationPolicyEvaluator` produces ordered findings from technical
   summaries and policy data.
5. `InspectionPairValidator` will orchestrate
   `validate_inspection_pair(inspection_id, policy)` and return a typed result.
6. `ValidationResultPersistence` will persist the completed result in a later
   database task.
7. `InspectionValidationStatusTransition` will apply a separately reviewed
   lifecycle transition after persistence.

Keeping these boundaries separate prevents filesystem inspection, policy
evaluation, persistence, and status mutation from becoming one monolithic
service.

## Future status transitions

No transition is implemented by this contract task. A future transactional
workflow may apply only:

- `RECEIVED` to `READY` for `VALIDATION_PASSED`;
- `RECEIVED` to `VALIDATION_FAILED` for `VALIDATION_FAILED`; and
- `RECEIVED` to `ERROR` for `VALIDATION_ERROR`.

Validation must not transition an inspection already in `PROCESSING`, `PASS`,
`FAIL`, or `UNCERTAIN`. Status selection must occur only after a complete
validation result is ready for persistence; persistence and transition failure
handling require a separate design.

## Synthetic examples and future stages

Passed, failed, and error examples under `contracts/examples` are explicitly
synthetic schema illustrations. Their hashes, IDs, dimensions, and timestamps
are not production data or performance claims.

Future tasks, each requiring separate review, are:

1. adapt and test the existing parsers for approved additional formats;
2. add validation result/policy persistence and migrations;
3. implement read-only artifact retrieval and execution orchestration;
4. implement the narrow guarded status transitions; and
5. add an API only after execution, persistence, idempotency, and error
   contracts are approved.

## Tests

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  .\backend\tests\test_inspection_validation_contract.py -q
.\.venv\Scripts\python.exe -m pytest .\backend\tests -q
```
