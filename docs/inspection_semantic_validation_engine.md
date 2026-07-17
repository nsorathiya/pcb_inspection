# Read-Only Paired Inspection Semantic-Validation Engine

Date: 2026-07-17

## Purpose and outcome meaning

The reusable engine under `backend/app/services/inspection_validation` reads
one persisted inspection's registered `RGB_RAW` and `HEIGHT_RAW` artifacts,
checks exact stored bytes and native technical metadata, applies an explicitly
selected versioned policy, and returns an `InspectionValidationResult`.

`VALIDATION_PASSED` means only that the pair is technically suitable to
continue under that policy. It is **not PCB PASS**, defect detection, inference,
or evidence of product quality. `VALIDATION_FAILED` describes blocking input,
integrity, format, policy, or evidence findings. `VALIDATION_ERROR` is reserved
for an unexpected internal failure and contains `VALIDATOR_INTERNAL_ERROR`.

The engine does not persist its result, update inspection status, append audit
events, create files, preprocess images, or expose an HTTP endpoint.

## Architecture

Responsibilities remain independently injectable:

- `DatabaseValidationArtifactRetriever` performs only repository reads for the
  inspection and all registered artifacts.
- `ManagedArtifactPathResolver` confines raw references to the expected
  `runtime/raw_uploads/<inspection-id>/<category>/` location.
- `StreamingFilesystemIntegrityInspector` streams SHA-256 and byte size and
  compares them with registered metadata.
- `PurposeSpecificNativeFormatInspector` delegates PNG, JPEG, BMP, TIFF, and
  NPY parsing to the existing dataset-validation parser adapters.
- `ValidationPolicyLoader` loads an explicit policy registry entry or path and
  validates it against the authoritative policy schema plus its cross-field
  semantic rules.
- `ContractValidationPolicyEvaluator` applies the supplied policy values; it
  does not copy the development policy into Python constants.
- `FindingFactory` constructs and sorts findings from
  `contracts/inspection_validation_findings.json`.
- `InspectionValidationService` orchestrates these components and calculates
  the outcome and summary counts.

The public operation is:

```python
result = await service.validate_inspection_pair(inspection_id, policy)
```

The inspection ID and generated validation ID must be canonical UUID strings.
Clock, validation-ID generation, and validator version are injected for
deterministic testing. `result.to_dict()` and `result_json(result)` produce the
path-free schema representation.

## Policy loading

The caller selects policy ID and version explicitly. The default registry has
only the development example at
`contracts/examples/inspection_validation_policy.development.json`. No policy
is inferred from board or recipe metadata.

The loader reads the authoritative JSON Schema for required fields, unknown
fields, types, bounds, enumerations, uniqueness, identifiers, and the contract
version. It also rejects inverted dimension bounds, an unusable NaN rule, and
a registered-transform rule that does not require registration evidence.
Missing IDs map to `POLICY_NOT_FOUND`; an unavailable version or malformed or
unsupported document maps to `POLICY_VERSION_UNSUPPORTED`. The service's
`validate_registered_policy` convenience method converts these expected
selection failures into safe failed results.

## Artifact retrieval and path safety

Exactly one `RGB_RAW` and one `HEIGHT_RAW` are required. Missing and duplicate
records produce the authoritative pair findings and `INCOMPLETE_RAW_PAIR`.
Optional `VALIDITY_MASK` and `CALIBRATION` records are read as technical
evidence only.

Paths must use portable relative POSIX syntax and the artifact-type-specific
raw category. Absolute, drive-qualified, backslash, traversal, wrong-inspection,
wrong-category, escaping, symbolic-link, and detectable Windows reparse-point
references fail safely. A resolved target must be a regular file. Absolute or
relative storage paths never enter the result or finding details.

## Integrity and native-format checks

SHA-256 and byte size are calculated in streaming reads. A missing file, size
mismatch, hash mismatch, unsafe path, redirect, or non-regular file is blocking.
The registered hash and size—not an internal resolved path—are returned in the
artifact summary.

RGB uses `inspect_rgb`; height uses `inspect_height`. A purpose-incompatible
but readable PNG may be inspected through the existing RGB parser solely to
report its technical channel and bit-depth reason safely. The engine never
adds another binary parser and never converts, rescales, normalizes, or writes
pixels. Extension/content and declared-media-type/content disagreement are
reported independently. Parser exception text is not exposed.

A required validity mask must have exactly one registered reference, intact
bytes, a readable single-channel raster, and the height dimensions. Calibration
evidence is checked only for safe path, readable bytes, and integrity because
no physical-calibration validity contract exists yet.

## Policy evaluation and findings

The evaluator applies allowed formats, RGB channels and bit depths, supported
RGB color modes, height storage types and minimum bit depth, dimension bounds,
single-channel height, pair dimension relationship, required mask,
calibration, and registration evidence, and `warning_as_blocking`.

Findings use only catalogue definitions. Catalogue order is the primary sort,
followed by artifact order, field, and safe message. Summary totals are derived
from the final ordered findings. When required calibration or registration is
missing, catalogue warning severity is retained but the finding is explicitly
blocking because the selected policy made that evidence mandatory.

`SAME_DIMENSIONS_REQUIRED` blocks unequal rasters.
`DIFFERENT_DIMENSIONS_ALLOWED` does not block only because dimensions differ.
`REGISTERED_TRANSFORM_REQUIRED` requires explicit injected evidence; equal
dimensions are never described as proof of registration.

## Current evidence limitations

Database schema version 1 has no registration-evidence field or transform
artifact type. Consequently, the database retriever always reports
registration evidence unavailable. A policy requiring it fails with
`REGISTRATION_EVIDENCE_MISSING`. An injected retriever can provide explicit
evidence for isolated validation tests, but the engine does not fabricate it or
apply a transform.

The database also has no selected height invalid/no-data policy field. Contract
1.0 therefore validates the policy's declared allowed strategies and available
mask evidence but cannot prove a per-inspection sentinel, NaN, units, scale, or
offset. A colorized-preview-specific finding is used only when internal test
evidence identifies such a reference; ordinary multichannel height is safely
reported as non-scalar without guessing its origin.

## Synthetic fixtures and read-only verification

Focused tests generate deterministic fixtures only below pytest temporary
directories, create managed runtime references, and compare expected technical
outcomes and blocking finding codes. A real temporary SQLite test confirms no
`INSERT`, `UPDATE`, or `DELETE` is executed during validation; status remains
`RECEIVED`, artifact rows and audit count remain unchanged, and source SHA-256,
size, and modification time remain unchanged.

Example construction outside an API workflow:

```python
findings = FindingFactory()
paths = RuntimePaths.from_root(runtime_root)
service = InspectionValidationService(
    DatabaseValidationArtifactRetriever(repositories.inspections, repositories.artifacts),
    StreamingFilesystemIntegrityInspector(ManagedArtifactPathResolver(paths)),
    PurposeSpecificNativeFormatInspector(findings),
    ContractValidationPolicyEvaluator(findings),
    findings,
)
policy = ValidationPolicyLoader().load("development-native-rgb-height", "1.0")
result = await service.validate_inspection_pair(inspection_id, policy)
```

No status transition or result persistence occurs after this call.

## Tests

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  .\backend\tests\test_inspection_semantic_validation_engine.py -q
.\.venv\Scripts\python.exe -m pytest .\backend\tests -q
```
