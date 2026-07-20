# Deterministic Synthetic Preprocessing Executor

## Purpose and development-only limitation

`SyntheticInspectionPreprocessingService` implements the Task-20 preprocessing
interfaces for the repository's generated software fixtures. It reads an
explicitly supplied, technically validated RGB/height pair, produces separate
in-memory float32 buffers, coordinates synthetic identity registration, and
returns the path-free `pcb-aoi-inspection-preprocessing/1.0` result.

The executor is mock, synthetic, and development-only. It rejects real input,
non-`READY` inspections, missing or non-passing validation outcomes,
production-approved policies, policies that disallow mock execution, and
implementations not explicitly identified as synthetic. It is not registered
in FastAPI application state.

## Supported fixture formats

The supported value-decoding surface is intentionally narrower than general
metadata inspection:

- RGB: generated three-channel 8-bit PNG and uncompressed classic RGB TIFF.
- Height: generated scalar uint16 classic TIFF, scalar 16-bit grayscale PNG,
  and two-dimensional C-order float32 NPY.

The decoder first uses the existing native inspectors and then invokes narrow
standard-library value helpers beside those parsers. It does not broaden the
accepted fixture subset to JPEG, BMP, BigTIFF, tiled TIFF, arbitrary compressed
TIFF, general NPY arrays, point clouds, or meshes.

## RGB decoding, normalization, and CHW layout

PNG samples and TIFF interleaved samples are decoded in RGB order. The output
is little-endian contiguous float32 with shape `[3, H, W]`. Buffer order is the
entire red plane, then the green plane, then the blue plane. BGR conversion is
never performed.

The only implemented RGB normalization is `UNIT_RANGE`:

```text
float32_output = integer_sample / ((2 ** bit_depth) - 1)
```

The active policy accepts 8-bit synthetic RGB, so the divisor is 255. No
mean/std normalization, resizing, cropping, padding, color conversion, or
interpolation executes.

## Height decoding and no-scaling behavior

Height output is little-endian contiguous float32 with shape `[1, H, W]` in
row-major scalar order. uint16 samples are converted numerically to float32;
they are not divided by 65535. float32 NPY values retain their finite numeric
values. `REJECT` invalid-value handling blocks NaN and infinity.

The synthetic policy uses scaling mode `NONE`. Results therefore set
`physical_unit` to null and `physical_scale_applied` to false. No Z scale,
offset, unit, sentinel value, calibration, clipping, or normalization is
invented. A uint16 sample is not treated as a physically calibrated height.

## Framework-neutral internal buffers

`InternalPreprocessedBuffer` contains immutable bytes, element count, byte
size, SHA-256, and a public-safe `PreprocessedBufferDescriptor`. Float values
use explicit little-endian IEEE-754 encoding via Python's `struct` module.
Shape, dimensions, layout, type, contiguity, finite-value status, byte length,
and content hash are checked before a result can succeed.

The public result contains only descriptors and safe statistics. It never
contains buffer bytes, memory views, arrays, framework tensors, or source
paths.

## Safe statistics

Statistics are calculated from the exact float32 values represented by the
output bytes:

- minimum and maximum;
- arithmetic mean;
- population standard deviation;
- finite and non-finite value counts; and
- total element count.

Statistics are compact technical summaries. Height statistics do not imply
physical meaning.

## Synthetic identity registration

Only `SYNTHETIC_IDENTITY_ONLY` is implemented. It requires synthetic input, a
development-only policy, explicit policy permission, and matching output
dimensions when the policy requires them. It estimates no transform and does
no resampling. The result states `transform_applied=false`,
`synthetic_identity=true`, and emits the nonblocking warning
`SYNTHETIC_IDENTITY_REGISTRATION_USED`.

Equal dimensions are only a fixture relationship; they do not prove physical
registration. Unequal fixture dimensions fail with
`OUTPUT_DIMENSION_RELATIONSHIP_INVALID` and are not automatically resized.

## Determinism and source immutability

Clock, preprocessing-ID generation, implementation ID, and implementation
version are injectable. With fixed providers and identical validated source
bytes, result JSON, RGB bytes, height bytes, hashes, statistics, and finding
order are identical.

Sources are opened read-only. Their SHA-256 and byte size must still match the
validated identity. The executor creates no converted files, previews, caches,
reports, or temporary files and does not modify source timestamps.

## Findings and outcomes

Findings are constructed only from
`contracts/inspection_preprocessing_findings.json` and sorted by catalogue
order plus stable ties. Expected prerequisite, policy, input, registration,
and output incompatibilities produce `PREPROCESSING_FAILED`. Unexpected
exceptions produce only the safe blocking `PREPROCESSING_INTERNAL_ERROR` and
`PREPROCESSING_ERROR`; exception text, tracebacks, and paths are not exposed.

`PREPROCESSING_SUCCEEDED` means two technically valid in-memory branch buffers
were produced. It is not PCB `PASS`, a classification, model confidence,
registration proof, calibration proof, or production suitability.

## Example Python usage

```python
from app.services.inspection_preprocessing import (
    SyntheticInspectionPreprocessingService,
    SyntheticPreprocessingPolicyLoader,
    ValidatedArtifactSource,
    ValidatedInspectionInput,
)

policy = SyntheticPreprocessingPolicyLoader().load(
    "synthetic-paired-rgb-height",
    "1.0",
)
validated = ValidatedInspectionInput(
    inspection_id=inspection_id,
    validation_id=validation_id,
    inspection_status="READY",
    validation_outcome="VALIDATION_PASSED",
    synthetic_input=True,
    rgb=ValidatedArtifactSource(rgb_identity, rgb_fixture_path),
    height=ValidatedArtifactSource(height_identity, height_fixture_path),
)
execution = await SyntheticInspectionPreprocessingService().preprocess_inspection(
    validated,
    policy,
)
```

The path-bearing source objects and internal buffers must remain inside trusted
execution code. Only `execution.result.to_dict()` is suitable for a future
public boundary.

## Unsupported integration and replacement points

The executor itself has no database, HTTP endpoint, application-state wiring,
status transition, audit event, inference call, model load, classification,
report, preview, or frontend integration. Schema version 3 provides a separate
result persistence/lifecycle coordinator. The trusted internal synthetic
processing orchestrator now constructs this executor's input from persisted
validation evidence, a verified generated-fixture manifest, and safely resolved
runtime artifacts; invokes it once for the winning new run; and retains its
buffers in memory for mock inference. The executor itself remains unaware of
those concerns. See `docs/synthetic_processing_orchestrator.md` and
`docs/inspection_processing_lifecycle.md`. It does not rerun semantic validation.

Real-data integration must replace or extend the policy registry, trusted
source reader, RGB decoder/processor, height decoder/processor, registration
processor, invalid-value rules, physical-scale evidence, and model-specific
output compatibility only after reviewed representative data and focused tests
exist. The synthetic executor must not be promoted into a production policy.
