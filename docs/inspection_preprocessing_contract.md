# Inspection Preprocessing Contract

## Purpose and boundary

This contract defines the future hand-off from a technically validated paired
inspection to separate RGB (2D) and height/depth (3D raster) preprocessing, a
registration coordinator, and eventually an inference engine. It provides
versioned policy and result shapes, typed replaceable interfaces, stable
technical findings, and synthetic examples. It does not provide an executable
preprocessor.

Preprocessing may be considered only after the inspection is `READY` and its
selected validation result is `VALIDATION_PASSED`. Validation establishes
technical readability and policy suitability; preprocessing must still verify
that its own policy accepts the validated metadata.

## Non-goals

Version 1 does not read artifact files, crop or resize images, normalize pixel
values, scale height values, create tensors, register geometry, estimate a
homography, run inference, persist results, expose an API, or change inspection
status. It does not accept point clouds or meshes. No mode listed in a schema is
a claim that an implementation exists.

## Contract versions

- Policy: `pcb-aoi-inspection-preprocessing-policy/1.0`
- Result: `pcb-aoi-inspection-preprocessing/1.0`
- Finding catalogue: `pcb-aoi-inspection-preprocessing-findings/1.0`

Policies and implementations must identify their own versions as well. A future
implementation must declare the subset of version-1 modes it supports and fail
with a stable finding when it cannot honor a selected mode.

## Outcomes

`PREPROCESSING_SUCCEEDED` means both technical branch outputs were produced for
future inference. It does not mean PCB `PASS`, prove physical calibration,
prove registration accuracy, or establish model suitability.

`PREPROCESSING_FAILED` is a known prerequisite, input, policy, registration, or
output incompatibility and requires a blocking finding.

`PREPROCESSING_ERROR` is an unexpected execution failure and requires the
blocking `PREPROCESSING_INTERNAL_ERROR` finding. These outcomes are technical
processing results; PCB `PASS`, `FAIL`, and `UNCERTAIN` are not preprocessing
outcomes.

## Development-only synthetic policy

The example policy is deliberately nonproduction. It accepts the repository's
generated 16-by-12 RGB/height fixtures, requires `READY` and
`VALIDATION_PASSED`, preserves the full frame without resizing, describes
float32 `CHW` outputs, uses RGB unit-range normalization, leaves height values
without physical scaling, rejects invalid height values, and requires matching
output dimensions. It permits a mock implementation and synthetic input,
rejects real input, and allows uncalibrated height.

The policy selects `SYNTHETIC_IDENTITY_ONLY`. That choice is suitable only for
generated paired fixtures and must emit
`SYNTHETIC_IDENTITY_REGISTRATION_USED`. The parameters are not recommendations
for a camera, scanner, production recipe, or trained model.

## Separate RGB and height branches

The RGB branch accepts validated `RGB_RAW` identity metadata and produces an
`RGBPreprocessingOutput`. Its policy independently states ROI, resize,
interpolation, output channel count, type, layout, aspect-ratio behavior, and
normalization. `MEAN_STD` has no implicit defaults: one mean and one positive
standard deviation per output channel are required.

The height branch accepts validated `HEIGHT_RAW` identity metadata and produces
a `HeightPreprocessingOutput`. Its policy independently states ROI, resize,
interpolation, output channel count, type, layout, scaling, and invalid-value
handling. It remains distinguishable from RGB through the future inference
boundary. Neither branch output contains public tensor bytes.

## ROI and resize rules

`FULL_FRAME`, `STATIC_RECTANGLE`, and `RECIPE_DEFINED` are policy concepts.
Full-frame mode must not invent rectangle coordinates. A static rectangle must
be explicit; recipe resolution is future work. `RESIZE`, `LETTERBOX`, and
`CENTER_CROP` require positive target dimensions. `NONE` requires null target
dimensions, avoiding a fabricated resize target. Matching RGB/height output
dimensions are controlled by the explicit
`require_matching_output_dimensions` flag rather than inferred from input
dimensions.

## Normalization, height scaling, and invalid values

RGB normalization modes are `NONE`, `UNIT_RANGE`, and `MEAN_STD`. Height scaling
modes are `NONE`, `DECLARED_PHYSICAL_SCALE`, `MIN_MAX`, and `STANDARD_SCORE`.
No clipping, normalization, scale, or offset may be implicit.

Native stored height values and physical height values are distinct. A uint16
sample type does not establish physical accuracy. `DECLARED_PHYSICAL_SCALE`
requires an explicit unit plus traceable scale and offset sources. A result
must state whether physical scaling was applied and must not claim millimetres
or micrometres without declared evidence. The synthetic policy uses `NONE`,
sets `physical_scale_applied` false, and reports no physical unit.

`MASK` requires a validity-mask input and requested validity-mask output.
`REPLACE_WITH_CONSTANT` requires an explicit replacement value.
`PRESERVE_NAN` is allowed only with float32 or float64 output.

## Registration boundary

Equal raster dimensions do not prove registration. Registration coordination
may inspect both branch descriptors but does not collapse them into a single
object. `USE_DECLARED_TRANSFORM` requires a real transform reference.
`NOT_PERFORMED` is valid only when an explicitly selected policy allows it.
`SYNTHETIC_IDENTITY_ONLY` requires a development-only, synthetic-input policy,
must never be used for real input, and always carries a warning. Production
policies cannot enable synthetic identity registration.

No feature matching, transform estimation, homography calculation, or geometric
transformation is implemented in this task.

## Framework-neutral output descriptor

`PreprocessedBufferDescriptor` describes shape, `HW`/`HWC`/`CHW`/`NCHW`
layout, data type, channels, dimensions, byte order, contiguity, finite-value
verification, and source artifact SHA-256. It deliberately contains neither
NumPy/PyTorch/ONNX types nor the actual bytes. A future internal implementation
may own an array or byte buffer while exposing only this safe descriptor in the
public result.

## Findings and deterministic order

The finding catalogue owns code, severity, category, message, default blocking
behavior, and order. Findings are ordered first by catalogue order, then by
stable branch, field, code, message, and canonical safe-details ties. Details
are primitive, path-free values. Codes describe prerequisites, policies, RGB,
height, registration, output, or internal failures—not defects, classifications,
model confidence, or PCB quality.

## Future service interfaces

The protocol-only package defines replaceable boundaries for policy loading,
validated-inspection metadata reading, RGB preprocessing, height preprocessing,
registration coordination, orchestration, and a future result sink. The
orchestrator signature accepts inspection ID, validation ID, and a typed policy.
No concrete loader, reader, processor, orchestrator, or sink exists yet.

## Future execution and lifecycle integration

A later coordinator may move `READY` to `PROCESSING` before preprocessing and
inference begin. An unexpected preprocessing failure may lead from `PROCESSING`
to `ERROR`. Only inference may later produce `PASS`, `FAIL`, or `UNCERTAIN` from
`PROCESSING`; preprocessing by itself must never set those statuses. This task
documents that future integration but makes no database or lifecycle change.

## Review required when real vision data arrives

Before any production policy is created, the vision team must review real RGB
color space and bit depth, sensor and height formats, invalid-value encoding,
ROI ownership, target dimensions, interpolation, normalization parameters,
physical unit/scale/offset evidence, calibration provenance, registration
method and accuracy evidence, output layout expected by the selected model,
finite-value rules, and whether aspect-ratio changes are acceptable. Model and
recipe compatibility also require separate versioned evidence. The synthetic
policy must not be promoted or relabelled as production configuration.
