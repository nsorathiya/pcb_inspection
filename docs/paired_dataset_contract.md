# Versioned Paired 2D/3D Dataset Contract

Date: 2026-07-17

## Purpose and version

The authoritative contract identifier is `pcb-aoi-dataset/1.0`. It describes
one RGB image and one native height/depth map captured for the same physical
inspection instance, plus traceable ground truth, calibration, provenance, and
leakage-safe split metadata.

An incompatible field or semantic change requires a new contract version.
Existing packages must never be silently reinterpreted under a newer version.
The JSON Schemas are:

- `contracts/pcb_aoi_sample.schema.json`
- `contracts/dataset_split_manifest.schema.json`
- `contracts/dataset_manifest.schema.json`
- `contracts/defect_taxonomy.json`

This contract is designed for future verified paired data. It does not convert
or assign the current legacy 2D images.

## Canonical package structure

```text
dataset_paired/
  dataset_manifest.json
  schemas/
    pcb_aoi_sample.schema.json
    dataset_split_manifest.schema.json
    dataset_manifest.schema.json
    defect_taxonomy.json
  samples/
    sample_000001/
      rgb.png
      height.tiff
      validity_mask.png       # optional
      calibration.json        # optional at technical validation
      metadata.json
  manifests/
    samples.jsonl
    split_manifest.json
```

The dataset manifest is at the package root so all references are portable,
relative paths without `..`. Each sample has its own directory and metadata,
making one physical inspection instance independently identifiable. The schema
files are immutable copies of the versions used to create the package.

Each `samples.jsonl` line is an index record containing `sample_id`,
`metadata_file`, and the metadata file's SHA-256. It is an index, not a source
of labels or split assignments. `split_manifest.json` is the sole authority for
split assignment.

## Sample fields

All schema-required fields are required for initial technical validation.
Optional fields remain optional until the delivery-stage policy below requires
them.

### Identity and ground truth

| Field | Meaning |
| --- | --- |
| `contract_version` | Must equal `pcb-aoi-dataset/1.0`. |
| `sample_id` | Stable unique identity for this inspection instance. |
| `board_id` | Physical board/assembly identity; never derive it from a random filename. |
| `panel_id` | Optional physical panel identity. |
| `component_ref` | Optional component designator such as `U1`. |
| `inspection_region_id` | Optional non-component inspection-region identity. |
| `recipe_id`, `recipe_version` | Exact PCB inspection recipe identity and version. |
| `ground_truth.label` | Primary business label: `OK` or `NOK`. |
| `ground_truth.defect_type` | `null` for OK; a controlled taxonomy value for NOK. |
| `ground_truth.taxonomy_version` | Must equal `pcb-aoi-defects/1.0`. |
| `ground_truth.label_source` | System, procedure, or authority that produced the label. |
| `ground_truth.review_status` | `unreviewed`, `in_review`, `approved`, or `rejected`. |
| `ground_truth.reviewed_by` | Reviewer identity; required when status is `approved`. |
| `ground_truth.reviewed_at` | Timezone-aware review timestamp; required when approved. |
| `ground_truth.review_notes` | Optional review rationale or uncertainty. |

The business label answers whether the inspected item is acceptable. The
defect type describes why an item is NOK. Model-output classes describe a
particular model's targets. These are different concepts and must not be
collapsed. The existing `contracts/class_labels.json` remains a legacy model
output contract; it is not changed by this dataset contract. In particular,
the legacy `no_defect` model class is represented here as `label=OK` and
`defect_type=null`, not as a defect taxonomy member.

Version `pcb-aoi-defects/1.0` includes only evidence-backed NOK types:
`dispense_error`, `misalignment`, and `missing_part`. Adding or changing a
meaning requires a reviewed taxonomy version; consumers must reject unknown
types rather than guess.

### Production and capture

| Field | Meaning |
| --- | --- |
| `production.lot_id` | Production lot identity. |
| `production.batch_id` | Optional batch identity within or across lots. |
| `production.capture_session_id` | Acquisition run/session shared by related captures. |
| `production.production_date` | Calendar production date, not a guessed filename date. |
| `production.station_id` | Inspection station identity. |
| `production.machine_id` | Optional production or inspection machine identity. |
| `production.sequential_group_id` | Optional reviewed group for related sequential frames. |
| `capture.captured_at` | Capture time with `Z` or an explicit UTC offset. |
| `capture.camera_2d_id` | Physical/logical 2D camera identity. |
| `capture.sensor_3d_id` | Physical/logical 3D sensor identity. |
| `capture.lighting_profile_id` | Optional controlled lighting profile. |
| `capture.exposure` | Optional recorded exposure time, gain, and/or aperture. |

### Files, 2D, and 3D

| Field | Meaning |
| --- | --- |
| `files.rgb_file` | Safe path relative to the sample directory for the original 2D image. |
| `files.height_file` | Safe relative path for the original native height/depth data. |
| `files.validity_mask_file` | Optional no-data mask; required for `validity_mask` policy. |
| `files.calibration_file` | Optional calibration/transform record. |
| `rgb.width`, `rgb.height` | Stored 2D pixel dimensions. |
| `rgb.channels` | Stored channel count. |
| `rgb.color_space` | Explicit `GRAY`, `RGB`, or supported Bayer order. |
| `rgb.bit_depth` | Stored per-channel bit depth. |
| `height_3d.representation` | `height_map` or `depth_map`; never a preview, point cloud, or mesh. |
| `height_3d.width`, `height_3d.height` | Native raster dimensions. |
| `height_3d.storage_format` | Native container: `tiff`, `npy`, `hdf5`, or `exr`. |
| `height_3d.storage_data_type` | Stored scalar type, separate from physical scaling. |
| `height_3d.physical_value_type` | Whether values represent `height` or `depth`. |
| `height_3d.z_unit` | Physical Z unit: `um`, `mm`, or `m`. |
| `height_3d.z_scale`, `height_3d.z_offset` | Physical conversion: `physical_z = stored_value * z_scale + z_offset`. |
| `height_3d.xy_unit` | Physical X/Y unit. |
| `height_3d.x_scale`, `height_3d.y_scale` | Physical spacing per raster column and row. |
| `height_3d.invalid_value` | Optional stored sentinel; required for `sentinel_value`. |
| `height_3d.no_data_policy` | `none`, `sentinel_value`, `validity_mask`, or `nan`. |

Original height/depth values must be retained exactly. `uint16` describes
storage only and does not prove physical accuracy, calibration, or units. A
colored visualization, screenshot, normalized image, or 8-bit preview is not a
height map and may only be supplied as a separate derived file outside the
required pair. Point clouds and meshes are rejected by version 1.0; they must
be explicitly named and governed by a future compatible schema, never renamed
or described as `height_map`.

### Registration, integrity, and provenance

| Field | Meaning |
| --- | --- |
| `registration.registration_status` | `aligned`, `requires_transform`, or `unverified`. |
| `registration.registration_method` | Documented alignment/registration procedure. |
| `registration.transform_reference` | Optional relative transform record; required when a transform is needed. |
| `registration.calibration_version` | Calibration set/version used for the pair. |
| `registration.coordinate_system` | Coordinate frame in which 3D values/transform are defined. |
| `integrity.rgb_sha256`, `integrity.height_sha256` | Lowercase SHA-256 of the original stored files. |
| `integrity.validity_mask_sha256` | Optional mask hash; required when a mask is required. |
| `integrity.calibration_sha256` | Optional calibration-file hash. |
| `provenance.source_system` | Acquisition/export system of record. |
| `provenance.source_export_version` | Optional exporter/version identity. |
| `provenance.imported_at` | Package import time with timezone. |
| `provenance.is_synthetic` | Explicitly distinguishes synthetic from production data. |
| `provenance.notes` | Optional provenance limitations or conversion history. |

Hashes are computed over the exact bytes delivered, before conversion,
normalization, metadata rewriting, or preview generation. File existence,
hash, decoded properties, and pair consistency will be checked by the future
full validator; schema validation alone does not read files.

Equal dimensions do not prove registration. The provider must guarantee that
both files describe the same physical inspection instance and record the
alignment state, method, calibration version, coordinate system, and transform
reference where applicable.

## Delivery-stage policy

| Stage | Minimum required evidence |
| --- | --- |
| Initial technical validation | Every schema-required field; both original files; parseable native metadata; declared no-data policy; hashes; explicit synthetic flag; label source; production/session identifiers. `unreviewed` labels and `unverified` registration may be delivered but are visibly blocked from training. |
| Before model training | All technical checks pass; `is_synthetic=false` unless a reviewed experiment explicitly permits synthetic data; approved labels with reviewer/time; verified defect taxonomy; verified same-instance pairing; aligned or validated transform; calibration artifact/hash; reviewed production grouping; authoritative split manifest passes uniqueness and group-exclusivity checks; no leakage-audit blockers. |
| Before production acceptance | Training minimum plus approved dataset manifest; locked test/holdout; traceable organization/team; validated calibration lifecycle; sensor/camera/recipe versions; acceptance criteria and limitations reviewed; repeatable hash verification; retention and change-control ownership. |
| Optional | Panel, batch, machine, sequential group, lighting, exposure, component/region, review notes, exporter version, and calibration hash only remain optional when the applicable stage does not need them. Never fabricate them. |

An unavailable optional value is omitted. A required-before-training or
required-before-production value blocks that stage; it is never replaced with
guessed text.

## Ground-truth workflow

1. Acquisition records an initial OK/NOK label, source, and taxonomy version.
2. A qualified reviewer checks the RGB/height pair and production context.
3. OK must retain `defect_type=null`; NOK must use one controlled defect type.
4. Ambiguous, conflicting, or incomplete samples remain `in_review` or become
   `rejected`; they are not silently accepted for training.
5. Approval records reviewer, timezone-aware timestamp, and relevant notes.
6. Corrections create a traceable dataset/manifest version rather than
   overwriting historical evidence without record.

## Authoritative grouped split workflow

The split manifest version is `pcb-aoi-split-manifest/1.0`. Each assignment
contains `sample_id`, one enumerated split, and the selected production grouping
value. The manifest records the grouping key type, creation method/time, and
optional approver.

Each sample ID must occur exactly once. A grouping value may occur in only one
protected split (`train`, `validation`, `test`, or `holdout`). Excluded samples
require a reason. `excluded` is not a model-evaluation split. Random image-level
splitting is prohibited whenever captures may be related. Split values must not
be copied into sample metadata as an independent authority.

The strongest proven key is selected in this order: board, panel, lot/batch,
capture session, production date plus station/recipe, then a human-reviewed
sequential group. The JSON Schema validates record shape; the two cross-record
rules identified by `x-semantic_rules` require semantic validation before
training or evaluation.

## Requested first package

Ask the vision team for 20-30 independently identifiable sample directories,
including both OK and evidence-backed NOK cases, the root dataset manifest,
the sample index, and a draft grouped split manifest. Request original RGB and
native height/depth files, masks/calibration records where applicable, actual
units/scales/no-data behavior, recipe and production identities, and ground
truth review evidence. Do not ask the team to manufacture missing values; list
missing stage-gated evidence as a limitation.

The files under `contracts/examples` are synthetic schema illustrations only.
Their identities, hashes, dimensions, scale values, and calibration references
are placeholders and are not a production dataset or calibration claim.
