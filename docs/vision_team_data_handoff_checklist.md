# Vision Team Paired-Data Handoff Checklist

Use this checklist for the first 20-30 sample technical-validation package.
Contract version: `pcb-aoi-dataset/1.0`.

## Before sending

- [ ] Package follows the documented root, `schemas/`, `samples/`, and
  `manifests/` structure.
- [ ] Every physical inspection has one stable, unique `sample_id` directory.
- [ ] Every sample contains the original 2D file, original native 3D
  height/depth file, and `metadata.json`.
- [ ] No screenshot, colored preview, or 8-bit conversion replaces native 3D
  values; previews are clearly separate derived files.
- [ ] Height/depth representation, container, stored data type, physical value
  type, Z unit/scale/offset, XY unit/scales, and no-data policy are recorded.
- [ ] Invalid/no-return values and any validity mask are documented and hashed.
- [ ] Point clouds or meshes are identified to the receiving team and are not
  labelled as height images; contract 1.0 will reject them.
- [ ] RGB and 3D files are guaranteed to come from the same physical inspection
  instance.
- [ ] Alignment status, method, coordinate system, calibration version, and any
  transform/calibration file are recorded.
- [ ] Exact lowercase SHA-256 hashes are provided for RGB, height/depth, and
  applicable mask/calibration files.

## Identity, labels, and provenance

- [ ] Board, recipe/version, lot, capture session, production date, station,
  capture time with timezone, camera ID, and 3D sensor ID are real source data.
- [ ] Unknown optional values are omitted, not guessed.
- [ ] Every sample has a primary `OK` or `NOK` label and a traceable
  `label_source`.
- [ ] OK samples use `defect_type: null`.
- [ ] NOK samples use only `dispense_error`, `misalignment`, or `missing_part`
  for taxonomy `pcb-aoi-defects/1.0`; proposed new types are supplied separately
  for review.
- [ ] Review status is honest. Approved labels include reviewer and
  timezone-aware review time.
- [ ] Acquisition/export system, export version when known, import time, and
  synthetic/real status are recorded.

## Grouping and manifests

- [ ] `dataset_manifest.json` reports dataset/version, organization/team,
  sample and OK/NOK counts, supported board/recipe IDs, status, approval, schema
  references, and known limitations.
- [ ] `samples.jsonl` indexes every sample metadata file and its hash.
- [ ] The strongest available grouping key is selected: board, panel, lot/batch,
  capture session, date/station/recipe, or reviewed sequential group.
- [ ] One `sample_id` appears only once in `split_manifest.json`.
- [ ] Related grouping values never cross train, validation, test, or holdout.
- [ ] Excluded samples have explicit reasons.
- [ ] No random image-level split was used for related captures.

## Receiving review

- [ ] Schema files and taxonomy versions match the declared contract.
- [ ] Package contains 20-30 independently identifiable samples, with both OK
  and available evidence-backed NOK examples.
- [ ] File hashes, decoded dimensions/types, scaling, masks, and pair references
  are verified by the receiving team when the full validator becomes available.
- [ ] Missing training-stage or production-stage evidence is logged as a
  blocker, not filled with placeholder values.
- [ ] No sample is approved for training solely because technical schema
  validation passed.

Do not use `contracts/examples` as data. Every example file is synthetic and
contains illustrative placeholder values only.
