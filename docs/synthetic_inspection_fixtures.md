# Deterministic Synthetic Inspection Fixtures

## Purpose and non-production warning

`scripts/generate_synthetic_inspection_fixtures.py` creates small, controlled
RGB/height file pairs for software tests, development demonstrations of the
intake API, and future semantic-validation tests.

> Synthetic fixture for software validation only.

The fixtures are not camera captures, production inspections, real defect
evidence, calibration evidence, training-approved data, production-approved
data, or model-accuracy evidence. They must never be mixed with real datasets,
used for model training, or used for accuracy measurements.

The application never runs this generator during startup. Generation occurs
only through an explicit library call or CLI command.

## Generation command

Run from the repository root in Windows PowerShell. The output directory is
required and must be outside the repository and real dataset directories:

```powershell
python .\scripts\generate_synthetic_inspection_fixtures.py `
  --output-root C:\pcb-aoi-demo\synthetic-fixtures
```

The default seed is the fixed integer `20260717`. Override it or select one or
more scenarios explicitly:

```powershell
python .\scripts\generate_synthetic_inspection_fixtures.py `
  --output-root C:\pcb-aoi-demo\synthetic-fixtures `
  --seed 12345 `
  --scenario valid_rgb_png_height_tiff `
  --scenario height_png_uint8
```

CLI exit codes are `0` for success, `2` for invalid usage or an unsafe output,
and `3` for an unexpected contained failure.

## Output structure

```text
synthetic-fixtures/
  SYNTHETIC_FIXTURES_MARKER.json
  generation_manifest.json
  scenarios/
    valid_rgb_png_height_tiff/
      rgb.png
      height.tiff
      scenario.json
    valid_rgb_tiff_height_png16/
      rgb.tiff
      height.png
      scenario.json
    ...
```

All manifest references use POSIX-style relative paths. Scenario records
conform to `pcb-aoi-synthetic-inspection-scenario/1.0` in
`contracts/synthetic_inspection_scenario.schema.json`. Each record contains
exact actual and declared SHA-256 values and byte sizes, expected intake and
technical-validation outcomes, expected finding codes, deterministic identity,
policy identity, and explicit synthetic/non-approval flags.

`generation_manifest.json` inventories every file below `scenarios/` and
provides a reproducibility digest over the ordered path/hash/size records. The
top-level marker hashes the generation manifest, avoiding self-referential
manifest hashes.

## Determinism

Generator version, seed, and selected scenario IDs fully determine:

- UUIDv5 scenario and generation identifiers;
- fixed fixture timestamp;
- dimensions and scalar values;
- RGB patterns;
- binary PNG, classic TIFF, and NPY bytes;
- canonical JSON bytes and ordering;
- hashes, sizes, and the scenario-tree digest.

PNG files use deterministic stored Deflate blocks rather than depending on a
platform compression heuristic. No current time, random UUID, absolute path,
or operating-system path separator is written to a manifest.

## Safe regeneration

The generator refuses filesystem roots, drive roots, home directories,
repository roots, repository source directories, legacy dataset directories,
backend test directories, and symbolic-link/reparse-point output paths. The CLI
does not permit runtime output. Tests can explicitly enable a test-only runtime
boundary through the library API.

An existing directory is never overwritten by default. To regenerate an
existing generator-owned directory:

```powershell
python .\scripts\generate_synthetic_inspection_fixtures.py `
  --output-root C:\pcb-aoi-demo\synthetic-fixtures `
  --overwrite-generated
```

Replacement proceeds only when:

- the ownership marker is valid for this generator version;
- the marker hash matches `generation_manifest.json`;
- every prior generated file still matches its recorded hash and size;
- no extra, missing, unknown, symlinked, or reparse-point file/directory exists.

New output is built in a sibling staging directory before the verified old
tree is removed file-by-file. Unknown or modified files cause regeneration to
stop; they are never deleted.

## Scenario catalogue

Technically valid native-file scenarios:

- `valid_rgb_png_height_tiff`
- `valid_rgb_tiff_height_png16`
- `valid_rgb_png_height_npy_float32`
- `valid_different_dimensions` (readable files; the default same-dimension
  policy blocks the pair and registration-transform evidence would be needed)
- `valid_with_mask_and_calibration_evidence`

Deliberately invalid or incomplete scenarios:

- `missing_rgb`, `missing_height`
- `corrupt_rgb`, `corrupt_height`
- `truncated_rgb_png`, `truncated_height_tiff`
- `height_png_uint8`, `height_png_rgb`, `height_png_rgba`
- `height_colorized_preview`
- `unsupported_rgb_extension`, `unsupported_height_extension`
- `hash_mismatch_rgb`, `hash_mismatch_height`
- `byte_size_mismatch_rgb`, `byte_size_mismatch_height`
- `dimension_mismatch_without_registration`
- `required_mask_missing`, `required_calibration_missing`,
  `required_registration_missing`
- `duplicate_rgb_reference`, `duplicate_height_reference`
- `unsafe_relative_path_reference`

Scenarios that cannot be represented as an ordinary pair express the intended
missing, duplicate, mismatched, or unsafe reference in `scenario.json`.

## Intake API demonstration

Only scenarios with `expected_intake_outcome=ACCEPTED` should be submitted as
ordinary multipart examples. For example:

```powershell
$root = "C:\pcb-aoi-demo\synthetic-fixtures\scenarios\valid_rgb_png_height_tiff"
$form = @{
  board_id = "SYNTHETIC-SOFTWARE-FIXTURE"
  recipe_id = "development-native-rgb-height"
  recipe_version = "1.0"
  rgb_image = Get-Item "$root\rgb.png"
  height_map = Get-Item "$root\height.tiff"
}
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/inspections" `
  -Form $form
```

An intake `RECEIVED` response still means only that bytes and metadata were
stored. It does not validate the native raster pair or classify PCB quality.

## Future semantic-validator use

The future validator can load a scenario record, choose the declared policy,
inspect the generated references, and compare technical outcome and finding
codes with the versioned expectations. This generator does not execute that
validation, access SQLite, create inspection records, or change statuses.

## What the fixtures do not simulate

The simple board rectangles, component blocks, fiducial-like marks, background
planes, raised/lower scalar regions, and optional invalid-mask cell are only
deterministic byte patterns. They do not simulate optics, illumination,
reflectivity, sensor noise, solder geometry, camera or height-sensor physics,
physical registration, calibrated Z measurements, production variation, or
real PCB defects.

## Focused tests

```powershell
.\.venv\Scripts\python.exe -m pytest `
  .\backend\tests\test_synthetic_inspection_fixtures.py -q
```

The tests generate only under pytest temporary directories and do not commit
binary fixtures.
