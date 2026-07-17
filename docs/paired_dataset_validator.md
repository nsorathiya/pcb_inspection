# Read-Only Paired Dataset Validator

Date: 2026-07-17

## Purpose

`scripts/validate_paired_dataset.py` validates a package governed by
`pcb-aoi-dataset/1.0`. It checks package structure, authoritative schemas,
sample metadata, exact hashes, native raster metadata, pairing evidence,
dataset counts, split semantics, and readiness for a requested delivery stage.

It does not modify, repair, convert, align, normalize, resize, relabel, train,
or load a model. Passing validation does not prove model accuracy, production
performance, physical calibration accuracy, or fitness for a specific AOI
decision threshold.

The legacy 2D-only dataset is not a valid input for this CLI.

## Installation

From the repository root, create/activate the project virtual environment and
install development requirements:

```powershell
python -m pip install -r .\backend\requirements-dev.txt
```

The validator uses the existing `jsonschema` development dependency and Python
standard-library binary inspection. It does not require OpenCV, PyTorch,
torchvision, Pillow, or a model checkpoint.

## Commands

Technical validation checks the minimum package and native-data evidence:

```powershell
python .\scripts\validate_paired_dataset.py `
  --dataset-root C:\data\dataset_paired `
  --stage technical-validation `
  --report-dir C:\data\validation_reports\technical
```

Model-training readiness adds approved ground truth, reviewed grouping,
non-synthetic status, registration/calibration evidence, verified hashes, and
leakage-safe split requirements:

```powershell
python .\scripts\validate_paired_dataset.py `
  --dataset-root C:\data\dataset_paired `
  --stage model-training `
  --report-dir C:\data\validation_reports\training
```

Production acceptance additionally requires an approved production-candidate
dataset, approved split manifest, test and holdout assignments, calibration and
source-export traceability, and a known-limitations declaration:

```powershell
python .\scripts\validate_paired_dataset.py `
  --dataset-root C:\data\dataset_paired `
  --stage production-acceptance `
  --report-dir C:\data\validation_reports\production
```

The report directory must be outside the supplied dataset root. The validator
never creates reports or any other file inside the package.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Validation passed for the requested stage. |
| 1 | Validation completed and blocking dataset findings exist. |
| 2 | CLI usage or configuration error, including a missing root or unsafe report location. |
| 3 | Unexpected validator failure. |

Warnings do not change a pass to blocked. Errors always produce exit code 1.

## Reports

Each run writes deterministic files under `--report-dir`:

```text
paired_dataset_validation.json
paired_dataset_validation.md
```

The reports contain validator/timestamp/stage identity, overall status,
dataset identity, OK/NOK and defect counts, board and recipe-version counts,
pair/hash/file/schema/registration/split/stage summaries, ordered findings,
and an ordered per-sample inventory. Dataset paths are relative; user-specific
absolute dataset paths are not included.

The timestamp reflects the validation run. Tests inject a fixed timestamp to
prove byte-for-byte repeatability.

## Safety and path rules

- All dataset files are opened read-only.
- Absolute paths, drive-qualified paths, backslashes, `..`, root escapes, and
  symbolic links in dataset references are rejected.
- Every reference is resolved and checked against the canonical dataset root.
- Sample data references are additionally confined to their sample directory.
- Only regular files are accepted.
- Hard-linked RGB/height files reused by unrelated sample IDs are detected by
  filesystem identity.
- SHA-256 is streamed over the exact bytes; mismatches are never rewritten.
- The tool does not claim that equal RGB/height dimensions prove registration.

## Supported native inspection

2D inspection is content-based, not extension-based:

- PNG: signature, chunks, CRC, compressed-stream readability, dimensions,
  channels/mode, and bit depth.
- BMP: baseline DIB dimensions, channels, and bit depth.
- JPEG: frame dimensions, components/mode, and precision.
- TIFF: the classic strip-based, contiguous, uncompressed/Deflate RGB or
  grayscale subset documented in the semantic-validation contract.

Native raster 3D inspection supports:

- Classic, strip-based scalar TIFF with uncompressed or Deflate storage.
- Non-interlaced PNG color type 0 with exactly one 16-bit unsigned scalar
  channel.
- Two-dimensional NumPy `.npy` arrays with contract-supported integer or
  floating scalar types.

The validator reads metadata and compressed bytes without converting or
normalizing values. It blocks color/multi-sample TIFFs and rejects 8-bit,
palette, RGB, RGBA, and grayscale-plus-alpha PNG previews as raw height data.
Dataset contract 1.0 still controls which declared storage formats are valid
for a package; parser readability alone does not override that schema.

Current explicit limitations:

- BigTIFF, tiled TIFF, LZW TIFF, HDF5, and EXR are blocked by validator 1.0.
- Point clouds and meshes are outside dataset contract 1.0.
- Vendor calibration semantics are evidenced by version/file/hash but are not
  physically recalibrated by this software.
- The current dataset manifest declares recipe IDs but has no dataset-level
  recipe-version allow-list; observed sample recipe versions are reported.

Add support only with representative files and focused tests. Never silently
interpret an unsupported container.

## Typical blocking findings

Examples include:

- `file.missing`: a referenced RGB, height, mask, or calibration file is absent.
- `integrity.rgb_hash_mismatch`: exact file bytes do not match metadata.
- `pair.height_storage_data_type_mismatch`: TIFF/NPY native type differs from
  `storage_data_type`.
- `taxonomy.nok_defect_unknown`: NOK uses an unknown or missing defect type.
- `split.duplicate_assignment`: a sample appears more than once.
- `split.protected_group_crossing`: a board, lot, session, panel, batch, or
  sequential group crosses protected splits.
- `stage.synthetic_dataset`: synthetic evidence was submitted for training or
  production acceptance.

The correct response is to correct the source package through the owning vision
team's reviewed process and rerun validation. The validator never repairs it.
