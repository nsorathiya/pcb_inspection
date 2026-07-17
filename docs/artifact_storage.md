# Immutable Inspection-Artifact Storage Foundation

Date: 2026-07-17

## Scope

This foundation stores exact artifact bytes below the configured runtime root
and can coordinate a successful write with an `inspection_artifacts` database
row. It does not add HTTP or multipart handling, create inspections, validate
image semantics, preprocess images, generate previews, run inference, load a
model, or implement retention cleanup.

Raw artifact storage proves byte integrity only. Paired RGB/height semantics,
native bit depth, dimensions, registration, calibration meaning, and dataset
readiness remain the responsibility of the separate input/dataset validators.

## Runtime layout

Every inspection ID is a canonical UUID generated or validated outside client
filenames. The deterministic layout is:

```text
runtime/
  raw_uploads/<inspection-id>/
    rgb/rgb_raw.<approved-extension>
    height/height_raw.<approved-extension>
    masks/validity_mask.<approved-extension>
    calibration/calibration.<approved-extension>
  previews/<inspection-id>/
    rgb_preview.<approved-extension>
    height_preview.<approved-extension>
  results/<inspection-id>/
    result_overlay.<approved-extension>
  reports/<inspection-id>/
    report.<approved-extension>
```

The database stores POSIX-style paths relative to `runtime`, such as
`raw_uploads/<inspection-id>/rgb/rgb_raw.png`. Public storage results never
contain the absolute runtime root.

## Artifact type authority

`app.db.models.ArtifactType` is the single type contract used by the database,
path router, size-limit router, storage result, and registration coordinator:

- `RGB_RAW`
- `HEIGHT_RAW`
- `VALIDITY_MASK`
- `CALIBRATION`
- `RGB_PREVIEW`
- `HEIGHT_PREVIEW`
- `RESULT_OVERLAY`
- `REPORT`

Storage routing checks that all enum values are covered. Unknown values fail;
there is no second independent artifact-type enum or silently selected default.

## Safe filenames and paths

The stored stem is generated entirely from the artifact type. Board IDs, lot
IDs, operator names, original filenames, and other user metadata are never
used as directories or stems. The original filename is informational metadata
only.

When present, only its final approved extension is retained and normalized to
lowercase. Raster, native-height, mask, calibration, and report types have
separate allow-lists. Double suffixes ending in executable or unsupported
types, including `.png.exe`, are rejected. A missing extension uses the safe
`.bin` fallback. Windows drive names, reserved device names, hidden names,
absolute paths, separators, and traversal text therefore cannot control the
physical destination.

Before storage or registration, the service verifies that the path is
relative, contains no `..`, backslash, colon, or drive prefix, resolves below
the configured category, and has a normalized `/` representation.

## Atomic write and immutability process

1. Create the managed inspection/category directories.
2. Reject existing symbolic links or Windows reparse points in the managed
   path.
3. Create a uniquely named temporary file in the final destination directory.
4. Stream bytes into the temporary file while calculating SHA-256 and size.
5. Flush and `fsync` the temporary file.
6. Verify expected hash, expected size, and the per-type size limit.
7. Recheck path confinement and redirection before finalization.
8. Create the final name atomically and exclusively with a same-filesystem hard
   link, then remove the temporary name.
9. On POSIX, `fsync` the parent directory.

The final filename is never opened for an in-place write. If it already exists,
the service streams its hash and size. Exact matching content returns an
idempotent result; different content raises a conflict and remains untouched.
Unique temporary names and exclusive filesystem finalization protect concurrent
processes without relying on a Python lock.

No update, overwrite, or general delete operation is exposed. Internal removal
is limited to a newly created file owned by the current operation when database
registration fails and the bytes still match that operation.

## Integrity and size limits

SHA-256 is lowercase hexadecimal and is calculated over exact source bytes.
Streams are read incrementally rather than loaded into memory. Expected hash
and expected byte size are optional; a mismatch rejects the operation before
the final name becomes visible and removes its temporary file.

Limits are positive integers and are enforced while streaming, before writing
the chunk that would exceed the limit:

| Variable | Development default | Applies to |
| --- | ---: | --- |
| `PCB_AOI_MAX_RGB_BYTES` | 52,428,800 (50 MiB) | `RGB_RAW` |
| `PCB_AOI_MAX_HEIGHT_BYTES` | 268,435,456 (256 MiB) | `HEIGHT_RAW` |
| `PCB_AOI_MAX_MASK_BYTES` | 67,108,864 (64 MiB) | `VALIDITY_MASK` |
| `PCB_AOI_MAX_CALIBRATION_BYTES` | 5,242,880 (5 MiB) | `CALIBRATION` |
| `PCB_AOI_MAX_GENERATED_ARTIFACT_BYTES` | 52,428,800 (50 MiB) | Previews, overlays, and reports |

These are conservative development limits, not production capacity claims.

## Database coordination and rollback

`ArtifactRegistrationService.store_and_register` stores bytes first and then
registers inspection ID, artifact type, relative path, SHA-256, byte size,
media type, and creation timestamp through the existing repository. The
inspection must already exist; the coordinator never creates one.

If registration fails, a final file created and still owned by that operation
is removed only when no matching database row exists and its current hash and
size still match. A pre-existing idempotent file is never deleted. Once a
database row succeeds, later caller or response failures are outside this
transaction boundary and do not remove the artifact.

The filesystem and SQLite cannot provide one distributed atomic transaction.
A process or machine crash in the narrow interval between finalization and
registration can leave an orphan. Startup reconciliation and retention tooling
are deliberately deferred; they must be evidence-driven and must never delete
registered immutable evidence blindly.

## Windows and filesystem notes

The service rejects portable symbolic links and checks the Windows reparse
attribute when the runtime reports it. Windows junctions, mount points,
network filesystems, antivirus/filter drivers, and races caused by a privileged
actor changing directory links cannot be made fully equivalent to POSIX
directory-descriptor confinement through portable Python path APIs.

Safe exclusive finalization requires same-filesystem hard-link support. This is
available on normal NTFS development volumes. A filesystem that cannot create
hard links fails safely instead of exposing a partially copied final file.
Windows does not provide the same portable directory `fsync` operation used on
POSIX; file contents are still flushed and `fsync`ed before finalization.

## Tests

From the repository root in Windows PowerShell:

```powershell
python -m pytest .\backend\tests\test_artifact_storage.py -q
python -m pytest .\backend\tests\test_database.py `
  .\backend\tests\test_runtime_paths.py `
  .\backend\tests\test_health.py -q
python -m pytest .\backend\tests -q
```

Tests use temporary runtime roots and real temporary SQLite databases. The
symbolic-link test skips only when the current Windows policy or filesystem
does not permit an unprivileged test symlink.
