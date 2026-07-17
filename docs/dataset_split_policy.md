# Dataset Split Policy

Date: 2026-07-17

## Purpose

This policy defines how future PCB AOI train, validation, and test partitions
must be created. It does not authorize moving, copying, relabeling, or splitting
the current legacy files. No split should be created until required production
grouping metadata has been reviewed.

## Non-negotiable rule

Random image-level splitting is prohibited whenever related frames, the same
physical board, the same panel, or the same production event could appear more
than once. Related samples must remain in exactly one partition.

Exact SHA-256 duplicates must also remain in one partition. A duplicate must not
be used to increase sample count, and no exact duplicate may cross from training
into validation or test.

## Required grouping hierarchy

Use the strongest trustworthy production identity available, in this order:

1. Physical board or assembly ID.
2. Panel ID, with every board/frame from that panel kept together.
3. Lot or production batch.
4. Capture session or acquisition run.
5. Production date and line/recipe combination.
6. Reviewed sequential image group when adjacent captures are known to describe
   the same board or production event.

If a stronger identifier exists, weaker values must not be used to separate
related samples. For example, frames from one board may not be split merely
because they have different filenames or timestamps.

## Split procedure

1. Preserve the original files and compute immutable SHA-256 hashes.
2. Attach reviewed ground truth and production grouping metadata.
3. Collapse exact duplicates and related captures into indivisible groups.
4. Resolve label conflicts within groups through human review.
5. Allocate complete groups—not individual images—to train, validation, or
   test.
6. Balance defect representation at group level where possible. Never break a
   group to improve class percentages.
7. Lock the test partition before model development. Do not use test results to
   tune preprocessing, thresholds, architecture, or hyperparameters.
8. Store a versioned split manifest containing relative path, SHA-256, class,
   group ID, group evidence, partition, reviewer, and decision rationale.
9. Re-run the leakage audit and require a zero blocking exit status before any
   training or evaluation.

Split ratios are intentionally not prescribed yet. They must be chosen only
after the number and class distribution of independent production groups is
known.

## Proven legacy information

The current repository proves only information directly present in files and
directories:

- Exact file bytes and SHA-256 hashes.
- File paths, filenames, extensions, sizes, and readable image dimensions.
- Canonical training labels represented by direct subdirectories of
  `backend/dataset`.
- Standalone test filenames that begin with a canonical class name.
- Raw-source hierarchy names such as `Crimp`, `Dispensing`, `PINS`, `OK`, `NG`,
  and `NOK`.

The raw hierarchy describes process/status organization; it does not prove a
board, panel, lot, batch, or capture-session identity.

## Heuristics requiring human review

Some filenames contain date-like or timestamp-like prefixes:

- `YYYY_MM_DD...` resembles a calendar date.
- Six-digit pairs such as `110219_143144` resemble a date/time but are
  ambiguous without a format definition.
- Sequentially similar filenames may be adjacent captures.
- `POS...GRADOS` appears to describe position or angle, not a production group.

These are hints only. The audit records them as `heuristic_only` and does not
assign a production group. A domain owner must confirm the naming convention
and map each sample to a reviewed group. Uncertain samples remain blocked from
splitting.

## Current legacy policy outcome

The current `backend/test` folder must not be treated as an independent test
set when exact hashes overlap `backend/dataset`. Current evaluation results are
not trustworthy when that overlap exists.

The legacy dataset is not ready for retraining until, at minimum:

- Cross-partition exact duplicates are resolved in a reviewed manifest.
- Independent production group IDs are established.
- Class imbalance and the one-sample minority class are addressed with verified
  data, not copied images.
- Filename/class inconsistencies are reviewed by the vision/domain team.
- A grouped split manifest passes the leakage audit.

## Audit usage

Run from the repository root:

```powershell
python .\scripts\audit_legacy_dataset.py
```

Default evidence outputs are deliberately stored under the repository-level
`reports` directory, separate from future production inspection reports under
the configurable runtime root:

```text
reports/legacy_dataset_audit.json
reports/legacy_dataset_audit.md
```

The command returns `0` only when no blocking audit issue is found. It returns a
non-zero status for cross-partition duplicates, unreadable images, unexpected
or missing classes, images outside the canonical contract, unsupported source
files, significant class imbalance, or similar blocking integrity failures.

To select explicit evidence paths:

```powershell
python .\scripts\audit_legacy_dataset.py `
  --json-output .\reports\legacy_dataset_audit.json `
  --markdown-output .\reports\legacy_dataset_audit.md
```
