# Class-Label Contract

Date: 2026-07-17

## Purpose

Model output indices have no intrinsic defect meaning. A named prediction is
safe only when training, the saved model, and inference all use the same
versioned index-to-name mapping. The authoritative mapping is stored in
`contracts/class_labels.json`; class order must never be copied into a separate
training or inference list.

## Canonical mapping

Contract schema version: `1.0`

| Index | Class name |
| ---: | --- |
| 0 | `dispense_error` |
| 1 | `misalignment` |
| 2 | `missing_part` |
| 3 | `no_defect` |

These are the four real top-level class directories currently present under
`backend/dataset`. No class was added, renamed, or inferred from filenames.

## Repository mapping evidence and mismatch

The training loader uses `torchvision.datasets.ImageFolder`. ImageFolder sorts
class directory names alphabetically, producing:

```text
dispense_error=0, misalignment=1, missing_part=2, no_defect=3
```

The previous predictor maintained this independent list:

```text
missing_part=0, dispense_error=1, misalignment=2, no_defect=3
```

The exact mismatch was:

| Output index | Training/ImageFolder meaning | Previous inference display |
| ---: | --- | --- |
| 0 | `dispense_error` | `missing_part` |
| 1 | `misalignment` | `dispense_error` |
| 2 | `missing_part` | `misalignment` |
| 3 | `no_defect` | `no_defect` |

Therefore indices 0, 1, and 2 could be displayed as the wrong defect name.

## Training enforcement

After ImageFolder discovers the dataset, `backend/data_loader.py` compares its
complete `class_to_idx` dictionary with contract version `1.0`. Training raises
a `ClassLabelContractError` before the first epoch when a class is added,
removed, renamed, or assigned a different index. Labels are never silently
reordered.

After a future training run saves a checkpoint, the trainer writes a sidecar
containing the validated mapping, contract version, and checkpoint SHA-256.
This task did not run training or alter model weights.

## Inference enforcement

`backend/predictor.py` loads the same contract and requires hash-bound model
metadata with status `verified`, the same contract version, and an identical
`class_to_idx` mapping. Compatibility is checked before model weights are
loaded. Model output indices are converted through the contract, and indices
outside the defined range raise `ModelOutputIndexError` instead of indexing a
free-standing label list.

Missing, malformed, hash-mismatched, unverified, or mapping-mismatched metadata
raises `ModelLabelCompatibilityError`; named inference does not continue.

## Existing checkpoint status

`backend/saved_model/best_model.pth` is **unverified** and is intentionally
blocked from named inference.

- SHA-256:
  `c690331eb0b8775fb11fe64efa87e8fdea834d2e3c5cd6b494e92ce03025c44d`
- The checkpoint is a PyTorch state dictionary containing layer tensors and no
  class names, `class_to_idx`, training manifest, or label schema version.
- The repository training code and dataset folders suggest ImageFolder order,
  but do not prove that this exact artifact was created by that code and data.
- `backend/saved_model/best_model.metadata.json` records that uncertainty; it
  does not claim or fabricate a checkpoint mapping.

The model may be classified as verified only after trustworthy provenance is
provided or a future controlled training run writes mapping metadata after
validating the dataset against the authoritative contract.

## Focused tests

From the repository root:

```powershell
python -m pytest .\backend\tests\test_class_labels.py -q
```

The tests validate contract uniqueness and contiguity, current ImageFolder
ordering, inference conversion, mismatch rejection, output bounds, and the
existing checkpoint's blocked status.
