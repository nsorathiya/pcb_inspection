import copy
import hashlib
import json
import os
import shutil
import struct
import zlib
from pathlib import Path

import pytest

from app.services.dataset_validation import (
    EXIT_BLOCKED,
    EXIT_USAGE,
    ValidationStage,
    validate_dataset,
)
from app.services.dataset_validation.reporting import write_reports
from scripts.validate_paired_dataset import main as cli_main

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_ROOT = REPOSITORY_ROOT / "contracts"
FIXED_TIMESTAMP = "2026-07-17T12:00:00Z"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _png_bytes(width: int, height: int, *, color_type: int = 2) -> bytes:
    channels = {0: 1, 2: 3}[color_type]
    raw = b"".join(b"\x00" + bytes(width * channels) for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind)
        crc = zlib.crc32(data, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _tiff_bytes(
    width: int,
    height: int,
    *,
    storage_data_type: str = "uint16",
) -> bytes:
    type_map = {
        "uint16": (16, 1),
        "int16": (16, 2),
        "uint32": (32, 1),
        "int32": (32, 2),
        "float32": (32, 3),
        "float64": (64, 3),
    }
    bits, sample_format = type_map[storage_data_type]
    entries: list[tuple[int, int, int, int]] = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 1, bits),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (273, 4, 1, 0),
        (277, 3, 1, 1),
        (278, 4, 1, height),
        (279, 4, 1, width * height * (bits // 8)),
        (339, 3, 1, sample_format),
    ]
    ifd_size = 2 + len(entries) * 12 + 4
    pixel_offset = 8 + ifd_size
    entries[5] = (273, 4, 1, pixel_offset)
    output = bytearray(b"II" + struct.pack("<H", 42) + struct.pack("<I", 8))
    output.extend(struct.pack("<H", len(entries)))
    for tag, field_type, count, value in entries:
        output.extend(struct.pack("<HHI", tag, field_type, count))
        if field_type == 3:
            output.extend(struct.pack("<H", value) + b"\x00\x00")
        else:
            output.extend(struct.pack("<I", value))
    output.extend(struct.pack("<I", 0))
    output.extend(bytes(width * height * (bits // 8)))
    return bytes(output)


def _sample_metadata(sample_number: int) -> dict:
    sample_id = f"synthetic_sample_{sample_number:06d}"
    return {
        "contract_version": "pcb-aoi-dataset/1.0",
        "sample_id": sample_id,
        "board_id": f"SYNTH-BOARD-{sample_number:04d}",
        "panel_id": f"SYNTH-PANEL-{sample_number:04d}",
        "recipe_id": "SYNTH-RECIPE-A",
        "recipe_version": "1.0",
        "ground_truth": {
            "label": "OK",
            "defect_type": None,
            "taxonomy_version": "pcb-aoi-defects/1.0",
            "label_source": "synthetic_test_fixture",
            "review_status": "approved",
            "reviewed_by": "synthetic_reviewer",
            "reviewed_at": "2026-07-17T10:00:00Z",
        },
        "production": {
            "lot_id": f"SYNTH-LOT-{sample_number:04d}",
            "batch_id": f"SYNTH-BATCH-{sample_number:04d}",
            "capture_session_id": f"SYNTH-SESSION-{sample_number:04d}",
            "production_date": "2026-07-17",
            "station_id": "SYNTH-STATION-01",
            "sequential_group_id": f"SYNTH-SEQUENCE-{sample_number:04d}",
        },
        "capture": {
            "captured_at": "2026-07-17T09:00:00Z",
            "camera_2d_id": "SYNTH-CAMERA-01",
            "sensor_3d_id": "SYNTH-SENSOR-01",
        },
        "files": {
            "rgb_file": "rgb.png",
            "height_file": "height.tiff",
            "validity_mask_file": "validity_mask.png",
            "calibration_file": "calibration.json",
        },
        "rgb": {
            "width": 2,
            "height": 2,
            "channels": 3,
            "color_space": "RGB",
            "bit_depth": 8,
        },
        "height_3d": {
            "representation": "height_map",
            "width": 2,
            "height": 2,
            "storage_format": "tiff",
            "storage_data_type": "uint16",
            "physical_value_type": "height",
            "z_unit": "mm",
            "z_scale": 0.001,
            "z_offset": 0,
            "xy_unit": "mm",
            "x_scale": 0.01,
            "y_scale": 0.01,
            "no_data_policy": "validity_mask",
        },
        "registration": {
            "registration_status": "aligned",
            "registration_method": "synthetic_test_calibration",
            "transform_reference": "calibration.json",
            "calibration_version": "SYNTH-CAL-1.0",
            "coordinate_system": "SYNTH-RGB-PIXEL",
        },
        "integrity": {},
        "provenance": {
            "source_system": "synthetic_test_fixture",
            "source_export_version": "synthetic-1.0",
            "imported_at": "2026-07-17T11:00:00Z",
            "is_synthetic": True,
        },
    }


def _build_package(tmp_path: Path, sample_count: int = 1) -> Path:
    root = tmp_path / "dataset_paired"
    schemas = root / "schemas"
    samples_root = root / "samples"
    manifests = root / "manifests"
    schemas.mkdir(parents=True)
    samples_root.mkdir()
    manifests.mkdir()
    schema_names = (
        "pcb_aoi_sample.schema.json",
        "dataset_split_manifest.schema.json",
        "dataset_manifest.schema.json",
        "defect_taxonomy.json",
    )
    for name in schema_names:
        shutil.copyfile(CONTRACTS_ROOT / name, schemas / name)

    index_records = []
    assignments = []
    board_ids = []
    for number in range(1, sample_count + 1):
        metadata = _sample_metadata(number)
        sample_id = metadata["sample_id"]
        sample_dir = samples_root / sample_id
        sample_dir.mkdir()
        (sample_dir / "rgb.png").write_bytes(_png_bytes(2, 2))
        (sample_dir / "height.tiff").write_bytes(_tiff_bytes(2, 2))
        (sample_dir / "validity_mask.png").write_bytes(
            _png_bytes(2, 2, color_type=0)
        )
        (sample_dir / "calibration.json").write_text(
            '{"synthetic": true}\n', encoding="utf-8"
        )
        metadata["integrity"] = {
            "rgb_sha256": _sha256(sample_dir / "rgb.png"),
            "height_sha256": _sha256(sample_dir / "height.tiff"),
            "validity_mask_sha256": _sha256(sample_dir / "validity_mask.png"),
            "calibration_sha256": _sha256(sample_dir / "calibration.json"),
        }
        metadata_path = sample_dir / "metadata.json"
        _write_json(metadata_path, metadata)
        index_records.append(
            {
                "sample_id": sample_id,
                "metadata_file": f"samples/{sample_id}/metadata.json",
                "metadata_sha256": _sha256(metadata_path),
            }
        )
        assignments.append(
            {
                "sample_id": sample_id,
                "split": "train",
                "grouping_key_value": metadata["board_id"],
            }
        )
        board_ids.append(metadata["board_id"])

    index_path = manifests / "samples.jsonl"
    index_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in index_records),
        encoding="utf-8",
    )
    split_manifest = {
        "contract_version": "pcb-aoi-dataset/1.0",
        "split_manifest_version": "pcb-aoi-split-manifest/1.0",
        "dataset_id": "synthetic_validator_fixture",
        "dataset_version": "0.0-synthetic",
        "created_at": "2026-07-17T11:00:00Z",
        "creation_method": "synthetic_example",
        "grouping_key_type": "board_id",
        "is_synthetic": True,
        "assignments": assignments,
    }
    _write_json(manifests / "split_manifest.json", split_manifest)
    dataset_manifest = {
        "contract_version": "pcb-aoi-dataset/1.0",
        "dataset_manifest_version": "pcb-aoi-dataset-manifest/1.0",
        "dataset_id": "synthetic_validator_fixture",
        "dataset_version": "0.0-synthetic",
        "created_at": "2026-07-17T11:00:00Z",
        "source": {
            "organization": "SYNTHETIC-ORG",
            "team": "SYNTHETIC-TEAM",
        },
        "sample_count": sample_count,
        "expected_class_counts": {"OK": sample_count, "NOK": 0},
        "supported_board_ids": board_ids,
        "supported_recipe_ids": ["SYNTH-RECIPE-A"],
        "schema_references": {
            "sample_schema": "schemas/pcb_aoi_sample.schema.json",
            "split_manifest_schema": "schemas/dataset_split_manifest.schema.json",
            "dataset_manifest_schema": "schemas/dataset_manifest.schema.json",
            "defect_taxonomy": "schemas/defect_taxonomy.json",
        },
        "samples_manifest_reference": "manifests/samples.jsonl",
        "split_manifest_reference": "manifests/split_manifest.json",
        "dataset_status": "draft",
        "approval_status": "not_reviewed",
        "known_limitations": ["SYNTHETIC TEST FIXTURE"],
        "is_synthetic": True,
    }
    _write_json(root / "dataset_manifest.json", dataset_manifest)
    return root


def _metadata_path(root: Path, number: int = 1) -> Path:
    return root / "samples" / f"synthetic_sample_{number:06d}" / "metadata.json"


def _rewrite_metadata(root: Path, metadata: dict, number: int = 1) -> None:
    path = _metadata_path(root, number)
    _write_json(path, metadata)
    index_path = root / "manifests" / "samples.jsonl"
    records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    records[number - 1]["metadata_sha256"] = _sha256(path)
    index_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def _validate(root: Path, stage: ValidationStage = ValidationStage.TECHNICAL):
    return validate_dataset(root, stage, validation_timestamp=FIXED_TIMESTAMP)


def test_valid_technical_validation_package_passes_and_is_read_only(tmp_path) -> None:
    root = _build_package(tmp_path)
    tracked = [path for path in root.rglob("*") if path.is_file()]
    before = {path: (_sha256(path), path.stat().st_mtime_ns) for path in tracked}

    report = _validate(root)

    after = {path: (_sha256(path), path.stat().st_mtime_ns) for path in tracked}
    assert report.exit_code == 0
    assert report.summary["valid_pairs"] == 1
    assert before == after


@pytest.mark.parametrize(
    ("filename", "expected_code"),
    [("rgb.png", "file.missing"), ("height.tiff", "file.missing")],
)
def test_missing_required_pair_file_fails(tmp_path, filename, expected_code) -> None:
    root = _build_package(tmp_path)
    os.remove(root / "samples" / "synthetic_sample_000001" / filename)

    report = _validate(root)

    assert expected_code in _codes(report)
    assert report.exit_code == EXIT_BLOCKED


@pytest.mark.parametrize("hash_field", ["rgb_sha256", "height_sha256"])
def test_incorrect_file_hash_fails(tmp_path, hash_field) -> None:
    root = _build_package(tmp_path)
    metadata = _json(_metadata_path(root))
    metadata["integrity"][hash_field] = "0" * 64
    _rewrite_metadata(root, metadata)

    report = _validate(root)

    assert any(finding.code.endswith("hash_mismatch") for finding in report.findings)


def test_declared_dimensions_must_match_actual_image(tmp_path) -> None:
    root = _build_package(tmp_path)
    metadata = _json(_metadata_path(root))
    metadata["rgb"]["width"] = 99
    _rewrite_metadata(root, metadata)

    assert "pair.rgb_width_mismatch" in _codes(_validate(root))


def test_declared_height_storage_type_must_match_actual(tmp_path) -> None:
    root = _build_package(tmp_path)
    metadata = _json(_metadata_path(root))
    metadata["height_3d"]["storage_data_type"] = "float32"
    _rewrite_metadata(root, metadata)

    assert "pair.height_storage_data_type_mismatch" in _codes(_validate(root))


def test_invalid_mask_policy_fails(tmp_path) -> None:
    root = _build_package(tmp_path)
    os.remove(root / "samples" / "synthetic_sample_000001" / "validity_mask.png")

    assert "pair.validity_mask_required" in _codes(_validate(root))


def test_required_transform_file_must_exist(tmp_path) -> None:
    root = _build_package(tmp_path)
    metadata = _json(_metadata_path(root))
    metadata["registration"]["registration_status"] = "requires_transform"
    metadata["registration"]["transform_reference"] = "missing-transform.json"
    _rewrite_metadata(root, metadata)

    assert "registration.transform_required" in _codes(_validate(root))


def test_sample_path_traversal_is_rejected(tmp_path) -> None:
    root = _build_package(tmp_path)
    metadata = _json(_metadata_path(root))
    metadata["files"]["rgb_file"] = "../outside.png"
    _rewrite_metadata(root, metadata)

    assert "path.traversal" in _codes(_validate(root))


def test_manifest_path_escape_is_rejected(tmp_path) -> None:
    root = _build_package(tmp_path)
    manifest = _json(root / "dataset_manifest.json")
    manifest["samples_manifest_reference"] = "../outside.jsonl"
    _write_json(root / "dataset_manifest.json", manifest)

    assert "path.traversal" in _codes(_validate(root))


def test_duplicate_sample_ids_fail(tmp_path) -> None:
    root = _build_package(tmp_path)
    index = root / "manifests" / "samples.jsonl"
    first = index.read_text(encoding="utf-8")
    index.write_text(first + first, encoding="utf-8")

    assert "sample.id_duplicate" in _codes(_validate(root))


def test_one_physical_file_referenced_by_unrelated_samples_is_reported(tmp_path) -> None:
    root = _build_package(tmp_path, sample_count=2)
    first = root / "samples" / "synthetic_sample_000001" / "rgb.png"
    second = root / "samples" / "synthetic_sample_000002" / "rgb.png"
    second.unlink()
    os.link(first, second)

    assert "file.shared_between_samples" in _codes(_validate(root))


@pytest.mark.parametrize(
    ("label", "defect_type", "expected_code"),
    [
        ("NOK", "unsupported_defect", "taxonomy.nok_defect_unknown"),
        ("OK", "misalignment", "taxonomy.ok_has_defect"),
        ("NOK", None, "taxonomy.nok_defect_unknown"),
    ],
)
def test_invalid_business_label_and_defect_combinations_fail(
    tmp_path, label, defect_type, expected_code
) -> None:
    root = _build_package(tmp_path)
    metadata = _json(_metadata_path(root))
    metadata["ground_truth"]["label"] = label
    metadata["ground_truth"]["defect_type"] = defect_type
    _rewrite_metadata(root, metadata)

    assert expected_code in _codes(_validate(root))


def test_dataset_sample_count_mismatch_fails(tmp_path) -> None:
    root = _build_package(tmp_path)
    manifest = _json(root / "dataset_manifest.json")
    manifest["sample_count"] = 2
    _write_json(root / "dataset_manifest.json", manifest)

    assert "manifest.sample_count_mismatch" in _codes(_validate(root))


def test_declared_class_counts_mismatch_fails(tmp_path) -> None:
    root = _build_package(tmp_path)
    manifest = _json(root / "dataset_manifest.json")
    manifest["expected_class_counts"] = {"OK": 0, "NOK": 1}
    _write_json(root / "dataset_manifest.json", manifest)

    assert "manifest.class_counts_mismatch" in _codes(_validate(root))


def test_unknown_split_sample_fails(tmp_path) -> None:
    root = _build_package(tmp_path)
    path = root / "manifests" / "split_manifest.json"
    split = _json(path)
    split["assignments"].append(
        {
            "sample_id": "unknown_sample",
            "split": "test",
            "grouping_key_value": "UNKNOWN-BOARD",
        }
    )
    _write_json(path, split)

    assert "split.unknown_sample" in _codes(_validate(root))


def test_duplicate_split_assignment_fails(tmp_path) -> None:
    root = _build_package(tmp_path)
    path = root / "manifests" / "split_manifest.json"
    split = _json(path)
    duplicate = copy.deepcopy(split["assignments"][0])
    duplicate["split"] = "test"
    split["assignments"].append(duplicate)
    _write_json(path, split)

    assert "split.duplicate_assignment" in _codes(_validate(root))


def test_protected_group_cannot_cross_splits(tmp_path) -> None:
    root = _build_package(tmp_path, sample_count=2)
    first_metadata = _json(_metadata_path(root, 1))
    second_metadata = _json(_metadata_path(root, 2))
    second_metadata["board_id"] = first_metadata["board_id"]
    _rewrite_metadata(root, second_metadata, number=2)
    manifest = _json(root / "dataset_manifest.json")
    manifest["supported_board_ids"] = [first_metadata["board_id"]]
    _write_json(root / "dataset_manifest.json", manifest)
    split_path = root / "manifests" / "split_manifest.json"
    split = _json(split_path)
    split["assignments"][1]["split"] = "test"
    split["assignments"][1]["grouping_key_value"] = first_metadata["board_id"]
    _write_json(split_path, split)

    assert "split.protected_group_crossing" in _codes(_validate(root))


def test_excluded_sample_without_reason_fails(tmp_path) -> None:
    root = _build_package(tmp_path)
    path = root / "manifests" / "split_manifest.json"
    split = _json(path)
    split["assignments"][0]["split"] = "excluded"
    _write_json(path, split)

    assert "split.exclusion_reason_required" in _codes(_validate(root))


@pytest.mark.parametrize(
    "stage", [ValidationStage.TRAINING, ValidationStage.PRODUCTION]
)
def test_synthetic_package_is_blocked_beyond_technical_validation(
    tmp_path, stage
) -> None:
    root = _build_package(tmp_path)

    report = _validate(root, stage)

    assert "stage.synthetic_dataset" in _codes(report)
    assert report.exit_code == EXIT_BLOCKED


def test_report_ordering_and_hashes_are_repeatable_with_fixed_timestamp(
    tmp_path,
) -> None:
    root = _build_package(tmp_path)
    first = _validate(root)
    second = _validate(root)
    first_paths = write_reports(first, root, tmp_path / "reports-one")
    second_paths = write_reports(second, root, tmp_path / "reports-two")

    assert first.to_dict() == second.to_dict()
    assert [_sha256(path) for path in first_paths] == [
        _sha256(path) for path in second_paths
    ]


def test_blocking_findings_return_exit_code_one(tmp_path) -> None:
    root = _build_package(tmp_path)
    os.remove(root / "samples" / "synthetic_sample_000001" / "rgb.png")

    assert _validate(root).exit_code == EXIT_BLOCKED


def test_invalid_cli_usage_returns_exit_code_two(tmp_path) -> None:
    missing = tmp_path / "missing"

    result = cli_main(
        [
            "--dataset-root",
            str(missing),
            "--stage",
            "technical-validation",
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert result == EXIT_USAGE


def test_valid_cli_run_writes_reports_outside_dataset(tmp_path) -> None:
    root = _build_package(tmp_path)
    report_directory = tmp_path / "cli-reports"

    result = cli_main(
        [
            "--dataset-root",
            str(root),
            "--stage",
            "technical-validation",
            "--report-dir",
            str(report_directory),
        ]
    )

    assert result == 0
    assert (report_directory / "paired_dataset_validation.json").is_file()
    assert (report_directory / "paired_dataset_validation.md").is_file()


def test_argparse_missing_required_arguments_uses_exit_code_two() -> None:
    with pytest.raises(SystemExit) as error:
        cli_main([])

    assert error.value.code == EXIT_USAGE


def test_report_directory_inside_dataset_is_rejected(tmp_path) -> None:
    root = _build_package(tmp_path)

    result = cli_main(
        [
            "--dataset-root",
            str(root),
            "--stage",
            "technical-validation",
            "--report-dir",
            str(root / "reports"),
        ]
    )

    assert result == EXIT_USAGE
