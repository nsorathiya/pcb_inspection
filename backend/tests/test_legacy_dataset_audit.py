import struct
from pathlib import Path

from app.core.class_labels import load_class_label_contract
from scripts.audit_legacy_dataset import (
    audit_exit_code,
    audit_repository,
    write_reports,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPOSITORY_ROOT / "contracts" / "class_labels.json"


def _bmp_bytes(value: int, width: int = 2, height: int = 2) -> bytes:
    row_stride = (width * 3 + 3) & ~3
    row = bytes([value, value, value]) * width
    row += b"\x00" * (row_stride - width * 3)
    pixel_data = row * height
    file_size = 54 + len(pixel_data)
    file_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    dib_header = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height,
        1,
        24,
        0,
        len(pixel_data),
        2835,
        2835,
        0,
        0,
    )
    return file_header + dib_header + pixel_data


def _build_clean_repository(tmp_path: Path) -> tuple[Path, list[str]]:
    contract = load_class_label_contract(CONTRACT_PATH)
    class_names = [label.name for label in contract.classes]
    training_root = tmp_path / "backend" / "dataset"
    for index, class_name in enumerate(class_names, start=1):
        class_dir = training_root / class_name
        class_dir.mkdir(parents=True)
        (class_dir / f"{class_name}.bmp").write_bytes(_bmp_bytes(index))
    for relative_dir in (
        "backend/dataset_raw",
        "backend/test",
        "backend/uploads",
        "backend/annotated_output",
    ):
        (tmp_path / relative_dir).mkdir(parents=True)
    return tmp_path, class_names


def _audit(repo_root: Path):
    return audit_repository(repo_root, contract_path=CONTRACT_PATH)


def test_exact_duplicates_are_detected(tmp_path) -> None:
    repo_root, class_names = _build_clean_repository(tmp_path)
    class_dir = repo_root / "backend" / "dataset" / class_names[0]
    original = class_dir / f"{class_names[0]}.bmp"
    (class_dir / "duplicate.bmp").write_bytes(original.read_bytes())

    report = _audit(repo_root)

    assert len(report["exact_duplicate_groups"]) == 1
    assert report["exact_duplicate_groups"][0]["partitions"] == ["training"]


def test_cross_partition_duplicates_are_detected(tmp_path) -> None:
    repo_root, class_names = _build_clean_repository(tmp_path)
    training_image = (
        repo_root
        / "backend"
        / "dataset"
        / class_names[0]
        / f"{class_names[0]}.bmp"
    )
    test_image = repo_root / "backend" / "test" / f"{class_names[0]}_copy.bmp"
    test_image.write_bytes(training_image.read_bytes())

    report = _audit(repo_root)

    assert len(report["cross_partition_duplicate_groups"]) == 1
    assert report["summary"]["training_test_overlap_count"] == 1


def test_same_filename_with_different_content_is_distinguished(tmp_path) -> None:
    repo_root, class_names = _build_clean_repository(tmp_path)
    training_path = (
        repo_root / "backend" / "dataset" / class_names[0] / "shared.bmp"
    )
    raw_path = repo_root / "backend" / "dataset_raw" / "shared.bmp"
    training_path.write_bytes(_bmp_bytes(201))
    raw_path.write_bytes(_bmp_bytes(202))

    report = _audit(repo_root)

    filename_groups = report["duplicate_filenames_different_content"]
    assert len(filename_groups) == 1
    assert filename_groups[0]["filename"] == "shared.bmp"
    assert len(filename_groups[0]["sha256_values"]) == 2


def test_unexpected_class_directories_are_reported(tmp_path) -> None:
    repo_root, _ = _build_clean_repository(tmp_path)
    unexpected_dir = repo_root / "backend" / "dataset" / "unexpected_class"
    unexpected_dir.mkdir()
    (unexpected_dir / "sample.bmp").write_bytes(_bmp_bytes(77))

    report = _audit(repo_root)

    assert report["unexpected_class_directories"] == ["unexpected_class"]
    assert "unexpected_class_directories" in {
        issue["code"] for issue in report["blocking_issues"]
    }


def test_unreadable_images_are_reported(tmp_path) -> None:
    repo_root, class_names = _build_clean_repository(tmp_path)
    unreadable = (
        repo_root / "backend" / "dataset" / class_names[0] / "unreadable.bmp"
    )
    unreadable.write_bytes(b"not a bitmap")

    report = _audit(repo_root)

    assert report["unreadable_files"] == [
        f"backend/dataset/{class_names[0]}/unreadable.bmp"
    ]
    assert report["summary"]["unreadable_image_files"] == 1


def test_extension_content_mismatch_is_readable_and_reported(tmp_path) -> None:
    repo_root, class_names = _build_clean_repository(tmp_path)
    mismatched = (
        repo_root / "backend" / "dataset" / class_names[0] / "renamed.jpg"
    )
    mismatched.write_bytes(_bmp_bytes(91))

    report = _audit(repo_root)
    record = next(
        item for item in report["files"] if item["relative_path"].endswith("renamed.jpg")
    )

    assert record["readable"] is True
    assert record["detected_format"] == "BMP"
    assert record["extension_matches_content"] is False
    assert report["extension_content_mismatches"] == [record["relative_path"]]


def test_output_ordering_and_reports_are_deterministic(tmp_path) -> None:
    repo_root, _ = _build_clean_repository(tmp_path)

    first_report = _audit(repo_root)
    second_report = _audit(repo_root)
    first_json = tmp_path / "first" / "audit.json"
    first_markdown = tmp_path / "first" / "audit.md"
    second_json = tmp_path / "second" / "audit.json"
    second_markdown = tmp_path / "second" / "audit.md"
    write_reports(first_report, first_json, first_markdown)
    write_reports(second_report, second_json, second_markdown)

    assert first_report == second_report
    assert [record["relative_path"] for record in first_report["files"]] == sorted(
        record["relative_path"] for record in first_report["files"]
    )
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_markdown.read_bytes() == second_markdown.read_bytes()


def test_blocking_issues_produce_nonzero_result(tmp_path) -> None:
    repo_root, class_names = _build_clean_repository(tmp_path)
    training_image = (
        repo_root
        / "backend"
        / "dataset"
        / class_names[0]
        / f"{class_names[0]}.bmp"
    )
    (repo_root / "backend" / "test" / "leak.bmp").write_bytes(
        training_image.read_bytes()
    )

    assert audit_exit_code(_audit(repo_root)) != 0


def test_clean_synthetic_input_produces_success(tmp_path) -> None:
    repo_root, _ = _build_clean_repository(tmp_path)

    report = _audit(repo_root)

    assert report["blocking_issues"] == []
    assert audit_exit_code(report) == 0
