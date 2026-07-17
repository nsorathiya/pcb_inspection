from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.class_labels import (  # noqa: E402
    ClassLabelContract,
    load_class_label_contract,
)

AUDIT_SCHEMA_VERSION = "1.0"
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
SIGNIFICANT_IMBALANCE_RATIO = 3.0


@dataclass(frozen=True)
class SourceSpec:
    name: str
    relative_root: str
    partition: str
    is_source: bool
    class_layout: bool = False
    generated_reason: str | None = None
    required: bool = False


SOURCE_SPECS = (
    SourceSpec(
        name="training",
        relative_root="backend/dataset",
        partition="training",
        is_source=True,
        class_layout=True,
        required=True,
    ),
    SourceSpec(
        name="raw_source",
        relative_root="backend/dataset_raw",
        partition="raw_source",
        is_source=True,
    ),
    SourceSpec(
        name="test",
        relative_root="backend/test",
        partition="test",
        is_source=True,
    ),
    SourceSpec(
        name="legacy_uploads",
        relative_root="backend/uploads",
        partition="legacy_uploads",
        is_source=False,
        generated_reason="Legacy runtime upload copy; not an authoritative dataset source.",
    ),
    SourceSpec(
        name="annotated_outputs",
        relative_root="backend/annotated_output",
        partition="annotated_outputs",
        is_source=False,
        generated_reason="Generated prediction/annotation output; never source data.",
    ),
)


class ImageInspectionError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bmp_metadata(path: Path) -> dict[str, Any]:
    header = path.read_bytes()[:54]
    if len(header) < 30 or header[:2] != b"BM":
        raise ImageInspectionError("invalid BMP header")
    dib_size = struct.unpack_from("<I", header, 14)[0]
    if dib_size == 12:
        width, height = struct.unpack_from("<HH", header, 18)
        bits_per_pixel = struct.unpack_from("<H", header, 24)[0]
    elif dib_size >= 40 and len(header) >= 30:
        width, signed_height = struct.unpack_from("<ii", header, 18)
        height = abs(signed_height)
        bits_per_pixel = struct.unpack_from("<H", header, 28)[0]
    else:
        raise ImageInspectionError(f"unsupported BMP DIB header size {dib_size}")
    if width <= 0 or height <= 0 or bits_per_pixel <= 0:
        raise ImageInspectionError("invalid BMP dimensions or bit depth")
    if bits_per_pixel <= 8:
        channels = 1
        mode = f"{bits_per_pixel}-bit indexed/grayscale"
    elif bits_per_pixel == 24:
        channels = 3
        mode = "RGB"
    elif bits_per_pixel == 32:
        channels = 4
        mode = "RGBA"
    else:
        channels = None
        mode = f"{bits_per_pixel}-bit"
    return {
        "width": width,
        "height": height,
        "channel_count": channels,
        "image_mode": mode,
        "inspection_method": "BMP header",
    }


def _jpeg_metadata(path: Path) -> dict[str, Any]:
    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    with path.open("rb") as image_file:
        if image_file.read(2) != b"\xff\xd8":
            raise ImageInspectionError("invalid JPEG start marker")
        while True:
            marker_start = image_file.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker_byte = image_file.read(1)
            while marker_byte == b"\xff":
                marker_byte = image_file.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]
            if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            raw_length = image_file.read(2)
            if len(raw_length) != 2:
                raise ImageInspectionError("truncated JPEG segment length")
            segment_length = struct.unpack(">H", raw_length)[0]
            if segment_length < 2:
                raise ImageInspectionError("invalid JPEG segment length")
            segment = image_file.read(segment_length - 2)
            if len(segment) != segment_length - 2:
                raise ImageInspectionError("truncated JPEG segment")
            if marker in start_of_frame_markers:
                if len(segment) < 6:
                    raise ImageInspectionError("truncated JPEG frame header")
                height, width = struct.unpack_from(">HH", segment, 1)
                channels = segment[5]
                if width <= 0 or height <= 0 or channels <= 0:
                    raise ImageInspectionError("invalid JPEG dimensions or channels")
                mode = {1: "L", 3: "RGB", 4: "CMYK"}.get(
                    channels,
                    f"{channels}-component",
                )
                return {
                    "width": width,
                    "height": height,
                    "channel_count": channels,
                    "image_mode": mode,
                    "inspection_method": "JPEG frame header",
                }
            if marker == 0xDA:
                break
    raise ImageInspectionError("JPEG frame header not found")


def _png_metadata(path: Path) -> dict[str, Any]:
    header = path.read_bytes()[:29]
    if len(header) < 29 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageInspectionError("invalid PNG header")
    if header[12:16] != b"IHDR":
        raise ImageInspectionError("PNG IHDR chunk not found")
    width, height = struct.unpack_from(">II", header, 16)
    bit_depth = header[24]
    color_type = header[25]
    channels_by_color_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    mode_by_color_type = {0: "L", 2: "RGB", 3: "P", 4: "LA", 6: "RGBA"}
    if width <= 0 or height <= 0 or color_type not in channels_by_color_type:
        raise ImageInspectionError("invalid PNG dimensions or color type")
    return {
        "width": width,
        "height": height,
        "channel_count": channels_by_color_type[color_type],
        "image_mode": f"{mode_by_color_type[color_type]} ({bit_depth}-bit)",
        "inspection_method": "PNG IHDR header",
    }


def inspect_image(path: Path) -> dict[str, Any]:
    extension = path.suffix.lower()
    signature = path.read_bytes()[:8]
    if signature.startswith(b"BM"):
        parser = _bmp_metadata
        detected_format = "BMP"
        matching_extensions = {".bmp"}
    elif signature.startswith(b"\xff\xd8"):
        parser = _jpeg_metadata
        detected_format = "JPEG"
        matching_extensions = {".jpeg", ".jpg"}
    elif signature == b"\x89PNG\r\n\x1a\n":
        parser = _png_metadata
        detected_format = "PNG"
        matching_extensions = {".png"}
    else:
        parser = {
            ".bmp": _bmp_metadata,
            ".jpeg": _jpeg_metadata,
            ".jpg": _jpeg_metadata,
            ".png": _png_metadata,
        }.get(extension)
        if parser is None:
            raise ImageInspectionError(
                f"unsupported extension {extension or '<none>'}"
            )
        detected_format = None
        matching_extensions = set()
    metadata = parser(path)
    metadata["detected_format"] = detected_format
    metadata["extension_matches_content"] = extension in matching_extensions
    return metadata


def _infer_test_label(filename: str, contract: ClassLabelContract) -> str | None:
    for class_name in sorted(contract.class_to_idx, key=len, reverse=True):
        if filename.startswith(f"{class_name}_"):
            return class_name
    return None


def _grouping_hint(filename: str, class_label: str | None) -> dict[str, str] | None:
    stem = Path(filename).stem
    if class_label and stem.startswith(f"{class_label}_"):
        stem = stem[len(class_label) + 1 :]
    calendar_match = re.match(r"^(\d{4}_\d{2}_\d{2})", stem)
    if calendar_match:
        return {
            "kind": "filename_date_like_prefix",
            "value": calendar_match.group(1),
            "certainty": "heuristic_only",
        }
    numeric_match = re.match(r"^(\d{6}_\d{6})", stem)
    if numeric_match:
        return {
            "kind": "numeric_timestamp_like_prefix",
            "value": numeric_match.group(1),
            "certainty": "heuristic_only",
        }
    return None


def _appears_generated(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith(("pred_", "temp_", "annotated_"))


def _inventory_file(
    repo_root: Path,
    source_root: Path,
    path: Path,
    spec: SourceSpec,
    contract: ClassLabelContract,
) -> dict[str, Any]:
    relative_to_source = path.relative_to(source_root)
    extension = path.suffix.lower()
    is_image_extension = extension in SUPPORTED_IMAGE_EXTENSIONS
    class_label = None
    class_label_basis = None
    if spec.class_layout and len(relative_to_source.parts) > 1:
        class_label = relative_to_source.parts[0]
        class_label_basis = "class_directory"
    elif spec.partition == "test" and is_image_extension:
        class_label = _infer_test_label(path.name, contract)
        if class_label:
            class_label_basis = "filename_prefix"

    try:
        file_size = path.stat().st_size
        sha256 = _sha256(path)
    except OSError as exc:
        file_size = None
        sha256 = None
        file_error = str(exc)
    else:
        file_error = None

    image_metadata: dict[str, Any] = {
        "width": None,
        "height": None,
        "channel_count": None,
        "image_mode": None,
        "inspection_method": None,
        "detected_format": None,
        "extension_matches_content": None,
    }
    readable = False
    if is_image_extension and file_error is None:
        try:
            image_metadata.update(inspect_image(path))
            readable = True
            readability_status = "readable"
            readability_error = None
        except (OSError, ImageInspectionError) as exc:
            readability_status = "unreadable"
            readability_error = str(exc)
    elif is_image_extension:
        readability_status = "unreadable"
        readability_error = file_error
    else:
        readability_status = "unsupported_extension"
        readability_error = f"unsupported extension {extension or '<none>'}"

    generated_by_location = not spec.is_source
    generated_by_name = _appears_generated(path)
    generated_reason = spec.generated_reason
    if generated_reason is None and generated_by_name:
        generated_reason = "Filename uses a known legacy generated-output prefix."

    return {
        "relative_path": path.relative_to(repo_root).as_posix(),
        "source": spec.name,
        "partition": spec.partition,
        "is_source": spec.is_source,
        "is_image_extension": is_image_extension,
        "class_label": class_label,
        "class_label_basis": class_label_basis,
        "extension": extension,
        "file_size": file_size,
        "sha256": sha256,
        **image_metadata,
        "readable": readable,
        "readability_status": readability_status,
        "readability_error": readability_error,
        "appears_generated": generated_by_location or generated_by_name,
        "generated_reason": generated_reason,
        "grouping_hint": _grouping_hint(path.name, class_label)
        if spec.is_source and is_image_extension
        else None,
        "production_group_proven": False,
        "requires_group_review": spec.is_source and is_image_extension,
    }


def _duplicate_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: record["relative_path"])
    return {
        "sha256": ordered[0]["sha256"],
        "paths": [record["relative_path"] for record in ordered],
        "partitions": sorted({record["partition"] for record in ordered}),
        "sources": sorted({record["source"] for record in ordered}),
    }


def _issue(code: str, count: int, message: str) -> dict[str, Any]:
    return {"code": code, "count": count, "message": message}


def audit_repository(
    repo_root: Path,
    *,
    contract_path: Path | None = None,
    source_specs: tuple[SourceSpec, ...] = SOURCE_SPECS,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    contract_path = contract_path or repo_root / "contracts" / "class_labels.json"
    contract = load_class_label_contract(contract_path)
    records: list[dict[str, Any]] = []
    missing_source_directories: list[str] = []

    for spec in source_specs:
        source_root = repo_root / spec.relative_root
        if not source_root.is_dir():
            if spec.required:
                missing_source_directories.append(spec.relative_root)
            continue
        for path in sorted(
            (candidate for candidate in source_root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(repo_root).as_posix(),
        ):
            records.append(
                _inventory_file(repo_root, source_root, path, spec, contract)
            )

    records.sort(key=lambda record: record["relative_path"])
    image_records = [record for record in records if record["is_image_extension"]]
    source_images = [
        record for record in image_records if record["is_source"]
    ]
    readable_images = [record for record in image_records if record["readable"]]

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in image_records:
        if record["sha256"]:
            by_hash[record["sha256"]].append(record)
    exact_duplicate_groups = [
        _duplicate_group(group)
        for _, group in sorted(by_hash.items())
        if len(group) > 1
    ]

    source_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_images:
        if record["sha256"]:
            source_by_hash[record["sha256"]].append(record)
    cross_partition_duplicate_groups = [
        _duplicate_group(group)
        for _, group in sorted(source_by_hash.items())
        if len({record["partition"] for record in group}) > 1
    ]

    training_test_overlaps = [
        _duplicate_group(group)
        for _, group in sorted(source_by_hash.items())
        if {"training", "test"}.issubset(
            {record["partition"] for record in group}
        )
    ]
    training_hashes = {
        record["sha256"]
        for record in source_images
        if record["partition"] == "training" and record["sha256"]
    }
    training_test_overlap_count = sum(
        1
        for record in source_images
        if record["partition"] == "test"
        and record["sha256"] in training_hashes
    )

    by_filename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in image_records:
        by_filename[Path(record["relative_path"]).name.casefold()].append(record)
    duplicate_filenames_different_content = []
    for filename, group in sorted(by_filename.items()):
        hashes = {record["sha256"] for record in group if record["sha256"]}
        if len(hashes) > 1:
            duplicate_filenames_different_content.append(
                {
                    "filename": filename,
                    "paths": sorted(record["relative_path"] for record in group),
                    "sha256_values": sorted(hashes),
                }
            )

    training_spec = next(
        (spec for spec in source_specs if spec.partition == "training"),
        None,
    )
    expected_classes = set(contract.class_to_idx)
    actual_class_directories: set[str] = set()
    empty_class_directories: list[str] = []
    if training_spec:
        training_root = repo_root / training_spec.relative_root
        if training_root.is_dir():
            actual_class_directories = {
                path.name for path in training_root.iterdir() if path.is_dir()
            }
            for class_name in sorted(actual_class_directories):
                class_dir = training_root / class_name
                if not any(
                    path.is_file()
                    and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                    for path in class_dir.rglob("*")
                ):
                    empty_class_directories.append(
                        class_dir.relative_to(repo_root).as_posix()
                    )
    unexpected_class_directories = sorted(
        actual_class_directories - expected_classes
    )
    missing_class_directories = sorted(expected_classes - actual_class_directories)
    images_outside_contract = sorted(
        record["relative_path"]
        for record in source_images
        if record["partition"] == "training"
        and record["class_label"] not in expected_classes
    )

    unreadable_files = sorted(
        record["relative_path"]
        for record in image_records
        if not record["readable"]
    )
    unsupported_files = sorted(
        record["relative_path"]
        for record in records
        if not record["is_image_extension"]
    )
    unsupported_source_files = sorted(
        record["relative_path"]
        for record in records
        if record["is_source"] and not record["is_image_extension"]
    )
    generated_files = sorted(
        record["relative_path"]
        for record in records
        if record["appears_generated"]
    )
    generated_files_in_source = sorted(
        record["relative_path"]
        for record in source_images
        if record["appears_generated"]
    )
    extension_content_mismatches = sorted(
        record["relative_path"]
        for record in image_records
        if record["readable"] and record["extension_matches_content"] is False
    )
    extension_content_mismatches_in_source = sorted(
        record["relative_path"]
        for record in source_images
        if record["readable"] and record["extension_matches_content"] is False
    )

    class_counts = {
        class_name: sum(
            1
            for record in source_images
            if record["partition"] == "training"
            and record["class_label"] == class_name
        )
        for class_name in contract.class_to_idx
    }
    nonzero_class_counts = [count for count in class_counts.values() if count > 0]
    if len(nonzero_class_counts) == len(class_counts) and nonzero_class_counts:
        imbalance_ratio = max(nonzero_class_counts) / min(nonzero_class_counts)
    else:
        imbalance_ratio = None
    significant_imbalance = (
        imbalance_ratio is None
        or imbalance_ratio > SIGNIFICANT_IMBALANCE_RATIO
    )

    source_counts = Counter(record["source"] for record in image_records)
    for spec in source_specs:
        source_counts.setdefault(spec.name, 0)
    partition_counts = Counter(record["partition"] for record in image_records)
    partition_counts.setdefault("validation", 0)

    blocking_issues: list[dict[str, Any]] = []
    if cross_partition_duplicate_groups:
        blocking_issues.append(
            _issue(
                "cross_partition_duplicates",
                len(cross_partition_duplicate_groups),
                "Exact source-image content occurs in more than one partition/source.",
            )
        )
    if training_test_overlap_count:
        blocking_issues.append(
            _issue(
                "training_test_overlap",
                training_test_overlap_count,
                "Test images are exact byte-for-byte matches of training images.",
            )
        )
    if unreadable_files:
        blocking_issues.append(
            _issue(
                "unreadable_images",
                len(unreadable_files),
                "Files with supported image extensions could not be inspected.",
            )
        )
    if unsupported_source_files:
        blocking_issues.append(
            _issue(
                "unsupported_source_files",
                len(unsupported_source_files),
                "Unsupported files are present inside source-data locations.",
            )
        )
    if unexpected_class_directories:
        blocking_issues.append(
            _issue(
                "unexpected_class_directories",
                len(unexpected_class_directories),
                "Training class directories differ from the canonical contract.",
            )
        )
    if missing_class_directories:
        blocking_issues.append(
            _issue(
                "missing_class_directories",
                len(missing_class_directories),
                "Canonical training class directories are missing.",
            )
        )
    if empty_class_directories:
        blocking_issues.append(
            _issue(
                "empty_class_directories",
                len(empty_class_directories),
                "Training class directories contain no supported images.",
            )
        )
    if images_outside_contract:
        blocking_issues.append(
            _issue(
                "images_outside_contract",
                len(images_outside_contract),
                "Training images are outside canonical class directories.",
            )
        )
    if generated_files_in_source:
        blocking_issues.append(
            _issue(
                "generated_files_in_source",
                len(generated_files_in_source),
                "Files that appear generated are inside source-data locations.",
            )
        )
    if extension_content_mismatches_in_source:
        blocking_issues.append(
            _issue(
                "extension_content_mismatches_in_source",
                len(extension_content_mismatches_in_source),
                "Source-image extensions do not match detected image content.",
            )
        )
    if significant_imbalance:
        blocking_issues.append(
            _issue(
                "significant_class_imbalance",
                len(class_counts),
                "Training class imbalance exceeds the documented 3:1 audit threshold.",
            )
        )
    if missing_source_directories:
        blocking_issues.append(
            _issue(
                "missing_required_source_directories",
                len(missing_source_directories),
                "Required source directories are missing.",
            )
        )

    source_group_review_count = sum(
        1 for record in source_images if record["requires_group_review"]
    )
    retraining_blockers = [issue["code"] for issue in blocking_issues]
    if source_group_review_count:
        retraining_blockers.append("production_group_metadata_not_proven")

    evaluation_trustworthy = (
        partition_counts["test"] > 0 and not blocking_issues
    )
    audit_exit_code = 1 if blocking_issues else 0

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "contract_schema_version": contract.schema_version,
        "scope": [
            {
                "source": spec.name,
                "relative_root": spec.relative_root,
                "partition": spec.partition,
                "is_source": spec.is_source,
                "generated_reason": spec.generated_reason,
            }
            for spec in source_specs
        ],
        "summary": {
            "total_files": len(records),
            "total_image_files": len(image_records),
            "source_image_files": len(source_images),
            "readable_image_files": len(readable_images),
            "unreadable_image_files": len(unreadable_files),
            "unsupported_files": len(unsupported_files),
            "generated_or_runtime_files": len(generated_files),
            "extension_content_mismatches": len(extension_content_mismatches),
            "exact_duplicate_groups": len(exact_duplicate_groups),
            "cross_partition_duplicate_groups": len(
                cross_partition_duplicate_groups
            ),
            "training_test_overlap_count": training_test_overlap_count,
            "evaluation_trustworthy": evaluation_trustworthy,
            "dataset_suitable_for_retraining": not retraining_blockers,
            "audit_exit_code": audit_exit_code,
        },
        "class_counts": class_counts,
        "source_counts": dict(sorted(source_counts.items())),
        "partition_counts": dict(sorted(partition_counts.items())),
        "class_imbalance": {
            "threshold_ratio": SIGNIFICANT_IMBALANCE_RATIO,
            "observed_max_to_min_ratio": imbalance_ratio,
            "significant": significant_imbalance,
        },
        "blocking_issues": blocking_issues,
        "retraining_blockers": retraining_blockers,
        "grouping_assessment": {
            "proven_production_group_count": 0,
            "source_images_requiring_human_group_review": source_group_review_count,
            "statement": (
                "Filename date/timestamp-like prefixes are heuristic hints only; "
                "no board, lot, panel, session, or batch grouping is proven."
            ),
        },
        "exact_duplicate_groups": exact_duplicate_groups,
        "cross_partition_duplicate_groups": cross_partition_duplicate_groups,
        "training_test_overlaps": training_test_overlaps,
        "duplicate_filenames_different_content": (
            duplicate_filenames_different_content
        ),
        "unreadable_files": unreadable_files,
        "unsupported_files": unsupported_files,
        "empty_class_directories": empty_class_directories,
        "unexpected_class_directories": unexpected_class_directories,
        "missing_class_directories": missing_class_directories,
        "missing_source_directories": missing_source_directories,
        "images_outside_contract": images_outside_contract,
        "generated_files": generated_files,
        "extension_content_mismatches": extension_content_mismatches,
        "files": records,
    }


def audit_exit_code(report: dict[str, Any]) -> int:
    return int(report["summary"]["audit_exit_code"])


def _markdown_paths(paths: list[str]) -> list[str]:
    return [f"- `{path}`" for path in paths] or ["- None"]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Legacy Dataset Leakage Audit",
        "",
        "Generated deterministically by `scripts/audit_legacy_dataset.py`.",
        "This report uses exact SHA-256 content hashes; it makes no near-duplicate claims.",
        "",
        "## Verdict",
        "",
        f"- Current evaluation trustworthy: **{str(summary['evaluation_trustworthy']).lower()}**",
        "- Dataset suitable for retraining: "
        f"**{str(summary['dataset_suitable_for_retraining']).lower()}**",
        f"- Audit exit code: `{summary['audit_exit_code']}`",
        "",
        "## Summary",
        "",
        "| Measure | Count |",
        "| --- | ---: |",
        f"| Inventoried files | {summary['total_files']} |",
        f"| Image files | {summary['total_image_files']} |",
        f"| Source image files | {summary['source_image_files']} |",
        f"| Readable image files | {summary['readable_image_files']} |",
        f"| Unreadable image files | {summary['unreadable_image_files']} |",
        f"| Unsupported files | {summary['unsupported_files']} |",
        f"| Generated/runtime files | {summary['generated_or_runtime_files']} |",
        "| Extension/content mismatches | "
        f"{summary['extension_content_mismatches']} |",
        f"| Exact duplicate groups | {summary['exact_duplicate_groups']} |",
        "| Cross-partition duplicate groups | "
        f"{summary['cross_partition_duplicate_groups']} |",
        f"| Training/test overlap images | {summary['training_test_overlap_count']} |",
        "",
        "## Training class counts",
        "",
        "| Class | Images |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{class_name}` | {count} |"
        for class_name, count in report["class_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Current sources and partitions",
            "",
            "| Source | Image files |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{source}` | {count} |"
        for source, count in report["source_counts"].items()
    )
    lines.extend(["", "Validation partition found: **no**.", ""])

    lines.extend(["## Blocking issues", ""])
    if report["blocking_issues"]:
        lines.extend(
            f"- `{issue['code']}` ({issue['count']}): {issue['message']}"
            for issue in report["blocking_issues"]
        )
    else:
        lines.append("- None")

    lines.extend(["", "## Exact training/test overlap", ""])
    if report["training_test_overlaps"]:
        for group in report["training_test_overlaps"]:
            lines.append(f"- SHA-256 `{group['sha256']}`")
            lines.extend(f"  - `{path}`" for path in group["paths"])
    else:
        lines.append("- None")

    lines.extend(["", "## Cross-partition duplicate groups", ""])
    if report["cross_partition_duplicate_groups"]:
        for group in report["cross_partition_duplicate_groups"]:
            partitions = ", ".join(group["partitions"])
            lines.append(
                f"- SHA-256 `{group['sha256']}`; partitions: `{partitions}`"
            )
            lines.extend(f"  - `{path}`" for path in group["paths"])
    else:
        lines.append("- None")

    lines.extend(["", "## Unexpected, unreadable, and unsupported files", ""])
    lines.append("### Unexpected class directories")
    lines.extend(_markdown_paths(report["unexpected_class_directories"]))
    lines.append("")
    lines.append("### Unreadable image files")
    lines.extend(_markdown_paths(report["unreadable_files"]))
    lines.append("")
    lines.append("### Unsupported files")
    lines.extend(_markdown_paths(report["unsupported_files"]))
    lines.append("")
    lines.append("### Extension/content mismatches")
    lines.extend(_markdown_paths(report["extension_content_mismatches"]))
    lines.append("")
    lines.append("### Empty class directories")
    lines.extend(_markdown_paths(report["empty_class_directories"]))
    lines.append("")
    lines.append("### Images outside the canonical class contract")
    lines.extend(_markdown_paths(report["images_outside_contract"]))

    lines.extend(["", "## Duplicate filenames with different content", ""])
    if report["duplicate_filenames_different_content"]:
        for group in report["duplicate_filenames_different_content"]:
            lines.append(f"- `{group['filename']}`")
            lines.extend(f"  - `{path}`" for path in group["paths"])
    else:
        lines.append("- None")

    grouping = report["grouping_assessment"]
    lines.extend(
        [
            "",
            "## Grouping evidence",
            "",
            f"- Proven production groups: {grouping['proven_production_group_count']}",
            "- Source images requiring human grouping review: "
            f"{grouping['source_images_requiring_human_group_review']}",
            f"- {grouping['statement']}",
            "",
            "## Conclusion",
            "",
        ]
    )
    if summary["evaluation_trustworthy"]:
        lines.append("No blocking training/test leakage was found by this exact-hash audit.")
    else:
        lines.append(
            "The current evaluation is not trustworthy because blocking leakage or "
            "dataset-integrity issues exist."
        )
    if summary["dataset_suitable_for_retraining"]:
        lines.append("The audited source set has no recorded retraining blocker.")
    else:
        lines.append(
            "The dataset is not ready for retraining. Resolve every recorded blocker "
            "and obtain reviewed production grouping metadata before splitting."
        )
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any],
    json_output: Path,
    markdown_output: Path,
) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown(report), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit legacy PCB image sources for exact-hash leakage.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository root containing backend/ and contracts/.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "legacy_dataset_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "legacy_dataset_audit.md",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_repository(args.repo_root)
    write_reports(report, args.json_output, args.markdown_output)
    print(
        "Legacy dataset audit: "
        f"{report['summary']['total_image_files']} images, "
        f"{report['summary']['training_test_overlap_count']} training/test overlaps, "
        f"exit={audit_exit_code(report)}"
    )
    print(f"JSON report: {args.json_output}")
    print(f"Markdown report: {args.markdown_output}")
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
