from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.services.dataset_validation.file_inspection import (
    FileInspectionError,
    InspectedRaster,
    PathSafetyError,
    inspect_height,
    inspect_rgb,
    rgb_color_space_compatible,
    safe_dataset_file,
    sha256_file,
)
from app.services.dataset_validation.models import (
    Finding,
    SampleValidation,
    ValidationReport,
    ValidationStage,
)
from app.services.dataset_validation.schema_validation import (
    SchemaBundle,
    load_schema_bundle,
    schema_findings,
)

PROTECTED_SPLITS = {"train", "validation", "test", "holdout"}
SCHEMA_REFERENCE_NAMES = {
    "sample_schema": "pcb_aoi_sample.schema.json",
    "split_manifest_schema": "dataset_split_manifest.schema.json",
    "dataset_manifest_schema": "dataset_manifest.schema.json",
    "defect_taxonomy": "defect_taxonomy.json",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("JSON document must contain an object")
    return value


def _raster_dict(value: InspectedRaster) -> dict[str, Any]:
    result: dict[str, Any] = {
        "format": value.format,
        "width": value.width,
        "height": value.height,
        "channels": value.channels,
        "bit_depth": value.bit_depth,
        "mode": value.mode,
    }
    if value.storage_data_type is not None:
        result["storage_data_type"] = value.storage_data_type
    return result


class _ValidationRun:
    def __init__(
        self,
        dataset_root: Path,
        stage: ValidationStage,
        timestamp: str,
        schemas: SchemaBundle,
    ) -> None:
        self.root = dataset_root
        self.stage = stage
        self.timestamp = timestamp
        self.schemas = schemas
        self.findings: list[Finding] = []
        self.samples: list[SampleValidation] = []
        self.samples_by_id: dict[str, SampleValidation] = {}
        self.metadata_by_id: dict[str, dict[str, Any]] = {}
        self.physical_references: dict[str, tuple[str, str]] = {}
        self.dataset_manifest: dict[str, Any] = {}
        self.split_manifest: dict[str, Any] = {}

    def add(self, finding: Finding, sample: SampleValidation | None = None) -> None:
        self.findings.append(finding)
        if sample is not None:
            sample.findings.append(finding)

    def safe_file(
        self,
        base: Path,
        relative_path: str,
        *,
        category: str,
        sample: SampleValidation | None = None,
        sample_id: str | None = None,
    ) -> tuple[Path, str] | None:
        try:
            return safe_dataset_file(self.root, base, relative_path)
        except PathSafetyError as exc:
            self.add(
                Finding(
                    code=exc.code,
                    category=category,
                    message=str(exc),
                    sample_id=sample_id,
                    path=relative_path,
                ),
                sample,
            )
            return None

    def load_dataset_manifest(self) -> bool:
        resolved = self.safe_file(
            self.root,
            "dataset_manifest.json",
            category="manifest",
        )
        if resolved is None:
            return False
        path, relative = resolved
        try:
            self.dataset_manifest = _json_object(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.add(
                Finding(
                    code="manifest.dataset_unreadable",
                    category="manifest",
                    message=f"Dataset manifest cannot be read: {exc}",
                    path=relative,
                )
            )
            return False
        for finding in schema_findings(
            self.schemas.dataset_validator,
            self.dataset_manifest,
            scope="dataset_manifest",
            path=relative,
        ):
            self.add(finding)
        return True

    def validate_schema_copies(self) -> None:
        references = self.dataset_manifest.get("schema_references")
        if not isinstance(references, dict):
            return
        authoritative = {
            "sample_schema": self.schemas.sample_schema,
            "split_manifest_schema": self.schemas.split_schema,
            "dataset_manifest_schema": self.schemas.dataset_schema,
            "defect_taxonomy": self.schemas.taxonomy,
        }
        for key in sorted(SCHEMA_REFERENCE_NAMES):
            value = references.get(key)
            if not isinstance(value, str):
                continue
            if Path(value).name != SCHEMA_REFERENCE_NAMES[key]:
                self.add(
                    Finding(
                        code="manifest.schema_reference_name",
                        category="manifest",
                        message=f"{key} must reference {SCHEMA_REFERENCE_NAMES[key]}",
                        path=value,
                    )
                )
            resolved = self.safe_file(self.root, value, category="manifest")
            if resolved is None:
                continue
            path, relative = resolved
            try:
                supplied = _json_object(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self.add(
                    Finding(
                        code="manifest.schema_copy_unreadable",
                        category="manifest",
                        message=f"Referenced schema cannot be read: {exc}",
                        path=relative,
                    )
                )
                continue
            if supplied != authoritative[key]:
                self.add(
                    Finding(
                        code="manifest.schema_copy_mismatch",
                        category="manifest",
                        message=f"Packaged {key} differs from the authoritative repository contract",
                        path=relative,
                    )
                )

    def load_samples(self) -> None:
        reference = self.dataset_manifest.get("samples_manifest_reference")
        if not isinstance(reference, str):
            return
        resolved = self.safe_file(self.root, reference, category="manifest")
        if resolved is None:
            return
        index_path, index_relative = resolved
        try:
            lines = index_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            self.add(
                Finding(
                    code="manifest.samples_index_unreadable",
                    category="manifest",
                    message=f"Samples index cannot be read: {exc}",
                    path=index_relative,
                )
            )
            return
        seen_ids: set[str] = set()
        indexed_metadata_paths: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                self.add(
                    Finding(
                        code="manifest.samples_index_json",
                        category="manifest",
                        message=f"Line {line_number} is invalid JSON: {exc.msg}",
                        path=index_relative,
                    )
                )
                continue
            if not isinstance(record, dict) or set(record) != {
                "sample_id",
                "metadata_file",
                "metadata_sha256",
            }:
                self.add(
                    Finding(
                        code="manifest.samples_index_record",
                        category="manifest",
                        message=f"Line {line_number} must contain sample_id, metadata_file, and metadata_sha256 only",
                        path=index_relative,
                    )
                )
                continue
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                self.add(
                    Finding(
                        code="sample.id_invalid",
                        category="metadata",
                        message=f"Line {line_number} has an invalid sample_id",
                        path=index_relative,
                    )
                )
                continue
            if sample_id in seen_ids:
                self.add(
                    Finding(
                        code="sample.id_duplicate",
                        category="metadata",
                        message=f"Duplicate sample_id in samples index: {sample_id}",
                        sample_id=sample_id,
                        path=index_relative,
                    )
                )
                continue
            seen_ids.add(sample_id)
            if isinstance(record.get("metadata_file"), str):
                indexed_metadata_paths.add(record["metadata_file"])
            self.load_sample(record, index_relative)
        samples_directory = self.root / "samples"
        if not samples_directory.is_dir() or samples_directory.is_symlink():
            self.add(
                Finding(
                    code="package.samples_directory_invalid",
                    category="manifest",
                    message="Package must contain a regular samples directory",
                    path="samples",
                )
            )
            return
        for child in sorted(samples_directory.iterdir(), key=lambda path: path.name):
            relative = child.relative_to(self.root).as_posix()
            if child.is_symlink() or not child.is_dir():
                self.add(
                    Finding(
                        code="package.sample_entry_invalid",
                        category="manifest",
                        message="Every direct samples entry must be a regular sample directory",
                        path=relative,
                    )
                )
                continue
            metadata_relative = (child / "metadata.json").relative_to(self.root).as_posix()
            if metadata_relative not in indexed_metadata_paths:
                self.add(
                    Finding(
                        code="package.unindexed_sample",
                        category="manifest",
                        message="Sample directory is not indexed by samples.jsonl",
                        path=relative,
                    )
                )

    def load_sample(self, record: dict[str, Any], index_relative: str) -> None:
        sample_id = record["sample_id"]
        metadata_reference = record.get("metadata_file")
        inventory = {
            "sample_id": sample_id,
            "sample_directory": None,
            "label": None,
            "defect_type": None,
            "board_id": None,
            "recipe_id": None,
            "recipe_version": None,
            "lot_id": None,
            "capture_session_id": None,
            "station_id": None,
            "rgb_file": None,
            "height_file": None,
            "validity_mask_file": None,
            "calibration_file": None,
            "pair_validation_status": "BLOCKED",
        }
        sample = SampleValidation(inventory=inventory)
        self.samples.append(sample)
        self.samples_by_id[sample_id] = sample
        if not isinstance(metadata_reference, str):
            self.add(
                Finding(
                    code="sample.metadata_reference_invalid",
                    category="metadata",
                    message="metadata_file must be a relative path",
                    sample_id=sample_id,
                    path=index_relative,
                ),
                sample,
            )
            return
        resolved = self.safe_file(
            self.root,
            metadata_reference,
            category="metadata",
            sample=sample,
            sample_id=sample_id,
        )
        if resolved is None:
            return
        metadata_path, metadata_relative = resolved
        parts = Path(metadata_relative).parts
        if len(parts) != 3 or parts[0] != "samples" or parts[-1] != "metadata.json":
            self.add(
                Finding(
                    code="package.sample_structure",
                    category="metadata",
                    message="Sample metadata must be samples/<sample-directory>/metadata.json",
                    sample_id=sample_id,
                    path=metadata_relative,
                ),
                sample,
            )
        inventory["sample_directory"] = metadata_path.parent.relative_to(self.root).as_posix()
        declared_metadata_hash = record.get("metadata_sha256")
        actual_metadata_hash = sha256_file(metadata_path)
        if declared_metadata_hash != actual_metadata_hash:
            self.add(
                Finding(
                    code="integrity.metadata_hash_mismatch",
                    category="integrity",
                    message="Metadata SHA-256 does not match samples.jsonl",
                    sample_id=sample_id,
                    path=metadata_relative,
                ),
                sample,
            )
        try:
            metadata = _json_object(metadata_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.add(
                Finding(
                    code="sample.metadata_unreadable",
                    category="metadata",
                    message=f"Sample metadata cannot be read: {exc}",
                    sample_id=sample_id,
                    path=metadata_relative,
                ),
                sample,
            )
            return
        for finding in schema_findings(
            self.schemas.sample_validator,
            metadata,
            scope="sample_metadata",
            path=metadata_relative,
            sample_id=sample_id,
        ):
            self.add(finding, sample)
        if metadata.get("sample_id") != sample_id:
            self.add(
                Finding(
                    code="sample.id_mismatch",
                    category="metadata",
                    message="Metadata sample_id does not match samples.jsonl",
                    sample_id=sample_id,
                    path=metadata_relative,
                ),
                sample,
            )
        self.metadata_by_id[sample_id] = metadata
        self.populate_inventory(sample, metadata)
        self.validate_taxonomy(sample, metadata)
        self.validate_sample_files(sample, metadata, metadata_path.parent)
        base_categories = {"metadata", "schema", "file", "integrity", "pair", "registration", "taxonomy"}
        if not any(
            finding.severity == "error" and finding.category in base_categories
            for finding in sample.findings
        ):
            inventory["pair_validation_status"] = "PASS"

    @staticmethod
    def populate_inventory(sample: SampleValidation, metadata: dict[str, Any]) -> None:
        ground_truth = metadata.get("ground_truth", {})
        production = metadata.get("production", {})
        files = metadata.get("files", {})
        sample.inventory.update(
            {
                "label": ground_truth.get("label"),
                "defect_type": ground_truth.get("defect_type"),
                "board_id": metadata.get("board_id"),
                "recipe_id": metadata.get("recipe_id"),
                "recipe_version": metadata.get("recipe_version"),
                "lot_id": production.get("lot_id"),
                "capture_session_id": production.get("capture_session_id"),
                "station_id": production.get("station_id"),
                "rgb_file": files.get("rgb_file"),
                "height_file": files.get("height_file"),
                "validity_mask_file": files.get("validity_mask_file"),
                "calibration_file": files.get("calibration_file"),
            }
        )

    def validate_taxonomy(self, sample: SampleValidation, metadata: dict[str, Any]) -> None:
        sample_id = sample.inventory["sample_id"]
        ground_truth = metadata.get("ground_truth", {})
        label = ground_truth.get("label")
        defect_type = ground_truth.get("defect_type")
        supported = set(
            self.schemas.taxonomy.get("$defs", {})
            .get("supported_defect_type", {})
            .get("enum", [])
        )
        if label == "OK" and defect_type is not None:
            self.add(
                Finding(
                    code="taxonomy.ok_has_defect",
                    category="taxonomy",
                    message="OK samples must have defect_type null",
                    sample_id=sample_id,
                ),
                sample,
            )
        if label == "NOK" and defect_type not in supported:
            self.add(
                Finding(
                    code="taxonomy.nok_defect_unknown",
                    category="taxonomy",
                    message="NOK samples must use an authoritative defect taxonomy value",
                    sample_id=sample_id,
                ),
                sample,
            )
        if ground_truth.get("taxonomy_version") != self.schemas.taxonomy.get("taxonomy_version"):
            self.add(
                Finding(
                    code="taxonomy.version_mismatch",
                    category="taxonomy",
                    message="Sample taxonomy version does not match the authoritative taxonomy",
                    sample_id=sample_id,
                ),
                sample,
            )

    def _referenced_file(
        self,
        sample: SampleValidation,
        sample_dir: Path,
        reference: Any,
        kind: str,
    ) -> tuple[Path, str] | None:
        sample_id = sample.inventory["sample_id"]
        if not isinstance(reference, str):
            self.add(
                Finding(
                    code=f"file.{kind}_reference_invalid",
                    category="file",
                    message=f"{kind} file reference is missing or invalid",
                    sample_id=sample_id,
                ),
                sample,
            )
            return None
        resolved = self.safe_file(
            sample_dir,
            reference,
            category="file",
            sample=sample,
            sample_id=sample_id,
        )
        if resolved is None:
            return None
        path, relative = resolved
        stat_result = path.stat()
        physical_key = f"{stat_result.st_dev}:{stat_result.st_ino}"
        previous = self.physical_references.get(physical_key)
        if previous and previous[0] != sample_id and kind in {"rgb", "height"}:
            message = (
                f"Physical file is referenced by unrelated samples {previous[0]!r} "
                f"and {sample_id!r}"
            )
            finding = Finding(
                code="file.shared_between_samples",
                category="file",
                message=message,
                sample_id=sample_id,
                path=relative,
            )
            self.add(finding, sample)
            previous_sample = self.samples_by_id.get(previous[0])
            if previous_sample is not None:
                self.add(
                    Finding(
                        code="file.shared_between_samples",
                        category="file",
                        message=message,
                        sample_id=previous[0],
                        path=relative,
                    ),
                    previous_sample,
                )
        else:
            self.physical_references.setdefault(physical_key, (sample_id, kind))
        return path, relative

    def _verify_hash(
        self,
        sample: SampleValidation,
        path: Path,
        relative: str,
        declared: Any,
        kind: str,
    ) -> None:
        if not isinstance(declared, str) or sha256_file(path) != declared:
            self.add(
                Finding(
                    code=f"integrity.{kind}_hash_mismatch",
                    category="integrity",
                    message=f"{kind} SHA-256 does not match the exact file bytes",
                    sample_id=sample.inventory["sample_id"],
                    path=relative,
                ),
                sample,
            )

    def _inspect(
        self,
        sample: SampleValidation,
        path: Path,
        relative: str,
        kind: str,
        inspector: Callable[[Path], InspectedRaster],
    ) -> InspectedRaster | None:
        try:
            return inspector(path)
        except (OSError, ValueError) as exc:
            self.add(
                Finding(
                    code=f"file.{kind}_unreadable",
                    category="file",
                    message=f"{kind} file cannot be inspected: {exc}",
                    sample_id=sample.inventory["sample_id"],
                    path=relative,
                ),
                sample,
            )
            return None

    def validate_sample_files(
        self,
        sample: SampleValidation,
        metadata: dict[str, Any],
        sample_dir: Path,
    ) -> None:
        files = metadata.get("files", {})
        integrity = metadata.get("integrity", {})
        rgb_declared = metadata.get("rgb", {})
        height_declared = metadata.get("height_3d", {})
        sample_id = sample.inventory["sample_id"]

        rgb_ref = self._referenced_file(sample, sample_dir, files.get("rgb_file"), "rgb")
        height_ref = self._referenced_file(sample, sample_dir, files.get("height_file"), "height")
        rgb_actual = height_actual = None
        if rgb_ref:
            rgb_path, rgb_relative = rgb_ref
            self._verify_hash(sample, rgb_path, rgb_relative, integrity.get("rgb_sha256"), "rgb")
            rgb_actual = self._inspect(sample, rgb_path, rgb_relative, "rgb", inspect_rgb)
            if rgb_actual:
                sample.inventory["actual_rgb"] = _raster_dict(rgb_actual)
                comparisons = {
                    "width": rgb_actual.width,
                    "height": rgb_actual.height,
                    "channels": rgb_actual.channels,
                    "bit_depth": rgb_actual.bit_depth,
                }
                for field, actual in comparisons.items():
                    if rgb_declared.get(field) != actual:
                        self.add(
                            Finding(
                                code=f"pair.rgb_{field}_mismatch",
                                category="pair",
                                message=f"Declared RGB {field} does not match actual value {actual}",
                                sample_id=sample_id,
                                path=rgb_relative,
                            ),
                            sample,
                        )
                declared_color = rgb_declared.get("color_space")
                if not isinstance(declared_color, str) or not rgb_color_space_compatible(declared_color, rgb_actual):
                    self.add(
                        Finding(
                            code="pair.rgb_color_space_mismatch",
                            category="pair",
                            message="Declared RGB color space is incompatible with actual image content",
                            sample_id=sample_id,
                            path=rgb_relative,
                        ),
                        sample,
                    )
        if height_ref:
            height_path, height_relative = height_ref
            self._verify_hash(sample, height_path, height_relative, integrity.get("height_sha256"), "height")
            if re.search(r"(?:preview|heatmap|colou?rized)", Path(height_relative).name, re.IGNORECASE):
                self.add(
                    Finding(
                        code="pair.height_preview_name",
                        category="pair",
                        message="A preview/colorized filename is not accepted as native height data",
                        sample_id=sample_id,
                        path=height_relative,
                    ),
                    sample,
                )
            height_actual = self._inspect(sample, height_path, height_relative, "height", inspect_height)
            if height_actual:
                sample.inventory["actual_height"] = _raster_dict(height_actual)
                if height_actual.channels != 1 or height_actual.mode != "SCALAR":
                    self.add(
                        Finding(
                            code="pair.height_not_scalar",
                            category="pair",
                            message="Native height/depth data must be a single-channel scalar raster",
                            sample_id=sample_id,
                            path=height_relative,
                        ),
                        sample,
                    )
                comparisons = {
                    "width": height_actual.width,
                    "height": height_actual.height,
                    "storage_format": height_actual.format,
                    "storage_data_type": height_actual.storage_data_type,
                }
                for field, actual in comparisons.items():
                    declared_field = "storage_format" if field == "storage_format" else field
                    if height_declared.get(declared_field) != actual:
                        self.add(
                            Finding(
                                code=f"pair.height_{field}_mismatch",
                                category="pair",
                                message=f"Declared height {field} does not match actual value {actual}",
                                sample_id=sample_id,
                                path=height_relative,
                            ),
                            sample,
                        )
        if rgb_ref and height_ref:
            rgb_stat = rgb_ref[0].stat()
            height_stat = height_ref[0].stat()
            same_physical_file = (
                rgb_stat.st_dev == height_stat.st_dev
                and rgb_stat.st_ino == height_stat.st_ino
            )
        else:
            same_physical_file = False
        if same_physical_file:
            self.add(
                Finding(
                    code="pair.same_physical_file",
                    category="pair",
                    message="RGB and height references resolve to the same physical file",
                    sample_id=sample_id,
                ),
                sample,
            )

        mask_reference = files.get("validity_mask_file")
        mask_ref = None
        if mask_reference is not None:
            mask_ref = self._referenced_file(sample, sample_dir, mask_reference, "validity_mask")
            if mask_ref:
                mask_path, mask_relative = mask_ref
                self._verify_hash(
                    sample,
                    mask_path,
                    mask_relative,
                    integrity.get("validity_mask_sha256"),
                    "validity_mask",
                )
                mask_actual = self._inspect(sample, mask_path, mask_relative, "validity_mask", inspect_rgb)
                if mask_actual and height_actual and (
                    mask_actual.width != height_actual.width
                    or mask_actual.height != height_actual.height
                    or mask_actual.channels != 1
                ):
                    self.add(
                        Finding(
                            code="pair.validity_mask_mismatch",
                            category="pair",
                            message="Validity mask must be single-channel and match height dimensions",
                            sample_id=sample_id,
                            path=mask_relative,
                        ),
                        sample,
                    )
        if height_declared.get("no_data_policy") == "validity_mask" and mask_ref is None:
            self.add(
                Finding(
                    code="pair.validity_mask_required",
                    category="pair",
                    message="validity_mask no-data policy requires a valid mask file",
                    sample_id=sample_id,
                ),
                sample,
            )
        if height_declared.get("no_data_policy") == "nan" and height_declared.get("storage_data_type") not in {"float32", "float64"}:
            self.add(
                Finding(
                    code="pair.nan_policy_non_float",
                    category="pair",
                    message="NaN no-data policy requires floating-point height storage",
                    sample_id=sample_id,
                ),
                sample,
            )

        calibration_ref = None
        if files.get("calibration_file") is not None:
            calibration_ref = self._referenced_file(
                sample, sample_dir, files.get("calibration_file"), "calibration"
            )
            if calibration_ref and integrity.get("calibration_sha256") is not None:
                self._verify_hash(
                    sample,
                    calibration_ref[0],
                    calibration_ref[1],
                    integrity.get("calibration_sha256"),
                    "calibration",
                )
        registration = metadata.get("registration", {})
        if registration.get("registration_status") == "requires_transform":
            transform = registration.get("transform_reference")
            if not isinstance(transform, str) or self._referenced_file(
                sample, sample_dir, transform, "transform"
            ) is None:
                self.add(
                    Finding(
                        code="registration.transform_required",
                        category="registration",
                        message="requires_transform registration needs an existing transform reference",
                        sample_id=sample_id,
                    ),
                    sample,
                )
        if rgb_actual and height_actual and rgb_actual.width == height_actual.width and rgb_actual.height == height_actual.height:
            self.add(
                Finding(
                    code="registration.dimensions_not_proof",
                    category="registration",
                    severity="warning",
                    message="Equal raster dimensions are not proof of physical registration",
                    sample_id=sample_id,
                ),
                sample,
            )

    def validate_dataset_consistency(self) -> None:
        unique_metadata = list(self.metadata_by_id.values())
        actual_count = len(self.samples)
        if self.dataset_manifest.get("sample_count") != actual_count:
            self.add(
                Finding(
                    code="manifest.sample_count_mismatch",
                    category="manifest",
                    message=f"Declared sample_count does not match indexed sample count {actual_count}",
                )
            )
        label_counts = Counter(
            metadata.get("ground_truth", {}).get("label") for metadata in unique_metadata
        )
        declared_counts = self.dataset_manifest.get("expected_class_counts")
        if isinstance(declared_counts, dict):
            actual_labels = {"OK": label_counts.get("OK", 0), "NOK": label_counts.get("NOK", 0)}
            if declared_counts != actual_labels:
                self.add(
                    Finding(
                        code="manifest.class_counts_mismatch",
                        category="manifest",
                        message=f"Declared OK/NOK counts do not match actual counts {actual_labels}",
                    )
                )
        board_ids = sorted(
            {metadata.get("board_id") for metadata in unique_metadata if metadata.get("board_id")}
        )
        if sorted(self.dataset_manifest.get("supported_board_ids", [])) != board_ids:
            self.add(
                Finding(
                    code="manifest.board_ids_mismatch",
                    category="manifest",
                    message="Declared supported_board_ids do not match sample metadata",
                )
            )
        recipe_ids = sorted(
            {metadata.get("recipe_id") for metadata in unique_metadata if metadata.get("recipe_id")}
        )
        if sorted(self.dataset_manifest.get("supported_recipe_ids", [])) != recipe_ids:
            self.add(
                Finding(
                    code="manifest.recipe_ids_mismatch",
                    category="manifest",
                    message="Declared supported_recipe_ids do not match sample metadata",
                )
            )
        versions = sorted(
            {
                f"{metadata.get('recipe_id')}@{metadata.get('recipe_version')}"
                for metadata in unique_metadata
                if metadata.get("recipe_id") and metadata.get("recipe_version")
            }
        )
        if versions:
            self.add(
                Finding(
                    code="manifest.recipe_versions_observed",
                    category="manifest",
                    severity="warning",
                    message="Observed recipe versions: " + ", ".join(versions),
                )
            )

    def load_and_validate_split(self) -> None:
        reference = self.dataset_manifest.get("split_manifest_reference")
        if not isinstance(reference, str):
            return
        resolved = self.safe_file(self.root, reference, category="split")
        if resolved is None:
            return
        path, relative = resolved
        try:
            self.split_manifest = _json_object(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.add(
                Finding(
                    code="split.unreadable",
                    category="split",
                    message=f"Split manifest cannot be read: {exc}",
                    path=relative,
                )
            )
            return
        for finding in schema_findings(
            self.schemas.split_validator,
            self.split_manifest,
            scope="split_manifest",
            path=relative,
        ):
            self.add(finding)
        for field in ("dataset_id", "dataset_version", "contract_version"):
            if self.split_manifest.get(field) != self.dataset_manifest.get(field):
                self.add(
                    Finding(
                        code=f"split.{field}_mismatch",
                        category="split",
                        message=f"Split manifest {field} does not match dataset manifest",
                        path=relative,
                    )
                )
        assignments = self.split_manifest.get("assignments")
        if not isinstance(assignments, list):
            return
        seen: set[str] = set()
        assignment_by_id: dict[str, dict[str, Any]] = {}
        for assignment in assignments:
            if not isinstance(assignment, dict):
                continue
            sample_id = assignment.get("sample_id")
            if not isinstance(sample_id, str):
                continue
            if sample_id in seen:
                self.add(
                    Finding(
                        code="split.duplicate_assignment",
                        category="split",
                        message=f"Duplicate split assignment for sample_id {sample_id}",
                        sample_id=sample_id,
                        path=relative,
                    )
                )
            seen.add(sample_id)
            assignment_by_id.setdefault(sample_id, assignment)
            if sample_id not in self.metadata_by_id:
                self.add(
                    Finding(
                        code="split.unknown_sample",
                        category="split",
                        message=f"Split assignment references unknown sample_id {sample_id}",
                        sample_id=sample_id,
                        path=relative,
                    )
                )
            if assignment.get("split") == "excluded" and not assignment.get("exclusion_reason"):
                self.add(
                    Finding(
                        code="split.exclusion_reason_required",
                        category="split",
                        message="Excluded samples require an exclusion reason",
                        sample_id=sample_id,
                        path=relative,
                    )
                )
        for sample_id in sorted(self.metadata_by_id):
            if sample_id not in assignment_by_id:
                self.add(
                    Finding(
                        code="split.assignment_missing",
                        category="split",
                        message=f"Sample {sample_id} has no split assignment",
                        sample_id=sample_id,
                        path=relative,
                    ),
                    self.samples_by_id.get(sample_id),
                )
        grouping_type = self.split_manifest.get("grouping_key_type")
        for sample_id, assignment in sorted(assignment_by_id.items()):
            metadata = self.metadata_by_id.get(sample_id)
            if metadata is None:
                continue
            expected = self.group_value(metadata, grouping_type)
            if expected is None or assignment.get("grouping_key_value") != expected:
                self.add(
                    Finding(
                        code="split.grouping_value_mismatch",
                        category="split",
                        message=f"Split grouping value does not match sample {grouping_type}",
                        sample_id=sample_id,
                        path=relative,
                    ),
                    self.samples_by_id.get(sample_id),
                )
        self.validate_group_crossing(assignment_by_id, relative)

    @staticmethod
    def group_value(metadata: dict[str, Any], grouping_type: Any) -> str | None:
        production = metadata.get("production", {})
        if grouping_type == "board_id":
            return metadata.get("board_id")
        if grouping_type == "panel_id":
            return metadata.get("panel_id")
        if grouping_type in {"lot_id", "batch_id", "capture_session_id", "sequential_group_id"}:
            return production.get(grouping_type)
        if grouping_type == "production_date_station_recipe":
            values = (
                production.get("production_date"),
                production.get("station_id"),
                metadata.get("recipe_id"),
                metadata.get("recipe_version"),
            )
            return ":".join(values) if all(isinstance(value, str) for value in values) else None
        return None

    def validate_group_crossing(
        self,
        assignments: dict[str, dict[str, Any]],
        path: str,
    ) -> None:
        group_types = [
            "board_id",
            "panel_id",
            "lot_id",
            "batch_id",
            "capture_session_id",
            "sequential_group_id",
        ]
        for group_type in group_types:
            group_splits: dict[str, set[str]] = defaultdict(set)
            group_samples: dict[str, list[str]] = defaultdict(list)
            for sample_id, assignment in assignments.items():
                split = assignment.get("split")
                metadata = self.metadata_by_id.get(sample_id)
                if split not in PROTECTED_SPLITS or metadata is None:
                    continue
                value = self.group_value(metadata, group_type)
                if value:
                    group_splits[value].add(split)
                    group_samples[value].append(sample_id)
            for value in sorted(group_splits):
                splits = sorted(group_splits[value])
                if len(splits) > 1:
                    self.add(
                        Finding(
                            code="split.protected_group_crossing",
                            category="split",
                            message=(
                                f"{group_type} {value!r} crosses protected splits "
                                f"{', '.join(splits)} for samples "
                                f"{', '.join(sorted(group_samples[value]))}"
                            ),
                            path=path,
                        )
                    )

    def validate_stage(self) -> None:
        if self.stage is ValidationStage.TECHNICAL:
            return
        dataset_synthetic = self.dataset_manifest.get("is_synthetic") is True
        if dataset_synthetic:
            self.add(
                Finding(
                    code="stage.synthetic_dataset",
                    category="stage",
                    message=f"Synthetic datasets cannot pass {self.stage.value}",
                )
            )
        if self.split_manifest.get("is_synthetic") is True:
            self.add(
                Finding(
                    code="stage.synthetic_split_manifest",
                    category="stage",
                    message=f"Synthetic split manifests cannot pass {self.stage.value}",
                )
            )
        if self.dataset_manifest.get("dataset_status") not in {"training_ready", "production_candidate"}:
            self.add(
                Finding(
                    code="stage.dataset_not_training_ready",
                    category="stage",
                    message="Dataset status is not ready for model training",
                )
            )
        for sample_id, metadata in sorted(self.metadata_by_id.items()):
            sample = self.samples_by_id[sample_id]
            ground_truth = metadata.get("ground_truth", {})
            registration = metadata.get("registration", {})
            files = metadata.get("files", {})
            integrity = metadata.get("integrity", {})
            if metadata.get("provenance", {}).get("is_synthetic") is True:
                self.add(
                    Finding(
                        code="stage.synthetic_sample",
                        category="stage",
                        message="Synthetic samples cannot be used for this stage",
                        sample_id=sample_id,
                    ),
                    sample,
                )
            if ground_truth.get("review_status") != "approved" or not ground_truth.get("reviewed_by") or not ground_truth.get("reviewed_at"):
                self.add(
                    Finding(
                        code="stage.ground_truth_not_approved",
                        category="stage",
                        message="Model training requires approved ground truth with reviewer and timestamp",
                        sample_id=sample_id,
                    ),
                    sample,
                )
            if registration.get("registration_status") == "unverified":
                self.add(
                    Finding(
                        code="stage.registration_unverified",
                        category="stage",
                        message="Model training requires verified registration or a validated transform",
                        sample_id=sample_id,
                    ),
                    sample,
                )
            if not files.get("calibration_file") or not integrity.get("calibration_sha256"):
                self.add(
                    Finding(
                        code="stage.calibration_evidence_missing",
                        category="stage",
                        message="Model training requires a calibration artifact and hash",
                        sample_id=sample_id,
                    ),
                    sample,
                )
        if self.stage is not ValidationStage.PRODUCTION:
            return
        if self.dataset_manifest.get("dataset_status") != "production_candidate":
            self.add(
                Finding(
                    code="stage.production_status_required",
                    category="stage",
                    message="Production acceptance requires dataset_status production_candidate",
                )
            )
        if self.dataset_manifest.get("approval_status") != "approved":
            self.add(
                Finding(
                    code="stage.dataset_approval_required",
                    category="stage",
                    message="Production acceptance requires an approved dataset manifest",
                )
            )
        if not self.split_manifest.get("approved_by"):
            self.add(
                Finding(
                    code="stage.split_approval_required",
                    category="stage",
                    message="Production acceptance requires split-manifest approval evidence",
                )
            )
        protected_present = {
            assignment.get("split")
            for assignment in self.split_manifest.get("assignments", [])
            if isinstance(assignment, dict)
        }
        if not ({"test", "holdout"} <= protected_present):
            self.add(
                Finding(
                    code="stage.locked_evaluation_partitions_required",
                    category="stage",
                    message="Production acceptance requires approved test and holdout assignments",
                )
            )
        if "known_limitations" not in self.dataset_manifest:
            self.add(
                Finding(
                    code="stage.known_limitations_required",
                    category="stage",
                    message="Production acceptance requires a known-limitations declaration",
                )
            )
        for sample_id, metadata in sorted(self.metadata_by_id.items()):
            if not metadata.get("provenance", {}).get("source_export_version"):
                self.add(
                    Finding(
                        code="stage.production_traceability_incomplete",
                        category="stage",
                        message="Production acceptance requires source export/version traceability",
                        sample_id=sample_id,
                    ),
                    self.samples_by_id[sample_id],
                )

    def report(self) -> ValidationReport:
        labels = Counter(
            sample.inventory.get("label")
            for sample in self.samples
            if sample.inventory.get("label") in {"OK", "NOK"}
        )
        defects = Counter(
            sample.inventory.get("defect_type")
            for sample in self.samples
            if sample.inventory.get("defect_type")
        )
        boards = Counter(
            sample.inventory.get("board_id")
            for sample in self.samples
            if sample.inventory.get("board_id")
        )
        recipes = Counter(
            f"{sample.inventory.get('recipe_id')}@{sample.inventory.get('recipe_version')}"
            for sample in self.samples
            if sample.inventory.get("recipe_id") and sample.inventory.get("recipe_version")
        )
        valid_pairs = sum(
            sample.inventory.get("pair_validation_status") == "PASS"
            for sample in self.samples
        )
        errors = [finding for finding in self.findings if finding.severity == "error"]
        warnings = [finding for finding in self.findings if finding.severity == "warning"]
        summary = {
            "total_samples": len(self.samples),
            "label_counts": {"OK": labels.get("OK", 0), "NOK": labels.get("NOK", 0)},
            "defect_type_counts": dict(sorted(defects.items())),
            "board_counts": dict(sorted(boards.items())),
            "recipe_version_counts": dict(sorted(recipes.items())),
            "valid_pairs": valid_pairs,
            "invalid_pairs": len(self.samples) - valid_pairs,
            "hash_failures": sum(f.code.startswith("integrity.") and "hash" in f.code for f in errors),
            "missing_files": sum(f.code == "file.missing" for f in errors),
            "metadata_schema_failures": sum(f.code == "schema.sample_metadata" for f in errors),
            "image_depth_metadata_mismatches": sum(f.category == "pair" for f in errors),
            "registration_calibration_issues": sum(f.category in {"registration", "stage"} and ("registration" in f.code or "calibration" in f.code) for f in errors),
            "split_leakage_issues": sum(f.code == "split.protected_group_crossing" for f in errors),
            "stage_readiness_blockers": sum(f.category == "stage" for f in errors),
            "blocking_findings": len(errors),
            "warnings": len(warnings),
        }
        dataset = {
            "dataset_id": self.dataset_manifest.get("dataset_id"),
            "dataset_version": self.dataset_manifest.get("dataset_version"),
            "contract_version": self.dataset_manifest.get("contract_version"),
            "dataset_status": self.dataset_manifest.get("dataset_status"),
            "approval_status": self.dataset_manifest.get("approval_status"),
        }
        return ValidationReport(
            dataset=dataset,
            requested_stage=self.stage,
            validation_timestamp=self.timestamp,
            summary=summary,
            findings=self.findings,
            samples=self.samples,
        )


def validate_dataset(
    dataset_root: Path,
    stage: ValidationStage | str,
    *,
    validation_timestamp: str | None = None,
) -> ValidationReport:
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("dataset_root must be an existing directory")
    selected_stage = stage if isinstance(stage, ValidationStage) else ValidationStage(stage)
    run = _ValidationRun(
        dataset_root=root,
        stage=selected_stage,
        timestamp=validation_timestamp or _utc_timestamp(),
        schemas=load_schema_bundle(),
    )
    if run.load_dataset_manifest():
        run.validate_schema_copies()
        run.load_samples()
        run.validate_dataset_consistency()
        run.load_and_validate_split()
        run.validate_stage()
    return run.report()
