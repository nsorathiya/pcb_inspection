from __future__ import annotations

import json
from pathlib import Path

from app.services.dataset_validation.file_inspection import is_within
from app.services.dataset_validation.models import ValidationReport

JSON_REPORT_NAME = "paired_dataset_validation.json"
MARKDOWN_REPORT_NAME = "paired_dataset_validation.md"


def validate_report_directory(dataset_root: Path, report_directory: Path) -> Path:
    root = dataset_root.resolve(strict=True)
    report = report_directory.resolve(strict=False)
    if report == root or is_within(report, root):
        raise ValueError("report_dir must be outside dataset_root")
    current = report
    while not current.exists() and current.parent != current:
        current = current.parent
    if current.exists() and current.is_symlink():
        raise ValueError("report_dir may not resolve through a symbolic link")
    return report


def render_markdown(report: ValidationReport) -> str:
    payload = report.to_dict()
    dataset = payload["dataset"]
    summary = payload["summary"]
    lines = [
        "# Paired 2D/3D Dataset Validation",
        "",
        f"- Validator version: `{payload['validator_version']}`",
        f"- Validation timestamp: `{payload['validation_timestamp']}`",
        f"- Requested stage: `{payload['requested_stage']}`",
        f"- Overall status: **{payload['overall_status']}**",
        f"- Dataset: `{dataset.get('dataset_id')}` version `{dataset.get('dataset_version')}`",
        f"- Contract: `{dataset.get('contract_version')}`",
        "",
        "## Summary",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Total samples | {summary['total_samples']} |",
        f"| OK | {summary['label_counts']['OK']} |",
        f"| NOK | {summary['label_counts']['NOK']} |",
        f"| Valid pairs | {summary['valid_pairs']} |",
        f"| Invalid pairs | {summary['invalid_pairs']} |",
        f"| Hash failures | {summary['hash_failures']} |",
        f"| Missing files | {summary['missing_files']} |",
        f"| Metadata/schema failures | {summary['metadata_schema_failures']} |",
        f"| Image/depth mismatches | {summary['image_depth_metadata_mismatches']} |",
        f"| Registration/calibration issues | {summary['registration_calibration_issues']} |",
        f"| Split leakage issues | {summary['split_leakage_issues']} |",
        f"| Stage-readiness blockers | {summary['stage_readiness_blockers']} |",
        f"| Blocking findings | {summary['blocking_findings']} |",
        f"| Warnings | {summary['warnings']} |",
        "",
        "## Counts",
        "",
        f"- Defect types: `{json.dumps(summary['defect_type_counts'], sort_keys=True)}`",
        f"- Boards: `{json.dumps(summary['board_counts'], sort_keys=True)}`",
        f"- Recipes: `{json.dumps(summary['recipe_version_counts'], sort_keys=True)}`",
        "",
        "## Findings",
        "",
    ]
    if payload["findings"]:
        for finding in payload["findings"]:
            context = []
            if finding.get("sample_id"):
                context.append(f"sample={finding['sample_id']}")
            if finding.get("path"):
                context.append(f"path={finding['path']}")
            suffix = f" ({', '.join(context)})" if context else ""
            lines.append(
                f"- **{finding['severity'].upper()}** `{finding['code']}`: "
                f"{finding['message']}{suffix}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Samples", ""])
    for sample in payload["samples"]:
        lines.extend(
            [
                f"### `{sample['sample_id']}`",
                "",
                f"- Directory: `{sample.get('sample_directory')}`",
                f"- Label/defect: `{sample.get('label')}` / `{sample.get('defect_type')}`",
                f"- Board: `{sample.get('board_id')}`",
                f"- Recipe: `{sample.get('recipe_id')}@{sample.get('recipe_version')}`",
                f"- Pair status: **{sample.get('pair_validation_status')}**",
                f"- Stage readiness: **{sample.get('stage_readiness_status')}**",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    report: ValidationReport,
    dataset_root: Path,
    report_directory: Path,
) -> tuple[Path, Path]:
    output = validate_report_directory(dataset_root, report_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / JSON_REPORT_NAME
    markdown_path = output / MARKDOWN_REPORT_NAME
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
