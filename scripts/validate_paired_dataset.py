from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.dataset_validation import (  # noqa: E402
    EXIT_UNEXPECTED,
    EXIT_USAGE,
    ValidationStage,
    validate_dataset,
)
from app.services.dataset_validation.reporting import (  # noqa: E402
    validate_report_directory,
    write_reports,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only validation of a versioned paired PCB AOI dataset package.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing dataset_manifest.json; never modified by this tool.",
    )
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in ValidationStage],
        required=True,
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Output directory outside dataset-root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset_root = args.dataset_root.resolve(strict=True)
        if not dataset_root.is_dir():
            raise ValueError("dataset-root must be an existing directory")
        validate_report_directory(dataset_root, args.report_dir)
    except (OSError, ValueError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        report = validate_dataset(dataset_root, args.stage)
        json_path, markdown_path = write_reports(
            report,
            dataset_root,
            args.report_dir,
        )
    except Exception as exc:  # pragma: no cover - final CLI containment boundary
        print(f"unexpected validator failure: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED
    print(
        f"Paired dataset validation: stage={args.stage} "
        f"status={'BLOCKED' if report.blocked else 'PASS'} "
        f"samples={report.summary['total_samples']} exit={report.exit_code}"
    )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
