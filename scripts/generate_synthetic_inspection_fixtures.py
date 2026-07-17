from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.testing.synthetic_aoi import (  # noqa: E402
    DEFAULT_SEED,
    GENERATOR_VERSION,
    SCENARIO_IDS,
    SyntheticFixtureError,
    generate_fixtures,
)

EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_UNEXPECTED = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic synthetic RGB/height fixtures for software "
            "validation only. No production or model-accuracy claims are created."
        )
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Explicit, safe directory outside the repository and real datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic non-negative seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIO_IDS,
        help="Generate one scenario; repeat to select multiple. Default: all scenarios.",
    )
    parser.add_argument(
        "--overwrite-generated",
        action="store_true",
        help="Replace only an intact directory proven to be owned by this generator.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = generate_fixtures(
            args.output_root,
            seed=args.seed,
            scenario_ids=args.scenario,
            overwrite_generated=args.overwrite_generated,
        )
    except (OSError, SyntheticFixtureError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # pragma: no cover - final CLI containment boundary
        print(f"unexpected synthetic fixture generation failure: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED
    print(
        "Synthetic fixtures generated: "
        f"generator_version={GENERATOR_VERSION} seed={result.seed} "
        f"scenarios={len(result.scenario_ids)}"
    )
    print(f"Output root: {result.output_root}")
    print(f"Scenario-tree SHA-256: {result.output_tree_sha256}")
    print("Synthetic fixture for software validation only.")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
