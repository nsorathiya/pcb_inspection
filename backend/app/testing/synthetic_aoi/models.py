from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GENERATOR_ID = "pcb-aoi-synthetic-inspection-fixtures"
GENERATOR_VERSION = "1.0.0"
SCENARIO_CONTRACT_VERSION = "pcb-aoi-synthetic-inspection-scenario/1.0"
GENERATION_MANIFEST_VERSION = "pcb-aoi-synthetic-fixture-generation/1.0"
MARKER_VERSION = "pcb-aoi-synthetic-fixture-marker/1.0"
DEFAULT_SEED = 20260717
FIXTURE_TIMESTAMP = "2026-07-17T00:00:00Z"
SYNTHETIC_STATEMENT = "Synthetic fixture for software validation only."
MARKER_FILENAME = "SYNTHETIC_FIXTURES_MARKER.json"
GENERATION_MANIFEST_FILENAME = "generation_manifest.json"


class SyntheticFixtureError(ValueError):
    """Expected generation or output-safety failure."""


@dataclass(frozen=True)
class ArtifactPlan:
    role: str
    references: tuple[str, ...]
    generated_file: str
    content: bytes
    media_type: str
    reference_expected_to_resolve: bool = True
    declared_sha256: str | None = None
    declared_byte_size: int | None = None


@dataclass(frozen=True)
class ScenarioPlan:
    scenario_id: str
    description: str
    expected_intake_outcome: str
    expected_technical_validation_outcome: str
    expected_finding_codes: tuple[str, ...]
    policy_id: str
    policy_version: str
    artifacts: tuple[ArtifactPlan, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GenerationResult:
    output_root: Path
    seed: int
    scenario_ids: tuple[str, ...]
    output_tree_sha256: str
