"""Deterministic, non-production RGB/height software-fixture generation."""

from app.testing.synthetic_aoi.generator import (
    generate_fixtures,
    validate_generated_fixtures,
)
from app.testing.synthetic_aoi.models import (
    DEFAULT_SEED,
    GENERATOR_ID,
    GENERATOR_VERSION,
    GenerationResult,
    SyntheticFixtureError,
)
from app.testing.synthetic_aoi.scenarios import SCENARIO_IDS

__all__ = [
    "DEFAULT_SEED",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "GenerationResult",
    "SCENARIO_IDS",
    "SyntheticFixtureError",
    "generate_fixtures",
    "validate_generated_fixtures",
]
