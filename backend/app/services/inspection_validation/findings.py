from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.db.models import ArtifactType
from app.services.inspection_validation.interfaces import (
    FindingCategory,
    FindingSeverity,
    ValidationFinding,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CATALOGUE_PATH = REPOSITORY_ROOT / "contracts" / "inspection_validation_findings.json"
_ARTIFACT_ORDER = {None: 0, ArtifactType.RGB_RAW: 1, ArtifactType.HEIGHT_RAW: 2,
                   ArtifactType.VALIDITY_MASK: 3, ArtifactType.CALIBRATION: 4}
_FORBIDDEN_DETAIL_KEYS = {"path", "absolute_path", "relative_path", "filename", "source_filename"}


@dataclass(frozen=True)
class FindingDefinition:
    order: int
    severity: FindingSeverity
    category: FindingCategory
    blocking: bool
    description: str


class FindingFactory:
    def __init__(self, catalogue_path: Path = DEFAULT_CATALOGUE_PATH) -> None:
        document = json.loads(catalogue_path.read_text(encoding="utf-8"))
        self._definitions = {
            item["code"]: FindingDefinition(
                order=item["order"],
                severity=FindingSeverity(item["default_severity"]),
                category=FindingCategory(item["category"]),
                blocking=item["default_blocking"],
                description=item["description"],
            )
            for item in document["findings"]
        }
        if set(self._definitions) != set(document["$defs"]["finding_code"]["enum"]):
            raise ValueError("finding catalogue definitions are inconsistent")

    def create(
        self,
        code: str,
        *,
        artifact_type: ArtifactType | None = None,
        field: str | None = None,
        details: Mapping[str, Any] | None = None,
        blocking: bool | None = None,
    ) -> ValidationFinding:
        definition = self._definitions[code]
        safe_details = None if details is None else dict(details)
        if safe_details is not None:
            if _FORBIDDEN_DETAIL_KEYS.intersection(safe_details):
                raise ValueError("finding details must not contain filesystem paths")
            if any(not isinstance(value, (str, int, float, bool, type(None), list, tuple)) for value in safe_details.values()):
                raise TypeError("finding details must contain safe primitive values")
        return ValidationFinding(
            code=code,
            severity=definition.severity,
            category=definition.category,
            message=definition.description,
            blocking=definition.blocking if blocking is None else blocking,
            artifact_type=artifact_type,
            field=field,
            details=safe_details,
        )

    def sort(self, findings: Sequence[ValidationFinding]) -> tuple[ValidationFinding, ...]:
        return tuple(sorted(findings, key=lambda item: (
            self._definitions[item.code].order,
            _ARTIFACT_ORDER.get(item.artifact_type, 99),
            item.field or "",
            item.message,
        )))
