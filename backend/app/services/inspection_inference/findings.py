from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from app.services.inspection_inference.models import (
    InferenceFinding,
    InferenceFindingCategory,
    InferenceFindingSeverity,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CATALOGUE_PATH = REPOSITORY_ROOT / "contracts" / "inspection_inference_findings.json"


def _path_shaped(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or "\\" in value
        or ".." in PurePosixPath(value).parts
    )


class InferenceFindingFactory:
    def __init__(self, catalogue_path: Path = DEFAULT_CATALOGUE_PATH) -> None:
        document = json.loads(Path(catalogue_path).read_text(encoding="utf-8"))
        self._document = document
        self._definitions = {item["code"]: item for item in document["findings"]}
        self._orders = {code: item["order"] for code, item in self._definitions.items()}

    @property
    def catalogue(self) -> Mapping[str, Any]:
        return self._document

    def create(
        self,
        code: str,
        *,
        branch: str | None = None,
        field: str | None = None,
        blocking: bool | None = None,
        details: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> InferenceFinding:
        definition = self._definitions.get(code)
        if definition is None:
            raise KeyError(f"unknown inference finding code: {code}")
        safe_details = dict(details or {})
        if any(_path_shaped(value) for value in safe_details.values()):
            raise ValueError("inference finding details must not contain paths")
        return InferenceFinding(
            code=code,
            severity=InferenceFindingSeverity(definition["severity"]),
            category=InferenceFindingCategory(definition["category"]),
            message=definition["message"],
            blocking=definition["default_blocking"] if blocking is None else blocking,
            branch=branch,
            field=field,
            details=safe_details,
        )

    def sort(self, findings: Sequence[InferenceFinding]) -> tuple[InferenceFinding, ...]:
        return tuple(sorted(findings, key=self.sort_key))

    def sort_key(self, finding: InferenceFinding) -> tuple[object, ...]:
        return (
            self._orders[finding.code],
            finding.branch or "",
            finding.field or "",
            finding.code,
            finding.message,
            json.dumps(dict(finding.details), sort_keys=True, separators=(",", ":")),
        )
