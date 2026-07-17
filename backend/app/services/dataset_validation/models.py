from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

VALIDATOR_VERSION = "1.0.0"
EXIT_PASSED = 0
EXIT_BLOCKED = 1
EXIT_USAGE = 2
EXIT_UNEXPECTED = 3


class ValidationStage(str, Enum):
    TECHNICAL = "technical-validation"
    TRAINING = "model-training"
    PRODUCTION = "production-acceptance"


@dataclass(frozen=True)
class Finding:
    code: str
    category: str
    message: str
    severity: str = "error"
    sample_id: str | None = None
    path: str | None = None

    def sort_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.sample_id or "",
            self.category,
            self.code,
            self.path or "",
            self.severity,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "category": self.category,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.sample_id is not None:
            result["sample_id"] = self.sample_id
        if self.path is not None:
            result["path"] = self.path
        return result


@dataclass
class SampleValidation:
    inventory: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        findings = sorted(self.findings, key=Finding.sort_key)
        blocked = any(finding.severity == "error" for finding in findings)
        return {
            **self.inventory,
            "validation_status": "BLOCKED" if blocked else "PASS",
            "stage_readiness_status": "BLOCKED" if blocked else "READY",
            "findings": [finding.to_dict() for finding in findings],
        }


@dataclass
class ValidationReport:
    dataset: dict[str, Any]
    requested_stage: ValidationStage
    validation_timestamp: str
    summary: dict[str, Any]
    findings: list[Finding]
    samples: list[SampleValidation]

    @property
    def blocked(self) -> bool:
        return any(finding.severity == "error" for finding in self.findings)

    @property
    def exit_code(self) -> int:
        return EXIT_BLOCKED if self.blocked else EXIT_PASSED

    def to_dict(self) -> dict[str, Any]:
        findings = sorted(self.findings, key=Finding.sort_key)
        samples = sorted(
            (sample.to_dict() for sample in self.samples),
            key=lambda sample: (sample.get("sample_id") or ""),
        )
        return {
            "validator_version": VALIDATOR_VERSION,
            "validation_timestamp": self.validation_timestamp,
            "requested_stage": self.requested_stage.value,
            "overall_status": "BLOCKED" if self.blocked else "PASS",
            "dataset": self.dataset,
            "summary": self.summary,
            "findings": [finding.to_dict() for finding in findings],
            "samples": samples,
        }
