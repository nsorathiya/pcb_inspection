from __future__ import annotations

from typing import Mapping


class InferenceKnownFailure(ValueError):
    """Expected incompatibility mapped to one authoritative finding."""

    def __init__(
        self,
        finding_code: str,
        *,
        branch: str | None = None,
        field: str | None = None,
        details: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        super().__init__(finding_code)
        self.finding_code = finding_code
        self.branch = branch
        self.field = field
        self.details = dict(details or {})


class InferencePolicyLoadError(InferenceKnownFailure):
    pass
