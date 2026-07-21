from __future__ import annotations

import json
from hashlib import sha256

from app.services.inspection_report.exceptions import DevelopmentReportCanonicalError
from app.services.inspection_report.models import DevelopmentReport


def canonical_report_bytes(report: DevelopmentReport) -> bytes:
    try:
        return json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DevelopmentReportCanonicalError("development report could not be serialized") from exc


def canonical_report_sha256(report: DevelopmentReport) -> str:
    return sha256(canonical_report_bytes(report)).hexdigest()
