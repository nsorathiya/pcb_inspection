from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from app.db.models import InspectionStatus
from app.db.processing_types import ProcessingRunStatus
from app.db.validation_types import ValidationOutcome
from app.services.inspection_audit.projection import project_audit_event
from app.services.inspection_inference.models import validate_inference_result_document
from app.services.inspection_preprocessing.models import validate_preprocessing_result_document
from app.services.inspection_report.canonical import canonical_report_sha256
from app.services.inspection_report.exceptions import (
    DevelopmentReportConsistencyError,
    DevelopmentReportRetrievalError,
)
from app.services.inspection_report.models import (
    DevelopmentReport,
    ReportArtifact,
    ReportAuditItem,
    ReportInspection,
    ReportProcessing,
    ReportValidation,
)
from app.services.inspection_report.repository import InspectionReportRecords, InspectionReportRepository
from app.services.inspection_validation.models import canonical_result_sha256, result_from_dict, result_to_dict

ROOT = Path(__file__).resolve().parents[4]
REPORT_CONTRACT_VERSION = "pcb-aoi-inspection-development-report/1.0"
BASE_LIMITATIONS = (
    "This development report is not a production quality certificate, calibration certificate, model-validation report, accuracy report, or legal disposition record.",
    "Physical calibration is not proven.",
    "Real 2D/3D registration is not proven.",
)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DevelopmentReportConsistencyError(f"persisted {label} JSON is invalid") from exc
    if not isinstance(parsed, dict):
        raise DevelopmentReportConsistencyError(f"persisted {label} JSON is not an object")
    return parsed


def _enum(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _finding_document(record: Any, *, validation: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": record.code,
        "severity": _enum(record.severity),
        "category": _enum(record.category),
        "message": record.message,
        "blocking": record.blocking,
    }
    if validation:
        if record.artifact_type is not None:
            result["artifact_type"] = _enum(record.artifact_type)
    elif record.branch is not None:
        result["branch"] = record.branch
    if record.field is not None:
        result["field"] = record.field
    details = _json_object(record.details_json, "finding details")
    if details:
        result["details"] = details
    return result


def _require_ordinals(records: Sequence[Any], label: str) -> list[Any]:
    ordered = sorted(records, key=lambda item: item.ordinal)
    if [item.ordinal for item in ordered] != list(range(len(ordered))):
        raise DevelopmentReportConsistencyError(f"{label} finding ordinals are inconsistent")
    return ordered


def _safe_inspection_error(inspection) -> dict[str, str] | None:
    if inspection.status not in {InspectionStatus.ERROR, InspectionStatus.VALIDATION_FAILED}:
        return None
    known = {
        ("INSPECTION_INTAKE_FAILED", "Paired artifact intake did not complete."),
        ("INPUT_VALIDATION_FAILED", "Inspection input validation failed."),
        ("VALIDATOR_INTERNAL_ERROR", "Inspection validation could not complete."),
        ("PREPROCESSING_FAILED", "Inspection preprocessing did not complete successfully."),
        ("PREPROCESSING_ERROR", "Inspection preprocessing did not complete successfully."),
        ("INFERENCE_FAILED", "Inspection inference did not complete successfully."),
        ("INFERENCE_ERROR", "Inspection inference did not complete successfully."),
    }
    if (inspection.error_code, inspection.error_message) in known:
        return {"code": inspection.error_code, "message": inspection.error_message}
    return {"code": "INSPECTION_ERROR", "message": "Inspection lifecycle did not complete successfully."}


class InspectionReportService:
    def __init__(self, repository: InspectionReportRepository) -> None:
        self._repository = repository
        pre_catalogue = self._json("contracts/inspection_preprocessing_findings.json")
        inf_catalogue = self._json("contracts/inspection_inference_findings.json")
        taxonomy = self._json("contracts/defect_taxonomy.json")
        pre_schema = self._json("contracts/inspection_preprocessing_result.schema.json")
        inf_schema = self._json("contracts/inspection_inference_result.schema.json")
        self._pre_catalogue = pre_catalogue
        self._inf_catalogue = inf_catalogue
        self._taxonomy = taxonomy
        self._pre_validator = Draft202012Validator(
            pre_schema,
            registry=Registry().with_resource(pre_catalogue["$id"], Resource.from_contents(pre_catalogue)),
            format_checker=FormatChecker(),
        )
        self._inf_validator = Draft202012Validator(
            inf_schema,
            registry=Registry().with_resources((
                (inf_catalogue["$id"], Resource.from_contents(inf_catalogue)),
                (taxonomy["$id"], Resource.from_contents(taxonomy)),
            )),
            format_checker=FormatChecker(),
        )

    @staticmethod
    def _json(relative_path: str) -> Mapping[str, Any]:
        return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))

    async def get_report(self, inspection_id: str) -> tuple[DevelopmentReport, str]:
        try:
            records = await self._repository.load(inspection_id)
        except Exception as exc:
            raise DevelopmentReportRetrievalError("development report evidence could not be retrieved") from exc
        if records.inspection is None:
            raise LookupError("inspection does not exist")
        report = self._assemble(records, inspection_id)
        return report, canonical_report_sha256(report)

    def _assemble(self, records: InspectionReportRecords, inspection_id: str) -> DevelopmentReport:
        inspection = records.inspection
        assert inspection is not None
        if inspection.id != inspection_id:
            raise DevelopmentReportConsistencyError("inspection identity is inconsistent")
        validation = self._validation(records.validation, inspection_id)
        processing = self._processing(records.processing, inspection_id, records.validation)
        self._lifecycle_consistency(inspection, records.validation, records.processing)
        if any(item.inspection_id != inspection_id for item in inspection.artifacts):
            raise DevelopmentReportConsistencyError("artifact ownership is inconsistent")
        audit = []
        for record in records.audits:
            item = project_audit_event(record, inspection_id)
            audit.append(ReportAuditItem(
                audit_event_id=item.audit_event_id,
                inspection_id=item.inspection_id,
                action=item.action,
                created_at=_timestamp(item.created_at),
                actor_id=item.actor_id,
                request_id=item.request_id,
                details=dict(item.details),
                details_redacted=item.details_redacted,
                development_only=item.development_only,
                mock_result=item.mock_result,
                production_approved=item.production_approved,
            ))
        synthetic = processing is not None
        mock_inference = processing is not None and processing.inference is not None
        limitations = [*BASE_LIMITATIONS]
        if processing is not None and processing.preprocessing is not None:
            limitations.append("Synthetic preprocessing was used.")
        if processing is not None and processing.inference is not None:
            limitations.extend(("No real AI model was executed.", "No confidence was produced."))
            if processing.inference.get("decision_basis") == "DETERMINISTIC_HASH_BUCKET":
                limitations.append("Deterministic digest bucketing was used.")
            if processing.final_decision is not None:
                limitations.append("Mock PASS/FAIL/UNCERTAIN is not production PCB disposition.")
        return DevelopmentReport(
            contract_version=REPORT_CONTRACT_VERSION,
            inspection_id=inspection_id,
            development_only=True,
            production_approved=False,
            synthetic_evidence_present=synthetic,
            mock_inference_present=mock_inference,
            inspection=ReportInspection(
                created_at=_timestamp(inspection.created_at),
                board_id=inspection.board_id,
                recipe_id=inspection.recipe_id,
                recipe_version=inspection.recipe_version,
                lot_id=inspection.lot_id,
                operator_id=inspection.operator_id,
                status=inspection.status.value,
                error=_safe_inspection_error(inspection),
            ),
            artifacts=[ReportArtifact(
                artifact_type=item.artifact_type.value,
                sha256=item.sha256,
                byte_size=item.byte_size,
                media_type=item.media_type,
                created_at=_timestamp(item.created_at),
            ) for item in sorted(inspection.artifacts, key=lambda value: (list(type(value.artifact_type)).index(value.artifact_type), value.id))],
            validation=validation,
            processing=processing,
            audit=audit,
            limitations=limitations,
        )

    def _validation(self, record, inspection_id: str) -> ReportValidation | None:
        if record is None:
            return None
        if record.inspection_id != inspection_id:
            raise DevelopmentReportConsistencyError("validation ownership is inconsistent")
        document = _json_object(record.result_json, "validation result")
        try:
            typed = result_from_dict(document)
        except Exception as exc:
            raise DevelopmentReportConsistencyError("validation result is inconsistent") from exc
        if result_to_dict(typed) != document or canonical_result_sha256(typed) != record.result_sha256:
            raise DevelopmentReportConsistencyError("validation result hash is inconsistent")
        expected_identity = (
            record.id, record.inspection_id, record.policy_id, record.policy_version,
            record.validator_version, record.outcome.value,
        )
        actual_identity = (
            typed.validation_id, typed.inspection_id, typed.validation_policy_id,
            typed.validation_policy_version, typed.validator_version, typed.outcome.value,
        )
        if expected_identity != actual_identity:
            raise DevelopmentReportConsistencyError("validation identity is inconsistent")
        if (
            _timestamp(record.started_at) != document.get("started_at")
            or _timestamp(record.completed_at) != document.get("completed_at")
            or record.contract_version != document.get("contract_version")
        ):
            raise DevelopmentReportConsistencyError("validation timestamps or contract are inconsistent")
        rows = _require_ordinals(record.findings, "validation")
        if any(item.validation_id != record.id for item in rows):
            raise DevelopmentReportConsistencyError("validation finding ownership is inconsistent")
        if [_finding_document(item, validation=True) for item in rows] != document["findings"]:
            raise DevelopmentReportConsistencyError("validation findings are inconsistent")
        summary = document["summary"]
        expected_counts = {
            "finding_count": len(rows),
            "info_count": sum(_enum(item.severity) == "INFO" for item in rows),
            "warning_count": sum(_enum(item.severity) == "WARNING" for item in rows),
            "error_count": sum(_enum(item.severity) == "ERROR" for item in rows),
            "blocking_count": sum(item.blocking for item in rows),
        }
        if any(summary.get(key) != value for key, value in expected_counts.items()):
            raise DevelopmentReportConsistencyError("validation summary is inconsistent")
        return ReportValidation(
            contract_version=document["contract_version"], validation_id=record.id,
            validation_key=record.validation_key, result_sha256=record.result_sha256,
            outcome=record.outcome.value,
            policy={"policy_id": record.policy_id, "policy_version": record.policy_version},
            validator_version=record.validator_version,
            started_at=_timestamp(record.started_at), completed_at=_timestamp(record.completed_at),
            rgb_technical_summary=document["rgb_artifact"],
            height_technical_summary=document["height_artifact"],
            findings=document["findings"], summary=summary,
        )

    def _processing(self, run, inspection_id: str, validation) -> ReportProcessing | None:
        if run is None:
            return None
        if (
            run.inspection_id != inspection_id
            or validation is None
            or validation.outcome is not ValidationOutcome.VALIDATION_PASSED
            or run.validation_id != validation.id
        ):
            raise DevelopmentReportConsistencyError("processing ownership or validation identity is inconsistent")
        pre = None
        inf = None
        if run.preprocessing_result is not None:
            record = run.preprocessing_result
            if record.processing_run_id != run.id:
                raise DevelopmentReportConsistencyError("preprocessing ownership is inconsistent")
            pre = _json_object(record.result_json, "preprocessing result")
            rows = _require_ordinals(record.findings, "preprocessing")
            if any(item.preprocessing_id != record.id for item in rows):
                raise DevelopmentReportConsistencyError("preprocessing finding ownership is inconsistent")
            self._validate_processing_document(pre, rows, record.result_sha256, self._pre_validator, self._pre_catalogue, False)
            if (
                pre.get("preprocessing_id") != record.id
                or pre.get("inspection_id") != inspection_id
                or pre.get("validation_id") != run.validation_id
                or pre.get("contract_version") != record.contract_version
                or pre.get("policy_id") != record.policy_id
                or pre.get("policy_version") != record.policy_version
                or pre.get("implementation_id") != record.implementation_id
                or pre.get("implementation_version") != record.implementation_version
                or pre.get("outcome") != record.outcome.value
                or pre.get("started_at") != _timestamp(record.started_at)
                or pre.get("completed_at") != _timestamp(record.completed_at)
                or (record.policy_id, record.policy_version) != (run.preprocessing_policy_id, run.preprocessing_policy_version)
                or (record.implementation_id, record.implementation_version) != (run.preprocessing_implementation_id, run.preprocessing_implementation_version)
                or pre.get("synthetic_input") is not True
                or pre.get("mock_implementation") is not True
                or pre.get("production_approved") is not False
            ):
                raise DevelopmentReportConsistencyError("preprocessing identity is inconsistent")
        if run.inference_result is not None:
            record = run.inference_result
            if record.processing_run_id != run.id or pre is None or record.preprocessing_id != pre.get("preprocessing_id"):
                raise DevelopmentReportConsistencyError("inference ownership is inconsistent")
            inf = _json_object(record.result_json, "inference result")
            rows = _require_ordinals(record.findings, "inference")
            if any(item.inference_id != record.id for item in rows):
                raise DevelopmentReportConsistencyError("inference finding ownership is inconsistent")
            self._validate_processing_document(inf, rows, record.result_sha256, self._inf_validator, self._inf_catalogue, True)
            if (
                inf.get("inference_id") != record.id
                or inf.get("inspection_id") != inspection_id
                or inf.get("validation_id") != run.validation_id
                or inf.get("preprocessing_id") != record.preprocessing_id
                or inf.get("contract_version") != record.contract_version
                or inf.get("policy_id") != record.policy_id
                or inf.get("policy_version") != record.policy_version
                or inf.get("engine_id") != record.engine_id
                or inf.get("engine_version") != record.engine_version
                or inf.get("engine_type") != record.engine_type
                or inf.get("execution_outcome") != record.execution_outcome.value
                or inf.get("decision") != (None if record.decision is None else record.decision.value)
                or inf.get("defect_type") != record.defect_type
                or inf.get("started_at") != _timestamp(record.started_at)
                or inf.get("completed_at") != _timestamp(record.completed_at)
                or (record.policy_id, record.policy_version) != (run.inference_policy_id, run.inference_policy_version)
                or (record.engine_id, record.engine_version, record.engine_type) != (run.engine_id, run.engine_version, run.engine_type)
                or inf.get("synthetic_input") is not True
                or inf.get("mock_preprocessing") is not True
                or inf.get("mock_inference") is not True
                or inf.get("production_approved") is not False
            ):
                raise DevelopmentReportConsistencyError("inference identity is inconsistent")
            if record.confidence is not None or "confidence" not in inf or inf["confidence"] is not None:
                raise DevelopmentReportConsistencyError("mock inference confidence is inconsistent")
            if run.final_decision is not None and inf.get("decision") != run.final_decision.value:
                raise DevelopmentReportConsistencyError("processing decision is inconsistent")
            inf = {key: value for key, value in inf.items() if key != "confidence"}
        if run.status is ProcessingRunStatus.STARTED:
            if pre is not None or inf is not None or run.completed_at is not None:
                raise DevelopmentReportConsistencyError("started processing evidence is inconsistent")
        else:
            if pre is None:
                raise DevelopmentReportConsistencyError("completed processing evidence is missing preprocessing")
            if run.status is ProcessingRunStatus.COMPLETED and inf is None:
                raise DevelopmentReportConsistencyError("completed processing evidence is missing inference")
            expected_completed = (
                pre.get("completed_at") if inf is None else inf.get("completed_at")
            )
            if run.completed_at is None or _timestamp(run.completed_at) != expected_completed:
                raise DevelopmentReportConsistencyError("processing completion timestamp is inconsistent")
        error = None
        if run.status is ProcessingRunStatus.ERROR:
            safe_messages = {
                "PREPROCESSING_FAILED": "Inspection preprocessing did not complete successfully.",
                "PREPROCESSING_ERROR": "Inspection preprocessing did not complete successfully.",
                "INFERENCE_FAILED": "Inspection inference did not complete successfully.",
                "INFERENCE_ERROR": "Inspection inference did not complete successfully.",
            }
            message = safe_messages.get(run.error_code)
            error = {"code": run.error_code, "message": message} if message and message == run.error_message else {"code": "PROCESSING_ERROR", "message": "Processing did not complete successfully."}
        return ReportProcessing(
            processing_run_id=run.id, validation_id=run.validation_id,
            processing_key=run.processing_key, lifecycle_status=run.status.value,
            preprocessing_policy={"policy_id": run.preprocessing_policy_id, "policy_version": run.preprocessing_policy_version},
            preprocessing_implementation={"implementation_id": run.preprocessing_implementation_id, "implementation_version": run.preprocessing_implementation_version},
            inference_policy={"policy_id": run.inference_policy_id, "policy_version": run.inference_policy_version},
            engine={"engine_id": run.engine_id, "engine_version": run.engine_version, "engine_type": run.engine_type},
            started_at=_timestamp(run.started_at), completed_at=None if run.completed_at is None else _timestamp(run.completed_at),
            final_decision=None if run.final_decision is None else run.final_decision.value,
            error=error, preprocessing=pre, inference=inf,
            synthetic_input=pre is not None and pre.get("synthetic_input") is True,
            mock_preprocessing=pre is not None and pre.get("mock_implementation") is True,
            mock_inference=inf is not None and inf.get("mock_inference") is True,
            production_approved=False,
        )

    def _validate_processing_document(self, document, rows, result_sha256, validator, catalogue, inference):
        from hashlib import sha256
        canonical = json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if sha256(canonical).hexdigest() != result_sha256:
            raise DevelopmentReportConsistencyError("processing result hash is inconsistent")
        try:
            validator.validate(document)
            if inference:
                validate_inference_result_document(document, catalogue, self._taxonomy)
            else:
                validate_preprocessing_result_document(document, catalogue)
        except Exception as exc:
            raise DevelopmentReportConsistencyError("processing result contract is inconsistent") from exc
        if [_finding_document(item) for item in rows] != document.get("findings"):
            raise DevelopmentReportConsistencyError("processing findings are inconsistent")

    @staticmethod
    def _lifecycle_consistency(inspection, validation, run) -> None:
        status = inspection.status
        if status is InspectionStatus.RECEIVED and (validation is not None or run is not None):
            raise DevelopmentReportConsistencyError("RECEIVED inspection has later lifecycle evidence")
        if status in {InspectionStatus.READY, InspectionStatus.VALIDATION_FAILED}:
            if validation is None or run is not None:
                raise DevelopmentReportConsistencyError("validation lifecycle evidence is inconsistent")
            expected = ValidationOutcome.VALIDATION_PASSED if status is InspectionStatus.READY else ValidationOutcome.VALIDATION_FAILED
            if validation.outcome is not expected:
                raise DevelopmentReportConsistencyError("validation outcome is inconsistent with inspection status")
        if status is InspectionStatus.PROCESSING:
            if validation is None or validation.outcome is not ValidationOutcome.VALIDATION_PASSED or run is None or run.status is not ProcessingRunStatus.STARTED:
                raise DevelopmentReportConsistencyError("processing lifecycle evidence is inconsistent")
        if status in {InspectionStatus.PASS, InspectionStatus.FAIL, InspectionStatus.UNCERTAIN}:
            if validation is None or validation.outcome is not ValidationOutcome.VALIDATION_PASSED or run is None or run.status is not ProcessingRunStatus.COMPLETED or run.final_decision is None or run.final_decision.value != status.value:
                raise DevelopmentReportConsistencyError("final lifecycle evidence is inconsistent")
            if inspection.completed_at is None or run.completed_at is None or _timestamp(inspection.completed_at) != _timestamp(run.completed_at):
                raise DevelopmentReportConsistencyError("final inspection timestamp is inconsistent")
        if status is InspectionStatus.ERROR:
            if run is not None and run.status is not ProcessingRunStatus.ERROR:
                raise DevelopmentReportConsistencyError("error lifecycle evidence is inconsistent")
            if run is None and validation is not None and validation.outcome is not ValidationOutcome.VALIDATION_ERROR:
                raise DevelopmentReportConsistencyError("validation error evidence is inconsistent")
            if run is not None and (inspection.completed_at is None or run.completed_at is None or _timestamp(inspection.completed_at) != _timestamp(run.completed_at)):
                raise DevelopmentReportConsistencyError("error inspection timestamp is inconsistent")
