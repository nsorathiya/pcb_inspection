from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from app.db.models import InspectionStatus
from app.db.repositories import AuditEventRepository
from app.db.processing_types import ProcessingRunStatus
from app.services.inspection_inference.findings import InferenceFindingFactory
from app.services.inspection_inference.models import (
    InferenceExecutionOutcome,
    InferenceFindingSeverity,
    InferenceSummary,
    InspectionInferencePolicy,
    InspectionInferenceResult,
    SyntheticInferenceInput,
)
from app.services.inspection_preprocessing.findings import PreprocessingFindingFactory
from app.services.inspection_preprocessing.models import (
    FindingSeverity,
    InspectionPreprocessingPolicy,
    InspectionPreprocessingResult,
    PreprocessingOutcome,
    PreprocessingSummary,
    RegistrationProcessingResult,
    SyntheticPreprocessingExecution,
)
from app.services.inspection_processing.exceptions import (
    ProcessingExecutionConsistencyError,
    ProcessingExecutionInProgressError,
)
from app.services.inspection_processing.execution_models import (
    InferenceEvidenceResult,
    PreprocessingEvidenceResult,
    ProcessingExecutionResult,
    ProcessingFindingResult,
    ProcessingSummaryResult,
)
from app.services.inspection_processing.input_builder import ProcessingInputSnapshot
from app.services.inspection_processing.persistence import (
    InspectionProcessingRepository,
    PersistedProcessingFinding,
    PersistedProcessingResult,
    PersistedProcessingRun,
)

ROOT = Path(__file__).resolve().parents[4]
PREPROCESSING_CONTRACT = "pcb-aoi-inspection-preprocessing/1.0"
INFERENCE_CONTRACT = "pcb-aoi-inspection-inference/1.0"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _at_or_after(clock: Callable[[], datetime], minimum: datetime) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("orchestration clock must include timezone information")
    value = value.astimezone(timezone.utc)
    minimum = minimum.astimezone(timezone.utc)
    return max(value, minimum)


def build_inference_input(
    execution: SyntheticPreprocessingExecution,
) -> SyntheticInferenceInput:
    result = execution.result
    return SyntheticInferenceInput(
        inspection_id=result.inspection_id,
        validation_id=result.validation_id,
        preprocessing_id=result.preprocessing_id,
        preprocessing_outcome=result.outcome.value,
        synthetic_input=result.synthetic_input,
        mock_preprocessing=result.mock_implementation,
        rgb_buffer=execution.rgb_buffer,
        height_buffer=execution.height_buffer,
    )


class SafeProcessingErrorResultFactory:
    """Construct only schema-shaped internal-error results after a guarded begin."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        preprocessing_id_generator: Callable[[], str] | None = None,
        inference_id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._clock = clock or _utc_now
        self._preprocessing_id = preprocessing_id_generator or (lambda: str(uuid4()))
        self._inference_id = inference_id_generator or (lambda: str(uuid4()))
        self._pre_findings = PreprocessingFindingFactory()
        self._inf_findings = InferenceFindingFactory()

    def preprocessing_error(
        self,
        snapshot: ProcessingInputSnapshot,
        policy: InspectionPreprocessingPolicy,
        *,
        implementation_id: str,
        implementation_version: str,
        not_before: datetime,
    ) -> InspectionPreprocessingResult:
        started = _at_or_after(self._clock, not_before)
        completed = _at_or_after(self._clock, started)
        finding = self._pre_findings.create("PREPROCESSING_INTERNAL_ERROR")
        return InspectionPreprocessingResult(
            contract_version=PREPROCESSING_CONTRACT,
            preprocessing_id=self._preprocessing_id(),
            inspection_id=snapshot.inspection_id,
            validation_id=snapshot.validation.validation_id,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            outcome=PreprocessingOutcome.ERROR,
            started_at=started,
            completed_at=completed,
            synthetic_input=True,
            mock_implementation=True,
            production_approved=False,
            rgb_input=snapshot.rgb_identity,
            height_input=snapshot.height_identity,
            rgb_output=None,
            height_output=None,
            registration=RegistrationProcessingResult(
                registration_mode=policy.registration.registration_mode.value,
                registration_status="FAILED",
                transform_applied=False,
                transform_reference=None,
                synthetic_identity=False,
                output_coordinate_reference=policy.registration.output_coordinate_reference,
                registration_warning=None,
            ),
            findings=(finding,),
            summary=PreprocessingSummary(
                total_findings=1,
                blocking_findings=1,
                warnings=0,
                errors=1,
            ),
        )

    def inference_error(
        self,
        preprocessing: InspectionPreprocessingResult,
        policy: InspectionInferencePolicy,
        *,
        engine_id: str,
        engine_version: str,
    ) -> InspectionInferenceResult:
        started = _at_or_after(self._clock, preprocessing.completed_at)
        completed = _at_or_after(self._clock, started)
        finding = self._inf_findings.create("INFERENCE_INTERNAL_ERROR")
        return InspectionInferenceResult(
            contract_version=INFERENCE_CONTRACT,
            inference_id=self._inference_id(),
            inspection_id=preprocessing.inspection_id,
            validation_id=preprocessing.validation_id,
            preprocessing_id=preprocessing.preprocessing_id,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            engine_id=engine_id,
            engine_version=engine_version,
            engine_type="MOCK",
            execution_outcome=InferenceExecutionOutcome.ERROR,
            started_at=started,
            completed_at=completed,
            synthetic_input=True,
            mock_preprocessing=True,
            mock_inference=True,
            production_approved=False,
            rgb_input=None,
            height_input=None,
            decision=None,
            defect_type=None,
            confidence=None,
            decision_basis=None,
            decision_digest=None,
            findings=(finding,),
            summary=InferenceSummary(
                total_findings=1,
                blocking_findings=1,
                warnings=0,
                errors=1,
            ),
        )


class SafeProcessingResultMapper:
    """Map current or immutable persisted lifecycle evidence to a path-free result."""

    def __init__(
        self,
        repository: InspectionProcessingRepository,
        audit_events: AuditEventRepository | None = None,
    ) -> None:
        self._repository = repository
        self._audit_events = audit_events
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
            registry=Registry().with_resource(
                pre_catalogue["$id"], Resource.from_contents(pre_catalogue)
            ),
            format_checker=FormatChecker(),
        )
        self._inf_validator = Draft202012Validator(
            inf_schema,
            registry=Registry().with_resources(
                (
                    (inf_catalogue["$id"], Resource.from_contents(inf_catalogue)),
                    (taxonomy["$id"], Resource.from_contents(taxonomy)),
                )
            ),
            format_checker=FormatChecker(),
        )

    @staticmethod
    def _json(relative: str) -> Mapping[str, Any]:
        return json.loads((ROOT / relative).read_text(encoding="utf-8"))

    async def replay(
        self,
        run: PersistedProcessingRun,
        inspection_status: InspectionStatus,
    ) -> ProcessingExecutionResult:
        return await self.restore(
            run,
            inspection_status,
            idempotent=True,
            started_now=False,
        )

    async def restore(
        self,
        run: PersistedProcessingRun,
        inspection_status: InspectionStatus,
        *,
        idempotent: bool,
        started_now: bool,
    ) -> ProcessingExecutionResult:
        pre = await self._repository.get_preprocessing_result(run.processing_run_id)
        inf = await self._repository.get_inference_result(run.processing_run_id)
        if run.status is ProcessingRunStatus.STARTED:
            if pre is not None or inf is not None or inspection_status is not InspectionStatus.PROCESSING:
                raise ProcessingExecutionConsistencyError(
                    "started processing lifecycle is internally inconsistent"
                )
            raise ProcessingExecutionInProgressError(
                "identical processing execution is already in progress"
            )
        if pre is None:
            raise ProcessingExecutionConsistencyError(
                "persisted processing lifecycle is missing preprocessing evidence"
            )
        pre_doc = await self._validate_pre(run, pre)
        if inf is None:
            inf_doc = None
        else:
            inf_doc = await self._validate_inf(run, pre_doc, inf)
        self._validate_lifecycle_shape(run, inspection_status, pre_doc, inf_doc)
        await self._verify_audits(run)
        return self._map_documents(
            run,
            inspection_status,
            pre_doc,
            inf_doc,
            idempotent=idempotent,
            started_now=started_now,
        )

    async def _verify_audits(self, run: PersistedProcessingRun) -> None:
        if self._audit_events is None:
            return
        try:
            records = await self._audit_events.list_for_entity(
                "inspection", run.inspection_id
            )
            matches: dict[str, int] = {}
            for record in records:
                if not record.action.startswith("INSPECTION_PROCESSING_") and not record.action.startswith("INSPECTION_MOCK_RESULT_"):
                    continue
                details = json.loads(record.details_json)
                if isinstance(details, dict) and details.get("processing_run_id") == run.processing_run_id:
                    matches[record.action] = matches.get(record.action, 0) + 1
            final_action = (
                "INSPECTION_PROCESSING_ERROR"
                if run.status is ProcessingRunStatus.ERROR
                else f"INSPECTION_MOCK_RESULT_{run.final_decision}"
            )
            if matches != {
                "INSPECTION_PROCESSING_STARTED": 1,
                final_action: 1,
            }:
                raise ValueError("processing lifecycle audits disagree")
        except Exception as exc:
            raise ProcessingExecutionConsistencyError(
                "persisted processing audit evidence is internally inconsistent"
            ) from exc

    async def _validate_pre(
        self, run: PersistedProcessingRun, persisted: PersistedProcessingResult
    ) -> Mapping[str, Any]:
        try:
            document = dict(persisted.result)
            self._pre_validator.validate(document)
            from app.services.inspection_preprocessing.models import validate_preprocessing_result_document

            validate_preprocessing_result_document(document, self._pre_catalogue)
            if (
                persisted.result_id != document["preprocessing_id"]
                or persisted.processing_run_id != run.processing_run_id
                or persisted.outcome != document["outcome"]
                or persisted.result_sha256 != self._document_sha(document)
                or document["inspection_id"] != run.inspection_id
                or document["validation_id"] != run.validation_id
                or (
                    document["policy_id"], document["policy_version"],
                    document["implementation_id"], document["implementation_version"],
                ) != (
                    run.preprocessing_policy_id, run.preprocessing_policy_version,
                    run.preprocessing_implementation_id, run.preprocessing_implementation_version,
                )
            ):
                raise ValueError("preprocessing columns disagree")
            findings = await self._repository.list_preprocessing_findings(
                persisted.result_id
            )
            self._verify_findings(document["findings"], findings, persisted.result_id)
            return document
        except Exception as exc:
            raise ProcessingExecutionConsistencyError(
                "persisted preprocessing evidence is internally inconsistent"
            ) from exc

    async def _validate_inf(
        self,
        run: PersistedProcessingRun,
        preprocessing: Mapping[str, Any],
        persisted: PersistedProcessingResult,
    ) -> Mapping[str, Any]:
        try:
            document = dict(persisted.result)
            self._inf_validator.validate(document)
            from app.services.inspection_inference.models import validate_inference_result_document

            validate_inference_result_document(
                document, self._inf_catalogue, self._taxonomy
            )
            if (
                persisted.result_id != document["inference_id"]
                or persisted.processing_run_id != run.processing_run_id
                or persisted.outcome != document["execution_outcome"]
                or persisted.result_sha256 != self._document_sha(document)
                or document["inspection_id"] != run.inspection_id
                or document["validation_id"] != run.validation_id
                or document["preprocessing_id"] != preprocessing["preprocessing_id"]
                or (
                    document["policy_id"], document["policy_version"],
                    document["engine_id"], document["engine_version"], document["engine_type"],
                ) != (
                    run.inference_policy_id, run.inference_policy_version,
                    run.engine_id, run.engine_version, run.engine_type,
                )
            ):
                raise ValueError("inference columns disagree")
            findings = await self._repository.list_inference_findings(persisted.result_id)
            self._verify_findings(document["findings"], findings, persisted.result_id)
            return document
        except Exception as exc:
            raise ProcessingExecutionConsistencyError(
                "persisted inference evidence is internally inconsistent"
            ) from exc

    @staticmethod
    def _document_sha(document: Mapping[str, Any]) -> str:
        payload = json.dumps(
            document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    @staticmethod
    def _verify_findings(
        embedded: Sequence[Mapping[str, Any]],
        rows: Sequence[PersistedProcessingFinding],
        parent_id: str,
    ) -> None:
        if len(embedded) != len(rows):
            raise ValueError("finding count disagrees")
        for ordinal, (document, row) in enumerate(zip(embedded, rows)):
            expected = {
                "code": row.code,
                "severity": row.severity,
                "category": row.category,
                "message": row.message,
                "blocking": row.blocking,
            }
            if row.branch is not None:
                expected["branch"] = row.branch
            if row.field is not None:
                expected["field"] = row.field
            if row.details:
                expected["details"] = dict(row.details)
            if (
                row.parent_result_id != parent_id
                or row.ordinal != ordinal
                or dict(document) != expected
            ):
                raise ValueError("finding rows disagree")

    @staticmethod
    def _validate_lifecycle_shape(run, inspection_status, pre, inf) -> None:
        if (
            pre["synthetic_input"] is not True
            or pre["mock_implementation"] is not True
            or pre["production_approved"] is not False
            or (
                inf is not None
                and (
                    inf["synthetic_input"] is not True
                    or inf["mock_preprocessing"] is not True
                    or inf["mock_inference"] is not True
                    or inf["production_approved"] is not False
                    or inf["confidence"] is not None
                )
            )
        ):
            raise ProcessingExecutionConsistencyError(
                "persisted processing safety flags are internally inconsistent"
            )
        pre_success = pre["outcome"] == "PREPROCESSING_SUCCEEDED"
        inf_success = inf is not None and inf["execution_outcome"] == "INFERENCE_SUCCEEDED"
        if run.status is ProcessingRunStatus.COMPLETED:
            if (
                not pre_success
                or not inf_success
                or run.final_decision != inf["decision"]
                or inspection_status.value != run.final_decision
                or run.error_code is not None
            ):
                raise ProcessingExecutionConsistencyError(
                    "completed processing lifecycle is internally inconsistent"
                )
        elif run.status is ProcessingRunStatus.ERROR:
            if (
                (pre_success and inf is None)
                or (not pre_success and inf is not None)
                or (pre_success and inf_success)
                or inspection_status is not InspectionStatus.ERROR
                or run.final_decision is not None
                or run.error_code is None
            ):
                raise ProcessingExecutionConsistencyError(
                    "error processing lifecycle is internally inconsistent"
                )
        else:
            raise ProcessingExecutionConsistencyError(
                "processing lifecycle status is unsupported"
            )

    @staticmethod
    def _map_documents(run, inspection_status, pre, inf, *, idempotent, started_now):
        def summary(document):
            value = document["summary"]
            return ProcessingSummaryResult(
                total_findings=value["total_findings"],
                blocking_findings=value["blocking_findings"],
                warnings=value["warnings"],
                errors=value["errors"],
            )

        def findings(document):
            return tuple(
                ProcessingFindingResult(
                    code=value["code"],
                    severity=value["severity"],
                    category=value["category"],
                    message=value["message"],
                    blocking=value["blocking"],
                    branch=value.get("branch"),
                    field=value.get("field"),
                    details=dict(value.get("details", {})),
                )
                for value in document["findings"]
            )

        preprocessing = PreprocessingEvidenceResult(
            preprocessing_id=pre["preprocessing_id"],
            policy_id=pre["policy_id"],
            policy_version=pre["policy_version"],
            implementation_id=pre["implementation_id"],
            implementation_version=pre["implementation_version"],
            outcome=pre["outcome"],
            summary=summary(pre),
            findings=findings(pre),
        )
        inference = None
        if inf is not None:
            inference = InferenceEvidenceResult(
                inference_id=inf["inference_id"],
                policy_id=inf["policy_id"],
                policy_version=inf["policy_version"],
                engine_id=inf["engine_id"],
                engine_version=inf["engine_version"],
                engine_type=inf["engine_type"],
                execution_outcome=inf["execution_outcome"],
                decision=inf["decision"],
                defect_type=inf["defect_type"],
                summary=summary(inf),
                findings=findings(inf),
            )
        return ProcessingExecutionResult(
            inspection_id=run.inspection_id,
            validation_id=run.validation_id,
            processing_run_id=run.processing_run_id,
            processing_key=run.processing_key,
            preprocessing_id=pre["preprocessing_id"],
            inference_id=None if inf is None else inf["inference_id"],
            preprocessing_outcome=pre["outcome"],
            inference_execution_outcome=None if inf is None else inf["execution_outcome"],
            mock_decision=None if inf is None else inf["decision"],
            defect_type=None if inf is None else inf["defect_type"],
            inspection_status=inspection_status,
            processing_status=run.status,
            synthetic_input_verified=pre["synthetic_input"] is True,
            mock_preprocessing=pre["mock_implementation"] is True,
            mock_inference=False if inf is None else inf["mock_inference"] is True,
            production_approved=False,
            lifecycle_idempotent_existing=idempotent,
            execution_started_now=started_now,
            started_at=run.started_at,
            completed_at=run.completed_at,
            preprocessing=preprocessing,
            inference=inference,
        )
