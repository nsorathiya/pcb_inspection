from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    InspectionInferenceResult as InferenceResultRecord,
    InspectionInferenceResultFinding,
    InspectionPreprocessingResult as PreprocessingResultRecord,
    InspectionPreprocessingResultFinding,
    InspectionProcessingRun,
)
from app.db.processing_types import (
    InferenceFindingCategory,
    PersistedInferenceOutcome,
    PersistedPreprocessingOutcome,
    PreprocessingFindingCategory,
    ProcessingFindingSeverity,
    ProcessingRunStatus,
)
from app.services.inspection_inference.models import (
    InferenceExecutionOutcome,
    InspectionInferenceResult,
    inference_result_to_dict,
    validate_inference_result_document,
)
from app.services.inspection_preprocessing.models import (
    InspectionPreprocessingResult,
    PreprocessingOutcome,
    preprocessing_result_to_dict,
    validate_preprocessing_result_document,
)
from app.services.inspection_processing.canonical import (
    canonical_inference_result_bytes,
    canonical_inference_result_sha256,
    canonical_preprocessing_result_bytes,
    canonical_preprocessing_result_sha256,
)
from app.services.inspection_processing.models import canonical_uuid, lowercase_sha256

ROOT = Path(__file__).resolve().parents[4]
PREPROCESSING_CONTRACT = "pcb-aoi-inspection-preprocessing/1.0"
INFERENCE_CONTRACT = "pcb-aoi-inspection-inference/1.0"


class ProcessingPersistenceError(Exception):
    pass


class ProcessingPersistenceConflictError(ProcessingPersistenceError):
    pass


class ProcessingPersistenceIntegrityError(ProcessingPersistenceError):
    pass


@dataclass(frozen=True)
class PersistedProcessingRun:
    processing_run_id: str
    inspection_id: str
    validation_id: str
    processing_key: str
    status: ProcessingRunStatus
    preprocessing_policy_id: str
    preprocessing_policy_version: str
    preprocessing_implementation_id: str
    preprocessing_implementation_version: str
    inference_policy_id: str
    inference_policy_version: str
    engine_id: str
    engine_version: str
    engine_type: str
    started_at: datetime
    completed_at: datetime | None
    final_decision: str | None
    error_code: str | None
    error_message: str | None
    created_at: datetime


@dataclass(frozen=True)
class PersistedProcessingResult:
    result_id: str
    processing_run_id: str
    outcome: str
    result: Mapping[str, Any]
    result_sha256: str
    started_at: datetime
    completed_at: datetime
    created_at: datetime


@dataclass(frozen=True)
class PersistedProcessingFinding:
    finding_id: str
    parent_result_id: str
    ordinal: int
    code: str
    severity: str
    category: str
    message: str
    branch: str | None
    field: str | None
    blocking: bool
    details: Mapping[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class _PreparedPreprocessing:
    result: InspectionPreprocessingResult
    document: Mapping[str, Any]
    canonical_bytes: bytes
    result_sha256: str
    created_at: datetime


@dataclass(frozen=True)
class _PreparedInference:
    result: InspectionInferenceResult
    document: Mapping[str, Any]
    canonical_bytes: bytes
    result_sha256: str
    created_at: datetime


@dataclass(frozen=True)
class _PersistedSet:
    preprocessing_sha256: str
    inference_sha256: str | None
    idempotent_existing: bool


def aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include timezone information")
    return value.astimezone(timezone.utc)


def retrieved_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def json_object_text(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("JSON value must be an object")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _path_shaped(value: Any, *, key: str | None = None) -> bool:
    if key is not None and any(token in key.lower() for token in ("path", "filename", "traceback", "sql")):
        return True
    if isinstance(value, Mapping):
        return any(_path_shaped(child, key=str(child_key)) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_path_shaped(child) for child in value)
    if isinstance(value, str):
        return (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or "\\" in value
            or ".." in PurePosixPath(value).parts
        )
    return False


class InspectionProcessingRepository:
    """Read-only public repository plus transaction-owned append helpers."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        clock: Callable[[], datetime] | None = None,
        finding_id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._sessions = session_factory
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._finding_id = finding_id_generator or (lambda: str(uuid4()))
        preprocessing_catalogue = self._json("contracts/inspection_preprocessing_findings.json")
        inference_catalogue = self._json("contracts/inspection_inference_findings.json")
        taxonomy = self._json("contracts/defect_taxonomy.json")
        preprocessing_schema = self._json("contracts/inspection_preprocessing_result.schema.json")
        inference_schema = self._json("contracts/inspection_inference_result.schema.json")
        self._preprocessing_catalogue = preprocessing_catalogue
        self._inference_catalogue = inference_catalogue
        self._taxonomy = taxonomy
        self._preprocessing_validator = Draft202012Validator(
            preprocessing_schema,
            registry=Registry().with_resource(
                preprocessing_catalogue["$id"], Resource.from_contents(preprocessing_catalogue)
            ),
            format_checker=FormatChecker(),
        )
        self._inference_validator = Draft202012Validator(
            inference_schema,
            registry=Registry().with_resources(
                (
                    (inference_catalogue["$id"], Resource.from_contents(inference_catalogue)),
                    (taxonomy["$id"], Resource.from_contents(taxonomy)),
                )
            ),
            format_checker=FormatChecker(),
        )

    @staticmethod
    def _json(relative_path: str) -> Mapping[str, Any]:
        return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))

    async def get_run_by_id(self, processing_run_id: str) -> PersistedProcessingRun | None:
        canonical_uuid(processing_run_id, "processing_run_id")
        async with self._sessions() as session:
            record = await session.get(InspectionProcessingRun, processing_run_id)
        return None if record is None else self._run(record)

    async def get_run_by_inspection_and_key(
        self, inspection_id: str, processing_key: str
    ) -> PersistedProcessingRun | None:
        canonical_uuid(inspection_id, "inspection_id")
        lowercase_sha256(processing_key, "processing_key")
        statement = select(InspectionProcessingRun).where(
            InspectionProcessingRun.inspection_id == inspection_id,
            InspectionProcessingRun.processing_key == processing_key,
        )
        async with self._sessions() as session:
            record = await session.scalar(statement)
        return None if record is None else self._run(record)

    async def get_latest_run_for_inspection(self, inspection_id: str) -> PersistedProcessingRun | None:
        canonical_uuid(inspection_id, "inspection_id")
        statement = (
            select(InspectionProcessingRun)
            .where(InspectionProcessingRun.inspection_id == inspection_id)
            .order_by(
                InspectionProcessingRun.started_at.desc(),
                InspectionProcessingRun.created_at.desc(),
                InspectionProcessingRun.id.asc(),
            )
            .limit(1)
        )
        async with self._sessions() as session:
            record = await session.scalar(statement)
        return None if record is None else self._run(record)

    async def get_preprocessing_result(self, processing_run_id: str) -> PersistedProcessingResult | None:
        canonical_uuid(processing_run_id, "processing_run_id")
        statement = select(PreprocessingResultRecord).where(
            PreprocessingResultRecord.processing_run_id == processing_run_id
        )
        async with self._sessions() as session:
            record = await session.scalar(statement)
        return None if record is None else self._result(record, record.outcome.value)

    async def get_inference_result(self, processing_run_id: str) -> PersistedProcessingResult | None:
        canonical_uuid(processing_run_id, "processing_run_id")
        statement = select(InferenceResultRecord).where(
            InferenceResultRecord.processing_run_id == processing_run_id
        )
        async with self._sessions() as session:
            record = await session.scalar(statement)
        return None if record is None else self._result(record, record.execution_outcome.value)

    async def list_preprocessing_findings(self, preprocessing_id: str) -> tuple[PersistedProcessingFinding, ...]:
        canonical_uuid(preprocessing_id, "preprocessing_id")
        statement = select(InspectionPreprocessingResultFinding).where(
            InspectionPreprocessingResultFinding.preprocessing_id == preprocessing_id
        ).order_by(InspectionPreprocessingResultFinding.ordinal.asc())
        async with self._sessions() as session:
            records = list(await session.scalars(statement))
        return tuple(self._finding(item, item.preprocessing_id) for item in records)

    async def list_inference_findings(self, inference_id: str) -> tuple[PersistedProcessingFinding, ...]:
        canonical_uuid(inference_id, "inference_id")
        statement = select(InspectionInferenceResultFinding).where(
            InspectionInferenceResultFinding.inference_id == inference_id
        ).order_by(InspectionInferenceResultFinding.ordinal.asc())
        async with self._sessions() as session:
            records = list(await session.scalars(statement))
        return tuple(self._finding(item, item.inference_id) for item in records)

    def _prepare_preprocessing(
        self, run: InspectionProcessingRun, result: InspectionPreprocessingResult
    ) -> _PreparedPreprocessing:
        if not isinstance(result, InspectionPreprocessingResult):
            raise TypeError("processing persistence requires an InspectionPreprocessingResult")
        canonical_uuid(result.preprocessing_id, "preprocessing_id")
        canonical_uuid(result.inspection_id, "result inspection_id")
        canonical_uuid(result.validation_id, "result validation_id")
        if (result.inspection_id, result.validation_id) != (run.inspection_id, run.validation_id):
            raise ValueError("preprocessing result inspection or validation identity does not match run")
        if result.contract_version != PREPROCESSING_CONTRACT:
            raise ValueError("preprocessing result contract version is unsupported")
        expected = (
            run.preprocessing_policy_id,
            run.preprocessing_policy_version,
            run.preprocessing_implementation_id,
            run.preprocessing_implementation_version,
        )
        actual = (result.policy_id, result.policy_version, result.implementation_id, result.implementation_version)
        if actual != expected:
            raise ValueError("preprocessing policy or implementation identity does not match run")
        if not isinstance(result.outcome, PreprocessingOutcome):
            raise ValueError("preprocessing outcome is invalid")
        if aware_utc(result.completed_at, "preprocessing completed_at") < aware_utc(result.started_at, "preprocessing started_at"):
            raise ValueError("preprocessing completed_at must not precede started_at")
        document = preprocessing_result_to_dict(result)
        if _path_shaped(document.get("findings", [])):
            raise ValueError("preprocessing finding details must not contain paths")
        try:
            self._preprocessing_validator.validate(document)
            validate_preprocessing_result_document(document, self._preprocessing_catalogue)
        except Exception as exc:
            raise ValueError("preprocessing result is not contract-valid") from exc
        canonical = canonical_preprocessing_result_bytes(result)
        return _PreparedPreprocessing(
            result, document, canonical, canonical_preprocessing_result_sha256(result),
            aware_utc(self._clock(), "created_at"),
        )

    def _prepare_inference(
        self,
        run: InspectionProcessingRun,
        preprocessing: InspectionPreprocessingResult,
        result: InspectionInferenceResult,
    ) -> _PreparedInference:
        if not isinstance(result, InspectionInferenceResult):
            raise TypeError("processing persistence requires an InspectionInferenceResult")
        for value, field in (
            (result.inference_id, "inference_id"),
            (result.inspection_id, "result inspection_id"),
            (result.validation_id, "result validation_id"),
            (result.preprocessing_id, "result preprocessing_id"),
        ):
            canonical_uuid(value, field)
        if (result.inspection_id, result.validation_id, result.preprocessing_id) != (
            run.inspection_id, run.validation_id, preprocessing.preprocessing_id
        ):
            raise ValueError("inference result identities do not match processing run")
        if result.contract_version != INFERENCE_CONTRACT:
            raise ValueError("inference result contract version is unsupported")
        expected = (run.inference_policy_id, run.inference_policy_version, run.engine_id, run.engine_version, run.engine_type)
        actual = (result.policy_id, result.policy_version, result.engine_id, result.engine_version, result.engine_type)
        if actual != expected:
            raise ValueError("inference policy or engine identity does not match run")
        if not isinstance(result.execution_outcome, InferenceExecutionOutcome):
            raise ValueError("inference execution outcome is invalid")
        if result.confidence is not None:
            raise ValueError("mock inference confidence must be null")
        if aware_utc(result.completed_at, "inference completed_at") < aware_utc(result.started_at, "inference started_at"):
            raise ValueError("inference completed_at must not precede started_at")
        document = inference_result_to_dict(result)
        if _path_shaped(document.get("findings", [])):
            raise ValueError("inference finding details must not contain paths")
        try:
            self._inference_validator.validate(document)
            validate_inference_result_document(document, self._inference_catalogue, self._taxonomy)
        except Exception as exc:
            raise ValueError("inference result is not contract-valid") from exc
        canonical = canonical_inference_result_bytes(result)
        return _PreparedInference(
            result, document, canonical, canonical_inference_result_sha256(result),
            aware_utc(self._clock(), "created_at"),
        )

    async def _persist_results(
        self,
        session: AsyncSession,
        run: InspectionProcessingRun,
        preprocessing: _PreparedPreprocessing,
        inference: _PreparedInference | None,
    ) -> _PersistedSet:
        existing_pre = await session.scalar(select(PreprocessingResultRecord).where(
            PreprocessingResultRecord.processing_run_id == run.id
        ))
        existing_inf = await session.scalar(select(InferenceResultRecord).where(
            InferenceResultRecord.processing_run_id == run.id
        ))
        if existing_pre is not None or existing_inf is not None:
            if existing_pre is None or existing_pre.result_sha256 != preprocessing.result_sha256:
                raise ProcessingPersistenceConflictError("processing run already has a different preprocessing result")
            if inference is None:
                if existing_inf is not None:
                    raise ProcessingPersistenceConflictError("processing run already has an inference result")
            elif existing_inf is None or existing_inf.result_sha256 != inference.result_sha256:
                raise ProcessingPersistenceConflictError("processing run already has a different inference result")
            return _PersistedSet(existing_pre.result_sha256, None if existing_inf is None else existing_inf.result_sha256, True)

        pre = preprocessing.result
        pre_doc = preprocessing.document
        pre_record = PreprocessingResultRecord(
            id=pre.preprocessing_id, processing_run_id=run.id,
            contract_version=pre.contract_version, policy_id=pre.policy_id,
            policy_version=pre.policy_version, implementation_id=pre.implementation_id,
            implementation_version=pre.implementation_version,
            outcome=PersistedPreprocessingOutcome(pre.outcome.value),
            started_at=aware_utc(pre.started_at, "started_at"),
            completed_at=aware_utc(pre.completed_at, "completed_at"),
            rgb_output_json=None if pre_doc["rgb_output"] is None else json_object_text(pre_doc["rgb_output"]),
            height_output_json=None if pre_doc["height_output"] is None else json_object_text(pre_doc["height_output"]),
            registration_json=None if pre_doc["registration"] is None else json_object_text(pre_doc["registration"]),
            summary_json=json_object_text(pre_doc["summary"]),
            result_json=preprocessing.canonical_bytes.decode("utf-8"),
            result_sha256=preprocessing.result_sha256, created_at=preprocessing.created_at,
        )
        session.add(pre_record)
        await session.flush()
        session.add_all(self._preprocessing_findings(pre, preprocessing.created_at))
        await session.flush()

        if inference is not None:
            inf = inference.result
            inf_doc = inference.document
            inf_record = InferenceResultRecord(
                id=inf.inference_id, processing_run_id=run.id, preprocessing_id=pre.preprocessing_id,
                contract_version=inf.contract_version, policy_id=inf.policy_id,
                policy_version=inf.policy_version, engine_id=inf.engine_id,
                engine_version=inf.engine_version, engine_type=inf.engine_type,
                execution_outcome=PersistedInferenceOutcome(inf.execution_outcome.value),
                decision=None if inf.decision is None else inf.decision.value,
                defect_type=inf.defect_type, confidence=None,
                decision_basis=inf.decision_basis, decision_digest=inf.decision_digest,
                started_at=aware_utc(inf.started_at, "started_at"),
                completed_at=aware_utc(inf.completed_at, "completed_at"),
                summary_json=json_object_text(inf_doc["summary"]),
                result_json=inference.canonical_bytes.decode("utf-8"),
                result_sha256=inference.result_sha256, created_at=inference.created_at,
            )
            session.add(inf_record)
            await session.flush()
            session.add_all(self._inference_findings(inf, inference.created_at))
            await session.flush()
        return _PersistedSet(preprocessing.result_sha256, None if inference is None else inference.result_sha256, False)

    def _preprocessing_findings(self, result: InspectionPreprocessingResult, created_at: datetime):
        return [InspectionPreprocessingResultFinding(
            id=canonical_uuid(self._finding_id(), "finding_id"), preprocessing_id=result.preprocessing_id,
            ordinal=ordinal, code=finding.code,
            severity=ProcessingFindingSeverity(finding.severity.value),
            category=PreprocessingFindingCategory(finding.category.value),
            message=finding.message, branch=finding.branch, field=finding.field,
            blocking=finding.blocking, details_json=json_object_text(finding.details), created_at=created_at,
        ) for ordinal, finding in enumerate(result.findings)]

    def _inference_findings(self, result: InspectionInferenceResult, created_at: datetime):
        return [InspectionInferenceResultFinding(
            id=canonical_uuid(self._finding_id(), "finding_id"), inference_id=result.inference_id,
            ordinal=ordinal, code=finding.code,
            severity=ProcessingFindingSeverity(finding.severity.value),
            category=InferenceFindingCategory(finding.category.value),
            message=finding.message, branch=finding.branch, field=finding.field,
            blocking=finding.blocking, details_json=json_object_text(finding.details), created_at=created_at,
        ) for ordinal, finding in enumerate(result.findings)]

    @staticmethod
    def _run(record: InspectionProcessingRun) -> PersistedProcessingRun:
        return PersistedProcessingRun(
            record.id, record.inspection_id, record.validation_id, record.processing_key,
            record.status, record.preprocessing_policy_id, record.preprocessing_policy_version,
            record.preprocessing_implementation_id, record.preprocessing_implementation_version,
            record.inference_policy_id, record.inference_policy_version, record.engine_id,
            record.engine_version, record.engine_type, retrieved_utc(record.started_at),
            retrieved_utc(record.completed_at),
            None if record.final_decision is None else record.final_decision.value,
            record.error_code, record.error_message, retrieved_utc(record.created_at),
        )

    @staticmethod
    def _result(record, outcome: str) -> PersistedProcessingResult:
        return PersistedProcessingResult(
            record.id, record.processing_run_id, outcome, json.loads(record.result_json),
            record.result_sha256, retrieved_utc(record.started_at),
            retrieved_utc(record.completed_at), retrieved_utc(record.created_at),
        )

    @staticmethod
    def _finding(record, parent_id: str) -> PersistedProcessingFinding:
        return PersistedProcessingFinding(
            record.id, parent_id, record.ordinal, record.code, record.severity.value,
            record.category.value, record.message, record.branch, record.field,
            record.blocking, json.loads(record.details_json), retrieved_utc(record.created_at),
        )
