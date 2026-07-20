import asyncio
import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from sqlalchemy import delete, func, select, update

from app.core.runtime_paths import RuntimePaths
from app.db.database import Database
from app.db.models import (
    ArtifactType,
    AuditEvent,
    Inspection,
    InspectionArtifact,
    InspectionInferenceResult as InferenceRecord,
    InspectionInferenceResultFinding,
    InspectionPreprocessingResult as PreprocessingRecord,
    InspectionPreprocessingResultFinding,
    InspectionProcessingRun,
    InspectionStatus,
    InspectionValidation,
    ModelVersion,
    SCHEMA_VERSION,
)
from app.db.repositories import InspectionCreate, Repositories
from app.db.validation_types import ValidationOutcome
from app.services.artifact_storage import (
    ArtifactInput,
    ArtifactPathPolicy,
    ArtifactRegistrationService,
    ArtifactSizeLimits,
    ArtifactStorageService,
)
from app.services.dataset_validation.file_inspection import inspect_height, inspect_rgb
from app.services.inspection_inference import SyntheticMockInferenceService
from app.services.inspection_inference.exceptions import InferenceKnownFailure
from app.services.inspection_preprocessing import SyntheticInspectionPreprocessingService
from app.services.inspection_preprocessing.policy_loader import SyntheticPreprocessingPolicyLoader
from app.services.inspection_processing import (
    InspectionProcessingOrchestrator,
    ProcessingExecutionInProgressError,
    ProcessingExecutionInspectionNotFoundError,
    ProcessingExecutionInspectionNotReadyError,
    ProcessingExecutionArtifactPairError,
    ProcessingExecutionValidationMissingError,
    ProcessingExecutionValidationNotPassedError,
    ProcessingExecutionPolicyError,
    ProcessingExecutionConflictError,
    ProcessingExecutionOrchestrationError,
    ProcessingExecutionRecoveryRequiredError,
    ProcessingLifecycleService,
    SyntheticFixtureProvenanceVerifier,
    SyntheticProvenanceMismatchError,
    SyntheticProvenanceUnavailableError,
)
from app.services.inspection_processing.execution_models import ProcessingExecutionResult
from app.services.inspection_processing.input_builder import (
    InspectionProcessingInputReader,
    ProcessingArtifactPreflight,
)
from app.services.inspection_validation import (
    ArtifactTechnicalSummary,
    InspectionValidationResult,
    ReadabilityStatus,
    ValidationCommitService,
    ValidationSummary,
)
from app.services.inspection_validation.artifact_reader import ManagedArtifactPathResolver
from app.testing.synthetic_aoi import generate_fixtures
from app.testing.synthetic_aoi.manifest import file_inventory, json_bytes, output_tree_sha256, sha256_bytes

FIXED_VALIDATION_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
FIXED_PREPROCESSING_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
FIXED_INFERENCE_ID = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
PRE_POLICY = ("synthetic-paired-rgb-height", "1.0")
INF_POLICY = ("synthetic-deterministic-mock-inference", "1.0")


class CountingPreprocessor:
    def __init__(self, delegate=None, entered=None, release=None):
        self.delegate = delegate or SyntheticInspectionPreprocessingService(
            preprocessing_id_generator=lambda: FIXED_PREPROCESSING_ID
        )
        self.count = 0
        self.entered = entered
        self.release = release

    async def preprocess_inspection(self, inputs, policy):
        self.count += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return await self.delegate.preprocess_inspection(inputs, policy)


class CountingInference:
    def __init__(self, delegate=None):
        self.delegate = delegate or SyntheticMockInferenceService(
            inference_id_generator=lambda: FIXED_INFERENCE_ID
        )
        self.count = 0

    async def run_inference(self, inputs, policy):
        self.count += 1
        return await self.delegate.run_inference(inputs, policy)


class RaisingPreprocessor:
    def __init__(self):
        self.count = 0

    async def preprocess_inspection(self, inputs, policy):
        self.count += 1
        raise RuntimeError("private path C:\\secret and SQL SELECT must stay hidden")


class RaisingInference:
    def __init__(self):
        self.count = 0

    async def run_inference(self, inputs, policy):
        self.count += 1
        raise RuntimeError("private traceback must stay hidden")


class KnownFailingInferenceValidator:
    def validate(self, inputs, policy):
        raise InferenceKnownFailure("RGB_BUFFER_REQUIRED", branch="RGB")


class ErroringInferenceEngine:
    engine_id = "synthetic-deterministic-mock-engine"
    engine_version = "1.0.0"

    async def infer(self, inputs, policy):
        raise RuntimeError("private engine failure")


class FailingLifecycle:
    def __init__(self, delegate, *, fail_begin=False, fail_complete=False):
        self.delegate = delegate
        self.fail_begin = fail_begin
        self.fail_complete = fail_complete

    async def begin_processing(self, *args, **kwargs):
        if self.fail_begin:
            raise RuntimeError("database internals")
        return await self.delegate.begin_processing(*args, **kwargs)

    async def complete_processing(self, *args, **kwargs):
        if self.fail_complete:
            raise RuntimeError("database internals")
        return await self.delegate.complete_processing(*args, **kwargs)


class CountingProvenance:
    def __init__(self, delegate):
        self.delegate = delegate
        self.count = 0

    def verify(self, snapshot):
        self.count += 1
        return self.delegate.verify(snapshot)


class CountingPreflight:
    def __init__(self, delegate):
        self.delegate = delegate
        self.count = 0

    async def resolve_and_verify(self, snapshot):
        self.count += 1
        return await self.delegate.resolve_and_verify(snapshot)


@dataclass
class Environment:
    database: Database
    repositories: Repositories
    paths: RuntimePaths
    fixture_root: Path
    scenario: dict
    inspection_id: str
    validation_id: str
    lifecycle: ProcessingLifecycleService
    preprocessor: CountingPreprocessor
    inference: CountingInference
    provenance: CountingProvenance
    preflight: CountingPreflight
    orchestrator: InspectionProcessingOrchestrator

    async def close(self):
        await self.database.dispose()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _technical(kind: ArtifactType, path: Path, media_type: str):
    raster = inspect_rgb(path) if kind is ArtifactType.RGB_RAW else inspect_height(path)
    return ArtifactTechnicalSummary(
        artifact_type=kind,
        sha256=_digest(path),
        byte_size=path.stat().st_size,
        declared_media_type=media_type,
        detected_format=raster.detected_format,
        width=raster.width,
        height=raster.height,
        channels=raster.channels,
        bit_depth=raster.bit_depth,
        storage_data_type=raster.storage_data_type,
        readability_status=ReadabilityStatus.READABLE,
    )


async def _environment(
    root: Path,
    scenario_id: str = "valid_rgb_png_height_tiff",
    *,
    inspection_status: InspectionStatus = InspectionStatus.RECEIVED,
    preprocessor=None,
    inference=None,
    inspection_id=None,
    register_evidence=False,
) -> Environment:
    paths = RuntimePaths.from_root(root / "runtime")
    paths.create_directories()
    database = Database(paths.database_file, busy_timeout_ms=5000)
    await database.initialize()
    repositories = Repositories.from_session_factory(database.session_factory)
    fixture_root = root / "fixtures"
    generate_fixtures(fixture_root, scenario_ids=(scenario_id,))
    scenario_root = fixture_root / "scenarios" / scenario_id
    scenario = json.loads((scenario_root / "scenario.json").read_text(encoding="utf-8"))
    inspection_id = inspection_id or str(uuid4())
    await repositories.inspections.create(
        InspectionCreate(
            id=inspection_id,
            status=inspection_status,
            board_id="SYNTHETIC",
            recipe_id="development-native-rgb-height",
            recipe_version="1.0",
        )
    )
    storage = ArtifactStorageService(
        ArtifactPathPolicy(paths),
        ArtifactSizeLimits(
            rgb_bytes=1024 * 1024,
            height_bytes=1024 * 1024,
            mask_bytes=1024 * 1024,
            calibration_bytes=1024 * 1024,
            generated_artifact_bytes=1024 * 1024,
        ),
    )
    registration = ArtifactRegistrationService(storage, repositories.artifacts)
    source_paths = {}
    for role, kind in (("rgb", ArtifactType.RGB_RAW), ("height", ArtifactType.HEIGHT_RAW)):
        record = scenario["artifacts"][role]
        source = scenario_root / record["generated_file"]
        source_paths[kind] = source
        await registration.store_and_register(
            ArtifactInput(
                inspection_id=inspection_id,
                artifact_type=kind,
                source=source.read_bytes(),
                original_filename=record["generated_file"],
                media_type=record["media_type"],
                expected_sha256=record["actual_sha256"],
                expected_byte_size=record["actual_byte_size"],
            )
        )
    if register_evidence:
        for role, kind in (("mask", ArtifactType.VALIDITY_MASK), ("calibration", ArtifactType.CALIBRATION)):
            if role not in scenario["artifacts"]:
                continue
            record = scenario["artifacts"][role]
            source = scenario_root / record["generated_file"]
            await registration.store_and_register(
                ArtifactInput(
                    inspection_id=inspection_id,
                    artifact_type=kind,
                    source=source.read_bytes(),
                    original_filename=record["generated_file"],
                    media_type=record["media_type"],
                    expected_sha256=record["actual_sha256"],
                    expected_byte_size=record["actual_byte_size"],
                )
            )

    if inspection_status is InspectionStatus.RECEIVED:
        now = datetime.now(timezone.utc)
        validation = InspectionValidationResult(
            contract_version="pcb-aoi-inspection-validation/1.0",
            validation_id=FIXED_VALIDATION_ID,
            inspection_id=inspection_id,
            validation_policy_id="development-native-rgb-height",
            validation_policy_version="1.0",
            outcome=ValidationOutcome.VALIDATION_PASSED,
            started_at=now,
            completed_at=now,
            validator_version="1.0.0",
            rgb_artifact=_technical(
                ArtifactType.RGB_RAW,
                source_paths[ArtifactType.RGB_RAW],
                scenario["artifacts"]["rgb"]["media_type"],
            ),
            height_artifact=_technical(
                ArtifactType.HEIGHT_RAW,
                source_paths[ArtifactType.HEIGHT_RAW],
                scenario["artifacts"]["height"]["media_type"],
            ),
            findings=(),
            summary=ValidationSummary(0, 0, 0, 0, 0, True, True),
        )
        await ValidationCommitService(
            database.session_factory,
            validation_repository=repositories.validations,
        ).commit_validation(validation, "a" * 64)

    lifecycle = ProcessingLifecycleService(
        database.session_factory, repository=repositories.processing
    )
    selected_pre = preprocessor or CountingPreprocessor()
    selected_inf = inference or CountingInference()
    provenance = CountingProvenance(SyntheticFixtureProvenanceVerifier(fixture_root))
    preflight = CountingPreflight(
        ProcessingArtifactPreflight(ManagedArtifactPathResolver(paths))
    )
    orchestrator = InspectionProcessingOrchestrator(
        repositories,
        paths,
        fixture_root,
        lifecycle,
        preprocessing_service=selected_pre,
        inference_service=selected_inf,
        provenance_verifier=provenance,
        artifact_preflight=preflight,
    )
    return Environment(
        database, repositories, paths, fixture_root, scenario, inspection_id,
        FIXED_VALIDATION_ID, lifecycle, selected_pre, selected_inf, provenance,
        preflight, orchestrator,
    )


async def _execute(environment):
    return await environment.orchestrator.execute_processing(
        environment.inspection_id,
        *PRE_POLICY,
        *INF_POLICY,
    )


@pytest.mark.parametrize(
    "scenario_id,preprocessing_outcome,inference_count",
    [
        ("valid_rgb_png_height_tiff", "PREPROCESSING_SUCCEEDED", 1),
        ("valid_rgb_tiff_height_png16", "PREPROCESSING_SUCCEEDED", 1),
        ("valid_rgb_png_height_npy_float32", "PREPROCESSING_SUCCEEDED", 1),
        ("valid_different_dimensions", "PREPROCESSING_FAILED", 0),
    ],
)
def test_generated_fixture_execution_uses_existing_services_and_lifecycle(
    tmp_path, scenario_id, preprocessing_outcome, inference_count
):
    async def scenario_test():
        environment = await _environment(tmp_path, scenario_id)
        try:
            result = await _execute(environment)
            assert result.preprocessing_outcome == preprocessing_outcome
            assert result.synthetic_input_verified is True
            assert result.mock_preprocessing is True
            assert result.production_approved is False
            assert not hasattr(result, "confidence")
            assert environment.preprocessor.count == 1
            assert environment.inference.count == inference_count
            if inference_count:
                assert result.mock_inference is True
                assert result.mock_decision in {"PASS", "FAIL", "UNCERTAIN"}
                assert result.inspection_status.value == result.mock_decision
                assert result.processing_status.value == "COMPLETED"
            else:
                assert result.mock_inference is False
                assert result.mock_decision is None
                assert result.inspection_status is InspectionStatus.ERROR
                assert result.processing_status.value == "ERROR"
        finally:
            await environment.close()

    asyncio.run(scenario_test())


@pytest.mark.parametrize(
    "inspection_id,expected",
    [
        ("00000000-0000-4000-8000-000000000003", "PASS"),
        ("00000000-0000-4000-8000-000000000001", "FAIL"),
        ("00000000-0000-4000-8000-000000000006", "UNCERTAIN"),
    ],
)
def test_controlled_identities_cover_all_mock_final_decisions(tmp_path, inspection_id, expected):
    async def scenario_test():
        environment = await _environment(tmp_path, inspection_id=inspection_id)
        try:
            result = await _execute(environment)
            assert result.mock_decision == expected
            assert result.inspection_status.value == expected
            assert result.mock_inference is True
            assert result.production_approved is False
            assert not hasattr(result, "confidence")
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_exact_retry_reconstructs_persisted_result_without_execution_or_file_reads(tmp_path):
    async def scenario_test():
        environment = await _environment(tmp_path)
        try:
            first = await _execute(environment)
            counts = (
                environment.preprocessor.count,
                environment.inference.count,
                environment.provenance.count,
                environment.preflight.count,
            )
            async with environment.database.session() as session:
                finding_counts_before = (
                    await session.scalar(select(func.count()).select_from(InspectionPreprocessingResultFinding)),
                    await session.scalar(select(func.count()).select_from(InspectionInferenceResultFinding)),
                )
            second = await _execute(environment)
            assert second.processing_run_id == first.processing_run_id
            assert second.preprocessing_id == first.preprocessing_id
            assert second.inference_id == first.inference_id
            assert second.mock_decision == first.mock_decision
            assert second.lifecycle_idempotent_existing is True
            assert second.execution_started_now is False
            assert counts == (1, 1, 1, 1)
            assert (
                environment.preprocessor.count,
                environment.inference.count,
                environment.provenance.count,
                environment.preflight.count,
            ) == counts
            async with environment.database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 1
                assert await session.scalar(select(func.count()).select_from(PreprocessingRecord)) == 1
                assert await session.scalar(select(func.count()).select_from(InferenceRecord)) == 1
                assert (
                    await session.scalar(select(func.count()).select_from(InspectionPreprocessingResultFinding)),
                    await session.scalar(select(func.count()).select_from(InspectionInferenceResultFinding)),
                ) == finding_counts_before
                assert await session.scalar(
                    select(func.count()).select_from(AuditEvent).where(
                        AuditEvent.entity_id == environment.inspection_id,
                        AuditEvent.action.like("INSPECTION_PROCESSING_%")
                        | AuditEvent.action.like("INSPECTION_MOCK_RESULT_%"),
                    )
                ) == 2
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_concurrent_identical_execution_has_one_database_winner(tmp_path):
    async def scenario_test():
        entered, release = asyncio.Event(), asyncio.Event()
        counter = CountingPreprocessor(entered=entered, release=release)
        environment = await _environment(tmp_path, preprocessor=counter)
        try:
            winner = asyncio.create_task(_execute(environment))
            await entered.wait()
            with pytest.raises(ProcessingExecutionInProgressError):
                await _execute(environment)
            release.set()
            result = await winner
            assert result.processing_status.value == "COMPLETED"
            assert environment.preprocessor.count == 1
            assert environment.inference.count == 1
            async with environment.database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 1
        finally:
            release.set()
            await environment.close()

    asyncio.run(scenario_test())


def test_concurrent_different_engine_version_has_one_lifecycle_winner(tmp_path):
    async def scenario_test():
        entered, release = asyncio.Event(), asyncio.Event()
        counter = CountingPreprocessor(entered=entered, release=release)
        environment = await _environment(tmp_path, preprocessor=counter)
        other = InspectionProcessingOrchestrator(
            environment.repositories,
            environment.paths,
            environment.fixture_root,
            environment.lifecycle,
            preprocessing_service=CountingPreprocessor(),
            inference_service=CountingInference(),
            engine_version="1.0.1",
        )
        try:
            winner = asyncio.create_task(_execute(environment))
            await entered.wait()
            with pytest.raises(ProcessingExecutionConflictError):
                await other.execute_processing(
                    environment.inspection_id, *PRE_POLICY, *INF_POLICY
                )
            release.set()
            await winner
            async with environment.database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 1
            assert environment.preprocessor.count == 1
            assert environment.inference.count == 1
        finally:
            release.set()
            await environment.close()

    asyncio.run(scenario_test())


@pytest.mark.parametrize("failure_stage", ["preprocessing", "inference"])
def test_unexpected_service_failure_is_safe_and_completes_error(tmp_path, failure_stage):
    async def scenario_test():
        preprocessor = RaisingPreprocessor() if failure_stage == "preprocessing" else None
        inference = RaisingInference() if failure_stage == "inference" else None
        environment = await _environment(
            tmp_path,
            preprocessor=preprocessor,
            inference=inference,
        )
        try:
            result = await _execute(environment)
            assert result.inspection_status is InspectionStatus.ERROR
            assert result.processing_status.value == "ERROR"
            if failure_stage == "preprocessing":
                assert result.preprocessing_outcome == "PREPROCESSING_ERROR"
                assert result.inference_id is None
                assert preprocessor.count == 1
            else:
                assert result.preprocessing_outcome == "PREPROCESSING_SUCCEEDED"
                assert result.inference_execution_outcome == "INFERENCE_ERROR"
                assert inference.count == 1
            assert "path" not in repr(result).lower()
            assert "traceback" not in repr(result).lower()
            assert "sql" not in repr(result).lower()
        finally:
            await environment.close()

    asyncio.run(scenario_test())


@pytest.mark.parametrize("outcome", ["INFERENCE_FAILED", "INFERENCE_ERROR"])
def test_typed_inference_failure_completes_existing_lifecycle_as_error(tmp_path, outcome):
    async def scenario_test():
        service = (
            SyntheticMockInferenceService(
                input_validator=KnownFailingInferenceValidator(),
                inference_id_generator=lambda: FIXED_INFERENCE_ID,
            )
            if outcome == "INFERENCE_FAILED"
            else SyntheticMockInferenceService(
                engine=ErroringInferenceEngine(),
                inference_id_generator=lambda: FIXED_INFERENCE_ID,
            )
        )
        counter = CountingInference(service)
        environment = await _environment(tmp_path, inference=counter)
        try:
            result = await _execute(environment)
            assert result.preprocessing_outcome == "PREPROCESSING_SUCCEEDED"
            assert result.inference_execution_outcome == outcome
            assert result.mock_decision is None
            assert result.inspection_status is InspectionStatus.ERROR
            assert result.processing_status.value == "ERROR"
            assert counter.count == 1
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_lifecycle_begin_and_completion_failures_are_safe(tmp_path):
    async def scenario_test():
        before = await _environment(tmp_path / "begin")
        begin_orchestrator = InspectionProcessingOrchestrator(
            before.repositories,
            before.paths,
            before.fixture_root,
            FailingLifecycle(before.lifecycle, fail_begin=True),
        )
        try:
            with pytest.raises(ProcessingExecutionOrchestrationError) as error:
                await begin_orchestrator.execute_processing(
                    before.inspection_id, *PRE_POLICY, *INF_POLICY
                )
            assert "database" not in str(error.value).lower()
            async with before.database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 0
        finally:
            await before.close()

        after = await _environment(tmp_path / "complete")
        complete_orchestrator = InspectionProcessingOrchestrator(
            after.repositories,
            after.paths,
            after.fixture_root,
            FailingLifecycle(after.lifecycle, fail_complete=True),
        )
        try:
            with pytest.raises(ProcessingExecutionRecoveryRequiredError) as error:
                await complete_orchestrator.execute_processing(
                    after.inspection_id, *PRE_POLICY, *INF_POLICY
                )
            assert "database" not in str(error.value).lower()
            inspection = await after.repositories.inspections.get(after.inspection_id)
            assert inspection.status is InspectionStatus.PROCESSING
            async with after.database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 1
                assert await session.scalar(select(func.count()).select_from(PreprocessingRecord)) == 0
        finally:
            await after.close()

    asyncio.run(scenario_test())


def test_post_begin_missing_artifact_finishes_as_persisted_error_and_replays(tmp_path):
    async def scenario_test():
        environment = await _environment(tmp_path)
        try:
            artifacts = await environment.repositories.artifacts.list_for_inspection(
                environment.inspection_id
            )
            rgb = next(item for item in artifacts if item.artifact_type is ArtifactType.RGB_RAW)
            target = environment.paths.root.joinpath(*Path(rgb.relative_path).parts)
            target.unlink()
            first = await _execute(environment)
            second = await _execute(environment)
            assert first.processing_status.value == "ERROR"
            assert first.inspection_status is InspectionStatus.ERROR
            assert first.preprocessing_outcome == "PREPROCESSING_ERROR"
            assert first.inference_id is None
            assert second.processing_run_id == first.processing_run_id
            assert second.preprocessing_id == first.preprocessing_id
            assert second.lifecycle_idempotent_existing is True
            assert environment.preprocessor.count == 0
            assert environment.inference.count == 0
            assert environment.preflight.count == 1
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_pre_begin_failures_leave_no_processing_lifecycle(tmp_path):
    async def scenario_test():
        environment = await _environment(tmp_path)
        try:
            with pytest.raises(ProcessingExecutionPolicyError):
                await environment.orchestrator.execute_processing(
                    environment.inspection_id, "missing", "1.0", *INF_POLICY
                )
            with pytest.raises(ProcessingExecutionInspectionNotFoundError):
                await environment.orchestrator.execute_processing(
                    str(uuid4()), *PRE_POLICY, *INF_POLICY
                )
            async with environment.database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 0
                inspection = await session.get(Inspection, environment.inspection_id)
                assert inspection.status is InspectionStatus.READY
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_unsupported_and_malformed_policy_fail_before_mutation(tmp_path):
    async def scenario_test():
        environment = await _environment(tmp_path)
        try:
            with pytest.raises(ProcessingExecutionPolicyError):
                await environment.orchestrator.execute_processing(
                    environment.inspection_id,
                    PRE_POLICY[0],
                    "9.9",
                    *INF_POLICY,
                )
            malformed = InspectionProcessingOrchestrator(
                environment.repositories,
                environment.paths,
                environment.fixture_root,
                environment.lifecycle,
                preprocessing_policy_loader=SyntheticPreprocessingPolicyLoader(
                    policy_document={"contract_version": "invalid"}
                ),
            )
            with pytest.raises(ProcessingExecutionPolicyError):
                await malformed.execute_processing(
                    environment.inspection_id, *PRE_POLICY, *INF_POLICY
                )
            async with environment.database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 0
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_prerequisite_failures_are_typed_and_do_not_begin(tmp_path):
    async def scenario_test():
        missing_validation = await _environment(
            tmp_path / "missing-validation", inspection_status=InspectionStatus.READY
        )
        try:
            with pytest.raises(ProcessingExecutionValidationMissingError):
                await _execute(missing_validation)
        finally:
            await missing_validation.close()

        not_ready = await _environment(tmp_path / "not-ready")
        try:
            async with not_ready.database.session() as session, session.begin():
                await session.execute(
                    update(Inspection)
                    .where(Inspection.id == not_ready.inspection_id)
                    .values(status=InspectionStatus.ERROR)
                )
            with pytest.raises(ProcessingExecutionInspectionNotReadyError):
                await _execute(not_ready)
        finally:
            await not_ready.close()

        not_passed = await _environment(tmp_path / "not-passed")
        try:
            async with not_passed.database.session() as session, session.begin():
                await session.execute(
                    update(InspectionValidation)
                    .where(InspectionValidation.id == not_passed.validation_id)
                    .values(outcome=ValidationOutcome.VALIDATION_FAILED)
                )
            with pytest.raises(ProcessingExecutionValidationNotPassedError):
                await _execute(not_passed)
        finally:
            await not_passed.close()

        incomplete = await _environment(tmp_path / "incomplete")
        try:
            async with incomplete.database.session() as session, session.begin():
                await session.execute(
                    delete(InspectionArtifact).where(
                        InspectionArtifact.inspection_id == incomplete.inspection_id,
                        InspectionArtifact.artifact_type == ArtifactType.HEIGHT_RAW,
                    )
                )
            with pytest.raises(ProcessingExecutionArtifactPairError):
                await _execute(incomplete)
        finally:
            await incomplete.close()

    asyncio.run(scenario_test())


def test_provenance_accepts_owned_tree_and_rejects_unknown_missing_or_mismatched(tmp_path):
    async def scenario_test():
        environment = await _environment(tmp_path)
        try:
            snapshot = await InspectionProcessingInputReader(environment.repositories).read(
                environment.inspection_id
            )
            verified = SyntheticFixtureProvenanceVerifier(environment.fixture_root).verify(snapshot)
            assert verified.generator_version == "1.0.0"
            with pytest.raises(SyntheticProvenanceUnavailableError):
                SyntheticFixtureProvenanceVerifier(tmp_path / "unknown").verify(snapshot)
            mismatch = replace(snapshot, rgb=replace(snapshot.rgb, sha256="0" * 64))
            with pytest.raises(SyntheticProvenanceMismatchError) as error:
                SyntheticFixtureProvenanceVerifier(environment.fixture_root).verify(mismatch)
            assert str(environment.fixture_root) not in str(error.value)
            marker = environment.fixture_root / "SYNTHETIC_FIXTURES_MARKER.json"
            marker.unlink()
            with pytest.raises(SyntheticProvenanceUnavailableError):
                SyntheticFixtureProvenanceVerifier(environment.fixture_root).verify(snapshot)
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_provenance_rejects_modified_manifest_and_scenario(tmp_path):
    async def scenario_test():
        first = await _environment(tmp_path / "manifest")
        try:
            snapshot = await InspectionProcessingInputReader(first.repositories).read(first.inspection_id)
            manifest = first.fixture_root / "generation_manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b" ")
            with pytest.raises(SyntheticProvenanceMismatchError):
                SyntheticFixtureProvenanceVerifier(first.fixture_root).verify(snapshot)
        finally:
            await first.close()

        second = await _environment(tmp_path / "scenario")
        try:
            snapshot = await InspectionProcessingInputReader(second.repositories).read(second.inspection_id)
            scenario_path = second.fixture_root / "scenarios" / second.scenario["scenario_id"] / "scenario.json"
            value = json.loads(scenario_path.read_text(encoding="utf-8"))
            value["production_approved"] = True
            scenario_path.write_text(json.dumps(value), encoding="utf-8")
            with pytest.raises(SyntheticProvenanceMismatchError):
                SyntheticFixtureProvenanceVerifier(second.fixture_root).verify(snapshot)
        finally:
            await second.close()

    asyncio.run(scenario_test())


def test_provenance_requires_exact_optional_evidence_identity(tmp_path):
    async def scenario_test():
        environment = await _environment(
            tmp_path,
            "valid_with_mask_and_calibration_evidence",
            register_evidence=True,
        )
        try:
            snapshot = await InspectionProcessingInputReader(environment.repositories).read(
                environment.inspection_id
            )
            assert len(snapshot.evidence) == 2
            verifier = SyntheticFixtureProvenanceVerifier(environment.fixture_root)
            verifier.verify(snapshot)
            changed = replace(
                snapshot,
                evidence=(replace(snapshot.evidence[0], byte_size=snapshot.evidence[0].byte_size + 1), *snapshot.evidence[1:]),
            )
            with pytest.raises(SyntheticProvenanceMismatchError):
                verifier.verify(changed)
            with pytest.raises(ProcessingExecutionConflictError):
                await _execute(environment)
            async with environment.database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 0
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_provenance_rejects_ambiguous_exact_scenario_match(tmp_path):
    async def scenario_test():
        environment = await _environment(tmp_path)
        try:
            original_id = environment.scenario["scenario_id"]
            duplicate_id = "duplicate_exact_identity"
            original = environment.fixture_root / "scenarios" / original_id
            duplicate = environment.fixture_root / "scenarios" / duplicate_id
            shutil.copytree(original, duplicate)
            scenario_path = duplicate / "scenario.json"
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            scenario["scenario_id"] = duplicate_id
            scenario["scenario_uuid"] = str(uuid5(NAMESPACE_URL, duplicate_id))
            scenario_path.write_bytes(json_bytes(scenario))

            manifest_path = environment.fixture_root / "generation_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            relative_files = [
                path.relative_to(environment.fixture_root)
                for path in (environment.fixture_root / "scenarios").rglob("*")
                if path.is_file()
            ]
            inventory = file_inventory(environment.fixture_root, relative_files)
            manifest["files"] = inventory
            manifest["scenario_ids"] = [original_id, duplicate_id]
            manifest["scenario_count"] = 2
            manifest["output_tree_sha256"] = output_tree_sha256(inventory)
            manifest_bytes = json_bytes(manifest)
            manifest_path.write_bytes(manifest_bytes)
            marker_path = environment.fixture_root / "SYNTHETIC_FIXTURES_MARKER.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["generation_manifest_sha256"] = sha256_bytes(manifest_bytes)
            marker_path.write_bytes(json_bytes(marker))

            snapshot = await InspectionProcessingInputReader(environment.repositories).read(
                environment.inspection_id
            )
            with pytest.raises(SyntheticProvenanceMismatchError):
                SyntheticFixtureProvenanceVerifier(environment.fixture_root).verify(snapshot)
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_execution_is_read_only_for_sources_and_creates_no_outputs(tmp_path):
    async def scenario_test():
        environment = await _environment(tmp_path)
        try:
            source_files = [path for path in environment.fixture_root.rglob("*") if path.is_file()]
            raw_files = [path for path in environment.paths.raw_uploads.rglob("*") if path.is_file()]

            def state(paths):
                return {path: (_digest(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths}

            before_sources, before_raw = state(source_files), state(raw_files)
            await _execute(environment)
            assert state(source_files) == before_sources
            assert state(raw_files) == before_raw
            assert not any(environment.paths.previews.rglob("*"))
            assert not any(environment.paths.results.rglob("*"))
            assert not any(environment.paths.reports.rglob("*"))
            assert not any(environment.paths.temporary.rglob("*"))
            async with environment.database.session() as session:
                assert await session.scalar(select(func.count()).select_from(ModelVersion)) == 0
                assert await session.scalar(select(func.count()).select_from(InspectionArtifact)) == 2
                assert await session.scalar(select(func.count()).select_from(InspectionValidation)) == 1
        finally:
            await environment.close()

    asyncio.run(scenario_test())


def test_boundary_remains_internal_and_schema_three(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "app" / "services" / "inspection_processing" / "orchestrator.py"
    ).read_text(encoding="utf-8").lower()
    assert "fastapi" not in source
    assert "torch" not in source
    assert "onnx" not in source
    assert "confidence" not in ProcessingExecutionResult.__dataclass_fields__
    assert SCHEMA_VERSION == 3
