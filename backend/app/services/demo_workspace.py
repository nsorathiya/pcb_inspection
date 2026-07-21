from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.runtime_paths import RuntimePaths
from app.db import Database, Repositories
from app.db.models import ArtifactType, InspectionStatus, Recipe, RecipeStatus
from app.db.repositories import RecipeCreate
from app.services.artifact_storage import ArtifactRegistrationService
from app.services.inspection_inference.service import SyntheticMockInferenceService
from app.services.inspection_intake import (
    InspectionIntakeCommand,
    InspectionIntakeCoordinator,
    InspectionIntakeFailure,
    IntakeArtifactSource,
)
from app.services.inspection_preprocessing.service import (
    SyntheticInspectionPreprocessingService,
)
from app.services.inspection_processing import (
    InspectionProcessingApiService,
    InspectionProcessingOrchestrator,
    ProcessingExecutionResultNotFoundError,
    ProcessingLifecycleService,
)
from app.services.inspection_processing.assembly import SafeProcessingResultMapper
from app.services.inspection_validation import (
    ContractValidationPolicyEvaluator,
    DatabaseValidationArtifactRetriever,
    FindingFactory,
    InspectionValidationOrchestrator,
    InspectionValidationService,
    ManagedArtifactPathResolver,
    PurposeSpecificNativeFormatInspector,
    StreamingFilesystemIntegrityInspector,
    ValidationCommitService,
    ValidationPolicyLoader,
    ValidationResultNotFoundError,
)
from app.services.inspection_validation.policy_loader import (
    DEVELOPMENT_POLICY_ID,
    DEVELOPMENT_POLICY_VERSION,
)
from app.services.inspection_preprocessing.policy_loader import (
    SYNTHETIC_POLICY_ID,
    SYNTHETIC_POLICY_VERSION,
)
from app.services.inspection_inference.policy_loader import (
    MOCK_POLICY_ID,
    MOCK_POLICY_VERSION,
)
from app.testing.synthetic_aoi import (
    DEFAULT_SEED,
    SyntheticFixtureError,
    generate_fixtures,
    validate_generated_fixtures,
)

DEMO_RECIPE_ID = "synthetic-e2e"
DEMO_RECIPE_NAME = "Synthetic E2E Recipe"
DEMO_REQUIRED_SCENARIOS = (
    "valid_rgb_png_height_tiff",
    "valid_different_dimensions",
)


class DemoWorkspaceError(Exception):
    """Base class for safe development-demo failures."""


class DemoWorkspaceDisabledError(DemoWorkspaceError):
    pass


class DemoWorkspaceNotConfiguredError(DemoWorkspaceError):
    pass


class DemoWorkspaceConsistencyError(DemoWorkspaceError):
    pass


class DemoWorkspaceLoadError(DemoWorkspaceError):
    pass


@dataclass(frozen=True)
class _DemoCase:
    key: str
    board_id: str
    scenario_id: str
    inspection_id: str
    validation_id: str
    preprocessing_id: str | None
    inference_id: str | None
    processing_run_id: str | None
    expected_status: InspectionStatus
    expected_validation_outcome: str
    expected_mock_decision: str | None
    force_preprocessing_error: bool = False


DEMO_CASES = (
    _DemoCase(
        "mock_pass", "DEMO-MOCK-PASS", "valid_rgb_png_height_tiff",
        "00000000-0000-4000-8000-000000000003",
        "bbbbbbbb-bbbb-4bbb-8bbb-000000000003",
        "cccccccc-cccc-4ccc-8ccc-000000000003",
        "dddddddd-dddd-4ddd-8ddd-000000000003",
        "eeeeeeee-eeee-4eee-8eee-000000000003",
        InspectionStatus.PASS, "VALIDATION_PASSED", "PASS",
    ),
    _DemoCase(
        "mock_fail", "DEMO-MOCK-FAIL", "valid_rgb_png_height_tiff",
        "00000000-0000-4000-8000-000000000001",
        "bbbbbbbb-bbbb-4bbb-8bbb-000000000001",
        "cccccccc-cccc-4ccc-8ccc-000000000001",
        "dddddddd-dddd-4ddd-8ddd-000000000001",
        "eeeeeeee-eeee-4eee-8eee-000000000001",
        InspectionStatus.FAIL, "VALIDATION_PASSED", "FAIL",
    ),
    _DemoCase(
        "mock_uncertain", "DEMO-MOCK-UNCERTAIN", "valid_rgb_png_height_tiff",
        "00000000-0000-4000-8000-000000000002",
        "bbbbbbbb-bbbb-4bbb-8bbb-000000000002",
        "cccccccc-cccc-4ccc-8ccc-000000000002",
        "dddddddd-dddd-4ddd-8ddd-000000000002",
        "eeeeeeee-eeee-4eee-8eee-000000000002",
        InspectionStatus.UNCERTAIN, "VALIDATION_PASSED", "UNCERTAIN",
    ),
    _DemoCase(
        "technical_error", "DEMO-TECHNICAL-ERROR", "valid_rgb_png_height_tiff",
        "00000000-0000-4000-8000-000000000004",
        "bbbbbbbb-bbbb-4bbb-8bbb-000000000004",
        "cccccccc-cccc-4ccc-8ccc-000000000004",
        None,
        "eeeeeeee-eeee-4eee-8eee-000000000004",
        InspectionStatus.ERROR, "VALIDATION_PASSED", None, True,
    ),
    _DemoCase(
        "validation_failure", "DEMO-VALIDATION-FAILURE", "valid_different_dimensions",
        "00000000-0000-4000-8000-000000000005",
        "bbbbbbbb-bbbb-4bbb-8bbb-000000000005",
        None, None, None,
        InspectionStatus.VALIDATION_FAILED, "VALIDATION_FAILED", None,
    ),
)


@dataclass(frozen=True)
class DemoInspectionState:
    key: str
    inspection_id: str
    board_id: str
    status: str | None
    validation_outcome: str | None
    processing_status: str | None
    preprocessing_outcome: str | None
    mock_decision: str | None
    complete: bool


@dataclass(frozen=True)
class DemoWorkspaceState:
    enabled: bool
    available: bool
    loaded: bool
    recipes_ready: bool
    inspections: tuple[DemoInspectionState, ...]
    synthetic: bool = True
    production_approved: bool = False
    idempotent_existing: bool | None = None


class _DemoFailureRgbProcessor:
    async def preprocess_rgb(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("controlled development demo preprocessing failure")


class DemoWorkspaceService:
    """Load one reserved, persistent synthetic workspace through real lifecycles."""

    def __init__(
        self,
        *,
        enabled: bool,
        fixture_root: Path | None,
        runtime_paths: RuntimePaths,
        database: Database,
        repositories: Repositories,
        artifact_registration: ArtifactRegistrationService,
        validation_reader: InspectionValidationOrchestrator,
        processing_reader: InspectionProcessingApiService,
    ) -> None:
        self._enabled = enabled
        self._fixture_root = fixture_root
        self._runtime_paths = runtime_paths
        self._database = database
        self._repositories = repositories
        self._artifact_registration = artifact_registration
        self._validation_reader = validation_reader
        self._processing_reader = processing_reader
        self._policy_loader = ValidationPolicyLoader()
        self._processing_mapper = SafeProcessingResultMapper(
            repositories.processing,
            repositories.audit_events,
        )
        self._load_lock = asyncio.Lock()

    async def get_state(self) -> DemoWorkspaceState:
        available = self._enabled and self._fixture_root is not None
        if not available:
            return DemoWorkspaceState(
                enabled=self._enabled,
                available=False,
                loaded=False,
                recipes_ready=False,
                inspections=(),
            )
        try:
            recipes_ready = await self._recipes_ready()
            inspections = tuple(
                [await self._inspection_state(case) for case in DEMO_CASES]
            )
        except DemoWorkspaceError:
            raise
        except Exception as exc:
            raise DemoWorkspaceLoadError(
                "demo workspace state could not be read"
            ) from exc
        return DemoWorkspaceState(
            enabled=True,
            available=True,
            loaded=recipes_ready and all(item.complete for item in inspections),
            recipes_ready=recipes_ready,
            inspections=inspections,
        )

    async def load(self, *, request_id: str) -> DemoWorkspaceState:
        if not self._enabled:
            raise DemoWorkspaceDisabledError("demo workspace is disabled")
        if self._fixture_root is None:
            raise DemoWorkspaceNotConfiguredError(
                "demo workspace fixture root is not configured"
            )
        async with self._load_lock:
            before = await self.get_state()
            if before.loaded:
                return DemoWorkspaceState(
                    **{**before.__dict__, "idempotent_existing": True}
                )
            try:
                await asyncio.to_thread(self._ensure_fixtures)
                for version, status, row_id in (
                    ("1.0", RecipeStatus.ACTIVE, "70000000-0000-4000-8000-000000000001"),
                    ("0.9", RecipeStatus.DRAFT, "70000000-0000-4000-8000-000000000002"),
                ):
                    await self._ensure_recipe(version, status, row_id)
                for case in DEMO_CASES:
                    await self._ensure_case(case, request_id=request_id)
                state = await self.get_state()
            except DemoWorkspaceError:
                raise
            except SyntheticFixtureError as exc:
                raise DemoWorkspaceNotConfiguredError(
                    "demo workspace fixtures are unavailable or invalid"
                ) from exc
            except Exception as exc:
                raise DemoWorkspaceLoadError(
                    "demo workspace could not be loaded"
                ) from exc
            if not state.loaded:
                raise DemoWorkspaceConsistencyError(
                    "demo workspace did not reach its expected state"
                )
            return DemoWorkspaceState(
                **{**state.__dict__, "idempotent_existing": False}
            )

    def _ensure_fixtures(self) -> None:
        assert self._fixture_root is not None
        if self._fixture_root.exists():
            result = validate_generated_fixtures(
                self._fixture_root,
                required_scenario_ids=DEMO_REQUIRED_SCENARIOS,
            )
            if result.seed != DEFAULT_SEED:
                raise SyntheticFixtureError(
                    "Demo workspace requires the repository default synthetic seed"
                )
            return
        generate_fixtures(
            self._fixture_root,
            seed=DEFAULT_SEED,
            scenario_ids=DEMO_REQUIRED_SCENARIOS,
        )

    @staticmethod
    def _recipe_configuration() -> dict[str, object]:
        return {
            "development_only": True,
            "production_approved": False,
            "note": "Recipe status does not establish production approval.",
        }

    async def _ensure_recipe(
        self,
        version: str,
        status: RecipeStatus,
        row_id: str,
    ) -> None:
        existing = await self._repositories.recipes.get_by_identity(
            DEMO_RECIPE_ID, version
        )
        if existing is None:
            try:
                await self._repositories.recipes.register(
                    RecipeCreate(
                        id=row_id,
                        recipe_id=DEMO_RECIPE_ID,
                        recipe_version=version,
                        name=DEMO_RECIPE_NAME,
                        configuration=self._recipe_configuration(),
                        status=status,
                    )
                )
            except Exception:
                existing = await self._repositories.recipes.get_by_identity(
                    DEMO_RECIPE_ID, version
                )
                if existing is None:
                    raise
            else:
                return
        self._verify_recipe(existing, version, status)

    def _verify_recipe(
        self,
        recipe: Recipe,
        version: str,
        status: RecipeStatus,
    ) -> None:
        expected_configuration = json.dumps(
            self._recipe_configuration(), sort_keys=True, separators=(",", ":")
        )
        if (
            recipe.recipe_id != DEMO_RECIPE_ID
            or recipe.recipe_version != version
            or recipe.name != DEMO_RECIPE_NAME
            or recipe.status is not status
            or recipe.configuration_json != expected_configuration
        ):
            raise DemoWorkspaceConsistencyError(
                "an existing recipe conflicts with the reserved demo identity"
            )

    async def _recipes_ready(self) -> bool:
        expected = (("1.0", RecipeStatus.ACTIVE), ("0.9", RecipeStatus.DRAFT))
        for version, status in expected:
            recipe = await self._repositories.recipes.get_by_identity(
                DEMO_RECIPE_ID, version
            )
            if recipe is None:
                return False
            self._verify_recipe(recipe, version, status)
        return True

    async def _ensure_case(self, case: _DemoCase, *, request_id: str) -> None:
        inspection = await self._repositories.inspections.get(case.inspection_id)
        if inspection is None:
            await self._receive_case(case, request_id=request_id)
            inspection = await self._repositories.inspections.get(case.inspection_id)
        if inspection is None:
            raise DemoWorkspaceConsistencyError("demo inspection intake is missing")
        self._verify_inspection_identity(case, inspection)

        validation = self._validation_orchestrator(case.validation_id)
        if inspection.status is InspectionStatus.RECEIVED:
            validation_result = await validation.execute_validation(
                case.inspection_id,
                DEVELOPMENT_POLICY_ID,
                DEVELOPMENT_POLICY_VERSION,
                actor_id=None,
                request_id=request_id,
            )
        else:
            validation_result = await validation.get_latest_validation(
                case.inspection_id
            )
        if validation_result.result.outcome.value != case.expected_validation_outcome:
            raise DemoWorkspaceConsistencyError(
                "demo validation outcome conflicts with its reserved scenario"
            )

        inspection = await self._repositories.inspections.get(case.inspection_id)
        if inspection is None:
            raise DemoWorkspaceConsistencyError("demo inspection disappeared")
        if case.processing_run_id is None:
            if inspection.status is not case.expected_status:
                raise DemoWorkspaceConsistencyError(
                    "demo validation failure has an unexpected lifecycle status"
                )
            return

        if inspection.status is InspectionStatus.READY:
            processing_result = await self._processing_orchestrator(case).execute_processing(
                case.inspection_id,
                SYNTHETIC_POLICY_ID,
                SYNTHETIC_POLICY_VERSION,
                MOCK_POLICY_ID,
                MOCK_POLICY_VERSION,
                actor_id=None,
                request_id=request_id,
            )
        else:
            processing_result = await self._processing_reader.get_latest_processing(
                case.inspection_id
            )
        if (
            processing_result.inspection_status is not case.expected_status
            or processing_result.mock_decision != case.expected_mock_decision
            or (
                case.force_preprocessing_error
                and processing_result.preprocessing_outcome != "PREPROCESSING_ERROR"
            )
        ):
            raise DemoWorkspaceConsistencyError(
                "demo processing result conflicts with its reserved scenario"
            )

    async def _receive_case(self, case: _DemoCase, *, request_id: str) -> None:
        assert self._fixture_root is not None
        scenario_root = self._fixture_root / "scenarios" / case.scenario_id
        try:
            record = json.loads(
                (scenario_root / "scenario.json").read_text(encoding="utf-8")
            )
            rgb = record["artifacts"]["rgb"]
            height = record["artifacts"]["height"]
            rgb_path = scenario_root / rgb["generated_file"]
            height_path = scenario_root / height["generated_file"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DemoWorkspaceNotConfiguredError(
                "demo scenario metadata is unavailable or invalid"
            ) from exc
        coordinator = InspectionIntakeCoordinator(
            self._repositories,
            self._artifact_registration,
            inspection_id_generator=lambda: case.inspection_id,
        )
        try:
            with rgb_path.open("rb") as rgb_stream, height_path.open("rb") as height_stream:
                await coordinator.receive_pair(
                    InspectionIntakeCommand(
                        board_id=case.board_id,
                        recipe_id=DEMO_RECIPE_ID,
                        recipe_version="1.0",
                        request_id=request_id,
                        operator_id="development-demo-workspace",
                        station_id="development-demo-workspace",
                        rgb=IntakeArtifactSource(
                            rgb_stream,
                            rgb["generated_file"],
                            rgb["media_type"],
                            rgb["actual_sha256"],
                            rgb["actual_byte_size"],
                        ),
                        height=IntakeArtifactSource(
                            height_stream,
                            height["generated_file"],
                            height["media_type"],
                            height["actual_sha256"],
                            height["actual_byte_size"],
                        ),
                    )
                )
        except InspectionIntakeFailure as exc:
            concurrent = await self._repositories.inspections.get(case.inspection_id)
            if concurrent is None:
                raise DemoWorkspaceLoadError("demo inspection intake failed") from exc
        inspection = await self._repositories.inspections.get(case.inspection_id)
        if inspection is None:
            raise DemoWorkspaceLoadError("demo inspection intake failed")
        self._verify_inspection_identity(case, inspection)
        artifacts = await self._repositories.artifacts.list_for_inspection(
            case.inspection_id
        )
        expected = {
            ArtifactType.RGB_RAW: (rgb["actual_sha256"], rgb["actual_byte_size"]),
            ArtifactType.HEIGHT_RAW: (
                height["actual_sha256"], height["actual_byte_size"]
            ),
        }
        if len(artifacts) != 2 or any(
            artifact.artifact_type not in expected
            or (artifact.sha256, artifact.byte_size) != expected[artifact.artifact_type]
            for artifact in artifacts
        ):
            raise DemoWorkspaceConsistencyError(
                "demo inspection artifacts conflict with their reserved scenario"
            )

    @staticmethod
    def _verify_inspection_identity(case: _DemoCase, inspection: Any) -> None:
        if (
            inspection.id != case.inspection_id
            or inspection.board_id != case.board_id
            or inspection.recipe_id != DEMO_RECIPE_ID
            or inspection.recipe_version != "1.0"
        ):
            raise DemoWorkspaceConsistencyError(
                "an existing inspection conflicts with a reserved demo identity"
            )

    def _validation_orchestrator(
        self,
        validation_id: str,
    ) -> InspectionValidationOrchestrator:
        findings = FindingFactory()
        engine = InspectionValidationService(
            DatabaseValidationArtifactRetriever(
                self._repositories.inspections,
                self._repositories.artifacts,
            ),
            StreamingFilesystemIntegrityInspector(
                ManagedArtifactPathResolver(self._runtime_paths)
            ),
            PurposeSpecificNativeFormatInspector(findings),
            ContractValidationPolicyEvaluator(findings),
            findings,
            policy_loader=self._policy_loader,
            validation_id_generator=lambda: validation_id,
        )
        return InspectionValidationOrchestrator(
            self._repositories,
            self._policy_loader,
            engine,
            ValidationCommitService(
                self._database.session_factory,
                validation_repository=self._repositories.validations,
            ),
        )

    def _processing_orchestrator(
        self,
        case: _DemoCase,
    ) -> InspectionProcessingOrchestrator:
        if case.preprocessing_id is None or case.processing_run_id is None:
            raise DemoWorkspaceConsistencyError(
                "demo processing identity is incomplete"
            )
        preprocessing = SyntheticInspectionPreprocessingService(
            rgb_processor=(
                _DemoFailureRgbProcessor()
                if case.force_preprocessing_error
                else None
            ),
            preprocessing_id_generator=lambda: case.preprocessing_id,
        )
        inference = SyntheticMockInferenceService(
            inference_id_generator=lambda: case.inference_id or case.preprocessing_id
        )
        return InspectionProcessingOrchestrator(
            self._repositories,
            self._runtime_paths,
            self._fixture_root,
            ProcessingLifecycleService(
                self._database.session_factory,
                repository=self._repositories.processing,
            ),
            preprocessing_service=preprocessing,
            inference_service=inference,
            result_mapper=self._processing_mapper,
            processing_run_id_generator=lambda: case.processing_run_id,
        )

    async def _inspection_state(self, case: _DemoCase) -> DemoInspectionState:
        inspection = await self._repositories.inspections.get(case.inspection_id)
        if inspection is None:
            return DemoInspectionState(
                case.key, case.inspection_id, case.board_id,
                None, None, None, None, None, False,
            )
        self._verify_inspection_identity(case, inspection)
        validation_outcome = None
        processing_status = None
        preprocessing_outcome = None
        mock_decision = None
        try:
            validation = await self._validation_reader.get_latest_validation(
                case.inspection_id
            )
            validation_outcome = validation.result.outcome.value
        except ValidationResultNotFoundError:
            pass
        try:
            processing = await self._processing_reader.get_latest_processing(
                case.inspection_id
            )
            processing_status = processing.processing_status.value
            preprocessing_outcome = processing.preprocessing_outcome
            mock_decision = processing.mock_decision
        except ProcessingExecutionResultNotFoundError:
            pass
        complete = (
            inspection.status is case.expected_status
            and validation_outcome == case.expected_validation_outcome
            and mock_decision == case.expected_mock_decision
            and (
                preprocessing_outcome
                == (
                    "PREPROCESSING_ERROR"
                    if case.force_preprocessing_error
                    else (
                        None
                        if case.processing_run_id is None
                        else "PREPROCESSING_SUCCEEDED"
                    )
                )
            )
            and (
                case.processing_run_id is None
                or processing_status in {"COMPLETED", "ERROR"}
            )
        )
        return DemoInspectionState(
            key=case.key,
            inspection_id=case.inspection_id,
            board_id=case.board_id,
            status=inspection.status.value,
            validation_outcome=validation_outcome,
            processing_status=processing_status,
            preprocessing_outcome=preprocessing_outcome,
            mock_decision=mock_decision,
            complete=complete,
        )
