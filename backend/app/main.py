from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.errors import ApiError, api_error_handler, request_validation_error_handler
from app.api.health import router as health_router
from app.api.demo_workspace import router as demo_workspace_router
from app.api.inspections import router as inspections_router
from app.api.recipes import router as recipes_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.request_context import RequestIdMiddleware
from app.core.runtime_paths import RuntimePaths
from app.db import Database, Repositories
from app.services.artifact_storage import (
    ArtifactPathPolicy,
    ArtifactRegistrationService,
    ArtifactSizeLimits,
    ArtifactStorageService,
)
from app.services.inspection_intake import InspectionIntakeCoordinator
from app.services.inspection_history import InspectionHistoryService
from app.services.inspection_audit import InspectionAuditRepository, InspectionAuditService
from app.services.inspection_processing import (
    InspectionProcessingApiService,
    InspectionProcessingOrchestrator,
    ProcessingLifecycleService,
)
from app.services.inspection_processing.assembly import SafeProcessingResultMapper
from app.services.inspection_validation import (
    ContractValidationPolicyEvaluator,
    DatabaseValidationArtifactRetriever,
    FindingFactory,
    InspectionValidationService,
    ManagedArtifactPathResolver,
    PurposeSpecificNativeFormatInspector,
    StreamingFilesystemIntegrityInspector,
    ValidationCommitService,
    ValidationPolicyLoader,
)
from app.services.inspection_validation.orchestrator import (
    InspectionValidationOrchestrator,
)
from app.services.inspection_validation.policy_loader import (
    DEVELOPMENT_POLICY_ID,
    DEVELOPMENT_POLICY_VERSION,
)
from app.services.recipe_catalogue import RecipeCatalogueService
from app.services.inspection_report import InspectionReportRepository, InspectionReportService
from app.services.demo_workspace import DemoWorkspaceService


def create_app(
    settings: Settings | None = None,
    *,
    validation_policy_loader: ValidationPolicyLoader | None = None,
    processing_orchestrator: InspectionProcessingOrchestrator | None = None,
) -> FastAPI:
    """Create the model-independent FastAPI application."""
    application_settings = settings or get_settings()
    logger = configure_logging(application_settings)
    runtime_paths = RuntimePaths.from_root(
        application_settings.runtime_root,
        application_settings.database_filename,
    )
    database = Database(
        runtime_paths.database_file,
        busy_timeout_ms=application_settings.sqlite_busy_timeout_ms,
        echo=application_settings.database_echo,
    )
    repositories = Repositories.from_session_factory(database.session_factory)
    artifact_storage = ArtifactStorageService(
        ArtifactPathPolicy(runtime_paths),
        ArtifactSizeLimits(
            rgb_bytes=application_settings.max_rgb_bytes,
            height_bytes=application_settings.max_height_bytes,
            mask_bytes=application_settings.max_mask_bytes,
            calibration_bytes=application_settings.max_calibration_bytes,
            generated_artifact_bytes=(
                application_settings.max_generated_artifact_bytes
            ),
        ),
    )
    artifact_registration = ArtifactRegistrationService(
        artifact_storage,
        repositories.artifacts,
    )
    inspection_intake = InspectionIntakeCoordinator(
        repositories,
        artifact_registration,
    )
    policy_loader = validation_policy_loader or ValidationPolicyLoader()
    # Validate the one explicitly registered application-owned development
    # policy during assembly. This reads no inspection artifacts and runs no
    # validation lifecycle.
    policy_loader.load(DEVELOPMENT_POLICY_ID, DEVELOPMENT_POLICY_VERSION)
    findings = FindingFactory()
    validation_engine = InspectionValidationService(
        DatabaseValidationArtifactRetriever(
            repositories.inspections,
            repositories.artifacts,
        ),
        StreamingFilesystemIntegrityInspector(
            ManagedArtifactPathResolver(runtime_paths)
        ),
        PurposeSpecificNativeFormatInspector(findings),
        ContractValidationPolicyEvaluator(findings),
        findings,
        policy_loader=policy_loader,
    )
    validation_commit = ValidationCommitService(
        database.session_factory,
        validation_repository=repositories.validations,
    )
    inspection_validation = InspectionValidationOrchestrator(
        repositories,
        policy_loader,
        validation_engine,
        validation_commit,
    )
    processing_result_mapper = SafeProcessingResultMapper(
        repositories.processing,
        repositories.audit_events,
    )
    configured_processing_orchestrator = None
    if (
        application_settings.enable_synthetic_processing_api
        and application_settings.synthetic_fixture_root is not None
    ):
        configured_processing_orchestrator = processing_orchestrator or (
            InspectionProcessingOrchestrator(
                repositories,
                runtime_paths,
                application_settings.synthetic_fixture_root,
                ProcessingLifecycleService(
                    database.session_factory,
                    repository=repositories.processing,
                ),
                result_mapper=processing_result_mapper,
            )
        )
    inspection_processing = InspectionProcessingApiService(
        repositories,
        processing_result_mapper,
        configured_processing_orchestrator,
    )
    inspection_history = InspectionHistoryService(database.session_factory)
    inspection_audit = InspectionAuditService(
        InspectionAuditRepository(database.session_factory)
    )
    inspection_report = InspectionReportService(
        InspectionReportRepository(database.session_factory)
    )
    recipe_catalogue = RecipeCatalogueService(database.session_factory)
    demo_workspace = DemoWorkspaceService(
        enabled=application_settings.enable_demo_workspace,
        fixture_root=application_settings.synthetic_fixture_root,
        runtime_paths=runtime_paths,
        database=database,
        repositories=repositories,
        artifact_registration=artifact_registration,
        validation_reader=inspection_validation,
        processing_reader=inspection_processing,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            runtime_paths.create_directories()
        except OSError:
            logger.exception(
                "Runtime directory initialization failed runtime_root=%s",
                runtime_paths.root,
            )
            raise

        try:
            await database.initialize()
        except Exception:
            logger.exception("Database initialization failed")
            await database.dispose()
            raise

        logger.info(
            "Application startup service=%s version=%s environment=%s "
            "runtime_root=%s",
            application_settings.application_name,
            application_settings.application_version,
            application_settings.environment,
            runtime_paths.root,
        )
        try:
            yield
        finally:
            await database.dispose()
            logger.info("Application shutdown")

    application = FastAPI(
        title=application_settings.application_name,
        version=application_settings.application_version,
        debug=application_settings.debug,
        lifespan=lifespan,
    )
    application.state.settings = application_settings
    application.state.logger = logger
    application.state.runtime_paths = runtime_paths
    application.state.database = database
    application.state.repositories = repositories
    application.state.artifact_storage = artifact_storage
    application.state.artifact_registration = artifact_registration
    application.state.inspection_intake = inspection_intake
    application.state.inspection_validation = inspection_validation
    application.state.inspection_processing = inspection_processing
    application.state.inspection_history = inspection_history
    application.state.inspection_audit = inspection_audit
    application.state.inspection_report = inspection_report
    application.state.recipe_catalogue = recipe_catalogue
    application.state.demo_workspace = demo_workspace
    application.add_exception_handler(ApiError, api_error_handler)
    application.add_exception_handler(
        RequestValidationError,
        request_validation_error_handler,
    )
    application.add_middleware(RequestIdMiddleware)
    application.include_router(
        health_router,
        prefix=application_settings.api_prefix,
    )
    application.include_router(
        inspections_router,
        prefix=application_settings.api_prefix,
    )
    application.include_router(
        recipes_router,
        prefix=application_settings.api_prefix,
    )
    application.include_router(
        demo_workspace_router,
        prefix=application_settings.api_prefix,
    )
    return application


app = create_app()
