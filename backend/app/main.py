from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.request_context import RequestIdMiddleware
from app.core.runtime_paths import RuntimePaths
from app.db import Database, Repositories


def create_app(settings: Settings | None = None) -> FastAPI:
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
    application.add_middleware(RequestIdMiddleware)
    application.include_router(
        health_router,
        prefix=application_settings.api_prefix,
    )
    return application


app = create_app()
