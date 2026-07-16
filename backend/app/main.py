from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.request_context import RequestIdMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the model-independent FastAPI application."""
    application_settings = settings or get_settings()
    logger = configure_logging(application_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info(
            "Application startup service=%s version=%s environment=%s",
            application_settings.application_name,
            application_settings.application_version,
            application_settings.environment,
        )
        try:
            yield
        finally:
            logger.info("Application shutdown")

    application = FastAPI(
        title=application_settings.application_name,
        version=application_settings.application_version,
        debug=application_settings.debug,
        lifespan=lifespan,
    )
    application.state.settings = application_settings
    application.state.logger = logger
    application.add_middleware(RequestIdMiddleware)
    application.include_router(
        health_router,
        prefix=application_settings.api_prefix,
    )
    return application


app = create_app()
