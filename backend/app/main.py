from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the model-independent FastAPI application."""
    application_settings = settings or get_settings()
    application = FastAPI(
        title=application_settings.application_name,
        version=application_settings.application_version,
        debug=application_settings.debug,
    )
    application.state.settings = application_settings
    application.include_router(
        health_router,
        prefix=application_settings.api_prefix,
    )
    return application


app = create_app()
