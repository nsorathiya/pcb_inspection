from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.core.config import Settings


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        service=settings.application_name,
        version=settings.application_version,
        environment=settings.environment,
    )
