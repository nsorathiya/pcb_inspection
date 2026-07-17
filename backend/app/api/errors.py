from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.request_context import get_request_id


class ApiErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or get_request_id() or "-"


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    payload = ApiErrorResponse(
        code=exc.code,
        message=exc.message,
        request_id=_request_id(request),
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def request_validation_error_handler(
    request: Request,
    _exc: RequestValidationError,
) -> JSONResponse:
    is_multipart = request.headers.get("content-type", "").lower().startswith(
        "multipart/form-data"
    )
    payload = ApiErrorResponse(
        code=(
            "INCOMPLETE_OR_INVALID_MULTIPART_REQUEST"
            if is_multipart
            else "INVALID_VALIDATION_REQUEST"
        ),
        message=(
            "Required multipart fields are missing or invalid."
            if is_multipart
            else "The validation request body is missing or invalid."
        ),
        request_id=_request_id(request),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())
