from __future__ import annotations

import contextvars
import json
import os
import sys
import threading
import time
from pathlib import Path
from uuid import UUID

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import uvicorn  # noqa: E402
from sqlalchemy import event  # noqa: E402

import app.services.inspection_intake as intake_module  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


def _uuid_values(prefix: str, order: tuple[int, ...]):
    values = iter(f"{prefix}-{value:012x}" for value in order)

    def next_value() -> str:
        try:
            return next(values)
        except StopIteration as exc:
            raise RuntimeError("The deterministic E2E identity sequence was exhausted") from exc

    return next_value


settings = Settings()
application = create_app(settings)

inspection_ids = _uuid_values("00000000-0000-4000-8000", (0x11, 0x19, 0x13, 0x14, 0x15, 0x16, 0x17))
validation_ids = _uuid_values("bbbbbbbb-bbbb-4bbb-8bbb", (0x11, 0x19, 0x13, 0x14, 0x15, 0x16))
preprocessing_ids = _uuid_values("cccccccc-cccc-4ccc-8ccc", (0x11, 0x19, 0x13, 0x16))
inference_ids = _uuid_values("dddddddd-dddd-4ddd-8ddd", (0x11, 0x19, 0x13, 0x16))
processing_run_ids = _uuid_values("eeeeeeee-eeee-4eee-8eee", (0x11, 0x19, 0x13, 0x14, 0x16))

intake_module.uuid4 = lambda: UUID(inspection_ids())
application.state.inspection_validation._engine._validation_id = validation_ids
orchestrator = application.state.inspection_processing._orchestrator
if orchestrator is None:
    raise RuntimeError("Synthetic processing must be enabled for the E2E backend")
orchestrator._preprocess._preprocessing_id = preprocessing_ids
orchestrator._infer._inference_id = inference_ids
orchestrator._run_id = processing_run_ids
orchestrator._errors._preprocessing_id = lambda: "cccccccc-cccc-4ccc-8ccc-000000000014"
orchestrator._errors._inference_id = lambda: "dddddddd-dddd-4ddd-8ddd-000000000014"

query_log = Path(os.environ["PCB_AOI_E2E_QUERY_LOG"])
query_log.parent.mkdir(parents=True, exist_ok=True)
query_statements: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "e2e_query_statements", default=None
)
log_lock = threading.Lock()


@event.listens_for(application.state.database.engine.sync_engine, "before_cursor_execute")
def _record_query(_connection, _cursor, statement, _parameters, _context, _many):
    statements = query_statements.get()
    if statements is not None:
        statements.append(statement.lstrip().split(None, 1)[0].upper())


@application.middleware("http")
async def _profile_request(request, call_next):
    statements: list[str] = []
    token = query_statements.set(statements)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        query_statements.reset(token)
        document = {
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "select_queries": sum(item in {"SELECT", "PRAGMA"} for item in statements),
            "write_queries": sum(item not in {"SELECT", "PRAGMA"} for item in statements),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        with log_lock, query_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(document, sort_keys=True) + "\n")


if __name__ == "__main__":
    uvicorn.run(
        application,
        host=os.environ.get("PCB_AOI_E2E_BACKEND_HOST", "127.0.0.1"),
        port=int(os.environ["PCB_AOI_E2E_BACKEND_PORT"]),
        log_level="warning",
        access_log=False,
    )
