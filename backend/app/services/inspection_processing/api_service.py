from __future__ import annotations

from app.db.repositories import Repositories
from app.services.inspection_processing.assembly import SafeProcessingResultMapper
from app.services.inspection_processing.exceptions import (
    ProcessingExecutionError,
    ProcessingExecutionInspectionNotFoundError,
    ProcessingExecutionOrchestrationError,
    ProcessingExecutionResultNotFoundError,
    SyntheticProcessingNotConfiguredError,
)
from app.services.inspection_processing.execution_models import ProcessingExecutionResult
from app.services.inspection_processing.orchestrator import InspectionProcessingOrchestrator


class InspectionProcessingApiService:
    """Application-owned execution gate and persisted-result retrieval boundary."""

    def __init__(
        self,
        repositories: Repositories,
        result_mapper: SafeProcessingResultMapper,
        orchestrator: InspectionProcessingOrchestrator | None,
    ) -> None:
        self._repositories = repositories
        self._mapper = result_mapper
        self._orchestrator = orchestrator

    async def execute_processing(
        self,
        inspection_id: str,
        preprocessing_policy_id: str,
        preprocessing_policy_version: str,
        inference_policy_id: str,
        inference_policy_version: str,
        *,
        request_id: str,
    ) -> ProcessingExecutionResult:
        if self._orchestrator is None:
            raise SyntheticProcessingNotConfiguredError(
                "synthetic processing execution is not configured"
            )
        return await self._orchestrator.execute_processing(
            inspection_id,
            preprocessing_policy_id,
            preprocessing_policy_version,
            inference_policy_id,
            inference_policy_version,
            actor_id=None,
            request_id=request_id,
        )

    async def get_latest_processing(
        self,
        inspection_id: str,
    ) -> ProcessingExecutionResult:
        try:
            inspection = await self._repositories.inspections.get(inspection_id)
            if inspection is None:
                raise ProcessingExecutionInspectionNotFoundError(
                    "inspection does not exist"
                )
            run = await self._repositories.processing.get_latest_run_for_inspection(
                inspection_id
            )
            if run is None:
                raise ProcessingExecutionResultNotFoundError(
                    "inspection has no processing result"
                )
            return await self._mapper.restore(
                run,
                inspection.status,
                idempotent=True,
                started_now=False,
            )
        except ProcessingExecutionError:
            raise
        except Exception as exc:
            raise ProcessingExecutionOrchestrationError(
                "processing result could not be read"
            ) from exc
