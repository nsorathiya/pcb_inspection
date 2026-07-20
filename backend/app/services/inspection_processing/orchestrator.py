from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from app.core.runtime_paths import RuntimePaths
from app.db.models import InspectionStatus
from app.db.repositories import Repositories
from app.services.inspection_inference.exceptions import InferencePolicyLoadError
from app.services.inspection_inference.mock_engine import ENGINE_ID, ENGINE_VERSION
from app.services.inspection_inference.models import (
    InspectionInferencePolicy,
    InspectionInferenceResult,
    SyntheticInferenceInput,
)
from app.services.inspection_inference.policy_loader import (
    SyntheticMockInferencePolicyLoader,
)
from app.services.inspection_inference.service import SyntheticMockInferenceService
from app.services.inspection_preprocessing.exceptions import PreprocessingPolicyLoadError
from app.services.inspection_preprocessing.models import (
    InspectionPreprocessingPolicy,
    PreprocessingOutcome,
    SyntheticPreprocessingExecution,
    ValidatedInspectionInput,
)
from app.services.inspection_preprocessing.policy_loader import (
    SyntheticPreprocessingPolicyLoader,
)
from app.services.inspection_preprocessing.service import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    SyntheticInspectionPreprocessingService,
)
from app.services.inspection_processing.assembly import (
    SafeProcessingErrorResultFactory,
    SafeProcessingResultMapper,
    build_inference_input,
)
from app.services.inspection_processing.exceptions import (
    ProcessingExecutionConflictError,
    ProcessingExecutionConsistencyError,
    ProcessingExecutionInspectionNotReadyError,
    ProcessingExecutionOrchestrationError,
    ProcessingExecutionPolicyError,
    ProcessingExecutionRecoveryRequiredError,
)
from app.services.inspection_processing.execution_models import ProcessingExecutionResult
from app.services.inspection_processing.input_builder import (
    InspectionProcessingInputReader,
    ProcessingArtifactPreflight,
    ProcessingInputSnapshot,
    build_validated_preprocessing_input,
)
from app.services.inspection_processing.lifecycle import (
    InvalidProcessingTransitionError,
    ProcessingInspectionNotFoundError,
    ProcessingLifecycleConflictError,
    ProcessingLifecycleError,
    ProcessingLifecycleService,
)
from app.services.inspection_processing.models import (
    ProcessingKeyArtifact,
    ProcessingStartIdentity,
    generate_processing_key,
)
from app.services.inspection_processing.provenance import (
    SyntheticFixtureProvenanceVerifier,
)
from app.services.inspection_validation.artifact_reader import ManagedArtifactPathResolver


class _PreprocessingExecutor(Protocol):
    async def preprocess_inspection(
        self,
        validated_input: ValidatedInspectionInput,
        policy: InspectionPreprocessingPolicy,
    ) -> SyntheticPreprocessingExecution: ...


class _InferenceExecutor(Protocol):
    async def run_inference(
        self,
        inputs: SyntheticInferenceInput,
        policy: InspectionInferencePolicy,
    ) -> InspectionInferenceResult: ...


class InspectionProcessingOrchestrator:
    """Coordinate trusted synthetic execution without transport or state logic."""

    def __init__(
        self,
        repositories: Repositories,
        runtime_paths: RuntimePaths,
        fixture_root: Path,
        lifecycle_service: ProcessingLifecycleService,
        *,
        preprocessing_policy_loader: SyntheticPreprocessingPolicyLoader | None = None,
        inference_policy_loader: SyntheticMockInferencePolicyLoader | None = None,
        preprocessing_service: _PreprocessingExecutor | None = None,
        inference_service: _InferenceExecutor | None = None,
        provenance_verifier: SyntheticFixtureProvenanceVerifier | None = None,
        input_reader: InspectionProcessingInputReader | None = None,
        artifact_preflight: ProcessingArtifactPreflight | None = None,
        result_mapper: SafeProcessingResultMapper | None = None,
        error_results: SafeProcessingErrorResultFactory | None = None,
        processing_run_id_generator: Callable[[], str] | None = None,
        preprocessing_implementation_id: str = IMPLEMENTATION_ID,
        preprocessing_implementation_version: str = IMPLEMENTATION_VERSION,
        engine_id: str = ENGINE_ID,
        engine_version: str = ENGINE_VERSION,
    ) -> None:
        self._repositories = repositories
        self._pre_policies = preprocessing_policy_loader or SyntheticPreprocessingPolicyLoader()
        self._inf_policies = inference_policy_loader or SyntheticMockInferencePolicyLoader()
        self._preprocess = preprocessing_service or SyntheticInspectionPreprocessingService()
        self._infer = inference_service or SyntheticMockInferenceService()
        self._lifecycle = lifecycle_service
        self._provenance = provenance_verifier or SyntheticFixtureProvenanceVerifier(
            fixture_root
        )
        self._inputs = input_reader or InspectionProcessingInputReader(repositories)
        self._preflight = artifact_preflight or ProcessingArtifactPreflight(
            ManagedArtifactPathResolver(runtime_paths)
        )
        self._mapper = result_mapper or SafeProcessingResultMapper(
            repositories.processing, repositories.audit_events
        )
        self._errors = error_results or SafeProcessingErrorResultFactory()
        self._run_id = processing_run_id_generator or (lambda: str(uuid4()))
        self._implementation_id = preprocessing_implementation_id
        self._implementation_version = preprocessing_implementation_version
        self._engine_id = engine_id
        self._engine_version = engine_version

    async def execute_processing(
        self,
        inspection_id: str,
        preprocessing_policy_id: str,
        preprocessing_policy_version: str,
        inference_policy_id: str,
        inference_policy_version: str,
        actor_id: str | None = None,
        request_id: str | None = None,
    ) -> ProcessingExecutionResult:
        preprocessing_policy, inference_policy = self._load_policies(
            preprocessing_policy_id,
            preprocessing_policy_version,
            inference_policy_id,
            inference_policy_version,
        )
        snapshot = await self._inputs.read(inspection_id)
        processing_key = self._processing_key(
            snapshot, preprocessing_policy, inference_policy
        )
        try:
            existing = await self._repositories.processing.get_run_by_inspection_and_key(
                inspection_id, processing_key
            )
        except Exception as exc:
            raise ProcessingExecutionOrchestrationError(
                "processing lifecycle evidence could not be read"
            ) from exc
        if existing is not None:
            return await self._mapper.replay(existing, snapshot.inspection_status)
        if snapshot.inspection_status is not InspectionStatus.READY:
            if snapshot.inspection_status is InspectionStatus.PROCESSING:
                raise ProcessingExecutionConflictError(
                    "another processing lifecycle is already in progress"
                )
            raise ProcessingExecutionInspectionNotReadyError(
                "inspection is not READY for new processing"
            )

        await asyncio.to_thread(self._provenance.verify, snapshot)
        # The schema-v3 key includes optional evidence, but validation persistence
        # currently has no technical summaries from which its preprocessing input
        # identity can be constructed. Refuse before mutation rather than invent it.
        if snapshot.evidence:
            raise ProcessingExecutionConflictError(
                "optional evidence is not supported by the selected synthetic executor"
            )
        identity = ProcessingStartIdentity(
            processing_run_id=self._run_id(),
            inspection_id=snapshot.inspection_id,
            validation_id=snapshot.validation.validation_id,
            validation_result_sha256=snapshot.validation.result_sha256,
            rgb_artifact=self._key_artifact(snapshot.rgb),
            height_artifact=self._key_artifact(snapshot.height),
            evidence_artifacts=tuple(self._key_artifact(item) for item in snapshot.evidence),
            preprocessing_policy_id=preprocessing_policy.policy_id,
            preprocessing_policy_version=preprocessing_policy.policy_version,
            preprocessing_implementation_id=self._implementation_id,
            preprocessing_implementation_version=self._implementation_version,
            inference_policy_id=inference_policy.policy_id,
            inference_policy_version=inference_policy.policy_version,
            engine_id=self._engine_id,
            engine_version=self._engine_version,
            engine_type="MOCK",
        )
        try:
            begun = await self._lifecycle.begin_processing(
                identity,
                processing_key,
                actor_id=actor_id,
                request_id=request_id,
            )
        except ProcessingInspectionNotFoundError as exc:
            raise ProcessingExecutionConsistencyError(
                "inspection disappeared before processing begin"
            ) from exc
        except (InvalidProcessingTransitionError, ProcessingLifecycleConflictError) as exc:
            raise ProcessingExecutionConflictError(
                "processing lifecycle conflicts with current state"
            ) from exc
        except (ProcessingLifecycleError, TypeError, ValueError) as exc:
            raise ProcessingExecutionOrchestrationError(
                "processing lifecycle could not begin"
            ) from exc
        except Exception as exc:
            raise ProcessingExecutionOrchestrationError(
                "processing lifecycle could not begin"
            ) from exc
        if begun.idempotent_existing:
            run = await self._repositories.processing.get_run_by_id(
                begun.processing_run_id
            )
            if run is None:
                raise ProcessingExecutionConsistencyError(
                    "processing lifecycle disappeared after begin"
                )
            inspection = await self._repositories.inspections.get(snapshot.inspection_id)
            if inspection is None:
                raise ProcessingExecutionConsistencyError(
                    "processing inspection disappeared after begin"
                )
            return await self._mapper.replay(run, inspection.status)

        return await self._execute_winning_run(
            snapshot,
            preprocessing_policy,
            inference_policy,
            begun.processing_run_id,
            begun.started_at,
            processing_key,
            actor_id,
            request_id,
        )

    def _load_policies(
        self,
        preprocessing_policy_id: str,
        preprocessing_policy_version: str,
        inference_policy_id: str,
        inference_policy_version: str,
    ) -> tuple[InspectionPreprocessingPolicy, InspectionInferencePolicy]:
        try:
            preprocessing = self._pre_policies.load(
                preprocessing_policy_id, preprocessing_policy_version
            )
            inference = self._inf_policies.load(
                inference_policy_id, inference_policy_version
            )
        except (PreprocessingPolicyLoadError, InferencePolicyLoadError) as exc:
            raise ProcessingExecutionPolicyError(
                "selected processing policy is unavailable or invalid"
            ) from exc
        except Exception as exc:
            raise ProcessingExecutionPolicyError(
                "selected processing policy could not be loaded"
            ) from exc
        if (
            not preprocessing.development_only
            or preprocessing.production_approved
            or not inference.development_only
            or inference.production_approved
            or inference.engine.engine_type != "MOCK"
        ):
            raise ProcessingExecutionPolicyError(
                "selected processing policy is not approved for synthetic mock execution"
            )
        return preprocessing, inference

    def _processing_key(
        self,
        snapshot: ProcessingInputSnapshot,
        preprocessing: InspectionPreprocessingPolicy,
        inference: InspectionInferencePolicy,
    ) -> str:
        try:
            return generate_processing_key(
                inspection_id=snapshot.inspection_id,
                validation_id=snapshot.validation.validation_id,
                validation_result_sha256=snapshot.validation.result_sha256,
                rgb_artifact=self._key_artifact(snapshot.rgb),
                height_artifact=self._key_artifact(snapshot.height),
                evidence_artifacts=tuple(
                    self._key_artifact(item) for item in snapshot.evidence
                ),
                preprocessing_policy_id=preprocessing.policy_id,
                preprocessing_policy_version=preprocessing.policy_version,
                preprocessing_implementation_id=self._implementation_id,
                preprocessing_implementation_version=self._implementation_version,
                inference_policy_id=inference.policy_id,
                inference_policy_version=inference.policy_version,
                engine_id=self._engine_id,
                engine_version=self._engine_version,
                engine_type="MOCK",
            )
        except ValueError as exc:
            raise ProcessingExecutionConsistencyError(
                "persisted processing identities cannot form a processing key"
            ) from exc

    @staticmethod
    def _key_artifact(artifact) -> ProcessingKeyArtifact:
        return ProcessingKeyArtifact(
            artifact.artifact_type.value, artifact.sha256, artifact.byte_size
        )

    async def _execute_winning_run(
        self,
        snapshot: ProcessingInputSnapshot,
        preprocessing_policy: InspectionPreprocessingPolicy,
        inference_policy: InspectionInferencePolicy,
        processing_run_id: str,
        begun_at,
        processing_key: str,
        actor_id: str | None,
        request_id: str | None,
    ) -> ProcessingExecutionResult:
        try:
            resolved = await self._preflight.resolve_and_verify(snapshot)
            validated = build_validated_preprocessing_input(snapshot, resolved)
            execution = await self._preprocess.preprocess_inspection(
                validated, preprocessing_policy
            )
            self._validate_preprocessing_execution(
                execution, snapshot, preprocessing_policy
            )
            preprocessing_result = execution.result
        except Exception:
            try:
                preprocessing_result = self._errors.preprocessing_error(
                    snapshot,
                    preprocessing_policy,
                    implementation_id=self._implementation_id,
                    implementation_version=self._implementation_version,
                    not_before=begun_at,
                )
            except Exception as exc:
                raise ProcessingExecutionRecoveryRequiredError(
                    "processing failed after begin and requires operational recovery"
                ) from exc
            return await self._complete(
                processing_run_id,
                processing_key,
                preprocessing_result,
                None,
                actor_id,
                request_id,
            )

        inference_result = None
        if preprocessing_result.outcome is PreprocessingOutcome.SUCCEEDED:
            try:
                inference_input = build_inference_input(execution)
                inference_result = await self._infer.run_inference(
                    inference_input, inference_policy
                )
                self._validate_inference_result(
                    inference_result, preprocessing_result, inference_policy
                )
            except Exception:
                try:
                    inference_result = self._errors.inference_error(
                        preprocessing_result,
                        inference_policy,
                        engine_id=self._engine_id,
                        engine_version=self._engine_version,
                    )
                except Exception as exc:
                    raise ProcessingExecutionRecoveryRequiredError(
                        "processing failed after begin and requires operational recovery"
                    ) from exc
        return await self._complete(
            processing_run_id,
            processing_key,
            preprocessing_result,
            inference_result,
            actor_id,
            request_id,
        )

    def _validate_preprocessing_execution(
        self,
        execution: SyntheticPreprocessingExecution,
        snapshot: ProcessingInputSnapshot,
        policy: InspectionPreprocessingPolicy,
    ) -> None:
        result = execution.result
        if (
            not isinstance(execution, SyntheticPreprocessingExecution)
            or result.inspection_id != snapshot.inspection_id
            or result.validation_id != snapshot.validation.validation_id
            or result.policy_id != policy.policy_id
            or result.policy_version != policy.policy_version
            or result.implementation_id != self._implementation_id
            or result.implementation_version != self._implementation_version
            or result.rgb_input != snapshot.rgb_identity
            or result.height_input != snapshot.height_identity
            or result.synthetic_input is not True
            or result.mock_implementation is not True
            or result.production_approved is not False
        ):
            raise ValueError("preprocessing execution identity is incompatible")
        if result.outcome is PreprocessingOutcome.SUCCEEDED and (
            execution.rgb_buffer is None or execution.height_buffer is None
        ):
            raise ValueError("successful preprocessing returned no private buffers")

    def _validate_inference_result(
        self,
        result: InspectionInferenceResult,
        preprocessing,
        policy: InspectionInferencePolicy,
    ) -> None:
        if (
            not isinstance(result, InspectionInferenceResult)
            or result.inspection_id != preprocessing.inspection_id
            or result.validation_id != preprocessing.validation_id
            or result.preprocessing_id != preprocessing.preprocessing_id
            or result.policy_id != policy.policy_id
            or result.policy_version != policy.policy_version
            or result.engine_id != self._engine_id
            or result.engine_version != self._engine_version
            or result.engine_type != "MOCK"
            or result.synthetic_input is not True
            or result.mock_preprocessing is not True
            or result.mock_inference is not True
            or result.production_approved is not False
            or result.confidence is not None
        ):
            raise ValueError("inference execution identity is incompatible")

    async def _complete(
        self,
        processing_run_id: str,
        processing_key: str,
        preprocessing_result,
        inference_result,
        actor_id,
        request_id,
    ) -> ProcessingExecutionResult:
        try:
            completed = await self._lifecycle.complete_processing(
                processing_run_id,
                preprocessing_result,
                inference_result,
                actor_id=actor_id,
                request_id=request_id,
            )
        except Exception as exc:
            raise ProcessingExecutionRecoveryRequiredError(
                "processing completion failed and requires operational recovery"
            ) from exc
        return self._mapper.current(
            processing_run_id=processing_run_id,
            processing_key=processing_key,
            inspection_status=completed.inspection_status,
            processing_status=completed.processing_status,
            preprocessing=preprocessing_result,
            inference=inference_result,
            completed_at=completed.completed_at,
        )
