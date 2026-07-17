from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.db.models import ArtifactType, Inspection, InspectionArtifact, InspectionStatus
from app.db.repositories import Repositories
from app.services.inspection_validation.exceptions import PolicyLoadError
from app.services.inspection_validation.interfaces import (
    InspectionValidationPolicy,
    InspectionValidationResult,
)
from app.services.inspection_validation.lifecycle import (
    InspectionNotFoundError,
    InvalidInspectionTransitionError,
    ValidationCommitError,
    ValidationCommitService,
    ValidationLifecycleConsistencyError,
)
from app.services.inspection_validation.models import (
    canonical_result_sha256,
    result_from_dict,
    result_to_dict,
)
from app.services.inspection_validation.persistence import (
    PersistedInspectionValidation,
    PersistedValidationFinding,
    ValidationPersistenceConflictError,
)
from app.services.inspection_validation.policy_loader import ValidationPolicyLoader
from app.services.inspection_validation.service import (
    DEFAULT_VALIDATOR_VERSION,
    VALIDATION_CONTRACT_VERSION,
    InspectionValidationService,
)
from app.services.inspection_validation.validation_key import (
    ValidationKeyArtifact,
    generate_validation_key,
)

_POLICY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_POLICY_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_EVIDENCE_TYPES = {ArtifactType.VALIDITY_MASK, ArtifactType.CALIBRATION}


class ValidationExecutionError(Exception):
    """Base class for safe validation orchestration failures."""


class InvalidValidationPolicySelectionError(ValidationExecutionError):
    pass


class ValidationPolicyNotFoundError(ValidationExecutionError):
    pass


class ValidationPolicyVersionUnsupportedError(ValidationExecutionError):
    pass


class ValidationInspectionNotFoundError(ValidationExecutionError):
    pass


class ValidationResultNotFoundError(ValidationExecutionError):
    pass


class ValidationExecutionConflictError(ValidationExecutionError):
    pass


class ValidationExecutionConsistencyError(ValidationExecutionConflictError):
    pass


class ValidationOrchestrationError(ValidationExecutionError):
    pass


@dataclass(frozen=True)
class ValidationExecutionResult:
    result: InspectionValidationResult
    validation_key: str
    inspection_status: InspectionStatus
    idempotent_existing: bool


class InspectionValidationOrchestrator:
    """Coordinate policy, identity, read-only execution, and atomic commit."""

    def __init__(
        self,
        repositories: Repositories,
        policy_loader: ValidationPolicyLoader,
        engine: InspectionValidationService,
        commit_service: ValidationCommitService,
        *,
        validator_version: str = DEFAULT_VALIDATOR_VERSION,
        contract_version: str = VALIDATION_CONTRACT_VERSION,
    ) -> None:
        self._repositories = repositories
        self._policies = policy_loader
        self._engine = engine
        self._commit = commit_service
        self._validator_version = validator_version
        self._contract_version = contract_version

    async def execute_validation(
        self,
        inspection_id: str,
        policy_id: str,
        policy_version: str,
        actor_id: str | None,
        request_id: str | None,
    ) -> ValidationExecutionResult:
        policy = self._load_policy(policy_id, policy_version)
        inspection, artifacts = await self._inspection_and_artifacts(inspection_id)
        validation_key = self._validation_key(
            inspection_id,
            artifacts,
            policy,
        )

        try:
            existing = await self._repositories.validations.get_by_inspection_and_key(
                inspection_id,
                validation_key,
            )
        except Exception as exc:
            raise ValidationOrchestrationError(
                "validation evidence could not be read"
            ) from exc

        if existing is not None:
            return await self._adopt_or_replay(
                existing,
                actor_id=actor_id,
                request_id=request_id,
            )

        if inspection.status is not InspectionStatus.RECEIVED:
            raise ValidationExecutionConflictError(
                "inspection is not eligible for a new validation"
            )

        try:
            result = await self._engine.validate_inspection_pair(
                inspection_id,
                policy,
            )
            self._validate_execution_identity(result, inspection_id, policy)
        except ValidationExecutionError:
            raise
        except Exception as exc:
            raise ValidationOrchestrationError(
                "technical validation could not be executed"
            ) from exc

        try:
            committed = await self._commit.commit_validation(
                result,
                validation_key,
                actor_id=actor_id,
                request_id=request_id,
            )
        except ValidationPersistenceConflictError:
            return await self._recover_concurrent_identical(
                inspection_id,
                validation_key,
                actor_id=actor_id,
                request_id=request_id,
            )
        except InspectionNotFoundError as exc:
            raise ValidationInspectionNotFoundError(
                "inspection does not exist"
            ) from exc
        except (InvalidInspectionTransitionError, ValidationLifecycleConsistencyError) as exc:
            raise ValidationExecutionConflictError(
                "inspection validation lifecycle conflicts with current state"
            ) from exc
        except ValidationCommitError as exc:
            raise ValidationOrchestrationError(
                "validation lifecycle could not be committed"
            ) from exc
        except Exception as exc:
            raise ValidationOrchestrationError(
                "validation lifecycle could not be committed"
            ) from exc

        return ValidationExecutionResult(
            result=result,
            validation_key=validation_key,
            inspection_status=committed.inspection_status,
            idempotent_existing=False,
        )

    async def get_latest_validation(
        self,
        inspection_id: str,
    ) -> ValidationExecutionResult:
        try:
            inspection = await self._repositories.inspections.get(inspection_id)
            if inspection is None:
                raise ValidationInspectionNotFoundError(
                    "inspection does not exist"
                )
            persisted = await self._repositories.validations.get_latest_for_inspection(
                inspection_id
            )
            if persisted is None:
                raise ValidationResultNotFoundError(
                    "inspection has no validation result"
                )
            result = await self._restore_persisted(persisted)
        except ValidationExecutionError:
            raise
        except Exception as exc:
            raise ValidationOrchestrationError(
                "validation result could not be read"
            ) from exc
        return ValidationExecutionResult(
            result=result,
            validation_key=persisted.validation_key,
            inspection_status=inspection.status,
            idempotent_existing=True,
        )

    def _load_policy(
        self,
        policy_id: str,
        policy_version: str,
    ) -> InspectionValidationPolicy:
        if not isinstance(policy_id, str) or not _POLICY_ID.fullmatch(policy_id):
            raise InvalidValidationPolicySelectionError("policy ID is invalid")
        if not isinstance(policy_version, str) or not _POLICY_VERSION.fullmatch(
            policy_version
        ):
            raise InvalidValidationPolicySelectionError("policy version is invalid")
        if not self._policies.supports_policy_id(policy_id):
            raise ValidationPolicyNotFoundError("policy is unavailable")
        if not self._policies.supports(policy_id, policy_version):
            raise ValidationPolicyVersionUnsupportedError(
                "policy version is unavailable"
            )
        try:
            return self._policies.load(policy_id, policy_version)
        except PolicyLoadError as exc:
            # The exact registry selection exists, so failure here means the
            # application-owned policy is malformed, not a client selection miss.
            raise ValidationOrchestrationError(
                "registered validation policy is invalid"
            ) from exc

    async def _inspection_and_artifacts(
        self,
        inspection_id: str,
    ) -> tuple[Inspection, list[InspectionArtifact]]:
        try:
            inspection = await self._repositories.inspections.get(inspection_id)
            if inspection is None:
                raise ValidationInspectionNotFoundError(
                    "inspection does not exist"
                )
            artifacts = await self._repositories.artifacts.list_for_inspection(
                inspection_id
            )
            return inspection, artifacts
        except ValidationExecutionError:
            raise
        except Exception as exc:
            raise ValidationOrchestrationError(
                "inspection validation inputs could not be read"
            ) from exc

    def _validation_key(
        self,
        inspection_id: str,
        artifacts: Sequence[InspectionArtifact],
        policy: InspectionValidationPolicy,
    ) -> str:
        rgb = [item for item in artifacts if item.artifact_type is ArtifactType.RGB_RAW]
        height = [
            item for item in artifacts if item.artifact_type is ArtifactType.HEIGHT_RAW
        ]
        if len(rgb) != 1 or len(height) != 1:
            raise ValidationExecutionConsistencyError(
                "registered raw artifact identity is incomplete or ambiguous"
            )

        def identity(item: InspectionArtifact) -> ValidationKeyArtifact:
            return ValidationKeyArtifact(
                artifact_type=item.artifact_type,
                sha256=item.sha256,
                byte_size=item.byte_size,
            )

        evidence = tuple(
            identity(item)
            for item in artifacts
            if item.artifact_type in _EVIDENCE_TYPES
        )
        try:
            return generate_validation_key(
                inspection_id=inspection_id,
                rgb_artifact=identity(rgb[0]),
                height_artifact=identity(height[0]),
                evidence_artifacts=evidence,
                contract_version=self._contract_version,
                policy_id=policy.policy_id,
                policy_version=policy.policy_version,
                validator_version=self._validator_version,
            )
        except ValueError as exc:
            raise ValidationExecutionConsistencyError(
                "registered artifact identity cannot form a validation key"
            ) from exc

    def _validate_execution_identity(
        self,
        result: InspectionValidationResult,
        inspection_id: str,
        policy: InspectionValidationPolicy,
    ) -> None:
        if (
            not isinstance(result, InspectionValidationResult)
            or result.inspection_id != inspection_id
            or result.contract_version != self._contract_version
            or result.validation_policy_id != policy.policy_id
            or result.validation_policy_version != policy.policy_version
            or result.validator_version != self._validator_version
        ):
            raise ValidationOrchestrationError(
                "validation engine returned incompatible result identity"
            )

    async def _adopt_or_replay(
        self,
        persisted: PersistedInspectionValidation,
        *,
        actor_id: str | None,
        request_id: str | None,
    ) -> ValidationExecutionResult:
        result = await self._restore_persisted(persisted)
        try:
            committed = await self._commit.commit_validation(
                result,
                persisted.validation_key,
                actor_id=actor_id,
                request_id=request_id,
            )
        except (ValidationPersistenceConflictError, ValidationLifecycleConsistencyError) as exc:
            raise ValidationExecutionConsistencyError(
                "persisted validation conflicts with inspection lifecycle"
            ) from exc
        except InvalidInspectionTransitionError as exc:
            raise ValidationExecutionConflictError(
                "inspection is not eligible for validation"
            ) from exc
        except InspectionNotFoundError as exc:
            raise ValidationInspectionNotFoundError(
                "inspection does not exist"
            ) from exc
        except ValidationCommitError as exc:
            raise ValidationOrchestrationError(
                "persisted validation lifecycle could not be completed"
            ) from exc
        if committed.lifecycle_idempotent_existing and committed.audit_action is None:
            raise ValidationExecutionConsistencyError(
                "persisted validation lifecycle audit is missing"
            )
        return ValidationExecutionResult(
            result=result,
            validation_key=persisted.validation_key,
            inspection_status=committed.inspection_status,
            idempotent_existing=True,
        )

    async def _recover_concurrent_identical(
        self,
        inspection_id: str,
        validation_key: str,
        *,
        actor_id: str | None,
        request_id: str | None,
    ) -> ValidationExecutionResult:
        try:
            persisted = await self._repositories.validations.get_by_inspection_and_key(
                inspection_id,
                validation_key,
            )
        except Exception as exc:
            raise ValidationExecutionConflictError(
                "a concurrent validation lifecycle conflict occurred"
            ) from exc
        if persisted is None:
            raise ValidationExecutionConflictError(
                "a concurrent validation lifecycle conflict occurred"
            )
        return await self._adopt_or_replay(
            persisted,
            actor_id=actor_id,
            request_id=request_id,
        )

    async def _restore_persisted(
        self,
        persisted: PersistedInspectionValidation,
    ) -> InspectionValidationResult:
        try:
            result = result_from_dict(persisted.result)
            document = result_to_dict(result)
            prepared = self._repositories.validations._prepare_validation(
                persisted.inspection_id,
                result,
                persisted.validation_key,
            )
            if (
                persisted.validation_id != result.validation_id
                or persisted.contract_version != result.contract_version
                or persisted.policy_id != result.validation_policy_id
                or persisted.policy_version != result.validation_policy_version
                or persisted.validator_version != result.validator_version
                or persisted.outcome is not result.outcome
                or persisted.result_sha256 != canonical_result_sha256(result)
                or prepared.result_sha256 != persisted.result_sha256
                or dict(persisted.rgb_summary) != document["rgb_artifact"]
                or dict(persisted.height_summary) != document["height_artifact"]
                or dict(persisted.summary) != document["summary"]
            ):
                raise ValueError("persisted validation columns disagree")
            findings = await self._repositories.validations.list_findings(
                persisted.validation_id
            )
            self._verify_findings(result, findings)
            return result
        except Exception as exc:
            raise ValidationExecutionConsistencyError(
                "persisted validation evidence is internally inconsistent"
            ) from exc

    @staticmethod
    def _verify_findings(
        result: InspectionValidationResult,
        persisted: Sequence[PersistedValidationFinding],
    ) -> None:
        if len(result.findings) != len(persisted):
            raise ValueError("persisted finding count disagrees")
        for ordinal, (expected, actual) in enumerate(zip(result.findings, persisted)):
            if (
                actual.ordinal != ordinal
                or actual.validation_id != result.validation_id
                or actual.code != expected.code
                or actual.severity is not expected.severity
                or actual.category is not expected.category
                or actual.message != expected.message
                or actual.artifact_type is not expected.artifact_type
                or actual.field != expected.field
                or actual.blocking is not expected.blocking
                or dict(actual.details) != dict(expected.details or {})
            ):
                raise ValueError("persisted finding rows disagree")
