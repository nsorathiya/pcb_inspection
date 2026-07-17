import asyncio
import hashlib
import json
import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from sqlalchemy import event, func, select

from app.core.runtime_paths import RuntimePaths
from app.db.database import Database
from app.db.models import ArtifactType, AuditEvent, InspectionArtifact, InspectionStatus
from app.db.repositories import InspectionArtifactCreate, InspectionCreate, Repositories
from app.services.inspection_validation import (
    ContractValidationPolicyEvaluator,
    DatabaseValidationArtifactRetriever,
    DimensionRelationship,
    FindingFactory,
    InspectionValidationService,
    ManagedArtifactPathResolver,
    PolicyLoadError,
    PurposeSpecificNativeFormatInspector,
    RetrievedInspectionArtifacts,
    StoredArtifactReference,
    StreamingFilesystemIntegrityInspector,
    ValidationOutcome,
    ValidationPolicyLoader,
    result_json,
)
from app.testing.synthetic_aoi import generate_fixtures

ROOT = Path(__file__).resolve().parents[2]
FIXED_TIME = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
FIXED_VALIDATION_ID = "11111111-1111-4111-8111-111111111111"
ROLE_TYPE = {
    "rgb": ArtifactType.RGB_RAW,
    "height": ArtifactType.HEIGHT_RAW,
    "mask": ArtifactType.VALIDITY_MASK,
    "calibration": ArtifactType.CALIBRATION,
}
ROLE_DIRECTORY = {"rgb": "rgb", "height": "height", "mask": "masks", "calibration": "calibration"}


class StaticRetriever:
    def __init__(self, context):
        self.context = context

    async def get_validation_artifacts(self, inspection_id):
        return self.context


class FailingRetriever:
    async def get_validation_artifacts(self, inspection_id):
        raise RuntimeError("contained test failure")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _scenario_setup(tmp_path, scenario_id):
    generated = tmp_path / "generated"
    generate_fixtures(generated, scenario_ids=(scenario_id,))
    record = _json(generated / "scenarios" / scenario_id / "scenario.json")
    inspection_id = record["scenario_uuid"]
    runtime = tmp_path / "runtime"
    references = []
    scenario_root = generated / "scenarios" / scenario_id
    for role, artifact in record["artifacts"].items():
        source = scenario_root / artifact["generated_file"]
        for reference in artifact["references"]:
            if ".." in PurePosixPath(reference).parts:
                relative = reference
            else:
                relative = f"raw_uploads/{inspection_id}/{ROLE_DIRECTORY[role]}/{reference}"
                target = runtime.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
            references.append(StoredArtifactReference(
                inspection_id=inspection_id,
                artifact_type=ROLE_TYPE[role],
                relative_path=relative,
                registered_sha256=artifact["declared_sha256"],
                registered_byte_size=artifact["declared_byte_size"],
                declared_media_type=artifact["media_type"],
            ))
    return record, runtime, RetrievedInspectionArtifacts(
        inspection_id=inspection_id,
        artifacts=tuple(references),
        synthetic_example=True,
    )


def _engine(runtime, context, *, retriever=None, policy_loader=None):
    findings = FindingFactory()
    formats = PurposeSpecificNativeFormatInspector(findings)
    return InspectionValidationService(
        retriever or StaticRetriever(context),
        StreamingFilesystemIntegrityInspector(ManagedArtifactPathResolver(RuntimePaths.from_root(runtime))),
        formats,
        ContractValidationPolicyEvaluator(findings),
        findings,
        clock=lambda: FIXED_TIME,
        validation_id_generator=lambda: FIXED_VALIDATION_ID,
        validator_version="1.0.0-test",
        policy_loader=policy_loader,
    )


def _policy():
    return ValidationPolicyLoader().load("development-native-rgb-height", "1.0")


def _run(tmp_path, scenario_id, policy=None, *, context_transform=None):
    record, runtime, context = _scenario_setup(tmp_path, scenario_id)
    if context_transform:
        context = context_transform(context)
    result = asyncio.run(_engine(runtime, context).validate_inspection_pair(record["scenario_uuid"], policy or _policy()))
    return record, runtime, context, result


@pytest.mark.parametrize("scenario_id", [
    "valid_rgb_png_height_tiff",
    "valid_rgb_tiff_height_png16",
    "valid_rgb_png_height_npy_float32",
])
def test_native_valid_synthetic_pairs_pass_development_policy(tmp_path, scenario_id):
    record, _, _, result = _run(tmp_path, scenario_id)
    assert result.outcome.value == record["expected_technical_validation_outcome"]
    assert result.summary.synthetic_example is True
    assert not [finding for finding in result.findings if finding.blocking]


def test_result_validates_schema_and_fixed_execution_is_byte_deterministic(tmp_path):
    record, runtime, context = _scenario_setup(tmp_path, "valid_rgb_png_height_tiff")
    service = _engine(runtime, context)
    first = asyncio.run(service.validate_inspection_pair(record["scenario_uuid"], _policy()))
    second = asyncio.run(service.validate_inspection_pair(record["scenario_uuid"], _policy()))
    assert result_json(first) == result_json(second)
    findings = _json(ROOT / "contracts" / "inspection_validation_findings.json")
    schema = _json(ROOT / "contracts" / "inspection_validation_result.schema.json")
    registry = Registry().with_resource(findings["$id"], Resource.from_contents(findings))
    Draft202012Validator(schema, registry=registry, format_checker=FormatChecker()).validate(first.to_dict())


@pytest.mark.parametrize(("scenario_id", "expected"), [
    ("missing_rgb", {"RGB_RAW_MISSING", "INCOMPLETE_RAW_PAIR"}),
    ("missing_height", {"HEIGHT_RAW_MISSING", "INCOMPLETE_RAW_PAIR"}),
    ("duplicate_rgb_reference", {"DUPLICATE_RGB_RAW", "INCOMPLETE_RAW_PAIR"}),
    ("duplicate_height_reference", {"DUPLICATE_HEIGHT_RAW", "INCOMPLETE_RAW_PAIR"}),
    ("hash_mismatch_rgb", {"ARTIFACT_SHA256_MISMATCH"}),
    ("hash_mismatch_height", {"ARTIFACT_SHA256_MISMATCH"}),
    ("byte_size_mismatch_rgb", {"ARTIFACT_SIZE_MISMATCH"}),
    ("byte_size_mismatch_height", {"ARTIFACT_SIZE_MISMATCH"}),
    ("unsafe_relative_path_reference", {"ARTIFACT_PATH_UNSAFE"}),
    ("corrupt_rgb", {"FILE_UNREADABLE"}),
    ("corrupt_height", {"FILE_UNREADABLE"}),
    ("truncated_rgb_png", {"FILE_UNREADABLE"}),
    ("truncated_height_tiff", {"FILE_UNREADABLE"}),
    ("height_png_uint8", {"HEIGHT_BIT_DEPTH_TOO_LOW"}),
    ("height_png_rgb", {"HEIGHT_NOT_SINGLE_CHANNEL"}),
    ("height_png_rgba", {"HEIGHT_NOT_SINGLE_CHANNEL"}),
    ("height_colorized_preview", {"HEIGHT_COLORIZED_PREVIEW_REJECTED"}),
    ("unsupported_rgb_extension", {"EXTENSION_CONTENT_MISMATCH"}),
    ("unsupported_height_extension", {"EXTENSION_CONTENT_MISMATCH"}),
    ("valid_different_dimensions", {"DIMENSION_RELATIONSHIP_UNSUPPORTED"}),
])
def test_synthetic_failure_scenarios_produce_expected_blocking_codes(tmp_path, scenario_id, expected):
    _, _, _, result = _run(tmp_path, scenario_id)
    assert result.outcome is ValidationOutcome.VALIDATION_FAILED
    assert {finding.code for finding in result.findings if finding.blocking} == expected


def test_missing_registered_file_fails(tmp_path):
    record, runtime, context = _scenario_setup(tmp_path, "valid_rgb_png_height_tiff")
    rgb = context.artifacts[0]
    runtime.joinpath(*PurePosixPath(rgb.relative_path).parts).unlink()
    result = asyncio.run(_engine(runtime, context).validate_inspection_pair(record["scenario_uuid"], _policy()))
    assert "ARTIFACT_FILE_MISSING" in {finding.code for finding in result.findings}
    assert result.outcome is ValidationOutcome.VALIDATION_FAILED


def test_symlink_escape_fails_where_supported(tmp_path):
    record, runtime, context = _scenario_setup(tmp_path, "valid_rgb_png_height_tiff")
    rgb = context.artifacts[0]
    target = runtime.joinpath(*PurePosixPath(rgb.relative_path).parts)
    outside = tmp_path / "outside.png"
    target.replace(outside)
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")
    result = asyncio.run(_engine(runtime, context).validate_inspection_pair(record["scenario_uuid"], _policy()))
    assert "ARTIFACT_SYMLINK_REJECTED" in {finding.code for finding in result.findings}


@pytest.mark.parametrize(("change", "code"), [
    ({"allowed_rgb_formats": ("TIFF",)}, "RGB_FORMAT_UNSUPPORTED"),
    ({"allowed_rgb_channels": (1,)}, "RGB_CHANNELS_UNSUPPORTED"),
    ({"allowed_rgb_bit_depths": (16,)}, "RGB_BIT_DEPTH_UNSUPPORTED"),
    ({"allowed_height_formats": ("PNG", "NPY")}, "HEIGHT_FORMAT_UNSUPPORTED"),
    ({"allowed_height_storage_types": ("uint16",)}, "HEIGHT_STORAGE_TYPE_UNSUPPORTED"),
])
def test_policy_property_failures_are_mapped(tmp_path, change, code):
    scenario = "valid_rgb_png_height_npy_float32" if code == "HEIGHT_STORAGE_TYPE_UNSUPPORTED" else "valid_rgb_png_height_tiff"
    _, _, _, result = _run(tmp_path, scenario, replace(_policy(), **change))
    assert code in {finding.code for finding in result.findings}
    assert result.outcome is ValidationOutcome.VALIDATION_FAILED


def test_media_type_warning_and_warning_as_blocking(tmp_path):
    record, runtime, context = _scenario_setup(tmp_path, "valid_rgb_png_height_tiff")
    artifacts = (replace(context.artifacts[0], declared_media_type="application/json"), *context.artifacts[1:])
    context = replace(context, artifacts=artifacts)
    normal = asyncio.run(_engine(runtime, context).validate_inspection_pair(record["scenario_uuid"], _policy()))
    blocking = asyncio.run(_engine(runtime, context).validate_inspection_pair(record["scenario_uuid"], replace(_policy(), warning_as_blocking=True)))
    assert normal.outcome is ValidationOutcome.VALIDATION_PASSED
    assert normal.findings[0].code == "MEDIA_TYPE_CONTENT_MISMATCH" and not normal.findings[0].blocking
    assert blocking.outcome is ValidationOutcome.VALIDATION_FAILED
    assert blocking.findings[0].blocking


def test_different_dimensions_allowed_and_required_evidence_rules(tmp_path):
    _, _, _, allowed = _run(tmp_path / "different", "valid_different_dimensions", replace(_policy(), dimension_relationship=DimensionRelationship.DIFFERENT_DIMENSIONS_ALLOWED))
    assert allowed.outcome is ValidationOutcome.VALIDATION_PASSED
    for field, code in [
        ("require_validity_mask", "VALIDITY_MASK_MISSING"),
        ("require_calibration_artifact", "CALIBRATION_EVIDENCE_MISSING"),
        ("require_registration_evidence", "REGISTRATION_EVIDENCE_MISSING"),
    ]:
        _, _, _, result = _run(tmp_path / field, "valid_rgb_png_height_tiff", replace(_policy(), **{field: True}))
        assert result.outcome is ValidationOutcome.VALIDATION_FAILED
        assert code in {finding.code for finding in result.findings if finding.blocking}


def test_present_mask_and_calibration_evidence_satisfy_required_policies(tmp_path):
    policy = replace(_policy(), require_validity_mask=True, require_calibration_artifact=True)
    _, _, _, result = _run(tmp_path, "valid_with_mask_and_calibration_evidence", policy)
    assert result.outcome is ValidationOutcome.VALIDATION_PASSED


def test_explicit_injected_registration_evidence_satisfies_transform_policy(tmp_path):
    policy = replace(
        _policy(),
        dimension_relationship=DimensionRelationship.REGISTERED_TRANSFORM_REQUIRED,
        require_registration_evidence=True,
    )
    transform = lambda context: replace(context, registration_evidence_available=True)
    _, _, _, result = _run(tmp_path, "valid_different_dimensions", policy, context_transform=transform)
    assert result.outcome is ValidationOutcome.VALIDATION_PASSED


def test_policy_loader_unknown_version_and_malformed_policy_fail_safely(tmp_path):
    loader = ValidationPolicyLoader()
    with pytest.raises(PolicyLoadError) as unknown:
        loader.load("missing-policy", "1.0")
    assert unknown.value.finding_code == "POLICY_NOT_FOUND"
    with pytest.raises(PolicyLoadError) as version:
        loader.load("development-native-rgb-height", "999")
    assert version.value.finding_code == "POLICY_VERSION_UNSUPPORTED"
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"contract_version":"wrong"}', encoding="utf-8")
    with pytest.raises(PolicyLoadError) as invalid:
        loader.load_path(malformed)
    assert invalid.value.finding_code == "POLICY_VERSION_UNSUPPORTED"


def test_registered_policy_selection_returns_safe_failed_result(tmp_path):
    record, runtime, context = _scenario_setup(tmp_path, "valid_rgb_png_height_tiff")
    service = _engine(runtime, context, policy_loader=ValidationPolicyLoader())
    result = asyncio.run(service.validate_registered_policy(record["scenario_uuid"], "missing-policy", "1.0"))
    assert result.outcome is ValidationOutcome.VALIDATION_FAILED
    assert [finding.code for finding in result.findings] == ["POLICY_NOT_FOUND"]


def test_findings_are_ordered_and_results_never_expose_paths(tmp_path):
    record, runtime, context = _scenario_setup(tmp_path, "valid_rgb_png_height_tiff")
    artifacts = (replace(context.artifacts[0], registered_sha256="0" * 64, declared_media_type="application/json"), *context.artifacts[1:])
    result = asyncio.run(_engine(runtime, replace(context, artifacts=artifacts)).validate_inspection_pair(record["scenario_uuid"], _policy()))
    orders = {item["code"]: item["order"] for item in _json(ROOT / "contracts" / "inspection_validation_findings.json")["findings"]}
    assert [orders[item.code] for item in result.findings] == sorted(orders[item.code] for item in result.findings)
    serialized = result_json(result)
    assert str(tmp_path) not in serialized
    assert "relative_path" not in serialized and "absolute_path" not in serialized


def test_unexpected_internal_error_returns_validation_error(tmp_path):
    inspection_id = str(uuid4())
    empty = RetrievedInspectionArtifacts(inspection_id, ())
    result = asyncio.run(_engine(tmp_path / "runtime", empty, retriever=FailingRetriever()).validate_inspection_pair(inspection_id, _policy()))
    assert result.outcome is ValidationOutcome.VALIDATION_ERROR
    assert [finding.code for finding in result.findings] == ["VALIDATOR_INTERNAL_ERROR"]


def test_noncanonical_inspection_id_is_rejected_before_retrieval(tmp_path):
    context = RetrievedInspectionArtifacts(str(uuid4()), ())
    with pytest.raises(ValueError, match="canonical UUID"):
        asyncio.run(_engine(tmp_path / "runtime", context).validate_inspection_pair("AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", _policy()))


def test_real_sqlite_validation_is_read_only_and_source_files_are_unchanged(tmp_path):
    async def scenario():
        record, runtime, context = _scenario_setup(tmp_path, "valid_rgb_png_height_tiff")
        paths = RuntimePaths.from_root(runtime)
        paths.database.mkdir(parents=True, exist_ok=True)
        database = Database(paths.database_file, busy_timeout_ms=5000)
        await database.initialize()
        repositories = Repositories.from_session_factory(database.session_factory)
        await repositories.inspections.create(InspectionCreate(
            id=record["scenario_uuid"], status=InspectionStatus.RECEIVED,
            board_id="SYNTHETIC", recipe_id="development-native-rgb-height", recipe_version="1.0",
        ))
        for reference in context.artifacts:
            await repositories.artifacts.create(InspectionArtifactCreate(
                inspection_id=reference.inspection_id, artifact_type=reference.artifact_type,
                relative_path=reference.relative_path, sha256=reference.registered_sha256,
                byte_size=reference.registered_byte_size, media_type=reference.declared_media_type,
            ))
        files = [runtime.joinpath(*PurePosixPath(item.relative_path).parts) for item in context.artifacts]
        before_files = [(hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
        async with database.session() as session:
            before_artifacts = await session.scalar(select(func.count()).select_from(InspectionArtifact))
            before_audits = await session.scalar(select(func.count()).select_from(AuditEvent))
        before_rows = [(item.id, item.inspection_id, item.artifact_type, item.relative_path, item.sha256, item.byte_size, item.media_type, item.created_at) for item in await repositories.artifacts.list_for_inspection(record["scenario_uuid"])]
        statements = []
        def record_sql(_conn, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.strip().split(None, 1)[0].upper())
        event.listen(database.engine.sync_engine, "before_cursor_execute", record_sql)
        try:
            findings = FindingFactory()
            service = InspectionValidationService(
                DatabaseValidationArtifactRetriever(repositories.inspections, repositories.artifacts),
                StreamingFilesystemIntegrityInspector(ManagedArtifactPathResolver(paths)),
                PurposeSpecificNativeFormatInspector(findings),
                ContractValidationPolicyEvaluator(findings), findings,
                clock=lambda: FIXED_TIME, validation_id_generator=lambda: FIXED_VALIDATION_ID,
            )
            result = await service.validate_inspection_pair(record["scenario_uuid"], _policy())
        finally:
            event.remove(database.engine.sync_engine, "before_cursor_execute", record_sql)
        inspection = await repositories.inspections.get(record["scenario_uuid"])
        async with database.session() as session:
            after_artifacts = await session.scalar(select(func.count()).select_from(InspectionArtifact))
            after_audits = await session.scalar(select(func.count()).select_from(AuditEvent))
        after_files = [(hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
        after_rows = [(item.id, item.inspection_id, item.artifact_type, item.relative_path, item.sha256, item.byte_size, item.media_type, item.created_at) for item in await repositories.artifacts.list_for_inspection(record["scenario_uuid"])]
        assert result.outcome is ValidationOutcome.VALIDATION_PASSED
        assert inspection.status is InspectionStatus.RECEIVED
        assert before_artifacts == after_artifacts == 2
        assert before_audits == after_audits == 0
        assert before_rows == after_rows
        assert before_files == after_files
        assert not {"INSERT", "UPDATE", "DELETE"}.intersection(statements)
        await database.dispose()
    asyncio.run(scenario())
