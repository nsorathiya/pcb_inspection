import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update

from app.core.config import Settings
from app.db.models import (
    ArtifactType,
    AuditEvent,
    InspectionArtifact,
    InspectionProcessingRun,
    InspectionStatus,
    InspectionValidation,
)
from app.main import create_app
from app.services.artifact_storage import ArtifactInput
from app.services.dataset_validation.file_inspection import (
    decode_height_values,
    decode_rgb_values,
)
from app.testing.synthetic_aoi import generate_fixtures
from app.testing.synthetic_aoi.raster_generation import encode_npy_float32

PASS_ID = "00000000-0000-4000-8000-000000000003"


def _application(
    tmp_path: Path,
    *,
    enabled: bool = True,
    demo: bool = False,
):
    fixture_root = tmp_path / "fixtures"
    return (
        create_app(
            Settings(
                _env_file=None,
                runtime_root=tmp_path / "runtime",
                enable_engineering_viewer=enabled,
                enable_demo_workspace=demo,
                synthetic_fixture_root=fixture_root,
            )
        ),
        fixture_root,
    )


def _scenario_files(root: Path, scenario_id: str):
    scenario_root = root / "scenarios" / scenario_id
    record = json.loads(
        (scenario_root / "scenario.json").read_text(encoding="utf-8")
    )
    rgb = record["artifacts"]["rgb"]
    height = record["artifacts"]["height"]
    return {
        "rgb_image": (
            rgb["generated_file"],
            (scenario_root / rgb["generated_file"]).read_bytes(),
            rgb["media_type"],
        ),
        "height_map": (
            height["generated_file"],
            (scenario_root / height["generated_file"]).read_bytes(),
            height["media_type"],
        ),
    }, scenario_root / rgb["generated_file"], scenario_root / height["generated_file"]


def _intake(client: TestClient, fixture_root: Path, scenario_id: str):
    generate_fixtures(fixture_root, scenario_ids=(scenario_id,))
    files, rgb_path, height_path = _scenario_files(fixture_root, scenario_id)
    response = client.post(
        "/api/v1/inspections",
        data={
            "board_id": f"ENGINEERING-{scenario_id}",
            "recipe_id": "engineering-viewer-test",
            "recipe_version": "1.0",
        },
        files=files,
    )
    assert response.status_code == 201, response.text
    return response.json()["inspection_id"], rgb_path, height_path


async def _side_effect_counts(application) -> tuple[int, int, int]:
    async with application.state.database.session() as session:
        counts = []
        for model in (AuditEvent, InspectionValidation, InspectionProcessingRun):
            value = await session.scalar(select(func.count()).select_from(model))
            counts.append(int(value or 0))
        return tuple(counts)


def _runtime_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for category in ("raw_uploads", "previews", "results", "reports")
        for path in sorted((root / category).rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("scenario_id", "rgb_format", "height_format", "height_storage"),
    [
        ("valid_rgb_png_height_tiff", "PNG", "TIFF", "uint16"),
        ("valid_rgb_tiff_height_png16", "TIFF", "PNG", "uint16"),
        ("valid_rgb_png_height_npy_float32", "PNG", "NPY", "float32"),
    ],
)
def test_supported_synthetic_formats_return_native_metadata_and_histogram(
    tmp_path,
    scenario_id,
    rgb_format,
    height_format,
    height_storage,
):
    application, fixtures = _application(tmp_path)
    with TestClient(application) as client:
        inspection_id, _rgb_source, height_source = _intake(
            client, fixtures, scenario_id
        )
        before = asyncio.run(_side_effect_counts(application))
        response = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view",
            headers={"X-Request-ID": "engineering-metadata"},
        )
        after = asyncio.run(_side_effect_counts(application))
        inspection = asyncio.run(
            application.state.repositories.inspections.get(inspection_id)
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    expected_values = decode_height_values(height_source).values
    assert payload["request_id"] == "engineering-metadata"
    assert payload["rgb"]["detected_format"] == rgb_format
    assert payload["rgb"]["channels"] == 3
    assert payload["height"]["detected_format"] == height_format
    assert payload["height"]["storage_data_type"] == height_storage
    assert payload["height_statistics"]["native_min"] == min(expected_values)
    assert payload["height_statistics"]["native_max"] == max(expected_values)
    assert payload["height_statistics"]["valid_count"] == len(expected_values)
    assert payload["height_statistics"]["invalid_count"] == 0
    histogram = payload["height_statistics"]["histogram"]
    assert histogram["bin_count"] == len(histogram["counts"]) == 64
    assert sum(histogram["counts"]) == len(expected_values)
    assert payload["physical_height_unit"] is None
    assert payload["calibration_status"] == "NOT_PROVIDED"
    assert payload["registration_status"] == "NOT_ESTABLISHED"
    assert payload["synthetic_input_verified"] is False
    assert "SYNTHETIC_INPUT_PROVENANCE_NOT_VERIFIED" in payload["warnings"]
    assert payload["production_approved"] is False
    assert payload["validation"]["available"] is False
    assert payload["processing"]["available"] is False
    assert "relative_path" not in response.text
    assert str(tmp_path.resolve()) not in response.text
    assert before == after
    assert inspection is not None and inspection.status is InspectionStatus.RECEIVED


def test_separate_coordinates_return_exact_native_rgb_and_height_values(tmp_path):
    scenario_id = "valid_rgb_tiff_height_png16"
    application, fixtures = _application(tmp_path)
    with TestClient(application) as client:
        inspection_id, rgb_source, height_source = _intake(
            client, fixtures, scenario_id
        )
        response = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view/sample",
            params={"rgb_x": 2, "rgb_y": 3, "height_x": 4, "height_y": 5},
        )
        out_of_bounds = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view/sample",
            params={"rgb_x": 16, "rgb_y": 0, "height_x": -1, "height_y": 0},
        )
        missing = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view/sample",
            params={"rgb_x": 0, "rgb_y": 0},
        )

    assert response.status_code == 200, response.text
    rgb = decode_rgb_values(rgb_source)
    height = decode_height_values(height_source)
    rgb_offset = (3 * rgb.metadata.width + 2) * 3
    height_offset = 5 * height.metadata.width + 4
    assert response.json()["rgb"] == {
        "x": 2,
        "y": 3,
        "storage_data_type": "uint8",
        "values": list(rgb.values[rgb_offset : rgb_offset + 3]),
    }
    assert response.json()["height"] == {
        "x": 4,
        "y": 5,
        "storage_data_type": "uint16",
        "value": height.values[height_offset],
        "valid": True,
        "physical_unit": None,
    }
    assert out_of_bounds.status_code == 422
    assert out_of_bounds.json()["code"] == "ENGINEERING_SAMPLE_OUT_OF_BOUNDS"
    assert missing.status_code == 422
    assert missing.json()["code"] == "INVALID_ENGINEERING_VIEW_QUERY"


def test_previews_are_browser_pngs_generated_in_memory_without_file_changes(tmp_path):
    application, fixtures = _application(tmp_path)
    with TestClient(application) as client:
        inspection_id, _rgb_source, _height_source = _intake(
            client, fixtures, "valid_rgb_png_height_npy_float32"
        )
        runtime_root = application.state.runtime_paths.root
        before = _runtime_tree(runtime_root)
        rgb = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view/rgb-preview"
        )
        height = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view/height-preview"
        )
        after = _runtime_tree(runtime_root)

    for response, kind in ((rgb, "RGB"), (height, "HEIGHT")):
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert response.headers["x-pcb-aoi-preview-derived"] == "true"
        assert response.headers["x-pcb-aoi-preview-persisted"] == "false"
        assert response.headers["x-pcb-aoi-preview-kind"] == kind
        assert response.headers["cache-control"] == "no-store"
    assert height.headers["x-pcb-aoi-preview-transform"] == "NATIVE_MIN_MAX_GRAYSCALE"
    assert height.headers["x-pcb-aoi-physical-units"] == "unavailable"
    assert before == after


def test_persisted_validation_and_processing_evidence_is_read_without_new_audit(
    tmp_path,
):
    application, fixtures = _application(tmp_path, demo=True)
    with TestClient(application) as client:
        loaded = client.post("/api/v1/development/demo-workspace/load")
        assert loaded.status_code == 200, loaded.text
        before_counts = asyncio.run(_side_effect_counts(application))
        before_tree = _runtime_tree(application.state.runtime_paths.root)
        response = client.get(
            f"/api/v1/inspections/{PASS_ID}/engineering-view"
        )
        after_counts = asyncio.run(_side_effect_counts(application))
        after_tree = _runtime_tree(application.state.runtime_paths.root)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["validation"]["available"] is True
    assert payload["validation"]["outcome"] == "VALIDATION_PASSED"
    assert payload["validation"]["technically_ready"] is True
    assert payload["processing"]["available"] is True
    assert payload["processing"]["processing_status"] == "COMPLETED"
    assert payload["processing"]["mock_decision"] == "PASS"
    assert payload["processing"]["production_approved"] is False
    assert payload["processing"]["synthetic_input_verified"] is True
    assert payload["synthetic_input_verified"] is True
    assert (
        "SYNTHETIC_INPUT_VERIFIED_BY_PERSISTED_PROCESSING"
        in payload["warnings"]
    )
    assert payload["registration_status"] == "SYNTHETIC_IDENTITY_ONLY"
    assert before_counts == after_counts
    assert before_tree == after_tree


@pytest.mark.parametrize("tamper_mode", ["byte_size", "sha256"])
def test_tampering_fails_closed_without_path_disclosure(tmp_path, tamper_mode):
    application, fixtures = _application(tmp_path)
    with TestClient(application) as client:
        first_id, _rgb, _height = _intake(
            client, fixtures, "valid_rgb_png_height_tiff"
        )
        records = asyncio.run(
            application.state.repositories.artifacts.list_for_inspection(first_id)
        )
        rgb_record = next(
            record for record in records if record.artifact_type is ArtifactType.RGB_RAW
        )
        rgb_path = application.state.runtime_paths.root.joinpath(
            *Path(rgb_record.relative_path).parts
        )
        original = rgb_path.read_bytes()
        if tamper_mode == "byte_size":
            rgb_path.write_bytes(original + b"tampered")
        else:
            changed = bytearray(original)
            changed[-1] ^= 0x01
            rgb_path.write_bytes(changed)
        tampered = client.get(
            f"/api/v1/inspections/{first_id}/engineering-view"
        )

    assert tampered.status_code == 409
    assert tampered.json()["code"] == "ENGINEERING_ARTIFACT_INTEGRITY_FAILED"
    assert str(tmp_path.resolve()) not in tampered.text
    assert "relative_path" not in tampered.text


def test_cross_inspection_registered_path_is_rejected_as_ownership_mismatch(tmp_path):
    scenario_id = "valid_rgb_png_height_tiff"
    application, fixtures = _application(tmp_path)
    with TestClient(application) as client:
        first_id, _rgb, _height = _intake(client, fixtures, scenario_id)
        files, _rgb_source, _height_source = _scenario_files(fixtures, scenario_id)
        second = client.post(
            "/api/v1/inspections",
            data={
                "board_id": "ENGINEERING-SECOND",
                "recipe_id": "engineering-viewer-test",
                "recipe_version": "1.0",
            },
            files=files,
        )
        assert second.status_code == 201, second.text
        second_id = second.json()["inspection_id"]
        second_records = asyncio.run(
            application.state.repositories.artifacts.list_for_inspection(second_id)
        )
        second_rgb = next(
            record
            for record in second_records
            if record.artifact_type is ArtifactType.RGB_RAW
        )

        async def cross_link() -> None:
            async with application.state.database.session() as session, session.begin():
                await session.execute(
                    update(InspectionArtifact)
                    .where(
                        InspectionArtifact.inspection_id == first_id,
                        InspectionArtifact.artifact_type == ArtifactType.RGB_RAW,
                    )
                    .values(relative_path=second_rgb.relative_path)
                )

        asyncio.run(cross_link())
        response = client.get(
            f"/api/v1/inspections/{first_id}/engineering-view"
        )

    assert response.status_code == 409
    assert response.json()["code"] == "ENGINEERING_ARTIFACT_INTEGRITY_FAILED"
    assert str(tmp_path.resolve()) not in response.text


def test_calibration_is_reported_present_but_never_interpreted_as_physical_units(
    tmp_path,
):
    scenario_id = "valid_with_mask_and_calibration_evidence"
    application, fixtures = _application(tmp_path)
    with TestClient(application) as client:
        inspection_id, _rgb, _height = _intake(client, fixtures, scenario_id)
        scenario_root = fixtures / "scenarios" / scenario_id
        record = json.loads(
            (scenario_root / "scenario.json").read_text(encoding="utf-8")
        )
        calibration = record["artifacts"]["calibration"]
        content = (scenario_root / calibration["generated_file"]).read_bytes()
        asyncio.run(
            application.state.artifact_registration.store_and_register(
                ArtifactInput(
                    inspection_id=inspection_id,
                    artifact_type=ArtifactType.CALIBRATION,
                    source=content,
                    original_filename=calibration["generated_file"],
                    media_type=calibration["media_type"],
                )
            )
        )
        response = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view"
        )

    assert response.status_code == 200, response.text
    assert response.json()["calibration_status"] == "PRESENT_UNINTERPRETED"
    assert response.json()["physical_height_unit"] is None
    assert "NO_PHYSICAL_HEIGHT_UNITS_AVAILABLE" in response.json()["warnings"]


def test_nonfinite_float32_height_values_are_counted_and_sampled_safely(tmp_path):
    scenario_id = "valid_rgb_png_height_npy_float32"
    application, fixtures = _application(tmp_path)
    generate_fixtures(fixtures, scenario_ids=(scenario_id,))
    files, _rgb_source, height_source = _scenario_files(fixtures, scenario_id)
    original = list(decode_height_values(height_source).values)
    original[0] = float("nan")
    original[1] = float("inf")
    height_bytes = encode_npy_float32(16, 12, tuple(original))
    files["height_map"] = ("height.npy", height_bytes, "application/octet-stream")

    with TestClient(application) as client:
        intake = client.post(
            "/api/v1/inspections",
            data={
                "board_id": "ENGINEERING-NONFINITE",
                "recipe_id": "engineering-viewer-test",
                "recipe_version": "1.0",
            },
            files=files,
        )
        assert intake.status_code == 201, intake.text
        inspection_id = intake.json()["inspection_id"]
        view = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view"
        )
        sample = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view/sample",
            params={"rgb_x": 0, "rgb_y": 0, "height_x": 0, "height_y": 0},
        )
        preview = client.get(
            f"/api/v1/inspections/{inspection_id}/engineering-view/height-preview"
        )

    assert view.status_code == 200, view.text
    assert view.json()["height_statistics"]["valid_count"] == 190
    assert view.json()["height_statistics"]["invalid_count"] == 2
    assert sum(view.json()["height_statistics"]["histogram"]["counts"]) == 190
    assert sample.status_code == 200
    assert sample.json()["height"]["value"] is None
    assert sample.json()["height"]["valid"] is False
    assert preview.status_code == 200


def test_disabled_viewer_and_invalid_or_unknown_inspection_fail_safely(tmp_path):
    disabled, _fixtures = _application(tmp_path / "disabled", enabled=False)
    with TestClient(disabled) as client:
        response = client.get(
            f"/api/v1/inspections/{PASS_ID}/engineering-view"
        )
    assert response.status_code == 404
    assert response.json()["code"] == "ENGINEERING_VIEWER_DISABLED"

    enabled, _fixtures = _application(tmp_path / "enabled")
    with TestClient(enabled) as client:
        unknown = client.get(
            f"/api/v1/inspections/{PASS_ID}/engineering-view"
        )
        invalid = client.get("/api/v1/inspections/not-a-uuid/engineering-view")
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "INSPECTION_NOT_FOUND"
    assert invalid.status_code == 400
    assert invalid.json()["code"] == "INVALID_INSPECTION_ID"
