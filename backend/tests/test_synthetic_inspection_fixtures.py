import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from app.services.dataset_validation.file_inspection import (
    FileInspectionError,
    inspect_height,
    inspect_rgb,
)
from app.testing.synthetic_aoi import (
    DEFAULT_SEED,
    GENERATOR_ID,
    GENERATOR_VERSION,
    SCENARIO_IDS,
    SyntheticFixtureError,
    generate_fixtures,
)
from app.testing.synthetic_aoi.models import (
    GENERATION_MANIFEST_FILENAME,
    MARKER_FILENAME,
    SYNTHETIC_STATEMENT,
)
from scripts import generate_synthetic_inspection_fixtures as generator_cli
from scripts.generate_synthetic_inspection_fixtures import main as cli_main

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "contracts" / "synthetic_inspection_scenario.schema.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    }


def _scenario(root: Path, scenario_id: str) -> dict:
    return _json(root / "scenarios" / scenario_id / "scenario.json")


def _artifact_path(root: Path, scenario_id: str, role: str) -> Path:
    record = _scenario(root, scenario_id)
    return root / "scenarios" / scenario_id / record["artifacts"][role]["generated_file"]


@pytest.fixture(scope="module")
def generated_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic-fixtures") / "generated"
    generate_fixtures(root)
    return root


def test_same_seed_produces_byte_identical_directory_trees(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = generate_fixtures(first, seed=12345)
    second_result = generate_fixtures(second, seed=12345)

    assert _tree_bytes(first) == _tree_bytes(second)
    assert first_result.output_tree_sha256 == second_result.output_tree_sha256
    assert not first.with_name(f".{first.name}.synthetic-aoi-staging").exists()
    assert not second.with_name(f".{second.name}.synthetic-aoi-staging").exists()


def test_different_seed_changes_controlled_content_and_identifiers(tmp_path) -> None:
    first = tmp_path / "seed-one"
    second = tmp_path / "seed-two"
    selected = ("valid_rgb_png_height_tiff",)
    generate_fixtures(first, seed=1, scenario_ids=selected)
    generate_fixtures(second, seed=2, scenario_ids=selected)

    assert _artifact_path(first, selected[0], "rgb").read_bytes() != _artifact_path(
        second, selected[0], "rgb"
    ).read_bytes()
    assert _scenario(first, selected[0])["scenario_uuid"] != _scenario(
        second, selected[0]
    )["scenario_uuid"]


def test_generated_identifiers_are_deterministic(tmp_path) -> None:
    first = tmp_path / "id-one"
    second = tmp_path / "id-two"
    selected = ("valid_rgb_tiff_height_png16",)
    generate_fixtures(first, seed=77, scenario_ids=selected)
    generate_fixtures(second, seed=77, scenario_ids=selected)

    assert _scenario(first, selected[0])["scenario_uuid"] == _scenario(
        second, selected[0]
    )["scenario_uuid"]
    assert _json(first / GENERATION_MANIFEST_FILENAME)["generation_uuid"] == _json(
        second / GENERATION_MANIFEST_FILENAME
    )["generation_uuid"]


def test_marker_and_generation_manifest_are_explicitly_synthetic(generated_root) -> None:
    marker = _json(generated_root / MARKER_FILENAME)
    manifest = _json(generated_root / GENERATION_MANIFEST_FILENAME)

    for value in (marker, manifest):
        assert value["synthetic"] is True
        assert value["training_approved"] is False
        assert value["production_approved"] is False
        assert value["model_accuracy_evidence"] is False
        assert value["fixture_statement"] == SYNTHETIC_STATEMENT
        assert value["generator_id"] == GENERATOR_ID
        assert value["generator_version"] == GENERATOR_VERSION
        assert value["scenario_id"]
        assert value["seed"] == DEFAULT_SEED
    assert marker["generation_manifest_sha256"] == _sha256(
        generated_root / GENERATION_MANIFEST_FILENAME
    )


def test_unknown_existing_directory_is_never_overwritten(tmp_path) -> None:
    root = tmp_path / "unknown"
    root.mkdir()
    important = root / "important.txt"
    important.write_text("user data", encoding="utf-8")

    with pytest.raises(SyntheticFixtureError, match="ownership marker"):
        generate_fixtures(root, overwrite_generated=True)

    assert important.read_text(encoding="utf-8") == "user data"


def test_owned_directory_requires_explicit_and_safe_regeneration(tmp_path) -> None:
    root = tmp_path / "owned"
    generate_fixtures(root, seed=88, scenario_ids=(SCENARIO_IDS[0],))
    before = _tree_bytes(root)

    with pytest.raises(SyntheticFixtureError, match="explicit overwrite-generated"):
        generate_fixtures(root, seed=88, scenario_ids=(SCENARIO_IDS[0],))

    generate_fixtures(
        root,
        seed=88,
        scenario_ids=(SCENARIO_IDS[0],),
        overwrite_generated=True,
    )
    assert _tree_bytes(root) == before


def test_modified_or_unknown_owned_output_is_not_deleted(tmp_path) -> None:
    root = tmp_path / "modified"
    generate_fixtures(root, scenario_ids=(SCENARIO_IDS[0],))
    unknown = root / "user-file.txt"
    unknown.write_text("must remain", encoding="utf-8")

    with pytest.raises(SyntheticFixtureError, match="extra, or unknown"):
        generate_fixtures(
            root,
            scenario_ids=(SCENARIO_IDS[0],),
            overwrite_generated=True,
        )

    assert unknown.read_text(encoding="utf-8") == "must remain"


@pytest.mark.parametrize(
    "unsafe_output",
    [
        REPOSITORY_ROOT,
        REPOSITORY_ROOT / "backend" / "dataset" / "synthetic",
        REPOSITORY_ROOT / "backend" / "dataset_raw" / "synthetic",
        REPOSITORY_ROOT / "backend" / "tests" / "generated",
    ],
)
def test_repository_and_dataset_outputs_are_rejected(unsafe_output) -> None:
    with pytest.raises(SyntheticFixtureError):
        generate_fixtures(unsafe_output, scenario_ids=(SCENARIO_IDS[0],))


def test_symbolic_link_output_is_rejected_where_supported(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")

    with pytest.raises(SyntheticFixtureError, match="symbolic links|reparse"):
        generate_fixtures(link, overwrite_generated=True)


def test_every_scenario_satisfies_versioned_schema(generated_root) -> None:
    schema = _json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    scenario_paths = sorted(generated_root.glob("scenarios/*/scenario.json"))

    assert len(scenario_paths) == len(SCENARIO_IDS)
    for path in scenario_paths:
        validator.validate(_json(path))


def test_scenario_schema_finding_codes_match_authoritative_catalogue() -> None:
    scenario_schema = _json(SCHEMA_PATH)
    findings = _json(REPOSITORY_ROOT / "contracts" / "inspection_validation_findings.json")

    assert scenario_schema["$defs"]["finding_code"]["enum"] == findings["$defs"][
        "finding_code"
    ]["enum"]


def test_all_references_hashes_and_sizes_match_declared_expectations(
    generated_root,
) -> None:
    for scenario_id in SCENARIO_IDS:
        scenario_root = generated_root / "scenarios" / scenario_id
        record = _scenario(generated_root, scenario_id)
        for artifact in record["artifacts"].values():
            path = scenario_root / artifact["generated_file"]
            assert path.is_file()
            assert artifact["actual_sha256"] == _sha256(path)
            assert artifact["actual_byte_size"] == path.stat().st_size
            if artifact["reference_expected_to_resolve"]:
                for reference in artifact["references"]:
                    pure = PurePosixPath(reference)
                    assert ".." not in pure.parts
                    assert (scenario_root / Path(*pure.parts)).is_file()


def test_generation_manifest_inventory_and_tree_digest_cover_scenario_tree(
    generated_root,
) -> None:
    manifest = _json(generated_root / GENERATION_MANIFEST_FILENAME)
    listed = {item["path"]: item for item in manifest["files"]}
    actual = {
        path.relative_to(generated_root).as_posix(): path
        for path in (generated_root / "scenarios").rglob("*")
        if path.is_file()
    }

    assert manifest["scenario_count"] == len(SCENARIO_IDS)
    assert manifest["scenario_ids"] == list(SCENARIO_IDS)
    assert set(listed) == set(actual)
    for relative, path in actual.items():
        assert "\\" not in relative
        assert not PurePosixPath(relative).is_absolute()
        assert listed[relative]["sha256"] == _sha256(path)
        assert listed[relative]["byte_size"] == path.stat().st_size
    serialized = (generated_root / GENERATION_MANIFEST_FILENAME).read_text(
        encoding="utf-8"
    )
    assert str(generated_root) not in serialized


def test_generated_valid_formats_are_accepted_by_existing_parsers(generated_root) -> None:
    png_tiff = "valid_rgb_png_height_tiff"
    tiff_png = "valid_rgb_tiff_height_png16"
    png_npy = "valid_rgb_png_height_npy_float32"

    assert inspect_rgb(_artifact_path(generated_root, png_tiff, "rgb")).detected_format == "PNG"
    assert inspect_height(_artifact_path(generated_root, png_tiff, "height")).storage_data_type == "uint16"
    assert inspect_rgb(_artifact_path(generated_root, tiff_png, "rgb")).detected_format == "TIFF"
    assert inspect_height(_artifact_path(generated_root, tiff_png, "height")).storage_data_type == "uint16"
    assert inspect_height(_artifact_path(generated_root, png_npy, "height")).storage_data_type == "float32"


@pytest.mark.parametrize(
    "scenario_id",
    ["height_png_uint8", "height_png_rgb", "height_png_rgba"],
)
def test_invalid_height_png_scenarios_are_rejected(generated_root, scenario_id) -> None:
    with pytest.raises(FileInspectionError):
        inspect_height(_artifact_path(generated_root, scenario_id, "height"))


@pytest.mark.parametrize(
    ("scenario_id", "role", "inspector"),
    [
        ("corrupt_rgb", "rgb", inspect_rgb),
        ("corrupt_height", "height", inspect_height),
        ("truncated_rgb_png", "rgb", inspect_rgb),
        ("truncated_height_tiff", "height", inspect_height),
    ],
)
def test_corrupt_and_truncated_scenarios_are_genuinely_unreadable(
    generated_root,
    scenario_id,
    role,
    inspector,
) -> None:
    with pytest.raises((FileInspectionError, OSError, ValueError)):
        inspector(_artifact_path(generated_root, scenario_id, role))


@pytest.mark.parametrize(
    ("scenario_id", "role", "field"),
    [
        ("hash_mismatch_rgb", "rgb", "declared_sha256"),
        ("hash_mismatch_height", "height", "declared_sha256"),
        ("byte_size_mismatch_rgb", "rgb", "declared_byte_size"),
        ("byte_size_mismatch_height", "height", "declared_byte_size"),
    ],
)
def test_declared_mismatch_scenarios_differ_from_actual(
    generated_root,
    scenario_id,
    role,
    field,
) -> None:
    artifact = _scenario(generated_root, scenario_id)["artifacts"][role]
    actual_field = field.replace("declared_", "actual_")
    assert artifact[field] != artifact[actual_field]


def test_scenarios_contain_no_pcb_classification_or_confidence(generated_root) -> None:
    forbidden_keys = {
        "classification",
        "confidence",
        "defect_prediction",
        "pcb_outcome",
    }

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    for scenario_id in SCENARIO_IDS:
        record = _scenario(generated_root, scenario_id)
        assert not forbidden_keys.intersection(keys(record))
        assert record["synthetic"] is True
        assert record["training_approved"] is False
        assert record["production_approved"] is False
        assert record["model_accuracy_evidence"] is False


def test_scenario_filter_uses_catalogue_order(tmp_path) -> None:
    root = tmp_path / "selected"
    requested = (SCENARIO_IDS[5], SCENARIO_IDS[0])
    result = generate_fixtures(root, scenario_ids=requested)
    manifest = _json(root / GENERATION_MANIFEST_FILENAME)

    assert result.scenario_ids == (SCENARIO_IDS[0], SCENARIO_IDS[5])
    assert manifest["scenario_ids"] == list(result.scenario_ids)
    assert sorted(path.name for path in (root / "scenarios").iterdir()) == sorted(
        result.scenario_ids
    )


def test_cli_success_and_usage_exit_codes(tmp_path) -> None:
    output = tmp_path / "cli-output"

    assert cli_main(["--output-root", str(output), "--scenario", SCENARIO_IDS[0]]) == 0
    assert (output / MARKER_FILENAME).is_file()

    unknown = tmp_path / "unknown-cli-output"
    unknown.mkdir()
    assert (
        cli_main(
            [
                "--output-root",
                str(unknown),
                "--overwrite-generated",
            ]
        )
        == 2
    )


def test_cli_missing_required_output_uses_argparse_exit_two() -> None:
    with pytest.raises(SystemExit) as error:
        cli_main([])

    assert error.value.code == 2


def test_cli_unexpected_generation_failure_returns_three(tmp_path, monkeypatch) -> None:
    def fail_unexpectedly(*args, **kwargs):
        raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(generator_cli, "generate_fixtures", fail_unexpectedly)

    assert generator_cli.main(["--output-root", str(tmp_path / "unexpected")]) == 3
