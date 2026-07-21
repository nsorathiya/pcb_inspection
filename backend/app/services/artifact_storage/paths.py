from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from app.core.runtime_paths import RuntimePaths
from app.db.models import ArtifactType
from app.services.artifact_storage.exceptions import (
    ArtifactPathError,
    ArtifactPathRedirectError,
    InvalidArtifactInputError,
    UnsupportedArtifactExtensionError,
    UnsupportedArtifactMediaTypeError,
    UnsupportedArtifactTypeError,
)


@dataclass(frozen=True)
class _ArtifactRoute:
    runtime_attribute: str
    subdirectory: str | None
    stored_stem: str
    allowed_extensions: frozenset[str]


_RASTER_EXTENSIONS = frozenset({".bin", ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
_HEIGHT_EXTENSIONS = frozenset({".bin", ".exr", ".h5", ".hdf5", ".npy", ".png", ".tif", ".tiff"})
_MASK_EXTENSIONS = frozenset({".bin", ".npy", ".png", ".tif", ".tiff"})
_CALIBRATION_EXTENSIONS = frozenset({".bin", ".json", ".txt", ".yaml", ".yml"})
_REPORT_EXTENSIONS = frozenset({".bin", ".csv", ".json", ".pdf", ".txt"})

_INTAKE_EXTENSIONS: dict[ArtifactType, frozenset[str]] = {
    ArtifactType.RGB_RAW: frozenset(
        {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    ),
    ArtifactType.HEIGHT_RAW: frozenset({".npy", ".png", ".tif", ".tiff"}),
}
_INTAKE_MEDIA_TYPES: dict[str, frozenset[str]] = {
    ".bmp": frozenset({"application/octet-stream", "image/bmp", "image/x-ms-bmp"}),
    ".jpeg": frozenset({"application/octet-stream", "image/jpeg"}),
    ".jpg": frozenset({"application/octet-stream", "image/jpeg"}),
    ".npy": frozenset(
        {"application/octet-stream", "application/x-npy", "application/x-numpy"}
    ),
    ".png": frozenset({"application/octet-stream", "image/png"}),
    ".tif": frozenset({"application/octet-stream", "image/tiff"}),
    ".tiff": frozenset({"application/octet-stream", "image/tiff"}),
}

_ARTIFACT_ROUTES: dict[ArtifactType, _ArtifactRoute] = {
    ArtifactType.RGB_RAW: _ArtifactRoute(
        "raw_uploads", "rgb", "rgb_raw", _RASTER_EXTENSIONS
    ),
    ArtifactType.HEIGHT_RAW: _ArtifactRoute(
        "raw_uploads", "height", "height_raw", _HEIGHT_EXTENSIONS
    ),
    ArtifactType.VALIDITY_MASK: _ArtifactRoute(
        "raw_uploads", "masks", "validity_mask", _MASK_EXTENSIONS
    ),
    ArtifactType.CALIBRATION: _ArtifactRoute(
        "raw_uploads", "calibration", "calibration", _CALIBRATION_EXTENSIONS
    ),
    ArtifactType.RGB_PREVIEW: _ArtifactRoute(
        "previews", None, "rgb_preview", _RASTER_EXTENSIONS
    ),
    ArtifactType.HEIGHT_PREVIEW: _ArtifactRoute(
        "previews", None, "height_preview", _RASTER_EXTENSIONS
    ),
    ArtifactType.RESULT_OVERLAY: _ArtifactRoute(
        "results", None, "result_overlay", _RASTER_EXTENSIONS
    ),
    ArtifactType.REPORT: _ArtifactRoute(
        "reports", None, "report", _REPORT_EXTENSIONS
    ),
}

if set(_ARTIFACT_ROUTES) != set(ArtifactType):
    raise RuntimeError("artifact storage routes do not cover the artifact contract")


@dataclass(frozen=True)
class _ArtifactPathPlan:
    artifact_type: ArtifactType
    category_root: Path
    destination: Path
    relative_path: str


def _is_redirect(path: Path) -> bool:
    if not os.path.lexists(path):
        return False
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & reparse_flag)


def _approved_extension(
    artifact_type: ArtifactType,
    original_filename: str | None,
) -> str:
    route = _ARTIFACT_ROUTES[artifact_type]
    if not original_filename:
        return ".bin"

    informational_name = original_filename.replace("\\", "/").rsplit("/", 1)[-1]
    extension = PurePosixPath(informational_name).suffix.lower()
    if not extension:
        return ".bin"
    if extension not in route.allowed_extensions:
        raise UnsupportedArtifactExtensionError(
            f"extension {extension!r} is not allowed for {artifact_type.value}"
        )
    return extension


def validate_intake_file(
    artifact_type: ArtifactType,
    original_filename: str | None,
    media_type: str | None,
) -> str:
    """Validate the conservative filename/media-type gate used by intake."""
    try:
        canonical_type = ArtifactType(artifact_type)
        allowed_extensions = _INTAKE_EXTENSIONS[canonical_type]
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedArtifactTypeError(
            "artifact type is not accepted by paired intake"
        ) from exc
    if not original_filename:
        raise UnsupportedArtifactExtensionError(
            f"a filename extension is required for {canonical_type.value}"
        )

    informational_name = original_filename.replace("\\", "/").rsplit("/", 1)[-1]
    extension = PurePosixPath(informational_name).suffix.lower()
    if extension not in allowed_extensions:
        raise UnsupportedArtifactExtensionError(
            f"extension {extension or '<none>'!r} is not accepted for "
            f"{canonical_type.value} intake"
        )

    normalized_media_type = (media_type or "application/octet-stream").lower()
    if normalized_media_type not in _INTAKE_MEDIA_TYPES[extension]:
        raise UnsupportedArtifactMediaTypeError(
            f"media type is not accepted for {canonical_type.value} {extension} intake"
        )
    return extension


def _canonical_inspection_id(inspection_id: str) -> str:
    try:
        parsed = UUID(inspection_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise InvalidArtifactInputError(
            "inspection_id must be a canonical UUID string"
        ) from exc
    canonical = str(parsed)
    if canonical != inspection_id:
        raise InvalidArtifactInputError(
            "inspection_id must be a canonical UUID string"
        )
    return canonical


class ArtifactPathPolicy:
    """Generate and validate managed paths without using client identifiers."""

    def __init__(self, runtime_paths: RuntimePaths) -> None:
        self._paths = runtime_paths

    def _plan(
        self,
        inspection_id: str,
        artifact_type: ArtifactType,
        original_filename: str | None,
    ) -> _ArtifactPathPlan:
        try:
            canonical_type = ArtifactType(artifact_type)
            route = _ARTIFACT_ROUTES[canonical_type]
        except (KeyError, TypeError, ValueError) as exc:
            raise UnsupportedArtifactTypeError("unknown artifact type") from exc

        canonical_id = _canonical_inspection_id(inspection_id)
        extension = _approved_extension(canonical_type, original_filename)
        category_root = getattr(self._paths, route.runtime_attribute)
        parent = category_root / canonical_id
        if route.subdirectory is not None:
            parent /= route.subdirectory
        destination = parent / f"{route.stored_stem}{extension}"
        self._assert_no_redirects(self._paths.root, destination.parent)
        self._assert_confined(destination, category_root)
        relative_path = destination.relative_to(self._paths.root).as_posix()
        self._validate_relative_path(relative_path)
        return _ArtifactPathPlan(
            artifact_type=canonical_type,
            category_root=category_root,
            destination=destination,
            relative_path=relative_path,
        )

    def _prepare_parent(self, plan: _ArtifactPathPlan) -> None:
        self._paths.root.mkdir(parents=True, exist_ok=True)
        self._assert_no_redirects(self._paths.root, plan.destination.parent)
        plan.destination.parent.mkdir(parents=True, exist_ok=True)
        self._assert_no_redirects(self._paths.root, plan.destination.parent)
        self._assert_confined(plan.destination, plan.category_root)

    def _validate_before_finalization(self, plan: _ArtifactPathPlan) -> None:
        self._assert_no_redirects(self._paths.root, plan.destination.parent)
        self._assert_confined(plan.destination, plan.category_root)
        if os.path.lexists(plan.destination) and _is_redirect(plan.destination):
            raise ArtifactPathRedirectError(
                "immutable artifact destination is a path redirection"
            )

    def _absolute_from_relative(self, relative_path: str) -> Path:
        self._validate_relative_path(relative_path)
        target = self._paths.root.joinpath(*PurePosixPath(relative_path).parts)
        self._assert_confined(target, self._paths.root)
        return target

    @staticmethod
    def _validate_relative_path(relative_path: str) -> None:
        path = PurePosixPath(relative_path)
        if (
            not relative_path
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in relative_path
            or ":" in relative_path
        ):
            raise ArtifactPathError("artifact database path is not safely relative")

    @staticmethod
    def _assert_no_redirects(root: Path, target: Path) -> None:
        try:
            relative = target.relative_to(root)
        except ValueError as exc:
            raise ArtifactPathError("artifact target escapes runtime root") from exc

        current = root
        for part in (".", *relative.parts):
            if part != ".":
                current /= part
            if _is_redirect(current):
                raise ArtifactPathRedirectError(
                    "symbolic links and reparse points are not allowed in managed paths"
                )

    @staticmethod
    def _assert_confined(target: Path, expected_root: Path) -> None:
        def normalized_resolved(path: Path) -> str:
            value = os.path.normcase(os.path.normpath(str(path.resolve(strict=False))))
            if value.startswith("\\\\?\\UNC\\"):
                return "\\\\" + value[8:]
            if value.startswith("\\\\?\\"):
                return value[4:]
            return value

        resolved_target = normalized_resolved(target)
        resolved_root = normalized_resolved(expected_root)
        try:
            common = os.path.commonpath((resolved_target, resolved_root))
        except ValueError as exc:
            raise ArtifactPathError(
                "artifact target escapes its expected runtime category"
            ) from exc
        if common != resolved_root:
            raise ArtifactPathError(
                "artifact target escapes its expected runtime category"
            )
