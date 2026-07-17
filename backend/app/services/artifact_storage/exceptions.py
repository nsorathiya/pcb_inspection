class ArtifactStorageError(Exception):
    """Base error for immutable artifact storage failures."""


class InvalidArtifactInputError(ArtifactStorageError, ValueError):
    """The requested artifact input is invalid."""


class UnsupportedArtifactTypeError(InvalidArtifactInputError):
    """The artifact type is not part of the authoritative contract."""


class UnsupportedArtifactExtensionError(InvalidArtifactInputError):
    """The source filename suffix is not approved for this artifact type."""


class ArtifactPathError(ArtifactStorageError):
    """A managed path is unsafe or outside its expected storage category."""


class ArtifactPathRedirectError(ArtifactPathError):
    """A symbolic link or filesystem reparse point redirects a managed path."""


class ArtifactIntegrityError(ArtifactStorageError):
    """Stored bytes do not match declared integrity metadata."""


class ArtifactHashMismatchError(ArtifactIntegrityError):
    """The calculated SHA-256 differs from the expected SHA-256."""


class ArtifactSizeMismatchError(ArtifactIntegrityError):
    """The calculated byte size differs from the expected byte size."""


class ArtifactSizeLimitError(ArtifactStorageError):
    """The artifact exceeded its configured per-type size limit."""


class ArtifactConflictError(ArtifactStorageError):
    """An immutable destination already contains different content."""


class ArtifactRegistrationError(ArtifactStorageError):
    """Filesystem storage could not be coordinated with database registration."""
