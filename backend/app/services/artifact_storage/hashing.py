from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Iterator

from app.services.artifact_storage.exceptions import InvalidArtifactInputError
from app.services.artifact_storage.models import BinarySource

DEFAULT_CHUNK_SIZE = 1024 * 1024


def iter_binary_chunks(
    source: BinarySource,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[bytes]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        view = memoryview(source)
        for offset in range(0, len(view), chunk_size):
            yield bytes(view[offset : offset + chunk_size])
        return

    stream: BinaryIO = source
    while True:
        chunk = stream.read(chunk_size)
        if chunk in (b"", None):
            return
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise InvalidArtifactInputError("binary stream read() must return bytes")
        yield bytes(chunk)


def hash_file(path: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[str, int]:
    digest = sha256()
    byte_size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
            byte_size += len(chunk)
    return digest.hexdigest(), byte_size
