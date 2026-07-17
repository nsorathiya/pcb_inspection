from __future__ import annotations

import ast
import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO


class PathSafetyError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class FileInspectionError(ValueError):
    pass


@dataclass(frozen=True)
class InspectedRaster:
    format: str
    width: int
    height: int
    channels: int
    bit_depth: int
    mode: str
    storage_data_type: str | None = None


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def safe_dataset_file(
    dataset_root: Path,
    base_directory: Path,
    relative_path: str,
) -> tuple[Path, str]:
    root = dataset_root.resolve(strict=True)
    base = base_directory.resolve(strict=True)
    if not is_within(base, root):
        raise PathSafetyError("path.base_escape", "Reference base escapes dataset root")
    if not isinstance(relative_path, str) or not relative_path:
        raise PathSafetyError("path.invalid", "Referenced path must be non-empty")
    if "\\" in relative_path or ":" in relative_path:
        raise PathSafetyError(
            "path.invalid",
            "Referenced paths must use portable relative forward-slash syntax",
        )
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise PathSafetyError(
            "path.traversal",
            f"Referenced path escapes its allowed base: {relative_path}",
        )
    candidate = base.joinpath(*pure_path.parts)
    unresolved = candidate.resolve(strict=False)
    if not is_within(unresolved, root):
        raise PathSafetyError(
            "path.root_escape",
            f"Referenced path resolves outside the dataset root: {relative_path}",
        )
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise PathSafetyError(
                "path.symlink",
                f"Symbolic links are not accepted in dataset references: {relative_path}",
            )
    if not candidate.exists():
        raise PathSafetyError("file.missing", f"Referenced file is missing: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if not is_within(resolved, root):
        raise PathSafetyError(
            "path.root_escape",
            f"Referenced path resolves outside the dataset root: {relative_path}",
        )
    if not resolved.is_file():
        raise PathSafetyError(
            "file.not_regular",
            f"Referenced path is not a regular file: {relative_path}",
        )
    return resolved, resolved.relative_to(root).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_png(path: Path) -> InspectedRaster:
    signature = b"\x89PNG\r\n\x1a\n"
    with path.open("rb") as source:
        if source.read(8) != signature:
            raise FileInspectionError("Invalid PNG signature")
        width = height = bit_depth = color_type = None
        decompressor = zlib.decompressobj()
        saw_idat = False
        saw_iend = False
        while True:
            raw_length = source.read(4)
            if not raw_length:
                break
            if len(raw_length) != 4:
                raise FileInspectionError("Truncated PNG chunk length")
            length = struct.unpack(">I", raw_length)[0]
            chunk_type = source.read(4)
            data = source.read(length)
            raw_crc = source.read(4)
            if len(chunk_type) != 4 or len(data) != length or len(raw_crc) != 4:
                raise FileInspectionError("Truncated PNG chunk")
            expected_crc = struct.unpack(">I", raw_crc)[0]
            actual_crc = zlib.crc32(chunk_type)
            actual_crc = zlib.crc32(data, actual_crc) & 0xFFFFFFFF
            if expected_crc != actual_crc:
                raise FileInspectionError("PNG chunk CRC mismatch")
            if chunk_type == b"IHDR":
                if length != 13:
                    raise FileInspectionError("Invalid PNG IHDR length")
                width, height, bit_depth, color_type = struct.unpack(">IIBB", data[:10])
                if width <= 0 or height <= 0:
                    raise FileInspectionError("Invalid PNG dimensions")
            elif chunk_type == b"IDAT":
                saw_idat = True
                try:
                    decompressor.decompress(data)
                except zlib.error as exc:
                    raise FileInspectionError(f"Unreadable PNG image data: {exc}") from exc
            elif chunk_type == b"IEND":
                saw_iend = True
                break
        if None in (width, height, bit_depth, color_type):
            raise FileInspectionError("PNG IHDR is missing")
        if not saw_idat or not saw_iend:
            raise FileInspectionError("PNG IDAT or IEND is missing")
        try:
            decompressor.flush()
        except zlib.error as exc:
            raise FileInspectionError(f"Unreadable PNG image data: {exc}") from exc
        if not decompressor.eof:
            raise FileInspectionError("Truncated PNG compressed image data")
    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    modes = {0: "GRAY", 2: "RGB", 3: "PALETTE", 4: "GRAY_ALPHA", 6: "RGBA"}
    if color_type not in channels_by_type:
        raise FileInspectionError(f"Unsupported PNG color type {color_type}")
    return InspectedRaster(
        format="png",
        width=int(width),
        height=int(height),
        channels=channels_by_type[color_type],
        bit_depth=int(bit_depth),
        mode=modes[color_type],
    )


def _inspect_bmp(path: Path) -> InspectedRaster:
    with path.open("rb") as source:
        header = source.read(54)
    if len(header) < 30 or header[:2] != b"BM":
        raise FileInspectionError("Invalid BMP header")
    declared_size = struct.unpack_from("<I", header, 2)[0]
    pixel_offset = struct.unpack_from("<I", header, 10)[0]
    actual_size = path.stat().st_size
    if declared_size > actual_size or pixel_offset >= actual_size:
        raise FileInspectionError("Truncated BMP pixel data")
    dib_size = struct.unpack_from("<I", header, 14)[0]
    if dib_size < 40:
        raise FileInspectionError("Unsupported BMP DIB header")
    width, signed_height = struct.unpack_from("<ii", header, 18)
    bit_depth = struct.unpack_from("<H", header, 28)[0]
    if width <= 0 or signed_height == 0 or bit_depth not in {8, 24, 32}:
        raise FileInspectionError("Unsupported BMP dimensions or bit depth")
    channels = 1 if bit_depth == 8 else bit_depth // 8
    mode = "GRAY" if channels == 1 else ("RGB" if channels == 3 else "RGBA")
    return InspectedRaster("bmp", width, abs(signed_height), channels, bit_depth // channels, mode)


def _inspect_jpeg(path: Path) -> InspectedRaster:
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    with path.open("rb") as source:
        source.seek(-2, 2)
        if source.read(2) != b"\xff\xd9":
            raise FileInspectionError("JPEG end marker is missing")
        source.seek(0)
        if source.read(2) != b"\xff\xd8":
            raise FileInspectionError("Invalid JPEG signature")
        while True:
            byte = source.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker = source.read(1)
            while marker == b"\xff":
                marker = source.read(1)
            if not marker:
                break
            value = marker[0]
            if value in {0xD8, 0xD9, 0x01} or 0xD0 <= value <= 0xD7:
                continue
            raw_length = source.read(2)
            if len(raw_length) != 2:
                break
            length = struct.unpack(">H", raw_length)[0]
            data = source.read(length - 2)
            if len(data) != length - 2:
                raise FileInspectionError("Truncated JPEG segment")
            if value in sof_markers:
                if len(data) < 6:
                    break
                bit_depth = data[0]
                height, width = struct.unpack_from(">HH", data, 1)
                channels = data[5]
                mode = {1: "GRAY", 3: "RGB", 4: "CMYK"}.get(channels, "UNKNOWN")
                return InspectedRaster("jpeg", width, height, channels, bit_depth, mode)
    raise FileInspectionError("JPEG frame header not found")


def inspect_rgb(path: Path) -> InspectedRaster:
    with path.open("rb") as source:
        signature = source.read(8)
    if signature == b"\x89PNG\r\n\x1a\n":
        return _inspect_png(path)
    if signature.startswith(b"BM"):
        return _inspect_bmp(path)
    if signature.startswith(b"\xff\xd8"):
        return _inspect_jpeg(path)
    raise FileInspectionError("Unsupported RGB image content; supported formats are PNG, BMP, and JPEG")


def _tiff_values(source: BinaryIO, endian: str, field_type: int, count: int, raw_value: bytes) -> list[int]:
    sizes = {1: 1, 3: 2, 4: 4}
    formats = {1: "B", 3: "H", 4: "I"}
    if field_type not in sizes:
        raise FileInspectionError(f"Unsupported TIFF field type {field_type}")
    byte_count = sizes[field_type] * count
    if byte_count <= 4:
        data = raw_value[:byte_count]
    else:
        offset = struct.unpack(endian + "I", raw_value)[0]
        position = source.tell()
        source.seek(offset)
        data = source.read(byte_count)
        source.seek(position)
        if len(data) != byte_count:
            raise FileInspectionError("Truncated TIFF field value")
    return list(struct.unpack(endian + formats[field_type] * count, data))


def _inspect_tiff(path: Path) -> InspectedRaster:
    with path.open("rb") as source:
        byte_order = source.read(2)
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise FileInspectionError("Invalid TIFF byte order")
        if struct.unpack(endian + "H", source.read(2))[0] != 42:
            raise FileInspectionError("BigTIFF or invalid TIFF is unsupported")
        ifd_offset = struct.unpack(endian + "I", source.read(4))[0]
        source.seek(ifd_offset)
        raw_count = source.read(2)
        if len(raw_count) != 2:
            raise FileInspectionError("Truncated TIFF IFD")
        entry_count = struct.unpack(endian + "H", raw_count)[0]
        tags: dict[int, list[int]] = {}
        for _ in range(entry_count):
            entry = source.read(12)
            if len(entry) != 12:
                raise FileInspectionError("Truncated TIFF IFD entry")
            tag, field_type, count = struct.unpack(endian + "HHI", entry[:8])
            if tag in {256, 257, 258, 259, 262, 273, 277, 278, 279, 322, 323, 324, 325, 339}:
                tags[tag] = _tiff_values(source, endian, field_type, count, entry[8:12])
        width = tags.get(256, [0])[0]
        height = tags.get(257, [0])[0]
        bits = tags.get(258, [0])
        samples = tags.get(277, [1])[0]
        compression = tags.get(259, [1])[0]
        sample_format = tags.get(339, [1])[0]
        if width <= 0 or height <= 0 or not bits:
            raise FileInspectionError("TIFF dimensions or bit depth are missing")
        if any(bit != bits[0] for bit in bits):
            raise FileInspectionError("Mixed TIFF sample bit depths are unsupported")
        if 322 in tags or 324 in tags:
            raise FileInspectionError("Tiled TIFF is not supported by this validator version")
        strip_offsets = tags.get(273)
        strip_counts = tags.get(279)
        if not strip_offsets or not strip_counts or len(strip_offsets) != len(strip_counts):
            raise FileInspectionError("TIFF strip offsets/byte counts are missing")
        if compression not in {1, 8, 32946}:
            raise FileInspectionError(f"Unsupported TIFF compression {compression}")
        decoded_size = 0
        for offset, byte_count in zip(strip_offsets, strip_counts):
            source.seek(offset)
            data = source.read(byte_count)
            if len(data) != byte_count:
                raise FileInspectionError("Truncated TIFF strip")
            if compression in {8, 32946}:
                try:
                    data = zlib.decompress(data)
                except zlib.error as exc:
                    raise FileInspectionError(f"Unreadable TIFF Deflate strip: {exc}") from exc
            decoded_size += len(data)
        minimum_size = math.ceil(width * height * samples * bits[0] / 8)
        if decoded_size < minimum_size:
            raise FileInspectionError("TIFF pixel data is truncated")
    kind = {1: "uint", 2: "int", 3: "float"}.get(sample_format)
    if kind is None or bits[0] not in {8, 16, 32, 64}:
        raise FileInspectionError("Unsupported TIFF sample format or bit depth")
    dtype = f"{kind}{bits[0]}"
    mode = "SCALAR" if samples == 1 else "MULTI_SAMPLE"
    return InspectedRaster("tiff", width, height, samples, bits[0], mode, dtype)


def _inspect_npy(path: Path) -> InspectedRaster:
    with path.open("rb") as source:
        if source.read(6) != b"\x93NUMPY":
            raise FileInspectionError("Invalid NPY signature")
        version = tuple(source.read(2))
        if version[0] == 1:
            header_length = struct.unpack("<H", source.read(2))[0]
        elif version[0] in {2, 3}:
            header_length = struct.unpack("<I", source.read(4))[0]
        else:
            raise FileInspectionError(f"Unsupported NPY version {version}")
        header = source.read(header_length).decode("latin1")
        try:
            metadata = ast.literal_eval(header.strip())
        except (SyntaxError, ValueError) as exc:
            raise FileInspectionError("Invalid NPY header") from exc
        shape = metadata.get("shape")
        descriptor = metadata.get("descr")
        if not isinstance(shape, tuple) or len(shape) != 2 or not all(
            isinstance(value, int) and value > 0 for value in shape
        ):
            raise FileInspectionError("NPY height/depth arrays must be two-dimensional")
        dtype_map = {
            "|u1": ("uint8", 8),
            "<u2": ("uint16", 16),
            ">u2": ("uint16", 16),
            "<i2": ("int16", 16),
            ">i2": ("int16", 16),
            "<u4": ("uint32", 32),
            ">u4": ("uint32", 32),
            "<i4": ("int32", 32),
            ">i4": ("int32", 32),
            "<f4": ("float32", 32),
            ">f4": ("float32", 32),
            "<f8": ("float64", 64),
            ">f8": ("float64", 64),
        }
        if descriptor not in dtype_map:
            raise FileInspectionError(f"Unsupported NPY data type {descriptor!r}")
        dtype, bit_depth = dtype_map[descriptor]
        data_offset = source.tell()
        expected_size = shape[0] * shape[1] * (bit_depth // 8)
    if path.stat().st_size < data_offset + expected_size:
        raise FileInspectionError("NPY array data is truncated")
    return InspectedRaster("npy", shape[1], shape[0], 1, bit_depth, "SCALAR", dtype)


def inspect_height(path: Path) -> InspectedRaster:
    with path.open("rb") as source:
        signature = source.read(8)
    if signature[:2] in {b"II", b"MM"}:
        return _inspect_tiff(path)
    if signature.startswith(b"\x93NUMPY"):
        return _inspect_npy(path)
    if signature.startswith(b"\x89HDF"):
        raise FileInspectionError("HDF5 requires a future dataset-layout contract")
    if signature.startswith(b"v/1\x01"):
        raise FileInspectionError("EXR is not supported by this validator version")
    if signature == b"\x89PNG\r\n\x1a\n":
        raise FileInspectionError("PNG previews are not accepted as native height/depth data")
    raise FileInspectionError("Unsupported native height/depth file content")


def rgb_color_space_compatible(declared: str, inspected: InspectedRaster) -> bool:
    if declared == "GRAY":
        return inspected.mode == "GRAY" and inspected.channels == 1
    if declared == "RGB":
        return inspected.mode == "RGB" and inspected.channels == 3
    if declared.startswith("BAYER_"):
        return inspected.mode == "GRAY" and inspected.channels == 1
    return False
