from __future__ import annotations

import ast
import hashlib
import math
import struct
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Mapping


class PathSafetyError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class FileInspectionError(ValueError):
    pass


@dataclass(frozen=True)
class InspectedRaster:
    """Read-only, path-free metadata returned by native raster inspectors."""

    detected_format: str
    width: int
    height: int
    channels: int
    bit_depth: int
    color_mode: str
    storage_data_type: str | None = None
    readability_status: str = "READABLE"
    safe_details: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        forbidden_keys = {
            "path",
            "absolute_path",
            "relative_path",
            "filename",
            "source_filename",
        }
        details = dict(self.safe_details)
        if forbidden_keys.intersection(details):
            raise ValueError("Native raster details must not contain filesystem paths")
        if any(
            not isinstance(value, (str, int, float, bool, type(None)))
            for value in details.values()
        ):
            raise TypeError("Native raster details must contain safe primitive values")
        object.__setattr__(self, "safe_details", details)
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def format(self) -> str:
        """Backward-compatible lowercase format used by dataset contract 1.0."""

        return self.detected_format.lower()

    @property
    def mode(self) -> str:
        """Backward-compatible alias used by the paired-dataset validator."""

        return self.color_mode


@dataclass(frozen=True)
class DecodedRaster:
    """Narrow native-value result for validated synthetic raster subsets."""

    metadata: InspectedRaster
    values: tuple[int | float, ...]

    def __post_init__(self) -> None:
        expected = self.metadata.width * self.metadata.height * self.metadata.channels
        if len(self.values) != expected:
            raise ValueError("decoded raster value count contradicts metadata")


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
    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    modes = {0: "GRAY", 2: "RGB", 3: "PALETTE", 4: "GRAY_ALPHA", 6: "RGBA"}
    allowed_bit_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    with path.open("rb") as source:
        if source.read(8) != signature:
            raise FileInspectionError("Invalid PNG signature")
        file_size = path.stat().st_size
        width = height = bit_depth = color_type = None
        compression_method = filter_method = interlace_method = None
        decompressor = zlib.decompressobj()
        decoded = bytearray()
        expected_decoded_size = None
        saw_ihdr = False
        saw_plte = False
        saw_idat = False
        saw_iend = False
        idat_ended = False
        idat_chunks = 0
        chunk_index = 0
        while True:
            raw_length = source.read(4)
            if not raw_length:
                break
            if len(raw_length) != 4:
                raise FileInspectionError("Truncated PNG chunk length")
            length = struct.unpack(">I", raw_length)[0]
            chunk_type = source.read(4)
            if len(chunk_type) != 4 or length > file_size - source.tell() - 4:
                raise FileInspectionError("Truncated PNG chunk")
            data = source.read(length)
            raw_crc = source.read(4)
            if len(data) != length or len(raw_crc) != 4:
                raise FileInspectionError("Truncated PNG chunk")
            expected_crc = struct.unpack(">I", raw_crc)[0]
            actual_crc = zlib.crc32(chunk_type)
            actual_crc = zlib.crc32(data, actual_crc) & 0xFFFFFFFF
            if expected_crc != actual_crc:
                raise FileInspectionError("PNG chunk CRC mismatch")
            if chunk_type == b"IHDR":
                if chunk_index != 0 or saw_ihdr:
                    raise FileInspectionError("PNG IHDR must be the first and only IHDR chunk")
                if length != 13:
                    raise FileInspectionError("Invalid PNG IHDR length")
                (
                    width,
                    height,
                    bit_depth,
                    color_type,
                    compression_method,
                    filter_method,
                    interlace_method,
                ) = struct.unpack(">IIBBBBB", data)
                if width <= 0 or height <= 0:
                    raise FileInspectionError("Invalid PNG dimensions")
                if color_type not in channels_by_type:
                    raise FileInspectionError(f"Unsupported PNG color type {color_type}")
                if bit_depth not in allowed_bit_depths[color_type]:
                    raise FileInspectionError(
                        f"Invalid PNG bit depth {bit_depth} for color type {color_type}"
                    )
                if compression_method != 0 or filter_method != 0:
                    raise FileInspectionError("Unsupported PNG compression or filter method")
                if interlace_method != 0:
                    raise FileInspectionError("Interlaced PNG is not supported by this inspector")
                row_bytes = math.ceil(
                    width * channels_by_type[color_type] * bit_depth / 8
                )
                expected_decoded_size = height * (row_bytes + 1)
                saw_ihdr = True
            elif not saw_ihdr:
                raise FileInspectionError("PNG IHDR must be the first chunk")
            elif chunk_type == b"PLTE":
                if saw_idat:
                    raise FileInspectionError("PNG PLTE must precede IDAT")
                if saw_plte:
                    raise FileInspectionError("PNG contains duplicate PLTE chunks")
                if color_type in {0, 4}:
                    raise FileInspectionError("PNG PLTE is invalid for grayscale color types")
                if length == 0 or length % 3 != 0 or length > 768:
                    raise FileInspectionError("Invalid PNG PLTE length")
                saw_plte = True
            elif chunk_type == b"IDAT":
                if idat_ended:
                    raise FileInspectionError("PNG IDAT chunks must be consecutive")
                if color_type == 3 and not saw_plte:
                    raise FileInspectionError("Palette PNG is missing PLTE before IDAT")
                saw_idat = True
                idat_chunks += 1
                try:
                    pending = data
                    while pending:
                        limit = max(
                            1,
                            int(expected_decoded_size) + 1 - len(decoded),
                        )
                        output = decompressor.decompress(pending, limit)
                        decoded.extend(output)
                        if len(decoded) > int(expected_decoded_size):
                            raise FileInspectionError(
                                "PNG image data exceeds declared dimensions"
                            )
                        remaining = decompressor.unconsumed_tail
                        if not remaining:
                            break
                        if remaining == pending and not output:
                            raise FileInspectionError("Unreadable PNG compressed image data")
                        pending = remaining
                except zlib.error as exc:
                    raise FileInspectionError(f"Unreadable PNG image data: {exc}") from exc
            elif chunk_type == b"IEND":
                if length != 0:
                    raise FileInspectionError("Invalid PNG IEND length")
                if not saw_idat:
                    raise FileInspectionError("PNG IEND appears before IDAT")
                saw_iend = True
                if source.read(1):
                    raise FileInspectionError("PNG contains trailing data after IEND")
                break
            else:
                if saw_idat:
                    idat_ended = True
                if 65 <= chunk_type[0] <= 90:
                    raise FileInspectionError(
                        f"Unsupported critical PNG chunk {chunk_type.decode('ascii', 'replace')}"
                    )
            chunk_index += 1
        if None in (width, height, bit_depth, color_type):
            raise FileInspectionError("PNG IHDR is missing")
        if not saw_idat or not saw_iend:
            raise FileInspectionError("PNG IDAT or IEND is missing")
        try:
            output = decompressor.flush(
                max(1, int(expected_decoded_size) + 1 - len(decoded))
            )
            decoded.extend(output)
        except zlib.error as exc:
            raise FileInspectionError(f"Unreadable PNG image data: {exc}") from exc
        if not decompressor.eof:
            raise FileInspectionError("Truncated PNG compressed image data")
        if decompressor.unused_data:
            raise FileInspectionError("PNG IDAT contains data after the compressed stream")
        if len(decoded) != expected_decoded_size:
            raise FileInspectionError("PNG image data does not match declared dimensions")
        row_bytes = math.ceil(width * channels_by_type[color_type] * bit_depth / 8)
        for row in range(height):
            if decoded[row * (row_bytes + 1)] not in {0, 1, 2, 3, 4}:
                raise FileInspectionError("PNG scanline uses an invalid filter type")
    return InspectedRaster(
        detected_format="PNG",
        width=int(width),
        height=int(height),
        channels=channels_by_type[color_type],
        bit_depth=int(bit_depth),
        color_mode=modes[color_type],
        storage_data_type=f"uint{bit_depth}" if bit_depth in {8, 16} else None,
        safe_details={
            "png_color_type": int(color_type),
            "png_compression_method": int(compression_method),
            "png_filter_method": int(filter_method),
            "png_interlace_method": int(interlace_method),
            "png_idat_chunk_count": idat_chunks,
            "chunk_crc_verified": True,
        },
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
    return InspectedRaster(
        "BMP",
        width,
        abs(signed_height),
        channels,
        bit_depth // channels,
        mode,
        f"uint{bit_depth // channels}",
    )


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
                return InspectedRaster(
                    "JPEG",
                    width,
                    height,
                    channels,
                    bit_depth,
                    mode,
                    f"uint{bit_depth}",
                )
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
    if signature[:2] in {b"II", b"MM"}:
        return _inspect_rgb_tiff(path)
    raise FileInspectionError(
        "Unsupported RGB image content; supported formats are PNG, BMP, JPEG, and TIFF"
    )


def _read_exact(source: BinaryIO, size: int, message: str) -> bytes:
    value = source.read(size)
    if len(value) != size:
        raise FileInspectionError(message)
    return value


def _tiff_values(
    source: BinaryIO,
    endian: str,
    field_type: int,
    count: int,
    raw_value: bytes,
    file_size: int,
) -> list[int]:
    sizes = {1: 1, 3: 2, 4: 4}
    formats = {1: "B", 3: "H", 4: "I"}
    if field_type not in sizes:
        raise FileInspectionError(f"Unsupported TIFF field type {field_type}")
    if count <= 0:
        raise FileInspectionError("TIFF field value count must be positive")
    byte_count = sizes[field_type] * count
    if byte_count <= 4:
        data = raw_value[:byte_count]
    else:
        offset = struct.unpack(endian + "I", raw_value)[0]
        if offset <= 0 or byte_count > file_size - offset:
            raise FileInspectionError("Truncated TIFF field value")
        position = source.tell()
        source.seek(offset)
        data = source.read(byte_count)
        source.seek(position)
        if len(data) != byte_count:
            raise FileInspectionError("Truncated TIFF field value")
    return list(struct.unpack(endian + formats[field_type] * count, data))


def _inspect_tiff(path: Path) -> InspectedRaster:
    with path.open("rb") as source:
        file_size = path.stat().st_size
        byte_order = source.read(2)
        if byte_order == b"II":
            endian = "<"
            byte_order_name = "LITTLE_ENDIAN"
        elif byte_order == b"MM":
            endian = ">"
            byte_order_name = "BIG_ENDIAN"
        else:
            raise FileInspectionError("Invalid TIFF byte order")
        magic = struct.unpack(
            endian + "H", _read_exact(source, 2, "Truncated TIFF header")
        )[0]
        if magic != 42:
            raise FileInspectionError("BigTIFF or invalid TIFF is unsupported")
        ifd_offset = struct.unpack(
            endian + "I", _read_exact(source, 4, "Truncated TIFF header")
        )[0]
        if ifd_offset < 8 or ifd_offset > file_size - 2:
            raise FileInspectionError("Invalid TIFF IFD offset")
        source.seek(ifd_offset)
        raw_count = _read_exact(source, 2, "Truncated TIFF IFD")
        entry_count = struct.unpack(endian + "H", raw_count)[0]
        tags: dict[int, list[int]] = {}
        tag_types: dict[int, int] = {}
        recognized_tags = {
            256,
            257,
            258,
            259,
            262,
            273,
            277,
            278,
            279,
            284,
            322,
            323,
            324,
            325,
            339,
        }
        for _ in range(entry_count):
            entry = _read_exact(source, 12, "Truncated TIFF IFD entry")
            tag, field_type, count = struct.unpack(endian + "HHI", entry[:8])
            if tag in recognized_tags:
                if tag in tags:
                    raise FileInspectionError(f"Duplicate TIFF tag {tag}")
                tags[tag] = _tiff_values(
                    source,
                    endian,
                    field_type,
                    count,
                    entry[8:12],
                    file_size,
                )
                tag_types[tag] = field_type
        next_ifd_offset = struct.unpack(
            endian + "I",
            _read_exact(source, 4, "Truncated TIFF next-IFD offset"),
        )[0]
        if next_ifd_offset != 0:
            raise FileInspectionError("Multi-page TIFF is unsupported")
        width = tags.get(256, [0])[0]
        height = tags.get(257, [0])[0]
        bits = tags.get(258, [0])
        samples = tags.get(277, [1])[0]
        compression = tags.get(259, [1])[0]
        photometric = tags.get(262, [-1])[0]
        sample_formats = tags.get(339, [1])
        sample_format = sample_formats[0]
        planar_configuration = tags.get(284, [1])[0]
        rows_per_strip = tags.get(278, [height])[0]
        if width <= 0 or height <= 0 or not bits or samples <= 0:
            raise FileInspectionError("TIFF dimensions or bit depth are missing")
        if tag_types.get(256) not in {3, 4} or tag_types.get(257) not in {3, 4}:
            raise FileInspectionError("TIFF dimensions use unsupported field types")
        if tag_types.get(258) != 3:
            raise FileInspectionError("TIFF BitsPerSample must use SHORT values")
        expected_tag_types = {
            259: {3},
            262: {3},
            273: {3, 4},
            277: {3},
            278: {3, 4},
            279: {3, 4},
            284: {3},
            339: {3},
        }
        for tag, allowed_types in expected_tag_types.items():
            if tag in tag_types and tag_types[tag] not in allowed_types:
                raise FileInspectionError(f"TIFF tag {tag} uses an invalid field type")
        if len(bits) not in {1, samples}:
            raise FileInspectionError(
                "TIFF BitsPerSample count contradicts SamplesPerPixel"
            )
        if any(bit != bits[0] for bit in bits):
            raise FileInspectionError("Mixed TIFF sample bit depths are unsupported")
        if 339 in tags and len(sample_formats) != samples:
            raise FileInspectionError(
                "TIFF SampleFormat count contradicts SamplesPerPixel"
            )
        if any(value != sample_format for value in sample_formats):
            raise FileInspectionError("Mixed TIFF sample formats are unsupported")
        if planar_configuration != 1:
            raise FileInspectionError(
                f"Unsupported TIFF planar configuration {planar_configuration}"
            )
        if any(tag in tags for tag in {322, 323, 324, 325}):
            raise FileInspectionError("Tiled TIFF is not supported by this validator version")
        if photometric == 3:
            raise FileInspectionError("Palette TIFF is unsupported for native RGB or height data")
        strip_offsets = tags.get(273)
        strip_counts = tags.get(279)
        if not strip_offsets or not strip_counts or len(strip_offsets) != len(strip_counts):
            raise FileInspectionError("TIFF strip offsets/byte counts are missing")
        if rows_per_strip <= 0:
            raise FileInspectionError("TIFF RowsPerStrip must be positive")
        expected_strip_count = math.ceil(height / rows_per_strip)
        if len(strip_offsets) != expected_strip_count:
            raise FileInspectionError(
                "TIFF strip count contradicts image height and RowsPerStrip"
            )
        if compression not in {1, 8, 32946}:
            raise FileInspectionError(f"Unsupported TIFF compression {compression}")
        decoded_size = 0
        row_bytes = math.ceil(width * samples * bits[0] / 8)
        for strip_index, (offset, byte_count) in enumerate(
            zip(strip_offsets, strip_counts)
        ):
            if offset <= 0 or byte_count <= 0 or byte_count > file_size - offset:
                raise FileInspectionError("Truncated TIFF strip")
            source.seek(offset)
            data = source.read(byte_count)
            if len(data) != byte_count:
                raise FileInspectionError("Truncated TIFF strip")
            if compression in {8, 32946}:
                try:
                    data = zlib.decompress(data)
                except zlib.error as exc:
                    raise FileInspectionError(f"Unreadable TIFF Deflate strip: {exc}") from exc
            strip_start_row = strip_index * rows_per_strip
            strip_rows = min(rows_per_strip, height - strip_start_row)
            expected_strip_size = row_bytes * strip_rows
            if len(data) != expected_strip_size:
                raise FileInspectionError(
                    "TIFF strip data size contradicts declared raster metadata"
                )
            decoded_size += len(data)
        expected_size = row_bytes * height
        if decoded_size != expected_size:
            raise FileInspectionError("TIFF pixel data is truncated")
    kind = {1: "uint", 2: "int", 3: "float"}.get(sample_format)
    if kind is None or bits[0] not in {8, 16, 32, 64}:
        raise FileInspectionError("Unsupported TIFF sample format or bit depth")
    dtype = f"{kind}{bits[0]}"
    mode = "SCALAR" if samples == 1 else "MULTI_SAMPLE"
    compression_name = "UNCOMPRESSED" if compression == 1 else "DEFLATE"
    photometric_name = {
        0: "WHITE_IS_ZERO",
        1: "BLACK_IS_ZERO",
        2: "RGB",
    }.get(photometric, "UNSUPPORTED_OR_UNSPECIFIED")
    return InspectedRaster(
        "TIFF",
        width,
        height,
        samples,
        bits[0],
        mode,
        dtype,
        safe_details={
            "tiff_byte_order": byte_order_name,
            "tiff_compression": compression_name,
            "tiff_compression_code": compression,
            "tiff_photometric": photometric_name,
            "tiff_photometric_code": photometric,
            "tiff_planar_configuration": planar_configuration,
            "tiff_rows_per_strip": rows_per_strip,
            "tiff_strip_count": len(strip_offsets),
            "tiff_sample_format": sample_format,
            "tiff_bits_per_sample_count": len(bits),
            "tiff_sample_format_count": len(sample_formats),
        },
    )


def _inspect_rgb_tiff(path: Path) -> InspectedRaster:
    inspected = _inspect_tiff(path)
    details = inspected.safe_details
    photometric = details["tiff_photometric_code"]
    if inspected.storage_data_type not in {"uint8", "uint16"}:
        raise FileInspectionError(
            "RGB TIFF samples must use unsigned 8-bit or 16-bit storage"
        )
    if inspected.channels == 1:
        if photometric not in {0, 1}:
            raise FileInspectionError(
                "Grayscale TIFF requires WhiteIsZero or BlackIsZero photometric interpretation"
            )
        warnings = (
            ("TIFF_WHITE_IS_ZERO_VALUES_ARE_NOT_INVERTED",)
            if photometric == 0
            else ()
        )
        return replace(inspected, color_mode="GRAY", warnings=warnings)
    if inspected.channels == 3:
        if photometric != 2:
            raise FileInspectionError(
                "Three-sample RGB TIFF requires RGB photometric interpretation"
            )
        if details["tiff_bits_per_sample_count"] != 3:
            raise FileInspectionError(
                "RGB TIFF BitsPerSample must declare one value per sample"
            )
        return replace(inspected, color_mode="RGB")
    raise FileInspectionError("RGB TIFF must contain one grayscale or three RGB samples")


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
    return InspectedRaster(
        "NPY",
        shape[1],
        shape[0],
        1,
        bit_depth,
        "SCALAR",
        dtype,
    )


def _inspect_height_png(path: Path) -> InspectedRaster:
    inspected = _inspect_png(path)
    color_type = inspected.safe_details["png_color_type"]
    if color_type != 0 or inspected.channels != 1:
        raise FileInspectionError(
            "Native height PNG must use grayscale color type 0 with one channel"
        )
    if inspected.bit_depth != 16:
        raise FileInspectionError("Native height PNG must use 16-bit grayscale storage")
    return replace(
        inspected,
        color_mode="SCALAR",
        storage_data_type="uint16",
    )


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
        return _inspect_height_png(path)
    raise FileInspectionError("Unsupported native height/depth file content")


def rgb_color_space_compatible(declared: str, inspected: InspectedRaster) -> bool:
    if declared == "GRAY":
        return inspected.mode == "GRAY" and inspected.channels == 1
    if declared == "RGB":
        return inspected.mode == "RGB" and inspected.channels == 3
    if declared.startswith("BAYER_"):
        return inspected.mode == "GRAY" and inspected.channels == 1
    return False


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _decoded_png_bytes(path: Path, inspected: InspectedRaster) -> bytes:
    """Extract and unfilter samples only after the authoritative PNG inspection."""

    compressed = bytearray()
    with path.open("rb") as source:
        if source.read(8) != b"\x89PNG\r\n\x1a\n":
            raise FileInspectionError("Invalid PNG signature")
        while True:
            raw_length = source.read(4)
            if not raw_length:
                break
            length = struct.unpack(">I", raw_length)[0]
            kind = _read_exact(source, 4, "Truncated PNG chunk")
            data = _read_exact(source, length, "Truncated PNG chunk")
            _read_exact(source, 4, "Truncated PNG chunk")
            if kind == b"IDAT":
                compressed.extend(data)
            if kind == b"IEND":
                break
    try:
        scanlines = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise FileInspectionError("Unreadable PNG image data") from exc
    bytes_per_sample = inspected.bit_depth // 8
    if bytes_per_sample not in {1, 2}:
        raise FileInspectionError("PNG value decoding requires 8-bit or 16-bit samples")
    bytes_per_pixel = inspected.channels * bytes_per_sample
    row_bytes = inspected.width * bytes_per_pixel
    if len(scanlines) != inspected.height * (row_bytes + 1):
        raise FileInspectionError("PNG image data does not match declared dimensions")
    output = bytearray()
    previous = bytes(row_bytes)
    for row_index in range(inspected.height):
        start = row_index * (row_bytes + 1)
        filter_type = scanlines[start]
        encoded = scanlines[start + 1 : start + 1 + row_bytes]
        decoded = bytearray(row_bytes)
        for index, value in enumerate(encoded):
            left = decoded[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            above = previous[index]
            upper_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 0:
                decoded[index] = value
            elif filter_type == 1:
                decoded[index] = (value + left) & 0xFF
            elif filter_type == 2:
                decoded[index] = (value + above) & 0xFF
            elif filter_type == 3:
                decoded[index] = (value + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                decoded[index] = (value + _paeth(left, above, upper_left)) & 0xFF
            else:
                raise FileInspectionError("PNG scanline uses an invalid filter type")
        output.extend(decoded)
        previous = bytes(decoded)
    return bytes(output)


def _decoded_tiff_bytes(path: Path, inspected: InspectedRaster) -> tuple[str, bytes]:
    """Read strips from the already-inspected classic contiguous TIFF subset."""

    with path.open("rb") as source:
        file_size = path.stat().st_size
        marker = _read_exact(source, 2, "Truncated TIFF header")
        endian = "<" if marker == b"II" else ">" if marker == b"MM" else ""
        if not endian:
            raise FileInspectionError("Invalid TIFF byte order")
        _read_exact(source, 2, "Truncated TIFF header")
        ifd_offset = struct.unpack(
            endian + "I", _read_exact(source, 4, "Truncated TIFF header")
        )[0]
        source.seek(ifd_offset)
        entry_count = struct.unpack(
            endian + "H", _read_exact(source, 2, "Truncated TIFF IFD")
        )[0]
        tags: dict[int, list[int]] = {}
        for _ in range(entry_count):
            entry = _read_exact(source, 12, "Truncated TIFF IFD entry")
            tag, field_type, count = struct.unpack(endian + "HHI", entry[:8])
            if tag in {259, 273, 279}:
                tags[tag] = _tiff_values(
                    source, endian, field_type, count, entry[8:12], file_size
                )
        if tags.get(259, [1])[0] != 1:
            raise FileInspectionError(
                "Synthetic value decoding supports uncompressed classic TIFF only"
            )
        offsets = tags.get(273, [])
        counts = tags.get(279, [])
        if not offsets or len(offsets) != len(counts):
            raise FileInspectionError("TIFF strip offsets/byte counts are missing")
        data = bytearray()
        for offset, count in zip(offsets, counts):
            source.seek(offset)
            data.extend(_read_exact(source, count, "Truncated TIFF strip"))
    expected = (
        inspected.width
        * inspected.height
        * inspected.channels
        * (inspected.bit_depth // 8)
    )
    if len(data) != expected:
        raise FileInspectionError("TIFF pixel data is truncated")
    return endian, bytes(data)


def _decoded_npy_float32(path: Path, inspected: InspectedRaster) -> tuple[float, ...]:
    with path.open("rb") as source:
        if source.read(6) != b"\x93NUMPY":
            raise FileInspectionError("Invalid NPY signature")
        version = tuple(_read_exact(source, 2, "Truncated NPY header"))
        if version[0] == 1:
            header_length = struct.unpack("<H", _read_exact(source, 2, "Truncated NPY header"))[0]
        elif version[0] in {2, 3}:
            header_length = struct.unpack("<I", _read_exact(source, 4, "Truncated NPY header"))[0]
        else:
            raise FileInspectionError("Unsupported NPY version")
        try:
            metadata = ast.literal_eval(
                _read_exact(source, header_length, "Truncated NPY header")
                .decode("latin1")
                .strip()
            )
        except (SyntaxError, ValueError) as exc:
            raise FileInspectionError("Invalid NPY header") from exc
        if metadata.get("fortran_order") is not False:
            raise FileInspectionError("Fortran-order NPY height arrays are unsupported")
        descriptor = metadata.get("descr")
        if descriptor not in {"<f4", ">f4"}:
            raise FileInspectionError("Synthetic NPY height decoding requires float32")
        count = inspected.width * inspected.height
        payload = source.read()
    if len(payload) != count * 4:
        raise FileInspectionError("NPY array data size contradicts declared shape")
    endian = "<" if descriptor == "<f4" else ">"
    return tuple(struct.unpack(endian + "f" * count, payload))


def decode_rgb_values(path: Path) -> DecodedRaster:
    """Decode only generated RGB PNG and classic RGB TIFF sample values."""

    inspected = inspect_rgb(path)
    if inspected.color_mode != "RGB" or inspected.channels != 3:
        raise FileInspectionError("Synthetic RGB decoding requires three RGB channels")
    if inspected.bit_depth not in {8, 16}:
        raise FileInspectionError("Synthetic RGB decoding requires 8-bit or 16-bit samples")
    if inspected.detected_format == "PNG":
        raw = _decoded_png_bytes(path, inspected)
        if inspected.bit_depth == 8:
            values: tuple[int | float, ...] = tuple(raw)
        else:
            count = inspected.width * inspected.height * inspected.channels
            values = tuple(struct.unpack(">" + "H" * count, raw))
    elif inspected.detected_format == "TIFF":
        endian, raw = _decoded_tiff_bytes(path, inspected)
        count = inspected.width * inspected.height * inspected.channels
        code = "B" if inspected.bit_depth == 8 else "H"
        values = tuple(struct.unpack(endian + code * count, raw))
    else:
        raise FileInspectionError("Synthetic RGB value format is unsupported")
    return DecodedRaster(inspected, values)


def decode_height_values(path: Path) -> DecodedRaster:
    """Decode only generated scalar uint16 TIFF/PNG and float32 NPY values."""

    inspected = inspect_height(path)
    if inspected.channels != 1:
        raise FileInspectionError("Synthetic height decoding requires one scalar channel")
    count = inspected.width * inspected.height
    if inspected.detected_format == "PNG":
        if inspected.storage_data_type != "uint16":
            raise FileInspectionError("Synthetic height PNG decoding requires uint16")
        raw = _decoded_png_bytes(path, inspected)
        values: tuple[int | float, ...] = tuple(struct.unpack(">" + "H" * count, raw))
    elif inspected.detected_format == "TIFF":
        if inspected.storage_data_type != "uint16":
            raise FileInspectionError("Synthetic height TIFF decoding requires uint16")
        endian, raw = _decoded_tiff_bytes(path, inspected)
        values = tuple(struct.unpack(endian + "H" * count, raw))
    elif inspected.detected_format == "NPY":
        if inspected.storage_data_type != "float32":
            raise FileInspectionError("Synthetic height NPY decoding requires float32")
        values = _decoded_npy_float32(path, inspected)
    else:
        raise FileInspectionError("Synthetic height value format is unsupported")
    return DecodedRaster(inspected, values)
