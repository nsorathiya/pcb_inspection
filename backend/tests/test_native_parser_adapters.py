import struct
import zlib
from pathlib import Path

import pytest

from app.services.dataset_validation.file_inspection import (
    FileInspectionError,
    inspect_height,
    inspect_rgb,
)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _png_bytes(
    width: int = 3,
    height: int = 2,
    *,
    bit_depth: int = 16,
    color_type: int = 0,
    interlace: int = 0,
    compressed_data: bytes | None = None,
) -> bytes:
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    row_bytes = (width * channels * bit_depth + 7) // 8
    raw = b"".join(b"\x00" + bytes(row_bytes) for _ in range(height))
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        color_type,
        0,
        0,
        interlace,
    )
    chunks = [_png_chunk(b"IHDR", ihdr)]
    if color_type == 3:
        chunks.append(_png_chunk(b"PLTE", b"\x00\x00\x00\xff\xff\xff"))
    chunks.extend(
        (
            _png_chunk(
                b"IDAT",
                zlib.compress(raw) if compressed_data is None else compressed_data,
            ),
            _png_chunk(b"IEND", b""),
        )
    )
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def _tiff_bytes(
    width: int = 3,
    height: int = 2,
    *,
    samples: int = 3,
    bits: int = 8,
    photometric: int = 2,
    compression: int = 1,
    byte_order: str = "little",
    planar_configuration: int = 1,
    bits_count: int | None = None,
    tiled: bool = False,
) -> bytes:
    endian = "<" if byte_order == "little" else ">"
    marker = b"II" if byte_order == "little" else b"MM"
    bits_values = [bits] * (samples if bits_count is None else bits_count)
    raw_pixels = bytes(width * height * samples * (bits // 8))
    strip_data = zlib.compress(raw_pixels) if compression in {8, 32946} else raw_pixels

    entries: list[tuple[int, int, list[int]]] = [
        (256, 4, [width]),
        (257, 4, [height]),
        (258, 3, bits_values),
        (259, 3, [compression]),
        (262, 3, [photometric]),
        (273, 4, [0]),
        (277, 3, [samples]),
        (278, 4, [height]),
        (279, 4, [len(strip_data)]),
        (284, 3, [planar_configuration]),
        (339, 3, [1] * samples),
    ]
    if tiled:
        entries.append((322, 4, [width]))
    entries.sort(key=lambda entry: entry[0])

    ifd_end = 8 + 2 + len(entries) * 12 + 4
    external = bytearray()
    external_offsets: dict[int, int] = {}
    for tag, field_type, values in entries:
        item_size = {3: 2, 4: 4}[field_type]
        if item_size * len(values) > 4:
            external_offsets[tag] = ifd_end + len(external)
            value_format = "H" if field_type == 3 else "I"
            external.extend(struct.pack(endian + value_format * len(values), *values))
    pixel_offset = ifd_end + len(external)
    entries = [
        (tag, field_type, [pixel_offset] if tag == 273 else values)
        for tag, field_type, values in entries
    ]

    output = bytearray(marker)
    output.extend(struct.pack(endian + "H", 42))
    output.extend(struct.pack(endian + "I", 8))
    output.extend(struct.pack(endian + "H", len(entries)))
    for tag, field_type, values in entries:
        output.extend(struct.pack(endian + "HHI", tag, field_type, len(values)))
        item_size = {3: 2, 4: 4}[field_type]
        if item_size * len(values) > 4:
            output.extend(struct.pack(endian + "I", external_offsets[tag]))
        else:
            value_format = "H" if field_type == 3 else "I"
            inline = struct.pack(endian + value_format * len(values), *values)
            output.extend(inline + bytes(4 - len(inline)))
    output.extend(struct.pack(endian + "I", 0))
    output.extend(external)
    output.extend(strip_data)
    return bytes(output)


def test_uncompressed_rgb_tiff_returns_typed_native_metadata(tmp_path) -> None:
    path = tmp_path / "rgb.tiff"
    path.write_bytes(_tiff_bytes(width=4, height=2))
    original = path.read_bytes()
    before_entries = tuple(tmp_path.iterdir())

    result = inspect_rgb(path)

    assert result.detected_format == "TIFF"
    assert result.format == "tiff"
    assert (result.width, result.height) == (4, 2)
    assert result.channels == 3
    assert result.bit_depth == 8
    assert result.color_mode == "RGB"
    assert result.storage_data_type == "uint8"
    assert result.readability_status == "READABLE"
    assert result.safe_details["tiff_photometric"] == "RGB"
    assert result.safe_details["tiff_compression"] == "UNCOMPRESSED"
    assert result.safe_details["tiff_strip_count"] == 1
    assert path.read_bytes() == original
    assert tuple(tmp_path.iterdir()) == before_entries
    assert str(tmp_path) not in repr(result)
    assert not any("path" in key or "filename" in key for key in result.safe_details)


def test_grayscale_tiff_is_allowed_and_white_is_zero_is_reported(tmp_path) -> None:
    path = tmp_path / "gray.tif"
    path.write_bytes(
        _tiff_bytes(samples=1, bits=16, photometric=0, byte_order="big")
    )

    result = inspect_rgb(path)

    assert result.channels == 1
    assert result.bit_depth == 16
    assert result.color_mode == "GRAY"
    assert result.storage_data_type == "uint16"
    assert result.safe_details["tiff_byte_order"] == "BIG_ENDIAN"
    assert result.warnings == ("TIFF_WHITE_IS_ZERO_VALUES_ARE_NOT_INVERTED",)


def test_big_endian_deflate_rgb_tiff_is_readable(tmp_path) -> None:
    path = tmp_path / "deflate-rgb.tiff"
    path.write_bytes(_tiff_bytes(compression=8, byte_order="big"))

    result = inspect_rgb(path)

    assert result.color_mode == "RGB"
    assert result.safe_details["tiff_byte_order"] == "BIG_ENDIAN"
    assert result.safe_details["tiff_compression"] == "DEFLATE"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (_tiff_bytes()[:-1], "Truncated TIFF strip"),
        (_tiff_bytes(tiled=True), "Tiled TIFF"),
        (b"II" + struct.pack("<H", 43) + bytes(12), "BigTIFF"),
        (
            _tiff_bytes(planar_configuration=2),
            "planar configuration",
        ),
        (
            _tiff_bytes(bits_count=2),
            "BitsPerSample count contradicts",
        ),
        (
            _tiff_bytes(samples=1, photometric=3),
            "Palette TIFF",
        ),
        (
            _tiff_bytes(compression=5),
            "Unsupported TIFF compression 5",
        ),
    ],
)
def test_unsupported_or_contradictory_rgb_tiff_is_rejected(
    tmp_path,
    content,
    message,
) -> None:
    path = tmp_path / "unsupported.tiff"
    path.write_bytes(content)

    with pytest.raises(FileInspectionError, match=message):
        inspect_rgb(path)


def test_scalar_16_bit_height_png_is_accepted_without_calibration_claims(
    tmp_path,
) -> None:
    path = tmp_path / "height.png"
    path.write_bytes(_png_bytes(width=5, height=3))
    original = path.read_bytes()
    before_entries = tuple(tmp_path.iterdir())

    result = inspect_height(path)

    assert result.detected_format == "PNG"
    assert result.format == "png"
    assert (result.width, result.height) == (5, 3)
    assert result.channels == 1
    assert result.bit_depth == 16
    assert result.color_mode == "SCALAR"
    assert result.storage_data_type == "uint16"
    assert result.readability_status == "READABLE"
    assert result.safe_details["png_color_type"] == 0
    assert result.safe_details["chunk_crc_verified"] is True
    assert not {
        "z_unit",
        "z_scale",
        "z_offset",
        "calibration",
        "registration",
        "no_data_value",
    }.intersection(result.safe_details)
    assert path.read_bytes() == original
    assert tuple(tmp_path.iterdir()) == before_entries
    assert str(tmp_path) not in repr(result)


@pytest.mark.parametrize(
    ("bit_depth", "color_type", "message"),
    [
        (8, 0, "16-bit grayscale"),
        (8, 2, "grayscale color type 0"),
        (8, 6, "grayscale color type 0"),
        (8, 3, "grayscale color type 0"),
        (8, 4, "grayscale color type 0"),
    ],
)
def test_non_scalar_or_8_bit_png_is_rejected_as_height(
    tmp_path,
    bit_depth,
    color_type,
    message,
) -> None:
    path = tmp_path / "not-native-height.png"
    path.write_bytes(_png_bytes(bit_depth=bit_depth, color_type=color_type))

    with pytest.raises(FileInspectionError, match=message):
        inspect_height(path)


def test_truncated_height_png_is_rejected(tmp_path) -> None:
    path = tmp_path / "truncated.png"
    path.write_bytes(_png_bytes()[:-3])

    with pytest.raises(FileInspectionError, match="Truncated PNG chunk"):
        inspect_height(path)


def test_png_crc_corruption_is_rejected(tmp_path) -> None:
    content = bytearray(_png_bytes())
    content[29] ^= 0x01
    path = tmp_path / "bad-crc.png"
    path.write_bytes(content)

    with pytest.raises(FileInspectionError, match="CRC mismatch"):
        inspect_height(path)


def test_invalid_png_compressed_stream_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad-stream.png"
    path.write_bytes(_png_bytes(compressed_data=b"not-a-zlib-stream"))

    with pytest.raises(FileInspectionError, match="Unreadable PNG image data"):
        inspect_height(path)


def test_interlaced_height_png_is_rejected(tmp_path) -> None:
    path = tmp_path / "interlaced.png"
    path.write_bytes(_png_bytes(interlace=1))

    with pytest.raises(FileInspectionError, match="Interlaced PNG"):
        inspect_height(path)


@pytest.mark.parametrize("iend_before_idat", [False, True])
def test_invalid_required_png_chunk_order_is_rejected(
    tmp_path,
    iend_before_idat,
) -> None:
    ihdr = _png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", 1, 1, 16, 0, 0, 0, 0),
    )
    idat = _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00"))
    iend = _png_chunk(b"IEND", b"")
    chunks = (ihdr, iend, idat) if iend_before_idat else (idat, ihdr, iend)
    path = tmp_path / "bad-order.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))

    with pytest.raises(FileInspectionError, match="IHDR|IEND appears before IDAT"):
        inspect_height(path)


def test_rgb_png_remains_rgb_only_and_is_not_accepted_as_height(tmp_path) -> None:
    path = tmp_path / "rgb.png"
    path.write_bytes(_png_bytes(bit_depth=8, color_type=2))

    rgb_result = inspect_rgb(path)

    assert rgb_result.color_mode == "RGB"
    assert rgb_result.channels == 3
    with pytest.raises(FileInspectionError, match="grayscale color type 0"):
        inspect_height(path)
