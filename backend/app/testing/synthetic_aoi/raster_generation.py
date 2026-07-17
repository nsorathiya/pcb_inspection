from __future__ import annotations

import hashlib
import struct
import zlib


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def deterministic_zlib_stream(data: bytes) -> bytes:
    """Return a zlib stream containing only deterministic stored Deflate blocks."""

    output = bytearray(b"\x78\x01")
    if not data:
        chunks = [b""]
    else:
        chunks = [data[index : index + 65535] for index in range(0, len(data), 65535)]
    for index, chunk in enumerate(chunks):
        output.append(1 if index == len(chunks) - 1 else 0)
        length = len(chunk)
        output.extend(struct.pack("<HH", length, length ^ 0xFFFF))
        output.extend(chunk)
    output.extend(struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF))
    return bytes(output)


def encode_png(
    width: int,
    height: int,
    *,
    bit_depth: int,
    color_type: int,
    pixel_bytes: bytes,
    interlace: int = 0,
    compressed_data: bytes | None = None,
) -> bytes:
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    row_bytes = (width * channels * bit_depth + 7) // 8
    if len(pixel_bytes) != row_bytes * height:
        raise ValueError("PNG pixel byte count does not match dimensions")
    scanlines = b"".join(
        b"\x00" + pixel_bytes[row * row_bytes : (row + 1) * row_bytes]
        for row in range(height)
    )
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
                deterministic_zlib_stream(scanlines)
                if compressed_data is None
                else compressed_data,
            ),
            _png_chunk(b"IEND", b""),
        )
    )
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def encode_classic_tiff(
    width: int,
    height: int,
    *,
    samples: int,
    bits: int,
    photometric: int,
    pixel_bytes: bytes,
    sample_format: int = 1,
    compression: int = 1,
    byte_order: str = "little",
    planar_configuration: int = 1,
    bits_count: int | None = None,
    tiled: bool = False,
) -> bytes:
    expected_size = width * height * samples * (bits // 8)
    if len(pixel_bytes) != expected_size:
        raise ValueError("TIFF pixel byte count does not match dimensions")
    endian = "<" if byte_order == "little" else ">"
    marker = b"II" if byte_order == "little" else b"MM"
    bits_values = [bits] * (samples if bits_count is None else bits_count)
    strip_data = (
        deterministic_zlib_stream(pixel_bytes)
        if compression in {8, 32946}
        else pixel_bytes
    )
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
        (339, 3, [sample_format] * samples),
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


def encode_npy_float32(width: int, height: int, values: tuple[float, ...]) -> bytes:
    if len(values) != width * height:
        raise ValueError("NPY value count does not match dimensions")
    prefix = b"\x93NUMPY\x01\x00"
    header = (
        "{'descr': '<f4', 'fortran_order': False, "
        f"'shape': ({height}, {width}), }}"
    ).encode("latin1")
    padding = (64 - ((len(prefix) + 2 + len(header) + 1) % 64)) % 64
    header_bytes = header + (b" " * padding) + b"\n"
    payload = struct.pack("<" + "f" * len(values), *values)
    return prefix + struct.pack("<H", len(header_bytes)) + header_bytes + payload


def _seed_byte(seed: int, scenario_id: str, label: str) -> int:
    material = f"{seed}:{scenario_id}:{label}".encode("utf-8")
    return hashlib.sha256(material).digest()[0]


def rgb_pattern(width: int, height: int, seed: int, scenario_id: str) -> bytes:
    variation = _seed_byte(seed, scenario_id, "rgb") % 31
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            color = (12 + variation, 18, 22)
            if 1 <= x < width - 1 and 1 <= y < height - 1:
                color = (18, 82 + variation, 34)
            if width // 4 <= x < width // 2 and height // 3 <= y < 2 * height // 3:
                color = (72 + variation, 72, 76)
            if 2 * width // 3 <= x < width - 2 and 2 <= y < max(3, height // 2):
                color = (124, 92 + variation, 34)
            if (x - 3) ** 2 + (y - 3) ** 2 <= 2:
                color = (232, 232, 220)
            pixels.extend(color)
    return bytes(pixels)


def height_uint16_values(
    width: int,
    height: int,
    seed: int,
    scenario_id: str,
) -> tuple[int, ...]:
    variation = _seed_byte(seed, scenario_id, "height")
    values: list[int] = []
    for y in range(height):
        for x in range(width):
            value = 1000 + variation
            if width // 4 <= x < width // 2 and height // 3 <= y < 2 * height // 3:
                value += 320
            elif 2 * width // 3 <= x < width - 1 and height // 2 <= y < height - 1:
                value -= 120
            values.append(value)
    return tuple(values)


def uint16_little_endian(values: tuple[int, ...]) -> bytes:
    return struct.pack("<" + "H" * len(values), *values)


def uint16_big_endian(values: tuple[int, ...]) -> bytes:
    return struct.pack(">" + "H" * len(values), *values)


def float32_height_values(
    width: int,
    height: int,
    seed: int,
    scenario_id: str,
) -> tuple[float, ...]:
    variation = _seed_byte(seed, scenario_id, "float-height") / 1024.0
    return tuple(
        1.0 + variation + (x * 0.03125) + (y * 0.0625)
        for y in range(height)
        for x in range(width)
    )
