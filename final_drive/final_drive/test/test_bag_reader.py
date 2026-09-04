from __future__ import annotations

import struct

import cv2
import numpy as np

from final_drive_vision.bag_reader import decode_compressed_image_message


def _align(payload: bytearray, alignment: int = 4) -> None:
    payload.extend(b"\0" * ((-len(payload)) % alignment))


def _write_string(payload: bytearray, value: str) -> None:
    _align(payload)
    encoded = value.encode("utf-8") + b"\0"
    payload.extend(struct.pack("<I", len(encoded)))
    payload.extend(encoded)


def test_decode_compressed_image_message() -> None:
    source = np.zeros((12, 16, 3), np.uint8)
    source[:, :, 1] = 180
    success, encoded = cv2.imencode(".jpg", source)
    assert success

    payload = bytearray(b"\x00\x01\x00\x00")
    payload.extend(struct.pack("<iI", 1, 2))
    _write_string(payload, "camera")
    _write_string(payload, "jpeg")
    _align(payload)
    payload.extend(struct.pack("<I", encoded.size))
    payload.extend(encoded.tobytes())

    decoded = decode_compressed_image_message(bytes(payload))
    assert decoded.shape == source.shape
    assert decoded[:, :, 1].mean() > 150
