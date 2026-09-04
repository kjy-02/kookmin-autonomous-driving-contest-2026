"""Small ROS 2 sqlite bag reader used by the offline tools.

Both ``sensor_msgs/msg/Image`` and ``sensor_msgs/msg/CompressedImage`` are
decoded.  Keeping this reader free of ROS imports lets the recorded data be
evaluated on the Windows development machine as well as on the vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import struct
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class BagFrame:
    index: int
    timestamp_ns: int
    frame_bgr: np.ndarray


class _CdrReader:
    def __init__(self, payload: bytes):
        if len(payload) < 4:
            raise ValueError("CDR payload is too short")
        # 0x0001 is the little-endian CDR encapsulation used by these bags.
        self._endian = "<" if payload[1] == 1 else ">"
        self._data = payload
        self._offset = 4

    def _align(self, alignment: int) -> None:
        self._offset = (self._offset + alignment - 1) & ~(alignment - 1)

    def u8(self) -> int:
        value = self._data[self._offset]
        self._offset += 1
        return value

    def u32(self) -> int:
        self._align(4)
        value = struct.unpack_from(self._endian + "I", self._data, self._offset)[0]
        self._offset += 4
        return value

    def i32(self) -> int:
        self._align(4)
        value = struct.unpack_from(self._endian + "i", self._data, self._offset)[0]
        self._offset += 4
        return value

    def string(self) -> str:
        length = self.u32()
        raw = self._data[self._offset : self._offset + length]
        self._offset += length
        return raw.rstrip(b"\0").decode("utf-8")

    def byte_sequence(self) -> memoryview:
        length = self.u32()
        start = self._offset
        end = start + length
        if end > len(self._data):
            raise ValueError("CDR byte sequence exceeds the message payload")
        self._offset = end
        return memoryview(self._data)[start:end]


def decode_image_message(payload: bytes) -> np.ndarray:
    reader = _CdrReader(payload)
    reader.i32()  # header.stamp.sec
    reader.u32()  # header.stamp.nanosec
    reader.string()  # header.frame_id
    height = reader.u32()
    width = reader.u32()
    encoding = reader.string().lower()
    is_bigendian = reader.u8()
    step = reader.u32()
    image_data = reader.byte_sequence()

    if is_bigendian:
        raise ValueError("Big-endian image messages are not supported")

    if encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding in {"mono8", "8uc1"}:
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")

    required = int(step) * int(height)
    if len(image_data) < required:
        raise ValueError("Image data is shorter than height * step")

    packed_width = int(width) * channels
    rows = np.frombuffer(image_data, dtype=np.uint8, count=required).reshape(height, step)
    packed = rows[:, :packed_width]

    if channels == 1:
        return cv2.cvtColor(packed.reshape(height, width), cv2.COLOR_GRAY2BGR)

    image = packed.reshape(height, width, channels)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def decode_compressed_image_message(payload: bytes) -> np.ndarray:
    reader = _CdrReader(payload)
    reader.i32()  # header.stamp.sec
    reader.u32()  # header.stamp.nanosec
    reader.string()  # header.frame_id
    reader.string()  # format, e.g. "jpeg"
    compressed = reader.byte_sequence()
    image = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the compressed image")
    return image


def _decode_message(payload: bytes, message_type: str) -> np.ndarray:
    if message_type == "sensor_msgs/msg/Image":
        return decode_image_message(payload)
    if message_type == "sensor_msgs/msg/CompressedImage":
        return decode_compressed_image_message(payload)
    raise ValueError(f"Unsupported image message type: {message_type}")


class RosbagImageReader:
    def __init__(self, bag_path: str | Path, topic: str = "/image_raw"):
        source = Path(bag_path)
        if source.is_dir():
            databases = sorted(source.glob("*.db3"))
            if len(databases) != 1:
                raise ValueError(f"Expected one db3 file in {source}, found {len(databases)}")
            source = databases[0]
        if not source.is_file():
            raise FileNotFoundError(source)
        self.database_path = source
        self.topic = topic

    def frame_count(self) -> int:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM messages AS m
                JOIN topics AS t ON t.id = m.topic_id
                WHERE t.name = ?
                """,
                (self.topic,),
            ).fetchone()
        return int(row[0])

    def frames(self, stride: int = 1) -> Iterator[BagFrame]:
        if stride < 1:
            raise ValueError("stride must be at least 1")
        connection = sqlite3.connect(self.database_path)
        try:
            cursor = connection.execute(
                """
                SELECT m.timestamp, m.data, t.type
                FROM messages AS m
                JOIN topics AS t ON t.id = m.topic_id
                WHERE t.name = ?
                ORDER BY m.timestamp
                """,
                (self.topic,),
            )
            output_index = 0
            for source_index, (timestamp, payload, message_type) in enumerate(cursor):
                if source_index % stride != 0:
                    continue
                yield BagFrame(
                    index=source_index,
                    timestamp_ns=int(timestamp),
                    frame_bgr=_decode_message(payload, str(message_type)),
                )
                output_index += 1
        finally:
            connection.close()
