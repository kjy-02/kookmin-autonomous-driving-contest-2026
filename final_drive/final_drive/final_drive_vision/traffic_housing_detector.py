"""OpenCV-DNN YOLO housing detector followed by fixed-slot classification."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .detectors import DetectionRoi, TrafficDetection
from .housing_classifier import TrafficHousingStateClassifier


class TrafficHousingDetector:
    """Detect the single four-lamp housing and classify its R/Y/L/G slots."""

    def __init__(
        self,
        model_path: str | Path,
        roi: DetectionRoi | None = None,
        max_y_fraction: float = 0.50,
        confidence_threshold: float = 0.35,
        input_size: int = 640,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"traffic housing model not found: {path}")
        if not 0.0 < max_y_fraction <= 1.0:
            raise ValueError("max_y_fraction must be within (0, 1]")
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be within (0, 1)")
        if input_size < 32 or input_size % 32:
            raise ValueError("input_size must be a multiple of 32")

        self.model_path = path
        self.roi = roi
        self.max_y_fraction = float(max_y_fraction)
        self.confidence_threshold = float(confidence_threshold)
        self.input_size = int(input_size)
        self._network = cv2.dnn.readNetFromONNX(str(path))
        self._network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._classifier = TrafficHousingStateClassifier()
        self.last_debug: dict[str, object] = {"detector": "yolo_housing"}

    def active_bounds(self, frame_shape: Sequence[int]) -> tuple[int, int, int, int]:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if self.roi is None:
            x0, y0, x1, y1 = 0, 0, width, height
        else:
            x0, y0, x1, y1 = self.roi.bounds(frame_shape)
        y1 = min(y1, int(round(height * self.max_y_fraction)))
        return x0, y0, x1, max(y0 + 1, y1)

    def _letterbox(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        height, width = frame_bgr.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            frame_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        canvas[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _prediction_rows(output: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        if isinstance(output, (list, tuple)):
            if not output:
                return np.empty((0, 5), dtype=np.float32)
            output = output[0]
        rows = np.asarray(output)
        if rows.ndim == 3:
            rows = rows[0]
        if rows.ndim != 2:
            raise ValueError(f"unexpected YOLO output shape: {rows.shape}")
        # YOLOv8 detection exports [channels, anchors].  Some OpenCV builds
        # return the transposed [anchors, channels] form.
        if rows.shape[0] <= 16 and rows.shape[1] > rows.shape[0]:
            rows = rows.T
        if rows.shape[1] < 5:
            raise ValueError(f"YOLO output has fewer than five channels: {rows.shape}")
        return rows

    def _best_bbox(
        self,
        output: np.ndarray | Sequence[np.ndarray],
        frame_shape: Sequence[int],
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> tuple[tuple[float, float, float, float], float, int] | None:
        rows = self._prediction_rows(output)
        x_min, y_min, x_max, y_max = self.active_bounds(frame_shape)
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        best: tuple[tuple[float, float, float, float], float, int] | None = None
        candidate_count = 0
        for row in rows:
            confidence = float(row[4])
            if confidence < self.confidence_threshold:
                continue
            center_x, center_y, width, height = (float(value) for value in row[:4])
            box_x = (center_x - 0.5 * width - pad_x) / scale
            box_y = (center_y - 0.5 * height - pad_y) / scale
            box_width = width / scale
            box_height = height / scale
            box_x1 = box_x + box_width
            box_y1 = box_y + box_height
            if box_width < 4.0 or box_height < 4.0:
                continue
            if box_x < x_min or box_y < y_min or box_x1 > x_max or box_y1 > y_max:
                continue
            if box_x1 <= 0.0 or box_y1 <= 0.0:
                continue
            if box_x >= frame_width or box_y >= frame_height:
                continue
            candidate_count += 1
            bbox = (box_x, box_y, box_width, box_height)
            if best is None or confidence > best[1]:
                best = (bbox, confidence, candidate_count)
        if best is None:
            return None
        return best[0], best[1], candidate_count

    def detect(self, frame_bgr: np.ndarray) -> TrafficDetection:
        if frame_bgr is None or frame_bgr.size == 0:
            return TrafficDetection()
        letterboxed, scale, pad_x, pad_y = self._letterbox(frame_bgr)
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        self._network.setInput(blob)
        output = self._network.forward()
        best = self._best_bbox(
            output,
            frame_bgr.shape,
            scale,
            pad_x,
            pad_y,
        )
        if best is None:
            self.last_debug = {
                "detector": "yolo_housing",
                "candidate_count": 0,
                "active_bounds": self.active_bounds(frame_bgr.shape),
            }
            return TrafficDetection()

        bbox, housing_confidence, candidate_count = best
        classified = self._classifier.classify(frame_bgr, bbox)
        self.last_debug = {
            "detector": "yolo_housing",
            "candidate_count": candidate_count,
            "housing_confidence": housing_confidence,
            "state_confidence": classified.confidence,
            "slot_brightness": tuple(slot.bright_mean for slot in classified.slots),
            "active_bounds": self.active_bounds(frame_bgr.shape),
        }
        return TrafficDetection(
            state=classified.state,
            confidence=housing_confidence,
            bbox=classified.bbox,
            slot_centers=tuple(slot.center for slot in classified.slots),
            radius=float(classified.slots[0].radius),
        )
