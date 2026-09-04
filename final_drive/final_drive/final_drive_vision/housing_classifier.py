"""Classify the four lamps inside a detected traffic-light housing.

The YOLO model used by final_drive finds one ``traffic_light`` housing.  This
module deliberately keeps the second stage small and explainable: the detected
box is divided into the fixed R/Y/L/G lamp order and the brightest 20 percent
of each circular lamp sample is measured.  The camera clips illuminated LEDs
close to white, so brightness and slot position are more reliable than hue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .detectors import (
    TRAFFIC_GREEN,
    TRAFFIC_GREEN_LEFT,
    TRAFFIC_NONE,
    TRAFFIC_RED,
    TRAFFIC_YELLOW,
)


SLOT_LABELS = ("R", "Y", "L", "G")


@dataclass(frozen=True)
class LampSlotMeasurement:
    """Brightness evidence for one physical lamp slot."""

    label: str
    center: tuple[float, float]
    radius: float
    bright_mean: float
    bright_fraction: float
    active: bool


@dataclass(frozen=True)
class HousingClassification:
    """State result and measurements retained for visual inspection."""

    state: int
    confidence: float
    bbox: tuple[int, int, int, int]
    slots: tuple[LampSlotMeasurement, ...]


class TrafficHousingStateClassifier:
    """Classify an already detected horizontal four-lamp housing.

    Thresholds were selected on the v2 training split and then checked against
    all 70 positive validation crops produced by the YOLO detector.  A single
    OFF frame can resemble a dim red lamp because of a ceiling reflection, so
    the ROS node must still require two consecutive state observations.
    """

    def __init__(
        self,
        active_bright_mean: float = 240.0,
        red_bright_mean: float = 215.0,
        red_inactive_margin: float = 30.0,
        red_yellow_spill_margin: float = 15.0,
        sample_radius_fraction: float = 0.38,
        top_fraction: float = 0.20,
        saturated_value: int = 240,
    ) -> None:
        if not 0.1 <= sample_radius_fraction <= 0.5:
            raise ValueError("sample_radius_fraction must be within [0.1, 0.5]")
        if not 0.05 <= top_fraction <= 0.5:
            raise ValueError("top_fraction must be within [0.05, 0.5]")
        self.active_bright_mean = float(active_bright_mean)
        self.red_bright_mean = float(red_bright_mean)
        self.red_inactive_margin = float(red_inactive_margin)
        self.red_yellow_spill_margin = float(red_yellow_spill_margin)
        self.sample_radius_fraction = float(sample_radius_fraction)
        self.top_fraction = float(top_fraction)
        self.saturated_value = int(saturated_value)

    @staticmethod
    def _clip_bbox(
        bbox: Sequence[float], frame_shape: Sequence[int]
    ) -> tuple[int, int, int, int]:
        if len(bbox) != 4:
            raise ValueError("bbox must contain x, y, width, height")
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        x, y, width, height = (float(value) for value in bbox)
        x0 = int(np.clip(np.floor(x), 0, frame_width - 1))
        y0 = int(np.clip(np.floor(y), 0, frame_height - 1))
        x1 = int(np.clip(np.ceil(x + width), x0 + 1, frame_width))
        y1 = int(np.clip(np.ceil(y + height), y0 + 1, frame_height))
        return x0, y0, x1 - x0, y1 - y0

    def measure(
        self, frame_bgr: np.ndarray, bbox: Sequence[float]
    ) -> tuple[tuple[int, int, int, int], tuple[LampSlotMeasurement, ...]]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame must not be empty")
        clipped_bbox = self._clip_bbox(bbox, frame_bgr.shape)
        x, y, width, height = clipped_bbox
        slot_width = width / 4.0
        radius = self.sample_radius_fraction * min(slot_width, float(height))
        radius = max(radius, 1.0)
        value_channel = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        measurements: list[LampSlotMeasurement] = []

        for index, label in enumerate(SLOT_LABELS):
            center_x = x + (index + 0.5) * slot_width
            center_y = y + 0.5 * height
            sample_x0 = max(0, int(np.floor(center_x - radius)))
            sample_y0 = max(0, int(np.floor(center_y - radius)))
            sample_x1 = min(frame_bgr.shape[1], int(np.ceil(center_x + radius + 1.0)))
            sample_y1 = min(frame_bgr.shape[0], int(np.ceil(center_y + radius + 1.0)))
            yy, xx = np.ogrid[sample_y0:sample_y1, sample_x0:sample_x1]
            disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
            values = value_channel[sample_y0:sample_y1, sample_x0:sample_x1][disk]
            if values.size == 0:
                bright_mean = 0.0
                bright_fraction = 0.0
            else:
                values_float = values.astype(np.float32)
                top_count = max(1, int(np.ceil(values_float.size * self.top_fraction)))
                brightest = np.partition(values_float, -top_count)[-top_count:]
                bright_mean = float(brightest.mean())
                bright_fraction = float(np.mean(values >= self.saturated_value))
            measurements.append(
                LampSlotMeasurement(
                    label=label,
                    center=(float(center_x), float(center_y)),
                    radius=float(radius),
                    bright_mean=bright_mean,
                    bright_fraction=bright_fraction,
                    active=bright_mean >= self.active_bright_mean,
                )
            )
        return clipped_bbox, tuple(measurements)

    def classify(
        self, frame_bgr: np.ndarray, bbox: Sequence[float]
    ) -> HousingClassification:
        clipped_bbox, slots = self.measure(frame_bgr, bbox)
        values = np.asarray([slot.bright_mean for slot in slots], dtype=np.float32)

        # LEFT+GREEN is checked first because both right-hand slots are active.
        if values[2] >= self.active_bright_mean and values[3] >= self.active_bright_mean:
            state = TRAFFIC_GREEN_LEFT
            evidence = min(values[2], values[3]) - self.active_bright_mean
            confidence = 0.70 + evidence / 70.0
        # A red lamp can be dimmer than the other LEDs and can spill into the
        # adjacent yellow sample.  The two rightmost inactive lamps provide the
        # independent contrast check.
        elif (
            values[0] >= self.red_bright_mean
            and values[0] - max(values[2], values[3]) >= self.red_inactive_margin
            and values[1] - values[0] <= self.red_yellow_spill_margin
        ):
            state = TRAFFIC_RED
            contrast = values[0] - max(values[2], values[3])
            evidence = min(
                values[0] - self.red_bright_mean,
                contrast - self.red_inactive_margin,
            )
            confidence = 0.55 + evidence / 70.0
        elif int(np.argmax(values)) == 1 and values[1] >= self.active_bright_mean:
            state = TRAFFIC_YELLOW
            evidence = values[1] - max(values[0], values[2], values[3])
            confidence = 0.65 + evidence / 100.0
        elif int(np.argmax(values)) == 3 and values[3] >= self.active_bright_mean:
            state = TRAFFIC_GREEN
            evidence = values[3] - max(values[0], values[1], values[2])
            confidence = 0.65 + evidence / 100.0
        else:
            state = TRAFFIC_NONE
            evidence = self.active_bright_mean - float(values.max())
            confidence = 0.55 + evidence / 80.0

        return HousingClassification(
            state=state,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            bbox=clipped_bbox,
            slots=slots,
        )
