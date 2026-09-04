"""Optional checkerboard detector kept for future course use.

The integrated drive disables this detector by default so traffic-light
processing is not delayed.  The implementation remains importable for the
offline viewer and for a later checkerboard mission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CheckerboardDetection:
    detected: bool = False
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None


class CheckerboardRoi(Protocol):
    """The ROI surface needed by :class:`CheckerboardDetector`."""

    def bounds(self, frame_shape: Sequence[int]) -> tuple[int, int, int, int]: ...


class CheckerboardDetector:
    """Detect the 4-by-2 alternating board placed below the traffic light."""

    def __init__(self, roi: CheckerboardRoi | None = None) -> None:
        self.roi = roi
        self.last_debug: dict[str, object] = {}

    @staticmethod
    def _sample_cell(
        gray: np.ndarray,
        point: np.ndarray,
        radius: float,
    ) -> float:
        del radius  # The image is already blurred over a 3-by-3 neighborhood.
        height, width = gray.shape
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        return float(gray[y, x])

    @classmethod
    def _score_triplet(
        cls,
        gray: np.ndarray,
        points: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[float, tuple[int, int, int, int]] | None:
        first, second, third = points
        horizontal = 0.5 * ((second - first) + (third - second))
        spacing = float(np.linalg.norm(horizontal))
        if spacing < 4.0:
            return None
        horizontal /= spacing
        vertical = np.asarray((-horizontal[1], horizontal[0]), dtype=np.float32)
        if vertical[1] < 0.0:
            vertical *= -1.0

        best: tuple[float, float, float, list[float], list[float]] | None = None
        cell_centers = tuple(first + (column - 0.5) * spacing * horizontal for column in range(4))
        sample_radius = max(1.5, 0.055 * spacing)
        for upper_fraction in (0.12, 0.20, 0.29):
            for lower_fraction in (0.14, 0.24, 0.35):
                upper = [
                    cls._sample_cell(
                        gray, center - upper_fraction * spacing * vertical, sample_radius
                    )
                    for center in cell_centers
                ]
                lower = [
                    cls._sample_cell(
                        gray, center + lower_fraction * spacing * vertical, sample_radius
                    )
                    for center in cell_centers
                ]
                parity_a = np.asarray((upper[0], upper[2], lower[1], lower[3]))
                parity_b = np.asarray((upper[1], upper[3], lower[0], lower[2]))
                contrast = float(abs(parity_a.mean() - parity_b.mean()))
                bright = parity_a if parity_a.mean() > parity_b.mean() else parity_b
                dark = parity_b if parity_a.mean() > parity_b.mean() else parity_a
                if contrast < 38.0 or float(bright.mean()) < 145.0:
                    continue
                # Each of the eight samples must broadly agree with its color;
                # this is what separates the board from lane-line reflections.
                bright_failures = int(np.count_nonzero(bright < float(bright.mean()) - 48.0))
                dark_failures = int(np.count_nonzero(dark > float(dark.mean()) + 48.0))
                if bright_failures + dark_failures > 1:
                    continue
                consistency = float(np.std(bright) + np.std(dark))
                score = contrast / 70.0 + max(0.0, 1.0 - consistency / 105.0)
                if best is None or score > best[0]:
                    best = (score, upper_fraction, lower_fraction, upper, lower)
        if best is None:
            return None

        score, upper_fraction, lower_fraction, _, _ = best
        outer_first = first - spacing * horizontal
        outer_last = third + spacing * horizontal
        polygon = np.asarray(
            (
                outer_first - upper_fraction * spacing * vertical,
                outer_last - upper_fraction * spacing * vertical,
                outer_last + lower_fraction * spacing * vertical,
                outer_first + lower_fraction * spacing * vertical,
            ),
            dtype=np.float32,
        )
        x, y, width, height = cv2.boundingRect(polygon)
        return score, (int(x), int(y), int(width), int(height))

    def detect(self, frame_bgr: np.ndarray) -> CheckerboardDetection:
        if frame_bgr is None or frame_bgr.size == 0:
            return CheckerboardDetection()
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi.bounds(frame_bgr.shape)
            # The board is mounted near the light.  Cap its structural search
            # at 70 percent image height so the newly included lower band does
            # not turn the floor checker pattern into a candidate.
            checker_y1 = min(y1, max(y0 + 1, int(round(0.70 * frame_bgr.shape[0]))))
            roi_frame = frame_bgr[y0:checker_y1, x0:x1]
            detection = self._detect_frame(roi_frame)
            bbox = detection.bbox
            if bbox is not None:
                bbox = (bbox[0] + x0, bbox[1] + y0, bbox[2], bbox[3])
            return CheckerboardDetection(
                detected=detection.detected,
                confidence=detection.confidence,
                bbox=bbox,
            )
        return self._detect_frame(frame_bgr)

    def _detect_frame(self, frame_bgr: np.ndarray) -> CheckerboardDetection:
        _, original_width = frame_bgr.shape[:2]
        processing_scale = min(1.0, 320.0 / original_width)
        if processing_scale < 1.0:
            processing_frame = cv2.resize(
                frame_bgr,
                None,
                fx=processing_scale,
                fy=processing_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            processing_frame = frame_bgr
        height, width = processing_frame.shape[:2]
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
        y0 = int(round(0.52 * height))
        y1 = int(round(0.90 * height))
        corners = cv2.goodFeaturesToTrack(
            gray[y0:y1],
            maxCorners=55,
            qualityLevel=0.035,
            minDistance=max(3.0, width / 210.0),
            blockSize=5,
            useHarrisDetector=True,
            k=0.04,
        )
        if corners is None:
            self.last_debug = {"corner_count": 0, "triplet_count": 0}
            return CheckerboardDetection()

        points = corners.reshape(-1, 2).astype(np.float32)
        points[:, 1] += y0
        points = points[np.argsort(points[:, 0])]
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        attempted: set[tuple[int, int, int]] = set()
        for first_index, first in enumerate(points):
            for second_index in range(first_index + 1, len(points)):
                second = points[second_index]
                dx = float(second[0] - first[0])
                if dx < 4.0:
                    continue
                if dx > 0.18 * width:
                    break
                dy = float(second[1] - first[1])
                if abs(dy) > max(4.0, 0.30 * dx):
                    continue
                predicted = second + (second - first)
                delta = points - predicted
                distance = np.hypot(delta[:, 0], 1.8 * delta[:, 1])
                for third_index in np.argsort(distance)[:2]:
                    third_index = int(third_index)
                    if third_index <= second_index or distance[third_index] > max(4.0, 0.25 * dx):
                        continue
                    key = (first_index, second_index, third_index)
                    if key in attempted:
                        continue
                    attempted.add(key)
                    first_gap = float(np.linalg.norm(second - first))
                    second_gap = float(np.linalg.norm(points[third_index] - second))
                    if not 0.72 <= second_gap / max(first_gap, 1.0) <= 1.38:
                        continue
                    result = self._score_triplet(gray, (first, second, points[third_index]))
                    if result is not None:
                        score, bbox = result
                        _, box_y, box_width, box_height = bbox
                        aspect = box_width / max(box_height, 1)
                        center_y = box_y + 0.5 * box_height
                        if not 0.045 * width <= box_width <= 0.42 * width:
                            continue
                        if not 4.5 <= aspect <= 24.0:
                            continue
                        if center_y < 0.55 * height:
                            continue
                        candidates.append((score, bbox))

        self.last_debug = {
            "corner_count": int(len(points)),
            "triplet_count": int(len(attempted)),
            "candidate_count": int(len(candidates)),
        }
        if not candidates:
            return CheckerboardDetection()
        score, bbox = max(candidates, key=lambda item: item[0])
        self.last_debug["score"] = float(score)
        confidence = float(np.clip((score - 1.0) / 1.5, 0.0, 1.0))
        if processing_scale < 1.0:
            x, y, box_width, box_height = bbox
            bbox = (
                int(round(x / processing_scale)),
                int(round(y / processing_scale)),
                int(round(box_width / processing_scale)),
                int(round(box_height / processing_scale)),
            )
        return CheckerboardDetection(True, confidence, bbox)
