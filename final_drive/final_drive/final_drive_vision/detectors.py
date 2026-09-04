"""OpenCV detector core with no ROS dependency."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable, Sequence

import cv2
import numpy as np

from .checkerboard_detector import CheckerboardDetection, CheckerboardDetector


TRAFFIC_NONE = 0
TRAFFIC_RED = 1
TRAFFIC_YELLOW = 2
TRAFFIC_GREEN = 3
TRAFFIC_GREEN_LEFT = 4


@dataclass(frozen=True)
class TrafficDetection:
    state: int = TRAFFIC_NONE
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    slot_centers: tuple[tuple[float, float], ...] = ()
    radius: float = 0.0


@dataclass(frozen=True)
class DetectionRoi:
    """Normalized image region used by both detectors.

    The defaults keep the center 70 percent horizontally and the upper
    85 percent vertically, matching the camera region requested for the car.
    """

    left: float = 0.15
    right: float = 0.15
    top: float = 0.0
    bottom: float = 0.15

    def __post_init__(self) -> None:
        values = (self.left, self.right, self.top, self.bottom)
        if any(value < 0.0 or value >= 1.0 for value in values):
            raise ValueError("ROI margins must be within [0, 1)")
        if self.left + self.right >= 1.0:
            raise ValueError("ROI left and right margins leave no image width")
        if self.top + self.bottom >= 1.0:
            raise ValueError("ROI top and bottom margins leave no image height")

    def bounds(self, frame_shape: Sequence[int]) -> tuple[int, int, int, int]:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        x0 = int(round(self.left * width))
        x1 = int(round((1.0 - self.right) * width))
        y0 = int(round(self.top * height))
        y1 = int(round((1.0 - self.bottom) * height))
        return (
            int(np.clip(x0, 0, width - 1)),
            int(np.clip(y0, 0, height - 1)),
            int(np.clip(x1, 1, width)),
            int(np.clip(y1, 1, height)),
        )

    def crop(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        x0, y0, x1, y1 = self.bounds(frame_bgr.shape)
        return frame_bgr[y0:y1, x0:x1], (x0, y0, x1, y1)

    def masked(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Hide excluded pixels without rescaling detector geometry."""

        x0, y0, x1, y1 = self.bounds(frame_bgr.shape)
        masked = frame_bgr.copy()
        if x0 > 0:
            masked[:, :x0] = masked[:, x0 : x0 + 1]
        if x1 < masked.shape[1]:
            masked[:, x1:] = masked[:, x1 - 1 : x1]
        if y0 > 0:
            masked[:y0, :] = masked[y0 : y0 + 1, :]
        if y1 < masked.shape[0]:
            masked[y1:, :] = masked[y1 - 1 : y1, :]
        return masked

    def contains_bbox(
        self, bbox: tuple[int, int, int, int] | None, frame_shape: Sequence[int]
    ) -> bool:
        if bbox is None:
            return True
        x0, y0, x1, y1 = self.bounds(frame_shape)
        box_x, box_y, box_width, box_height = bbox
        return (
            box_x >= x0
            and box_y >= y0
            and box_x + box_width <= x1
            and box_y + box_height <= y1
        )

    @staticmethod
    def restore_traffic(
        detection: TrafficDetection, offset_x: int, offset_y: int
    ) -> TrafficDetection:
        bbox = detection.bbox
        if bbox is not None:
            bbox = (bbox[0] + offset_x, bbox[1] + offset_y, bbox[2], bbox[3])
        centers = tuple(
            (center[0] + offset_x, center[1] + offset_y)
            for center in detection.slot_centers
        )
        return TrafficDetection(
            state=detection.state,
            confidence=detection.confidence,
            bbox=bbox,
            slot_centers=centers,
            radius=detection.radius,
        )

    @staticmethod
    def restore_checker(
        detection: CheckerboardDetection, offset_x: int, offset_y: int
    ) -> CheckerboardDetection:
        bbox = detection.bbox
        if bbox is not None:
            bbox = (bbox[0] + offset_x, bbox[1] + offset_y, bbox[2], bbox[3])
        return CheckerboardDetection(
            detected=detection.detected,
            confidence=detection.confidence,
            bbox=bbox,
        )


class TemporalStateFilter:
    """Reject single-frame state changes while clearing stale detections quickly."""

    def __init__(self, window_size: int = 5, required_hits: int = 3, clear_hits: int = 2):
        if not 1 <= required_hits <= window_size:
            raise ValueError("required_hits must be within the window")
        self.window_size = window_size
        self.required_hits = required_hits
        self.clear_hits = clear_hits
        self._history: Deque[int] = deque(maxlen=window_size)
        self._state = TRAFFIC_NONE
        self._none_run = 0

    @property
    def state(self) -> int:
        return self._state

    def reset(self) -> None:
        self._history.clear()
        self._state = TRAFFIC_NONE
        self._none_run = 0

    def update(self, observed_state: int) -> int:
        self._history.append(int(observed_state))
        if observed_state == TRAFFIC_NONE:
            self._none_run += 1
        else:
            self._none_run = 0

        if self._none_run >= self.clear_hits:
            self._state = TRAFFIC_NONE
            self._history.clear()
            return self._state

        nonzero = Counter(value for value in self._history if value != TRAFFIC_NONE)
        if nonzero:
            candidate, hits = nonzero.most_common(1)[0]
            if hits >= self.required_hits:
                self._state = candidate
        return self._state


class TemporalBoolFilter:
    """Debounce a boolean observation with independent on/off thresholds."""

    def __init__(self, on_hits: int = 2, off_hits: int = 2):
        if on_hits < 1 or off_hits < 1:
            raise ValueError("hit counts must be positive")
        self.on_hits = on_hits
        self.off_hits = off_hits
        self._state = False
        self._run_value = False
        self._run_length = 0

    @property
    def state(self) -> bool:
        return self._state

    def reset(self) -> None:
        self._state = False
        self._run_value = False
        self._run_length = 0

    def update(self, observed: bool) -> bool:
        observed = bool(observed)
        if observed == self._run_value:
            self._run_length += 1
        else:
            self._run_value = observed
            self._run_length = 1
        required = self.on_hits if observed else self.off_hits
        if self._run_length >= required:
            self._state = observed
        return self._state


def _color_excess_maps(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, g, r = cv2.split(frame_bgr.astype(np.float32))
    red = np.maximum(r - np.maximum(g, b), 0.0)
    yellow = np.maximum(np.minimum(r, g) - b - 0.25 * np.abs(r - g), 0.0)
    cyan = np.maximum(np.minimum(g, b) - r - 0.20 * np.abs(g - b), 0.0)
    return red, yellow, cyan


def _disk_strength(color_map: np.ndarray, center: tuple[float, float], radius: float) -> float:
    height, width = color_map.shape
    x0 = max(0, int(np.floor(center[0] - radius)))
    y0 = max(0, int(np.floor(center[1] - radius)))
    x1 = min(width, int(np.ceil(center[0] + radius + 1)))
    y1 = min(height, int(np.ceil(center[1] + radius + 1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    values = color_map[y0:y1, x0:x1][mask]
    if values.size == 0:
        return 0.0
    # Suppress weak chromatic noise while retaining small LED/arrow cores.
    return float(np.maximum(values - 12.0, 0.0).mean())


def _ring_mean(image: np.ndarray, center: tuple[float, float], radius: float) -> float:
    samples = []
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    height, width = image.shape
    for scale in (0.78, 1.0, 1.20):
        xs = np.rint(center[0] + scale * radius * np.cos(angles)).astype(np.int32)
        ys = np.rint(center[1] + scale * radius * np.sin(angles)).astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if np.any(valid):
            samples.append(image[ys[valid], xs[valid]])
    if not samples:
        return 0.0
    return float(np.concatenate(samples).mean())


def _disk_mean(image: np.ndarray, center: tuple[float, float], radius: float) -> float:
    height, width = image.shape
    x0 = max(0, int(np.floor(center[0] - radius)))
    y0 = max(0, int(np.floor(center[1] - radius)))
    x1 = min(width, int(np.ceil(center[0] + radius + 1)))
    y1 = min(height, int(np.ceil(center[1] + radius + 1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    values = image[y0:y1, x0:x1][mask]
    return float(values.mean()) if values.size else 0.0


def _local_color_strength(
    color_map: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> float:
    best = 0.0
    for dx in (-0.42 * radius, 0.0, 0.42 * radius):
        for dy in (-0.32 * radius, 0.0, 0.32 * radius):
            best = max(
                best,
                _disk_strength(color_map, (center[0] + dx, center[1] + dy), 0.75 * radius),
            )
    return best


def _dark_ring_coverage(
    gray: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> float:
    """Measure the dark lamp rim against the brighter housing behind it.

    The inactive lamps are dark circular recesses.  Sampling paired points on
    the recess and just outside it rejects window frames and ceiling fixtures
    that can otherwise resemble four aligned circles in the edge image.
    """

    height, width = gray.shape
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    coverages: list[float] = []
    for inner_scale, outer_scale in ((0.85, 1.40), (1.00, 1.50)):
        inner_x = np.rint(center[0] + inner_scale * radius * np.cos(angles)).astype(np.int32)
        inner_y = np.rint(center[1] + inner_scale * radius * np.sin(angles)).astype(np.int32)
        outer_x = np.rint(center[0] + outer_scale * radius * np.cos(angles)).astype(np.int32)
        outer_y = np.rint(center[1] + outer_scale * radius * np.sin(angles)).astype(np.int32)
        valid = (
            (inner_x >= 0)
            & (inner_x < width)
            & (inner_y >= 0)
            & (inner_y < height)
            & (outer_x >= 0)
            & (outer_x < width)
            & (outer_y >= 0)
            & (outer_y < height)
        )
        if not np.any(valid):
            continue
        difference = gray[outer_y[valid], outer_x[valid]].astype(np.int16)
        difference -= gray[inner_y[valid], inner_x[valid]].astype(np.int16)
        coverages.append(float(np.mean(difference > 15)))
    return max(coverages, default=0.0)


class TrafficLightDetector:
    """Find a four-lamp housing and classify its illuminated slots."""

    def __init__(
        self,
        roi: DetectionRoi | None = None,
        max_y_fraction: float = 0.50,
    ) -> None:
        if not 0.0 < max_y_fraction <= 1.0:
            raise ValueError("max_y_fraction must be within (0, 1]")
        self.roi = roi
        self.max_y_fraction = float(max_y_fraction)
        self.last_debug: dict[str, object] = {}

    def active_bounds(self, frame_shape: Sequence[int]) -> tuple[int, int, int, int]:
        """Return the signal-search area capped at an absolute image Y ratio."""

        height, width = int(frame_shape[0]), int(frame_shape[1])
        if self.roi is None:
            x0, y0, x1, y1 = 0, 0, width, height
        else:
            x0, y0, x1, y1 = self.roi.bounds(frame_shape)
        max_y = max(y0 + 1, int(round(height * self.max_y_fraction)))
        return x0, y0, x1, min(y1, max_y)

    @staticmethod
    def _color_components(
        color_map: np.ndarray,
        y_limit: int,
        scale: float,
    ) -> list[tuple[float, float, float, tuple[int, int, int, int]]]:
        smooth = cv2.GaussianBlur(color_map[:y_limit], (0, 0), max(0.8, scale))
        binary = (smooth > 18.0).astype(np.uint8)
        kernel_size = max(3, int(round(3.0 * scale)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        output = []
        min_area = max(2, int(round(3.0 * scale * scale)))
        max_area = int(round(3500.0 * scale * scale))
        max_dimension = int(round(80.0 * scale))
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            if not min_area <= area <= max_area:
                continue
            if width > max_dimension or height > max_dimension:
                continue
            aspect = width / max(height, 1)
            if not 0.25 <= aspect <= 3.5:
                continue
            mask = (labels == label).astype(np.uint8)
            mean_excess = cv2.mean(smooth, mask=mask)[0]
            score = float(mean_excess * area)
            if score < 450.0 * scale * scale:
                continue
            output.append(
                (
                    float(centroids[label, 0]),
                    float(centroids[label, 1]),
                    score,
                    (x, y, width, height),
                )
            )
        output.sort(key=lambda item: item[2], reverse=True)
        return output[:12]

    @staticmethod
    def _hough_circles(gray: np.ndarray, y_limit: int, scale: float) -> np.ndarray:
        # Dark lamp rims form compact near-circular contours.  Contour search is
        # an order of magnitude faster than a full Hough accumulator and, unlike
        # Hough, does not return hundreds of ceiling/chair circles per frame.
        roi = cv2.GaussianBlur(gray[:y_limit], (3, 3), 0.7 * scale)
        candidates: list[tuple[float, float, float, float]] = []
        min_area = 12.0 * scale * scale
        max_area = 5000.0 * scale * scale
        min_dimension = max(4.0, 4.0 * scale)
        max_dimension = 72.0 * scale
        for threshold in (90, 120, 150, 180):
            binary = cv2.inRange(roi, 0, threshold)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if not min_area <= area <= max_area or perimeter <= 0.0:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                if not min_dimension <= width <= max_dimension:
                    continue
                if not min_dimension <= height <= max_dimension:
                    continue
                aspect = width / max(height, 1)
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                if not 0.60 <= aspect <= 1.58 or circularity < 0.34:
                    continue
                candidates.append(
                    (x + 0.5 * width, y + 0.5 * height, 0.25 * (width + height), circularity)
                )

        candidates.sort(key=lambda value: (value[3], value[2]), reverse=True)
        deduplicated: list[tuple[float, float, float]] = []
        for cx, cy, radius, _ in candidates:
            duplicate = False
            for other_x, other_y, other_radius in deduplicated:
                if np.hypot(cx - other_x, cy - other_y) < 0.40 * max(radius, other_radius):
                    duplicate = True
                    break
            if not duplicate:
                deduplicated.append((cx, cy, radius))
        if not deduplicated:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(deduplicated, dtype=np.float32)

    @staticmethod
    def _nearest_anchor_circles(
        circles: np.ndarray,
        component: tuple[float, float, float, tuple[int, int, int, int]],
        scale: float,
    ) -> list[np.ndarray]:
        cx, cy, _, (_, _, width, height) = component
        output: list[np.ndarray] = []
        if circles.size:
            distances = np.hypot(circles[:, 0] - cx, circles[:, 1] - cy)
            limit = max(8.0 * scale, 0.9 * max(width, height) + 5.0 * scale)
            indices = np.flatnonzero(distances <= limit)
            indices = sorted(indices, key=lambda index: float(distances[index]))
            output.extend(circles[index] for index in indices[:3])

        # The illuminated disk can erase its own dark contour.  Add synthetic
        # anchors from the chromatic component while the remaining inactive
        # rims provide the independent structural check.
        bright_size = float(max(width, height))
        for radius in (0.50 * bright_size, 0.67 * bright_size, 0.82 * bright_size):
            radius = float(np.clip(radius, 3.0 * scale, 32.0 * scale))
            anchor = np.asarray((cx, cy, radius), dtype=np.float32)
            if not any(np.hypot(*(anchor[:2] - value[:2])) < 2.0 for value in output):
                output.append(anchor)
        return output

    @staticmethod
    def _match_group(
        circles: np.ndarray,
        anchor: np.ndarray,
        active_slot: int,
        gradient: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> tuple[float, tuple[tuple[float, float], ...], float] | None:
        height, width = frame_shape
        anchor_x, anchor_y, anchor_radius = (float(value) for value in anchor)
        best = None
        for spacing_ratio in np.linspace(1.90, 3.10, 13):
            spacing = spacing_ratio * anchor_radius
            centers = tuple(
                (anchor_x + (slot - active_slot) * spacing, anchor_y)
                for slot in range(4)
            )
            if centers[0][0] < 2.0 or centers[-1][0] >= width - 2.0:
                continue

            matched = 0
            match_cost = 0.0
            used: set[int] = set()
            for center in centers:
                dx = np.abs(circles[:, 0] - center[0])
                dy = np.abs(circles[:, 1] - center[1])
                dr = np.abs(circles[:, 2] - anchor_radius)
                cost = dx / max(anchor_radius * 0.65, 3.0)
                cost += dy / max(anchor_radius * 0.55, 3.0)
                cost += dr / max(anchor_radius * 0.60, 2.0)
                for index in np.argsort(cost):
                    idx = int(index)
                    if idx in used:
                        continue
                    if cost[idx] <= 2.5:
                        matched += 1
                        match_cost += float(cost[idx])
                        used.add(idx)
                    break

            # The active lamp can erase its own dark contour, so the three
            # inactive rims are sufficient when color and housing checks agree.
            if matched < 3:
                continue
            ring_edges = [_ring_mean(gradient, center, anchor_radius) for center in centers]
            minimum_edge = min(ring_edges)
            mean_edge = float(np.mean(ring_edges))
            if minimum_edge < 38.0 or mean_edge < 65.0:
                continue
            score = 2.2 * matched - 0.35 * match_cost
            score += min(minimum_edge / 65.0, 2.0)
            score += min(mean_edge / 100.0, 2.0)
            candidate = (score, centers, anchor_radius)
            if best is None or score > best[0]:
                best = candidate
        return best

    @staticmethod
    def _housing_metrics(
        hsv: np.ndarray,
        centers: tuple[tuple[float, float], ...],
        radius: float,
    ) -> tuple[float, float, float]:
        height, width = hsv.shape[:2]
        x0 = max(0, int(np.floor(centers[0][0] - 1.35 * radius)))
        x1 = min(width, int(np.ceil(centers[-1][0] + 1.35 * radius + 1)))
        y0 = max(0, int(np.floor(centers[0][1] - 1.50 * radius)))
        y1 = min(height, int(np.ceil(centers[0][1] + 1.50 * radius + 1)))
        if x1 <= x0 or y1 <= y0:
            return 0.0, 255.0, 255.0
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        mask[:] = 255
        for center in centers:
            cv2.circle(
                mask,
                (int(round(center[0] - x0)), int(round(center[1] - y0))),
                max(1, int(round(1.10 * radius))),
                0,
                -1,
            )
        if cv2.countNonZero(mask) < 10:
            return 0.0, 255.0, 255.0
        patch = hsv[y0:y1, x0:x1]
        saturation = cv2.mean(patch[:, :, 1], mask=mask)[0]
        value = cv2.mean(patch[:, :, 2], mask=mask)[0]
        _, value_std = cv2.meanStdDev(patch[:, :, 2], mask=mask)
        return float(value), float(saturation), float(value_std[0, 0])

    @staticmethod
    def _guided_group(
        component: tuple[float, float, float, tuple[int, int, int, int]],
        state: int,
        active_position: float,
        gradient: np.ndarray,
        hsv: np.ndarray,
        color_maps: tuple[np.ndarray, np.ndarray, np.ndarray],
        frame_shape: tuple[int, int],
        scale: float,
    ) -> tuple[float, tuple[tuple[float, float], ...], float, list[float]] | None:
        height, width = frame_shape
        component_x, component_y, component_score, (_, _, box_width, box_height) = component
        component_size = float(max(box_width, box_height))
        minimum_size = float(min(box_width, box_height))

        radius_min = max(4.0 * scale, 0.24 * minimum_size)
        radius_max = min(30.0 * scale, max(8.0 * scale, 0.88 * component_size))
        if radius_max <= radius_min:
            radii = np.asarray((radius_min,), dtype=np.float32)
        else:
            radii = np.linspace(radius_min, radius_max, 6)

        value_channel = hsv[:, :, 2]
        best = None
        for radius in radii:
            radius = float(radius)
            for spacing_ratio in (2.25, 2.50, 2.75, 3.00):
                spacing = spacing_ratio * radius
                for x_shift_ratio in (-0.40, 0.0, 0.40):
                    for y_shift_ratio in (-1.10, -0.55, 0.0, 0.45):
                        anchor_x = component_x + x_shift_ratio * radius
                        anchor_y = component_y + y_shift_ratio * radius
                        first_x = anchor_x - active_position * spacing
                        centers = tuple(
                            (first_x + slot * spacing, anchor_y)
                            for slot in range(4)
                        )
                        if centers[0][0] < 2.0 or centers[-1][0] >= width - 2.0:
                            continue
                        if anchor_y - 1.6 * radius < 1.0 or anchor_y + 1.6 * radius >= height:
                            continue

                        ring_edges = [_ring_mean(gradient, center, radius) for center in centers]
                        minimum_edge = min(ring_edges)
                        mean_edge = float(np.mean(ring_edges))
                        if minimum_edge < 72.0 or mean_edge < 108.0:
                            continue

                        red_strengths = [
                            _disk_strength(color_maps[0], center, 1.22 * radius)
                            for center in centers
                        ]
                        yellow_strengths = [
                            _disk_strength(color_maps[1], center, 0.72 * radius)
                            for center in centers
                        ]
                        cyan_strengths = [
                            _disk_strength(color_maps[2], center, 0.72 * radius)
                            for center in centers
                        ]

                        if state == TRAFFIC_RED:
                            state_scores = red_strengths
                            color_score = red_strengths[0]
                            inactive_slots = (1, 2, 3)
                            active_slots = (0,)
                            color_minimum = 2.2
                            if color_score < 1.8 * max(red_strengths[1:] + [0.1]):
                                continue
                        elif state == TRAFFIC_YELLOW:
                            state_scores = yellow_strengths
                            color_score = yellow_strengths[1]
                            inactive_slots = (0, 2, 3)
                            active_slots = (1,)
                            color_minimum = 7.0
                            others = [yellow_strengths[index] for index in inactive_slots]
                            if color_score < 1.8 * max(others + [0.1]):
                                continue
                        elif state == TRAFFIC_GREEN:
                            state_scores = cyan_strengths
                            color_score = cyan_strengths[3]
                            inactive_slots = (0, 1, 2)
                            active_slots = (3,)
                            color_minimum = 7.0
                            if cyan_strengths[2] > max(3.5, 0.30 * cyan_strengths[3]):
                                continue
                            if color_score < 1.8 * max(cyan_strengths[:3] + [0.1]):
                                continue
                        else:
                            state_scores = cyan_strengths
                            color_score = min(cyan_strengths[2], cyan_strengths[3])
                            inactive_slots = (0, 1)
                            active_slots = (2, 3)
                            color_minimum = 3.5
                            if min(cyan_strengths[2], cyan_strengths[3]) < 1.6 * max(
                                cyan_strengths[0], cyan_strengths[1], 0.1
                            ):
                                continue
                        if color_score < color_minimum:
                            continue

                        housing_value, housing_saturation, _ = TrafficLightDetector._housing_metrics(
                            hsv, centers, radius
                        )
                        if housing_value < 92.0 or housing_saturation > 112.0:
                            continue

                        center_values = [
                            _disk_mean(value_channel, center, 0.52 * radius)
                            for center in centers
                        ]
                        inactive_mean = float(
                            np.mean([center_values[index] for index in inactive_slots])
                        )
                        active_mean = float(
                            np.mean([center_values[index] for index in active_slots])
                        )
                        if inactive_mean > housing_value + 22.0:
                            continue
                        if active_mean - inactive_mean < 38.0:
                            continue

                        structure_score = minimum_edge / 90.0 + mean_edge / 130.0
                        color_term = min(color_score / max(color_minimum, 1.0), 4.0)
                        contrast_term = min((active_mean - inactive_mean) / 65.0, 2.0)
                        housing_term = min(housing_value / 140.0, 1.5)
                        source_term = min(np.log1p(component_score) / 10.0, 1.2)
                        score = (
                            structure_score
                            + color_term
                            + contrast_term
                            + housing_term
                            + source_term
                        )
                        candidate = (score, centers, radius, state_scores)
                        if best is None or score > best[0]:
                            best = candidate
        return best

    @staticmethod
    def _refine_bright_component(
        value_channel: np.ndarray,
        component: tuple[float, float, float, tuple[int, int, int, int]],
        scale: float,
    ) -> tuple[tuple[float, float], float]:
        center_x, center_y, _, (_, _, width, height) = component
        search_radius = int(max(24.0 * scale, 0.8 * max(width, height) + 10.0 * scale))
        image_height, image_width = value_channel.shape
        x0 = max(0, int(np.floor(center_x - search_radius)))
        y0 = max(0, int(np.floor(center_y - search_radius)))
        x1 = min(image_width, int(np.ceil(center_x + search_radius + 1)))
        y1 = min(image_height, int(np.ceil(center_y + search_radius + 1)))
        crop = value_channel[y0:y1, x0:x1]
        if crop.size == 0:
            return (center_x, center_y), float(max(width, height))
        percentile = float(np.percentile(crop, 85.0))
        threshold = max(215.0, percentile)
        binary = (crop > threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        best = None
        for label in range(1, count):
            _, _, item_width, item_height, area = (int(value) for value in stats[label])
            if not 3 <= area <= int(2600.0 * scale * scale):
                continue
            global_x = float(centroids[label, 0] + x0)
            global_y = float(centroids[label, 1] + y0)
            distance = float(np.hypot(global_x - center_x, global_y - center_y))
            if distance > 0.95 * search_radius:
                continue
            aspect = item_width / max(item_height, 1)
            if not 0.30 <= aspect <= 3.2:
                continue
            fill = area / max(item_width * item_height, 1)
            score = area * min(fill / 0.35, 1.2) / (1.0 + 0.025 * distance * distance)
            if best is None or score > best[0]:
                best = (score, (global_x, global_y), float(max(item_width, item_height)))
        if best is None:
            return (center_x, center_y), float(max(width, height))
        return best[1], best[2]

    @staticmethod
    def _evaluate_fixed_hypothesis(
        state_family: int,
        center: tuple[float, float],
        bright_size: float,
        gray: np.ndarray,
        gradient: np.ndarray,
        hsv: np.ndarray,
        color_maps: tuple[np.ndarray, np.ndarray, np.ndarray],
        frame_shape: tuple[int, int],
        y_limit: int,
        scale: float,
        assumed_active_slot: int | None = None,
    ) -> tuple[float, int, tuple[tuple[float, float], ...], float, list[float]] | None:
        height, width = frame_shape
        if state_family == TRAFFIC_RED:
            active_slot = 0
            denominator = 1.55
        elif state_family == TRAFFIC_YELLOW:
            active_slot = 1
            denominator = 1.52
        else:
            active_slot = 3 if assumed_active_slot is None else assumed_active_slot
            denominator = 1.51

        nominal_radius = float(np.clip(bright_size / denominator, 5.0 * scale, 30.0 * scale))
        best = None
        for radius_factor in (0.86, 1.0, 1.14):
            radius = nominal_radius * radius_factor
            for spacing_ratio in (2.30, 2.50, 2.70):
                spacing = spacing_ratio * radius
                first_x = center[0] - active_slot * spacing
                centers = tuple((first_x + slot * spacing, center[1]) for slot in range(4))
                if centers[0][0] < 2.0 or centers[-1][0] >= width - 2.0:
                    continue
                body_width = centers[-1][0] - centers[0][0] + 2.7 * radius
                body_center_x = 0.5 * (centers[0][0] + centers[-1][0])
                # The decision zone is deliberately narrower than the camera.
                # Very small/far rows and the fixed LED control box at the far
                # right are treated as NONE until the real four-lamp board is
                # large enough for a reliable decision.
                if not 72.0 * scale <= body_width <= 215.0 * scale:
                    continue
                if not 170.0 / 640.0 * width <= body_center_x <= 505.0 / 640.0 * width:
                    continue
                # The candidate must fit inside the configured signal ROI.
                # This replaces the former hidden Y=176 center cutoff.
                if not 48.0 / 480.0 * height <= center[1] <= y_limit - 1.6 * radius:
                    continue
                if centers[0][0] - 1.35 * radius < 175.0 / 640.0 * width:
                    continue
                if centers[-1][0] + 1.35 * radius > 545.0 / 640.0 * width:
                    continue

                ring_edges = [_ring_mean(gradient, item, radius) for item in centers]
                minimum_edge = min(ring_edges)
                mean_edge = float(np.mean(ring_edges))
                if minimum_edge < 64.0 or mean_edge < 100.0:
                    continue

                dark_rings = [_dark_ring_coverage(gray, item, radius) for item in centers]
                minimum_dark_ring = min(dark_rings)
                mean_dark_ring = float(np.mean(dark_rings))
                if minimum_dark_ring < 0.32 or mean_dark_ring < 0.45:
                    continue

                red_strengths = [
                    _disk_strength(color_maps[0], item, 1.28 * radius) for item in centers
                ]
                yellow_strengths = [
                    _disk_strength(color_maps[1], item, 0.75 * radius) for item in centers
                ]
                cyan_strengths = [
                    _disk_strength(color_maps[2], item, 0.75 * radius) for item in centers
                ]
                cyan_strengths[2] = max(
                    cyan_strengths[2],
                    _local_color_strength(color_maps[2], centers[2], radius),
                )
                cyan_strengths[3] = max(
                    cyan_strengths[3],
                    _local_color_strength(color_maps[2], centers[3], radius),
                )

                if state_family == TRAFFIC_RED:
                    state = TRAFFIC_RED
                    state_scores = red_strengths
                    color_score = red_strengths[0]
                    inactive_slots = (1, 2, 3)
                    active_slots = (0,)
                    if color_score < 2.0 or color_score < 1.7 * max(red_strengths[1:] + [0.1]):
                        continue
                elif state_family == TRAFFIC_YELLOW:
                    state = TRAFFIC_YELLOW
                    state_scores = yellow_strengths
                    color_score = yellow_strengths[1]
                    inactive_slots = (0, 2, 3)
                    active_slots = (1,)
                    others = [yellow_strengths[index] for index in inactive_slots]
                    if color_score < 6.0 or color_score < 1.7 * max(others + [0.1]):
                        continue
                else:
                    state_scores = cyan_strengths
                    green_score = cyan_strengths[3]
                    arrow_score = cyan_strengths[2]
                    if green_score < 6.0:
                        continue
                    preceding_red = max(
                        _local_color_strength(
                            color_maps[0],
                            (centers[0][0] - offset * spacing, centers[0][1]),
                            radius,
                        )
                        for offset in (1, 2, 3)
                    )
                    if preceding_red >= 4.0:
                        continue
                    if arrow_score >= max(2.6, 0.12 * green_score):
                        if green_score < 0.65 * arrow_score:
                            continue
                        state = TRAFFIC_GREEN_LEFT
                        color_score = min(arrow_score, green_score)
                        inactive_slots = (0, 1)
                        active_slots = (2, 3)
                    else:
                        # When the component used as an anchor is the arrow,
                        # a second green component appears one lamp spacing to
                        # its right.  Do not mislabel that shifted layout as a
                        # standalone green signal.
                        outside_right = _local_color_strength(
                            color_maps[2],
                            (centers[3][0] + spacing, centers[3][1]),
                            radius,
                        )
                        if active_slot == 2 or outside_right >= max(3.0, 0.18 * green_score):
                            continue
                        state = TRAFFIC_GREEN
                        color_score = green_score
                        inactive_slots = (0, 1, 2)
                        active_slots = (3,)
                    if green_score < 1.7 * max(cyan_strengths[:2] + [0.1]):
                        continue

                housing_value, housing_saturation, _ = TrafficLightDetector._housing_metrics(
                    hsv, centers, radius
                )
                if housing_value < 88.0 or housing_saturation > 115.0:
                    continue
                center_values = [
                    _disk_mean(hsv[:, :, 2], item, 0.52 * radius) for item in centers
                ]
                inactive_mean = float(np.mean([center_values[index] for index in inactive_slots]))
                active_mean = float(np.mean([center_values[index] for index in active_slots]))
                if inactive_mean > housing_value + 24.0:
                    continue
                if active_mean - inactive_mean < 34.0:
                    continue

                score = minimum_edge / 85.0 + mean_edge / 125.0
                score += min(color_score / 8.0, 4.0)
                score += min((active_mean - inactive_mean) / 60.0, 2.0)
                score += min(housing_value / 135.0, 1.5)
                score += min(mean_dark_ring / 0.55, 1.3)
                candidate = (score, state, centers, radius, state_scores)
                if best is None or score > best[0]:
                    best = candidate
        return best

    def detect(self, frame_bgr: np.ndarray) -> TrafficDetection:
        if frame_bgr is None or frame_bgr.size == 0:
            return TrafficDetection()
        x0, y0, x1, active_y1 = self.active_bounds(frame_bgr.shape)
        if self.roi is not None:
            masked = self.roi.masked(frame_bgr)
        else:
            masked = frame_bgr.copy()

        # Preserve full-frame coordinates for the detector's geometric rules,
        # but replace the ignored lower band so it cannot produce candidates.
        if active_y1 < masked.shape[0]:
            masked[active_y1:, :] = masked[active_y1 - 1 : active_y1, :]
        detection = self._detect_frame(masked, y_limit=active_y1)
        self.last_debug["active_bounds"] = (x0, y0, x1, active_y1)
        if detection.bbox is not None:
            box_x, box_y, box_width, box_height = detection.bbox
            if (
                box_x < x0
                or box_y < y0
                or box_x + box_width > x1
                or box_y + box_height > active_y1
            ):
                return TrafficDetection()
        return detection

    def _detect_frame(
        self,
        frame_bgr: np.ndarray,
        y_limit: int | None = None,
    ) -> TrafficDetection:
        height, width = frame_bgr.shape[:2]
        scale = 0.5 * (width / 640.0 + height / 480.0)
        if y_limit is None:
            y_limit = min(height, int(round(0.40 * height)))
        else:
            y_limit = int(np.clip(y_limit, 1, height))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)
        red_map, yellow_map, cyan_map = _color_excess_maps(frame_bgr)
        color_maps = (red_map, yellow_map, cyan_map)
        components = [self._color_components(color_map, y_limit, scale) for color_map in color_maps]
        hypotheses: list[tuple[float, int, tuple[tuple[float, float], ...], float, list[float]]] = []
        for color_index, color_components in enumerate(components):
            if color_index == 0:
                state_family = TRAFFIC_RED
            elif color_index == 1:
                state_family = TRAFFIC_YELLOW
            else:
                state_family = TRAFFIC_GREEN
            for component in color_components[:6]:
                center, bright_size = self._refine_bright_component(
                    hsv[:, :, 2], component, scale
                )
                active_slots = (3, 2) if state_family == TRAFFIC_GREEN else (None,)
                for active_slot in active_slots:
                    group = self._evaluate_fixed_hypothesis(
                        state_family,
                        center,
                        bright_size,
                        gray,
                        gradient,
                        hsv,
                        color_maps,
                        (height, width),
                        y_limit,
                        scale,
                        assumed_active_slot=active_slot,
                    )
                    if group is None:
                        continue
                    score, state, centers, radius, state_scores = group
                    hypotheses.append((score, state, centers, radius, state_scores))

        self.last_debug = {
            "component_counts": tuple(len(value) for value in components),
            "hypothesis_count": len(hypotheses),
            "best_by_state": {
                state: float(max(item[0] for item in hypotheses if item[1] == state))
                for state in sorted({item[1] for item in hypotheses})
            },
        }
        if not hypotheses:
            return TrafficDetection()

        score, state, centers, radius, state_scores = max(hypotheses, key=lambda value: value[0])
        x0 = int(round(centers[0][0] - 1.35 * radius))
        y0 = int(round(centers[0][1] - 1.55 * radius))
        x1 = int(round(centers[-1][0] + 1.35 * radius))
        y1 = int(round(centers[-1][1] + 1.55 * radius))
        bbox = (max(0, x0), max(0, y0), min(width, x1) - max(0, x0), min(height, y1) - max(0, y0))
        confidence = float(np.clip((score - 7.0) / 5.0, 0.0, 1.0))
        self.last_debug["state_scores"] = tuple(float(value) for value in state_scores)
        self.last_debug["score"] = float(score)
        return TrafficDetection(
            state=state,
            confidence=confidence,
            bbox=bbox,
            slot_centers=centers,
            radius=float(radius),
        )
