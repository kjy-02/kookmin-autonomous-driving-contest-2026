"""Evaluate the detector on the supplied sqlite ROS 2 bags without ROS imports."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .bag_reader import RosbagImageReader
from .detectors import (
    CheckerboardDetector,
    DetectionRoi,
    TemporalBoolFilter,
    TemporalStateFilter,
    TrafficLightDetector,
    TRAFFIC_GREEN,
    TRAFFIC_GREEN_LEFT,
    TRAFFIC_RED,
    TRAFFIC_YELLOW,
)


def expected_state(name: str) -> int:
    lowered = name.lower()
    if "leftgreen" in lowered:
        return TRAFFIC_GREEN_LEFT
    if "yellow" in lowered:
        return TRAFFIC_YELLOW
    if "green" in lowered:
        return TRAFFIC_GREEN
    if "red" in lowered:
        return TRAFFIC_RED
    raise ValueError(f"cannot infer traffic state from bag name: {name}")


def evaluate_bag(path: Path, stride: int) -> dict[str, object]:
    detection_roi = DetectionRoi()
    traffic_detector = TrafficLightDetector(roi=detection_roi)
    checker_detector = CheckerboardDetector(roi=detection_roi)
    traffic_filter = TemporalStateFilter()
    checker_filter = TemporalBoolFilter()
    raw_counts: Counter[int] = Counter()
    stable_counts: Counter[int] = Counter()
    checker_hits: list[int] = []
    stable_checker_hits: list[int] = []
    expected = expected_state(path.name)

    for frame in RosbagImageReader(path).frames(stride=stride):
        traffic_state = traffic_detector.detect(frame.frame_bgr).state
        stable_state = traffic_filter.update(traffic_state)
        raw_counts[traffic_state] += 1
        stable_counts[stable_state] += 1

        checker = checker_detector.detect(frame.frame_bgr).detected
        stable_checker = checker_filter.update(checker)
        if checker:
            checker_hits.append(frame.index)
        if stable_checker:
            stable_checker_hits.append(frame.index)

    nonzero = sum(value for key, value in stable_counts.items() if key != 0)
    correct = stable_counts[expected]
    return {
        "bag": path.name,
        "expected_state": expected,
        "sampled_frames": sum(stable_counts.values()),
        "raw_traffic_counts": dict(sorted(raw_counts.items())),
        "stable_traffic_counts": dict(sorted(stable_counts.items())),
        "nonzero_state_accuracy": correct / nonzero if nonzero else None,
        "checkerboard_raw_hits": len(checker_hits),
        "checkerboard_stable_hits": len(stable_checker_hits),
        "checkerboard_first_last_frame": (
            [min(stable_checker_hits), max(stable_checker_hits)]
            if stable_checker_hits
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_root", type=Path, help="folder containing traffic_* bag folders")
    parser.add_argument("--stride", type=int, default=10, help="evaluate every Nth frame")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    arguments = parser.parse_args()
    if arguments.stride < 1:
        parser.error("--stride must be at least 1")

    bag_paths = sorted(path for path in arguments.bag_root.iterdir() if path.is_dir())
    results = [evaluate_bag(path, arguments.stride) for path in bag_paths]
    report = {"stride": arguments.stride, "bags": results}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if arguments.output:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
