"""Interactive frame-by-frame viewer for image-only ROS 2 sqlite bags."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from .bag_reader import decode_image_message
from .detectors import (
    CheckerboardDetection,
    CheckerboardDetector,
    DetectionRoi,
    TemporalBoolFilter,
    TemporalStateFilter,
    TrafficDetection,
    TrafficLightDetector,
)


STATE_NAMES = {
    0: "NONE",
    1: "RED",
    2: "YELLOW",
    3: "GREEN",
    4: "GREEN+LEFT",
}
STATE_COLORS = {
    0: (170, 170, 170),
    1: (40, 40, 255),
    2: (0, 210, 255),
    3: (40, 220, 40),
    4: (255, 210, 30),
}


@dataclass(frozen=True)
class FrameAnalysis:
    traffic: TrafficDetection
    published_traffic: int
    checker: CheckerboardDetection
    published_checker: bool


class IndexedImageBag:
    """Random-access wrapper around one image topic in a sqlite rosbag."""

    def __init__(self, bag_path: Path, topic: str = "/image_raw"):
        databases = sorted(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
        if len(databases) != 1 or not databases[0].is_file():
            raise ValueError(f"expected one db3 file in {bag_path}")
        self.path = bag_path
        self.name = bag_path.name
        self.topic = topic
        self._connection = sqlite3.connect(databases[0])
        self._rows = self._connection.execute(
            """
            SELECT m.id, m.timestamp
            FROM messages AS m
            JOIN topics AS t ON t.id = m.topic_id
            WHERE t.name = ?
            ORDER BY m.timestamp
            """,
            (topic,),
        ).fetchall()
        if not self._rows:
            raise ValueError(f"no {topic} messages in {bag_path}")
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self._rows)

    def frame(self, index: int) -> np.ndarray:
        index = int(np.clip(index, 0, len(self) - 1))
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached.copy()
        message_id = int(self._rows[index][0])
        row = self._connection.execute(
            "SELECT data FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise IndexError(index)
        frame = decode_image_message(row[0])
        self._cache[index] = frame
        self._cache.move_to_end(index)
        while len(self._cache) > 48:
            self._cache.popitem(last=False)
        return frame.copy()

    def close(self) -> None:
        self._connection.close()


def discover_imagetrack_bags(root: Path) -> list[Path]:
    candidates: list[Path] = []
    if root.is_dir() and list(root.glob("*.db3")):
        return [root]
    for outer in sorted(root.glob("imagetrack*")):
        if not outer.is_dir():
            continue
        nested = outer / outer.name
        candidate = nested if nested.is_dir() else outer
        if list(candidate.glob("*.db3")):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"no imagetrack rosbag folders below {root}")
    return candidates


class BagFrameViewer:
    window_name = "imagetrack traffic viewer"

    def __init__(
        self, bag_paths: list[Path], detection_roi: DetectionRoi | None = None
    ):
        self.bag_paths = bag_paths
        self.bag_position = 0
        self.bag = IndexedImageBag(bag_paths[0])
        self.detection_roi = detection_roi or DetectionRoi()
        self.traffic_detector = TrafficLightDetector(roi=self.detection_roi)
        self.checker_detector = CheckerboardDetector(roi=self.detection_roi)
        self.traffic_filter = TemporalStateFilter()
        self.checker_filter = TemporalBoolFilter()
        self.checker_detection = CheckerboardDetection()
        self.checker_state = False
        self.current_index = -1
        self.requested_index = 0
        self.playing = False
        self._last_analysis: FrameAnalysis | None = None
        self._window_initialized = False

    def close(self) -> None:
        self.bag.close()

    def select_bag(self, position: int) -> None:
        position = int(np.clip(position, 0, len(self.bag_paths) - 1))
        if position == self.bag_position:
            return
        self.bag.close()
        self.bag_position = position
        self.bag = IndexedImageBag(self.bag_paths[position])
        self.current_index = -1
        self.requested_index = 0
        self.playing = False
        if self._window_initialized:
            cv2.setTrackbarMax("Frame", self.window_name, len(self.bag) - 1)
            cv2.setTrackbarPos("Frame", self.window_name, 0)

    def _reset_filters(self) -> None:
        self.traffic_filter.reset()
        self.checker_filter.reset()
        self.checker_detection = CheckerboardDetection()
        self.checker_state = False

    def _process_one(self, index: int) -> FrameAnalysis:
        frame = self.bag.frame(index)
        traffic = self.traffic_detector.detect(frame)
        published_traffic = self.traffic_filter.update(traffic.state)
        if index % 3 == 0:
            self.checker_detection = self.checker_detector.detect(frame)
            self.checker_state = self.checker_filter.update(
                self.checker_detection.detected
            )
        return FrameAnalysis(
            traffic=traffic,
            published_traffic=published_traffic,
            checker=self.checker_detection,
            published_checker=self.checker_state,
        )

    def analyze(self, index: int) -> FrameAnalysis:
        index = int(np.clip(index, 0, len(self.bag) - 1))
        if index == self.current_index + 1 and self.current_index >= 0:
            analysis = self._process_one(index)
        else:
            # Twelve preceding frames are enough to rebuild both debounce
            # filters while keeping random slider jumps responsive.
            self._reset_filters()
            start = max(0, index - 12)
            analysis = FrameAnalysis(
                TrafficDetection(), 0, CheckerboardDetection(), False
            )
            for frame_index in range(start, index + 1):
                analysis = self._process_one(frame_index)
        self.current_index = index
        self.requested_index = index
        self._last_analysis = analysis
        return analysis

    def render(self, index: int) -> np.ndarray:
        analysis = self.analyze(index)
        frame = self.bag.frame(index)
        x0, y0, x1, y1 = self.detection_roi.bounds(frame.shape)
        shaded = frame.copy()
        shaded[:y0, :] = 0
        shaded[y1:, :] = 0
        shaded[y0:y1, :x0] = 0
        shaded[y0:y1, x1:] = 0
        frame = cv2.addWeighted(frame, 0.45, shaded, 0.55, 0.0)
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (255, 180, 0), 2)
        traffic = analysis.traffic
        color = STATE_COLORS[traffic.state]

        if traffic.bbox is not None:
            x, y, width, height = traffic.bbox
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            for label, center in zip(("R", "Y", "L", "G"), traffic.slot_centers):
                point = (int(round(center[0])), int(round(center[1])))
                cv2.circle(frame, point, max(3, int(round(traffic.radius))), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (point[0] - 5, point[1] - int(round(traffic.radius)) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if analysis.checker.bbox is not None:
            x, y, width, height = analysis.checker.bbox
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 255), 3)
            cv2.putText(
                frame,
                "CHECKER",
                (x, max(52, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (20, 20, 20), -1)
        published_color = STATE_COLORS[analysis.published_traffic]
        cv2.putText(
            frame,
            f"{self.bag.name}  frame {index}/{len(self.bag) - 1}  ROI x=15-85% y=0-85%",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"raw={traffic.state}  publish={analysis.published_traffic} "
                f"{STATE_NAMES[analysis.published_traffic]}   "
                f"checker={str(analysis.published_checker).lower()}"
            ),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            published_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "A/D:1  W/S:10  J/L:100  SPACE:play  1-5:bag  C:capture  Q:quit",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return frame

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 720)
        self._window_initialized = True

        def on_trackbar(value: int) -> None:
            self.requested_index = value

        cv2.createTrackbar(
            "Frame", self.window_name, 0, len(self.bag) - 1, on_trackbar
        )
        displayed_index = -1
        while True:
            if self.playing:
                self.requested_index = min(self.requested_index + 1, len(self.bag) - 1)
                if self.requested_index >= len(self.bag) - 1:
                    self.playing = False
            if self.requested_index != displayed_index:
                image = self.render(self.requested_index)
                displayed_index = self.requested_index
                cv2.setTrackbarPos("Frame", self.window_name, displayed_index)
                cv2.imshow(self.window_name, image)

            key = cv2.waitKeyEx(20)
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord(" "),):
                self.playing = not self.playing
            elif key in (ord("a"), ord("A"), 2424832):
                self.playing = False
                self.requested_index = max(0, self.requested_index - 1)
            elif key in (ord("d"), ord("D"), 2555904):
                self.playing = False
                self.requested_index = min(len(self.bag) - 1, self.requested_index + 1)
            elif key in (ord("w"), ord("W")):
                self.playing = False
                self.requested_index = max(0, self.requested_index - 10)
            elif key in (ord("s"), ord("S")):
                self.playing = False
                self.requested_index = min(len(self.bag) - 1, self.requested_index + 10)
            elif key in (ord("j"), ord("J")):
                self.playing = False
                self.requested_index = max(0, self.requested_index - 100)
            elif key in (ord("l"), ord("L")):
                self.playing = False
                self.requested_index = min(len(self.bag) - 1, self.requested_index + 100)
            elif ord("1") <= key <= ord("5"):
                position = key - ord("1")
                if position < len(self.bag_paths):
                    self.select_bag(position)
                    displayed_index = -1
            elif key in (ord("c"), ord("C")):
                capture_directory = Path.cwd() / "viewer_captures"
                capture_directory.mkdir(exist_ok=True)
                capture = self.render(self.requested_index)
                output = capture_directory / f"{self.bag.name}_{self.requested_index:05d}.png"
                cv2.imwrite(str(output), capture)
                print(output)

        cv2.destroyWindow(self.window_name)
        self._window_initialized = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bag_root",
        type=Path,
        help="directory containing imagetrack1, imagetrack2, ...",
    )
    parser.add_argument("--bag", default="imagetrack1", help="initial bag name")
    parser.add_argument("--frame", type=int, default=0, help="initial frame index")
    parser.add_argument("--save", type=Path, help="render one frame and exit")
    arguments = parser.parse_args()

    bag_paths = discover_imagetrack_bags(arguments.bag_root.resolve())
    viewer = BagFrameViewer(bag_paths)
    try:
        names = [path.name for path in bag_paths]
        if arguments.bag in names:
            viewer.bag_position = names.index(arguments.bag)
            viewer.bag.close()
            viewer.bag = IndexedImageBag(bag_paths[viewer.bag_position])
        viewer.requested_index = int(np.clip(arguments.frame, 0, len(viewer.bag) - 1))
        if arguments.save:
            output = arguments.save.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), viewer.render(viewer.requested_index))
            print(output)
        else:
            viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
