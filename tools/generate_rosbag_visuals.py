#!/usr/bin/env python3
"""Generate portfolio-ready evidence images from the local ROS 2 bags.

The raw bag directory is intentionally gitignored because several SQLite files
exceed GitHub's file limit.  This script keeps the published evidence
reproducible without copying the 4.68 GiB source dataset into the repository.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import shutil
import sqlite3
import struct
import sys
import tempfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "final_drive" / "final_drive"
sys.path.insert(0, str(PACKAGE_ROOT))

from final_drive_vision.bag_reader import RosbagImageReader  # noqa: E402
from final_drive_vision.detectors import (  # noqa: E402
    TRAFFIC_GREEN,
    TRAFFIC_GREEN_LEFT,
    TRAFFIC_RED,
    TRAFFIC_YELLOW,
)
from final_drive_vision.traffic_housing_detector import (  # noqa: E402
    TrafficHousingDetector,
)


WHITE = (245, 247, 250)
INK = (31, 41, 55)
MUTED = (100, 116, 139)
BLUE = (220, 110, 30)
GREEN = (65, 170, 65)
RED = (55, 65, 220)
YELLOW = (30, 190, 235)
ORANGE = (20, 130, 245)
PANEL_BORDER = (215, 220, 228)


def put(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = INK,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 7])
    if not encoded:
        raise RuntimeError(f"could not write {path}")
    path.write_bytes(buffer.tobytes())


def float_array(payload: bytes) -> tuple[float, ...]:
    """Decode std_msgs/Float32MultiArray payloads from these little-endian bags."""

    if len(payload) < 16:
        return ()
    count = struct.unpack_from("<I", payload, 12)[0]
    end = 16 + count * 4
    if count > 1024 or end > len(payload):
        return ()
    return struct.unpack_from(f"<{count}f", payload, 16)


def topic_rows(database: Path, topic: str) -> list[tuple[int, tuple[float, ...]]]:
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT id FROM topics WHERE name=?", (topic,)
        ).fetchone()
        if row is None:
            raise ValueError(f"topic {topic} not found in {database}")
        return [
            (int(timestamp), float_array(payload))
            for timestamp, payload in connection.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (int(row[0]),),
            )
        ]


def nearest_row(
    rows: list[tuple[int, tuple[float, ...]]], timestamp: int
) -> tuple[int, tuple[float, ...]]:
    times = [row[0] for row in rows]
    index = bisect.bisect_left(times, timestamp)
    choices = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(rows)]
    return min((rows[candidate] for candidate in choices), key=lambda row: abs(row[0] - timestamp))


def hough_candidates(frame: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Run the same ROI/Canny/Hough settings used by camera_drive_node.cpp."""

    resized = cv2.resize(frame, (640, 480))
    roi = resized[300:380]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 50, minLineLength=50, maxLineGap=20)
    output: list[tuple[np.ndarray, bool]] = []
    if lines is None:
        return output
    for line in lines[:, 0]:
        x1, y1, x2, y2 = (int(value) for value in line)
        slope = 1000.0 if x1 == x2 else (y2 - y1) / (x2 - x1)
        if abs(slope) <= 0.15:
            continue
        if slope < 0.0 and x1 < 320:
            output.append((line, True))
        elif slope > 0.0 and x1 > 320:
            output.append((line, False))
    return output


def lane_panel(
    frame: np.ndarray,
    state: tuple[float, ...],
    elapsed: float,
    source_index: int,
) -> np.ndarray:
    image = cv2.resize(frame, (640, 480))
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 300), (639, 379), (255, 180, 60), -1)
    image = cv2.addWeighted(overlay, 0.12, image, 0.88, 0)

    for line, is_left in hough_candidates(image):
        x1, y1, x2, y2 = (int(value) for value in line)
        color = BLUE if is_left else YELLOW
        cv2.line(image, (x1, y1 + 300), (x2, y2 + 300), color, 2, cv2.LINE_AA)

    found, left_x, right_x, center_x, width = state[:5]
    evaluation_y = 335
    cv2.rectangle(image, (0, 300), (639, 379), (255, 210, 80), 2)
    cv2.line(image, (0, evaluation_y), (639, evaluation_y), WHITE, 1, cv2.LINE_AA)
    for x, color in ((left_x, BLUE), (right_x, YELLOW), (center_x, GREEN)):
        cv2.circle(image, (int(round(x)), evaluation_y), 8, color, -1, cv2.LINE_AA)
        cv2.line(image, (int(round(x)), evaluation_y - 24), (int(round(x)), evaluation_y + 24), color, 2)
    cv2.line(image, (320, 470), (int(round(center_x)), evaluation_y), GREEN, 3, cv2.LINE_AA)

    panel = np.full((560, 640, 3), WHITE, np.uint8)
    panel[:480] = image
    cv2.rectangle(panel, (0, 0), (639, 559), PANEL_BORDER, 2)
    put(panel, f"t={elapsed:.2f}s  frame={source_index}", (18, 505), 0.55, INK, 2)
    put(
        panel,
        f"detected={bool(found)}  L={left_x:.0f}px  R={right_x:.0f}px  center={center_x:.0f}px",
        (18, 530),
        0.48,
        INK,
        1,
    )
    put(panel, f"lane width={width:.0f}px", (18, 550), 0.46, MUTED, 1)
    return panel


def generate_lane(data_root: Path, output: Path) -> dict[str, object]:
    database = data_root / "rosbag" / "run_03_slow" / "run_03_slow_0.db3"
    lane_rows = topic_rows(database, "/final_drive/camera_lane_state")
    detected = [row for row in lane_rows if len(row[1]) >= 5 and row[1][0] > 0.5]
    quantiles = (0.40, 0.65, 0.85)
    targets = [detected[round(q * (len(detected) - 1))][0] for q in quantiles]
    target_rows = [nearest_row(lane_rows, target) for target in targets]

    reader = RosbagImageReader(database, topic="/image_raw/compressed")
    selected: list[tuple[object, tuple[float, ...]]] = []
    best: list[tuple[int, object] | None] = [None] * len(targets)
    for frame in reader.frames():
        for index, target in enumerate(targets):
            distance = abs(frame.timestamp_ns - target)
            if best[index] is None or distance < best[index][0]:
                best[index] = (distance, frame)
    origin = min(lane_rows[0][0], min(item[1].timestamp_ns for item in best if item))
    for index, item in enumerate(best):
        assert item is not None
        frame = item[1]
        _, state = nearest_row(lane_rows, frame.timestamp_ns)
        selected.append((frame, state))

    panels = [
        lane_panel(
            frame.frame_bgr,
            state,
            (frame.timestamp_ns - origin) / 1e9,
            frame.index,
        )
        for frame, state in selected
    ]
    header = np.full((82, 640 * len(panels), 3), WHITE, np.uint8)
    put(header, "Lane detection replay - recorded camera_lane_state", (24, 32), 0.82, INK, 2)
    put(
        header,
        "blue: left Hough candidates   yellow: right candidates   green: recorded lane center   shaded: ROI y=300..379",
        (24, 63),
        0.50,
        MUTED,
        1,
    )
    canvas = np.vstack([header, np.hstack(panels)])
    path = output / "lane-detection-replay.png"
    save_png(path, canvas)
    return {
        "output": path.relative_to(ROOT).as_posix(),
        "source": database.relative_to(ROOT).as_posix(),
        "topic": "/final_drive/camera_lane_state",
        "frames": [
            {
                "index": frame.index,
                "timestamp_ns": frame.timestamp_ns,
                "state": [round(float(value), 3) for value in state],
            }
            for frame, state in selected
        ],
    }


TRAFFIC_LABELS = {
    TRAFFIC_RED: "RED",
    TRAFFIC_YELLOW: "YELLOW",
    TRAFFIC_GREEN: "GREEN",
    TRAFFIC_GREEN_LEFT: "GREEN + LEFT",
}


def traffic_panel(frame: np.ndarray, detection, debug: dict[str, object], index: int) -> np.ndarray:
    image = cv2.resize(frame, (640, 480))
    if detection.bbox:
        x, y, width, height = detection.bbox
        cv2.rectangle(image, (x, y), (x + width, y + height), GREEN, 3, cv2.LINE_AA)
    values = tuple(float(value) for value in debug.get("slot_brightness", ()))
    for slot_index, center in enumerate(detection.slot_centers):
        point = tuple(int(round(value)) for value in center)
        active = slot_index < len(values) and values[slot_index] >= 240.0
        color = GREEN if active else MUTED
        cv2.circle(image, point, max(4, int(round(detection.radius))), color, 2 if active else 1, cv2.LINE_AA)
        if slot_index < len(values):
            put(image, f"{('R','Y','L','G')[slot_index]} {values[slot_index]:.0f}", (point[0] - 18, point[1] - 12), 0.38, color, 1)

    panel = np.full((552, 640, 3), WHITE, np.uint8)
    panel[:480] = image
    cv2.rectangle(panel, (0, 0), (639, 551), PANEL_BORDER, 2)
    state_confidence = float(debug.get("state_confidence", 0.0))
    put(panel, TRAFFIC_LABELS[detection.state], (18, 508), 0.68, INK, 2)
    put(
        panel,
        f"frame={index}  housing={detection.confidence:.3f}  state={state_confidence:.3f}",
        (18, 538),
        0.50,
        MUTED,
        1,
    )
    return panel


def generate_traffic(data_root: Path, output: Path) -> dict[str, object]:
    traffic_root = data_root / "traffic_rosbag"
    sources = [
        ("traffic_red_03", TRAFFIC_RED),
        ("traffic_yellow_03", TRAFFIC_YELLOW),
        ("traffic_green_03", TRAFFIC_GREEN),
        ("traffic_leftgreen_03", TRAFFIC_GREEN_LEFT),
    ]
    # OpenCV-DNN on Windows cannot open a model through a non-ASCII parent
    # directory. Load an ASCII temporary copy, then discard it once the network
    # is resident in memory.
    with tempfile.TemporaryDirectory(prefix="final-drive-model-") as temp_dir:
        temporary_model = Path(temp_dir) / "traffic_housing_v2_best.onnx"
        shutil.copy2(
            PACKAGE_ROOT / "models" / "traffic_housing_v2_best.onnx",
            temporary_model,
        )
        detector = TrafficHousingDetector(temporary_model)
    panels: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for name, expected in sources:
        database = traffic_root / name / f"{name}_0.db3"
        reader = RosbagImageReader(database)
        best = None
        for frame in reader.frames(stride=6):
            detection = detector.detect(frame.frame_bgr)
            if detection.state != expected:
                continue
            debug = dict(detector.last_debug)
            score = detection.confidence + float(debug.get("state_confidence", 0.0))
            if best is None or score > best[0]:
                best = (score, frame, detection, debug)
        if best is None:
            raise RuntimeError(f"no expected {TRAFFIC_LABELS[expected]} detection in {database}")
        _, frame, detection, debug = best
        panels.append(traffic_panel(frame.frame_bgr, detection, debug, frame.index))
        records.append(
            {
                "source": database.relative_to(ROOT).as_posix(),
                "frame": frame.index,
                "state": TRAFFIC_LABELS[expected],
                "housing_confidence": round(float(detection.confidence), 4),
                "state_confidence": round(float(debug.get("state_confidence", 0.0)), 4),
                "slot_brightness": [
                    round(float(value), 2) for value in debug.get("slot_brightness", ())
                ],
            }
        )

    body = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    header = np.full((82, body.shape[1], 3), WHITE, np.uint8)
    put(header, "Traffic-light replay - YOLO housing + fixed R/Y/L/G slots", (24, 32), 0.82, INK, 2)
    put(
        header,
        "box: detected housing   circles: sampled lamp slots   value: brightest 20% mean",
        (24, 63),
        0.50,
        MUTED,
        1,
    )
    canvas = np.vstack([header, body])
    path = output / "traffic-light-detection-replay.png"
    save_png(path, canvas)
    return {
        "output": path.relative_to(ROOT).as_posix(),
        "model": "final_drive/final_drive/models/traffic_housing_v2_best.onnx",
        "sampling_stride": 6,
        "frames": records,
    }


def cone_panel(frame: dict[str, object], caption: str) -> np.ndarray:
    width, height = 520, 580
    top, bottom, margin = 62, 530, 48
    image = np.full((height, width, 3), WHITE, np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), PANEL_BORDER, 2)
    put(image, caption, (18, 26), 0.62, INK, 2)
    put(
        image,
        f"t={float(frame['t']):.2f}s  status={frame['status']}  angle={float(frame['angle']):.1f}  speed={float(frame['speed']):.1f}",
        (18, 51),
        0.46,
        MUTED,
        1,
    )

    x_min, x_max = -1.0, 1.0
    y_min, y_max = -0.18, 1.30

    def project(point) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        px = margin + int(round((x_max - x) / (x_max - x_min) * (width - 2 * margin)))
        py = bottom - int(round((y - y_min) / (y_max - y_min) * (bottom - top)))
        return px, py

    for distance in (0.0, 0.5, 1.0):
        y = project((0.0, distance))[1]
        cv2.line(image, (margin, y), (width - margin, y), PANEL_BORDER, 1)
        put(image, f"{distance:.1f}m", (6, y + 4), 0.38, MUTED, 1)
    center_x = project((0.0, 0.0))[0]
    cv2.line(image, (center_x, top), (center_x, bottom), PANEL_BORDER, 1)

    for point in frame.get("scan", []):
        x, y = float(point[0]), float(point[1])
        if x_min <= x <= x_max and y_min <= y <= y_max:
            cv2.circle(image, project(point), 2, (175, 180, 188), -1)
    for point in frame.get("cones", []):
        cv2.circle(image, project(point), 7, ORANGE, 2, cv2.LINE_AA)

    for key, color in (("left", BLUE), ("right", RED)):
        points = sorted(frame.get(key, []), key=lambda point: float(point[1]))
        projected = [project(point) for point in points]
        if len(projected) >= 2:
            cv2.polylines(image, [np.asarray(projected, np.int32)], False, color, 3, cv2.LINE_AA)
        for point in projected:
            cv2.circle(image, point, 7, color, -1, cv2.LINE_AA)

    vehicle = np.asarray(
        [project((-0.12, -0.08)), project((0.12, -0.08)), project((0.0, 0.12))],
        np.int32,
    )
    cv2.fillConvexPoly(image, vehicle, INK, cv2.LINE_AA)
    angle = math.radians(float(frame.get("angle") or 0.0))
    arrow_end = project((0.45 * math.sin(angle), 0.45 * math.cos(angle)))
    cv2.arrowedLine(image, project((0.0, 0.10)), arrow_end, GREEN, 4, cv2.LINE_AA, tipLength=0.2)
    put(image, "+X left", (width - 98, height - 26), 0.42, MUTED, 1)
    put(image, "+Y forward", (18, height - 26), 0.42, MUTED, 1)
    return image


def generate_cones(data_root: Path, output: Path) -> dict[str, object]:
    source = data_root / "analysis" / "cone_visualization_data.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    run = next(item for item in data["runs"] if item["id"] == "03")
    frames = [
        frame
        for frame in run["frames"]
        if frame.get("status") == "BOTH" and abs(float(frame.get("speed") or 0.0)) > 0.1
    ]
    selected = [
        min(frames, key=lambda frame: float(frame.get("angle") or 0.0)),
        min(frames, key=lambda frame: abs(float(frame.get("angle") or 0.0))),
        max(frames, key=lambda frame: float(frame.get("angle") or 0.0)),
    ]
    captions = ("strong right command", "near-straight command", "left command")
    panels = [cone_panel(frame, caption) for frame, caption in zip(selected, captions)]
    header = np.full((82, panels[0].shape[1] * len(panels), 3), WHITE, np.uint8)
    put(header, "LiDAR cone replay - clusters, lane identity, and steering command", (24, 32), 0.82, INK, 2)
    put(
        header,
        "gray: scan points   orange ring: cone candidate   blue/red: tracked cone rows   green arrow: command direction",
        (24, 63),
        0.50,
        MUTED,
        1,
    )
    canvas = np.vstack([header, np.hstack(panels)])
    path = output / "lidar-cone-detection-replay.png"
    save_png(path, canvas)
    return {
        "output": path.relative_to(ROOT).as_posix(),
        "source": source.relative_to(ROOT).as_posix(),
        "run": run["id"],
        "frames": [
            {
                "index": frame["i"],
                "time_sec": frame["t"],
                "status": frame["status"],
                "angle": frame["angle"],
                "speed": frame["speed"],
                "cone_count": len(frame.get("cones", [])),
            }
            for frame in selected
        ],
    }


def copy_overtake(data_root: Path, output: Path) -> list[str]:
    source_root = data_root / "analysis" / "overtake"
    mappings = {
        "ghlvl1_montage.png": "lidar-overtake-frames.png",
        "ghlvl1_timeline.png": "lidar-overtake-timeline.png",
    }
    copied: list[str] = []
    for source_name, output_name in mappings.items():
        target = output / output_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / source_name, target)
        copied.append(target.relative_to(ROOT).as_posix())
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "로스백 데이터들",
        help="directory containing the local ROS bag collection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "media" / "rosbag-analysis",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data directory not found: {data_root}")

    summary = {
        "lane": generate_lane(data_root, output),
        "traffic_light": generate_traffic(data_root, output),
        "cones": generate_cones(data_root, output),
        "overtake_outputs": copy_overtake(data_root, output),
        "note": "Raw db3 files remain local; outputs are derived evidence for the portfolio.",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
