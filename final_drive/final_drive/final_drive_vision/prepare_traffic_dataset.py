"""Create a reviewed-first YOLO traffic-light dataset from the recorded bags.

The recordings were made by moving one camera toward one four-lamp signal.
Each sequence uses two manually checked anchor boxes.  Sparse Lucas-Kanade
optical flow propagates and blends those boxes over the reviewed interval.
Frames outside the user-reviewed interval are deliberately excluded instead of
being turned into false-negative background examples.

This utility is deliberately offline-only.  It does not change the detector
used by the live driving launch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Sequence

import cv2
import numpy as np

from .bag_reader import RosbagImageReader


CLASS_NAMES = ("red", "yellow", "green", "left_green", "off")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class SequenceAnchor:
    class_name: str
    frame_index: int
    bbox: tuple[float, float, float, float]
    near_frame: int
    near_bbox: tuple[float, float, float, float]
    first_labeled: int
    last_labeled: int


# The two boxes below were checked against a middle and near frame.  The labeled
# bounds are source-frame indices converted from the 1-based positions reviewed
# in the original-quality viewer on 2026-08-22.  Only those reviewed bounds are
# exported.  Re-run with a new config when a differently captured dataset is
# used.
DEFAULT_ANCHORS: dict[str, SequenceAnchor] = {
    "traffic_red_01": SequenceAnchor(
        "red", 136, (266, 190, 72, 27), 240, (191, 63, 176, 50), 108, 270
    ),
    "traffic_red_02": SequenceAnchor(
        "red", 109, (321, 182, 80, 27), 150, (246, 108, 154, 45), 90, 168
    ),
    "traffic_red_03": SequenceAnchor(
        "red", 113, (341, 189, 56, 22), 168, (267, 107, 135, 42), 102, 186
    ),
    "traffic_yellow_01": SequenceAnchor(
        "yellow", 90, (249, 187, 57, 22), 156, (235, 34, 189, 58), 84, 156
    ),
    "traffic_yellow_02": SequenceAnchor(
        "yellow", 88, (282, 191, 61, 24), 150, (247, 83, 145, 44), 78, 156
    ),
    "traffic_yellow_03": SequenceAnchor(
        "yellow", 96, (286, 169, 78, 27), 138, (248, 99, 126, 39), 90, 150
    ),
    "traffic_green_01": SequenceAnchor(
        "green", 290, (301, 181, 66, 24), 576, (198, 84, 168, 48), 156, 564
    ),
    "traffic_green_02": SequenceAnchor(
        "green", 181, (272, 167, 76, 27), 360, (143, 21, 219, 67), 162, 342
    ),
    "traffic_green_03": SequenceAnchor(
        "green", 175, (292, 183, 55, 21), 348, (159, 35, 208, 62), 198, 324
    ),
    "traffic_leftgreen_01": SequenceAnchor(
        "left_green", 96, (279, 196, 55, 21), 180, (236, 28, 209, 64), 96, 180
    ),
    "traffic_leftgreen_02": SequenceAnchor(
        "left_green", 83, (273, 189, 55, 22), 144, (247, 65, 169, 51), 96, 150
    ),
    "traffic_leftgreen_03": SequenceAnchor(
        "left_green", 95, (297, 191, 55, 22), 162, (252, 88, 148, 45), 114, 174
    ),
    "traffic_none_01": SequenceAnchor(
        "off", 92, (276, 196, 52, 21), 156, (259, 122, 119, 37), 102, 168
    ),
    "traffic_none_02": SequenceAnchor(
        "off", 91, (270, 189, 53, 21), 156, (238, 88, 150, 46), 78, 162
    ),
    "traffic_none_03": SequenceAnchor(
        "off", 234, (275, 176, 69, 23), 294, (184, 13, 209, 64), 234, 294
    ),
}


def _clamp_box(
    box: Sequence[float], frame_shape: Sequence[int]
) -> np.ndarray:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x, y, box_width, box_height = (float(value) for value in box)
    x0 = max(0.0, min(x, width - 2.0))
    y0 = max(0.0, min(y, height - 2.0))
    x1 = max(x0 + 2.0, min(x + box_width, width - 1.0))
    y1 = max(y0 + 2.0, min(y + box_height, height - 1.0))
    return np.asarray((x0, y0, x1 - x0, y1 - y0), dtype=np.float32)


def _feature_points(gray: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    x, y, width, height = (float(value) for value in box)
    margin_x = 0.08 * width
    margin_y = 0.15 * height
    x0 = max(0, int(np.floor(x - margin_x)))
    y0 = max(0, int(np.floor(y - margin_y)))
    x1 = min(gray.shape[1], int(np.ceil(x + width + margin_x + 1)))
    y1 = min(gray.shape[0], int(np.ceil(y + height + margin_y + 1)))
    mask = np.zeros(gray.shape, np.uint8)
    mask[y0:y1, x0:x1] = 255
    return cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=2.0,
        blockSize=3,
    )


def track_box_step(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    box: Sequence[float],
) -> np.ndarray | None:
    """Move one box between adjacent frames using a robust similarity fit."""

    points = _feature_points(previous_gray, box)
    if points is None or len(points) < 4:
        return None
    moved, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if moved is None or status is None:
        return None
    good = status.reshape(-1).astype(bool)
    if error is not None:
        good &= error.reshape(-1) < 30.0
    source = points.reshape(-1, 2)[good]
    target = moved.reshape(-1, 2)[good]
    if len(source) < 4:
        return None

    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=1000,
        confidence=0.98,
    )
    if matrix is None or inliers is None or int(inliers.sum()) < 4:
        return None
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if not 0.94 <= scale <= 1.08:
        return None

    x, y, width, height = (float(value) for value in box)
    center = np.asarray((((x + 0.5 * width), (y + 0.5 * height)),), np.float32)
    moved_center = cv2.transform(center.reshape(-1, 1, 2), matrix).reshape(2)
    new_width = width * scale
    new_height = height * scale
    return _clamp_box(
        (
            moved_center[0] - 0.5 * new_width,
            moved_center[1] - 0.5 * new_height,
            new_width,
            new_height,
        ),
        current_gray.shape,
    )


def propagate_boxes(
    gray_frames: Sequence[np.ndarray],
    anchor_index: int,
    anchor_box: Sequence[float],
) -> list[np.ndarray | None]:
    """Track an anchor over an entire sequence in both time directions."""

    if not 0 <= anchor_index < len(gray_frames):
        raise ValueError(f"Anchor frame {anchor_index} is outside the sequence")
    boxes: list[np.ndarray | None] = [None] * len(gray_frames)
    boxes[anchor_index] = _clamp_box(anchor_box, gray_frames[anchor_index].shape)
    for direction in (-1, 1):
        index = anchor_index
        box = boxes[index]
        assert box is not None
        while 0 <= index + direction < len(gray_frames):
            next_index = index + direction
            moved = track_box_step(gray_frames[index], gray_frames[next_index], box)
            if moved is None:
                break
            boxes[next_index] = moved
            box = moved
            index = next_index
    return boxes


def propagate_two_anchor_boxes(
    gray_frames: Sequence[np.ndarray],
    anchor: SequenceAnchor,
) -> list[np.ndarray | None]:
    """Blend a far/middle track with a near track to reject rail parallax.

    The horizontal rail overlaps the signal in close views and eventually
    dominates a single optical-flow track.  A second checked box near the end
    keeps the track attached to the housing.  Smooth blending between the two
    tracks avoids a discontinuity in the generated labels.
    """

    if anchor.near_frame <= anchor.frame_index:
        raise ValueError("near_frame must be later than frame_index")
    middle_track = propagate_boxes(gray_frames, anchor.frame_index, anchor.bbox)
    near_track = propagate_boxes(gray_frames, anchor.near_frame, anchor.near_bbox)
    output: list[np.ndarray | None] = []
    duration = float(anchor.near_frame - anchor.frame_index)
    for index, (middle_box, near_box) in enumerate(zip(middle_track, near_track)):
        if index <= anchor.frame_index:
            output.append(middle_box)
            continue
        if index >= anchor.near_frame:
            output.append(near_box)
            continue
        if middle_box is None:
            output.append(near_box)
            continue
        if near_box is None:
            output.append(middle_box)
            continue
        ratio = (index - anchor.frame_index) / duration
        smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        output.append((1.0 - smooth_ratio) * middle_box + smooth_ratio * near_box)
    return output


def _normalized_yolo_box(
    box: Sequence[float], frame_shape: Sequence[int]
) -> tuple[float, float, float, float]:
    height, width = float(frame_shape[0]), float(frame_shape[1])
    x, y, box_width, box_height = (float(value) for value in box)
    return (
        (x + 0.5 * box_width) / width,
        (y + 0.5 * box_height) / height,
        box_width / width,
        box_height / height,
    )


def _thumbnail(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)


def _write_data_yaml(output: Path) -> None:
    names = "\n".join(
        f"  {index}: {json.dumps(name)}" for index, name in enumerate(CLASS_NAMES)
    )
    (output / "data.yaml").write_text(
        # Omitting ``path`` makes Ultralytics resolve train/val from this YAML's
        # own directory, so the dataset remains portable between Windows,
        # Colab, and the Ubuntu vehicle computer.
        "train: images/train\nval: images/val\n"
        + f"names:\n{names}\n",
        encoding="utf-8",
    )


def prepare_dataset(
    bag_root: Path,
    output: Path,
    *,
    stride: int = 6,
    jpeg_quality: int = 92,
    minimum_frame_difference: float = 0.8,
    overwrite: bool = False,
) -> dict[str, object]:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    skipped_duplicates = 0
    excluded_outside_review = 0
    for bag_name, anchor in DEFAULT_ANCHORS.items():
        source = bag_root / bag_name
        if not source.exists():
            raise FileNotFoundError(f"Required bag is missing: {source}")
        reader = RosbagImageReader(source)
        gray_frames = [
            cv2.cvtColor(item.frame_bgr, cv2.COLOR_BGR2GRAY) for item in reader.frames()
        ]
        boxes = propagate_two_anchor_boxes(gray_frames, anchor)
        reviewed_indices = range(anchor.first_labeled, anchor.last_labeled + 1)
        if any(boxes[index] is None for index in reviewed_indices):
            first_missing = next(
                index
                for index in reviewed_indices
                if boxes[index] is None
            )
            raise RuntimeError(
                f"{bag_name}: tracking stopped at frame {first_missing}; "
                "the anchor needs review"
            )

        split = "val" if bag_name.endswith("_03") else "train"
        previous_kept: np.ndarray | None = None
        for frame in reader.frames(stride=stride):
            if not anchor.first_labeled <= frame.index <= anchor.last_labeled:
                excluded_outside_review += 1
                continue
            gray = cv2.cvtColor(frame.frame_bgr, cv2.COLOR_BGR2GRAY)
            thumb = _thumbnail(gray)
            is_boundary = frame.index in {
                anchor.first_labeled,
                anchor.last_labeled,
                anchor.frame_index,
                anchor.near_frame,
            }
            difference = (
                float("inf")
                if previous_kept is None
                else float(cv2.absdiff(thumb, previous_kept).mean())
            )
            if not is_boundary and difference < minimum_frame_difference:
                skipped_duplicates += 1
                continue
            previous_kept = thumb

            stem = f"{bag_name}_{frame.index:06d}"
            image_relative = Path("images") / split / f"{stem}.jpg"
            label_relative = Path("labels") / split / f"{stem}.txt"
            image_path = output / image_relative
            label_path = output / label_relative
            if not cv2.imwrite(
                str(image_path),
                frame.frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise RuntimeError(f"Failed to write {image_path}")

            box = boxes[frame.index]
            assert box is not None
            class_id = CLASS_IDS[anchor.class_name]
            normalized = _normalized_yolo_box(box, frame.frame_bgr.shape)
            label_path.write_text(
                f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized) + "\n",
                encoding="utf-8",
            )
            manifest_box: list[float] = [round(float(value), 3) for value in box]
            manifest.append(
                {
                    "bag": bag_name,
                    "frame": frame.index,
                    "timestamp_ns": frame.timestamp_ns,
                    "split": split,
                    "class": anchor.class_name,
                    "bbox": manifest_box,
                    "image": image_relative.as_posix(),
                    "label": label_relative.as_posix(),
                    "difference": None if not np.isfinite(difference) else round(difference, 3),
                }
            )
        del gray_frames, boxes

    _write_data_yaml(output)
    (output / "anchors.json").write_text(
        json.dumps(
            {name: asdict(anchor) for name, anchor in DEFAULT_ANCHORS.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    class_counts = {
        name: sum(1 for item in manifest if item["class"] == name)
        for name in (*CLASS_NAMES, "background")
    }
    split_counts = {
        split: sum(1 for item in manifest if item["split"] == split)
        for split in ("train", "val")
    }
    summary: dict[str, object] = {
        "source": str(bag_root.resolve()),
        "output": str(output.resolve()),
        "stride": stride,
        "minimum_frame_difference": minimum_frame_difference,
        "images": len(manifest),
        "skipped_duplicates": skipped_duplicates,
        "excluded_outside_review": excluded_outside_review,
        "class_counts": class_counts,
        "split_counts": split_counts,
        "items": manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a tracked traffic-light YOLO dataset from traffic_rosbag"
    )
    parser.add_argument("bag_root", type=Path, help="Folder containing traffic_* bag folders")
    parser.add_argument("output", type=Path, help="New dataset output folder")
    parser.add_argument("--stride", type=int, default=6, help="Keep every Nth source frame")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--min-frame-difference", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = prepare_dataset(
        args.bag_root,
        args.output,
        stride=args.stride,
        jpeg_quality=args.jpeg_quality,
        minimum_frame_difference=args.min_frame_difference,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
