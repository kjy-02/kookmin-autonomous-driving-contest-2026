"""Build the v2 single-class traffic-light housing dataset.

The v1 experiment proved that the housing can be localized at the requested
distance, while direct five-state classification confuses lamps that result in
the same driving action.  This v2 dataset labels every reviewed signal state as
one ``traffic_light`` class.  State classification is intentionally deferred to
the four-lamp crop classifier used after detection.

Only manually reviewed empty scenes are written as negative examples.  Empty
labels are never inferred merely because a frame lies outside a positive
review interval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Sequence

import cv2
import numpy as np

from .bag_reader import RosbagImageReader
from .prepare_traffic_dataset import (
    DEFAULT_ANCHORS,
    _normalized_yolo_box,
    propagate_two_anchor_boxes,
)


CLASS_NAME = "traffic_light"

# These ranges were visually checked on the full-resolution source frames.
# The signal housing is completely outside the image throughout every range.
# Missing entries deliberately contribute no tail backgrounds.
CONFIRMED_TAIL_BACKGROUNDS: dict[str, tuple[int, int]] = {
    "traffic_red_02": (192, 218),
    "traffic_red_03": (204, 225),
    "traffic_yellow_01": (174, 180),
    "traffic_yellow_02": (174, 176),
    "traffic_yellow_03": (168, 185),
    "traffic_leftgreen_02": (162, 166),
    "traffic_leftgreen_03": (186, 190),
    "traffic_none_01": (180, 184),
    "traffic_none_02": (174, 182),
    "traffic_none_03": (300, 307),
}


def _write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        'train: images/train\nval: images/val\nnames:\n  0: "traffic_light"\n',
        encoding="utf-8",
    )


def _write_item(
    output: Path,
    *,
    stem: str,
    split: str,
    frame_bgr: np.ndarray,
    timestamp_ns: int,
    source_name: str,
    source_frame: int,
    source_kind: str,
    review: str,
    source_state: str | None,
    box: Sequence[float] | None,
    jpeg_quality: int,
) -> dict[str, object]:
    image_relative = Path("images") / split / f"{stem}.jpg"
    label_relative = Path("labels") / split / f"{stem}.txt"
    image_path = output / image_relative
    label_path = output / label_relative
    if not cv2.imwrite(
        str(image_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    ):
        raise RuntimeError(f"Failed to write {image_path}")

    if box is None:
        label_path.write_text("", encoding="utf-8")
        manifest_box: list[float] | None = None
        item_class = "background"
    else:
        normalized = _normalized_yolo_box(box, frame_bgr.shape)
        label_path.write_text(
            "0 " + " ".join(f"{value:.6f}" for value in normalized) + "\n",
            encoding="utf-8",
        )
        manifest_box = [round(float(value), 3) for value in box]
        item_class = CLASS_NAME

    return {
        "source": source_name,
        "source_kind": source_kind,
        "frame": source_frame,
        "timestamp_ns": timestamp_ns,
        "split": split,
        "class": item_class,
        "source_state": source_state,
        "review": review,
        "bbox": manifest_box,
        "image": image_relative.as_posix(),
        "label": label_relative.as_posix(),
    }


def prepare_housing_dataset(
    traffic_bag_root: Path,
    output: Path,
    *,
    driving_background_bag: Path | None = None,
    driving_background_topic: str = "/image_raw/compressed",
    traffic_stride: int = 6,
    driving_background_stride: int = 30,
    jpeg_quality: int = 92,
    overwrite: bool = False,
) -> dict[str, object]:
    if traffic_stride < 1 or driving_background_stride < 1:
        raise ValueError("strides must be at least 1")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for bag_name, anchor in DEFAULT_ANCHORS.items():
        source = traffic_bag_root / bag_name
        if not source.exists():
            raise FileNotFoundError(f"Required bag is missing: {source}")
        reader = RosbagImageReader(source)
        gray_frames = [
            cv2.cvtColor(item.frame_bgr, cv2.COLOR_BGR2GRAY) for item in reader.frames()
        ]
        boxes = propagate_two_anchor_boxes(gray_frames, anchor)
        reviewed_indices = range(anchor.first_labeled, anchor.last_labeled + 1)
        if any(boxes[index] is None for index in reviewed_indices):
            first_missing = next(index for index in reviewed_indices if boxes[index] is None)
            raise RuntimeError(
                f"{bag_name}: tracking stopped at frame {first_missing}; "
                "the anchor needs review"
            )

        split = "val" if bag_name.endswith("_03") else "train"
        background_range = CONFIRMED_TAIL_BACKGROUNDS.get(bag_name)
        for frame in reader.frames(stride=traffic_stride):
            if anchor.first_labeled <= frame.index <= anchor.last_labeled:
                box = boxes[frame.index]
                assert box is not None
                review = "user_reviewed_positive"
                source_state: str | None = anchor.class_name
            elif (
                background_range is not None
                and background_range[0] <= frame.index <= background_range[1]
            ):
                box = None
                review = "manually_confirmed_empty_tail"
                source_state = None
            else:
                continue

            manifest.append(
                _write_item(
                    output,
                    stem=f"{bag_name}_{frame.index:06d}",
                    split=split,
                    frame_bgr=frame.frame_bgr,
                    timestamp_ns=frame.timestamp_ns,
                    source_name=bag_name,
                    source_frame=frame.index,
                    source_kind="traffic_sequence",
                    review=review,
                    source_state=source_state,
                    box=box,
                    jpeg_quality=jpeg_quality,
                )
            )
        del gray_frames, boxes

    if driving_background_bag is not None:
        background_reader = RosbagImageReader(
            driving_background_bag, topic=driving_background_topic
        )
        source_name = driving_background_bag.name
        for frame in background_reader.frames(stride=driving_background_stride):
            manifest.append(
                _write_item(
                    output,
                    stem=f"background_{source_name}_{frame.index:06d}",
                    split="train",
                    frame_bgr=frame.frame_bgr,
                    timestamp_ns=frame.timestamp_ns,
                    source_name=source_name,
                    source_frame=frame.index,
                    source_kind="driving_background",
                    review="full_sequence_confirmed_no_signal",
                    source_state=None,
                    box=None,
                    jpeg_quality=jpeg_quality,
                )
            )

    _write_data_yaml(output)
    (output / "anchors.json").write_text(
        json.dumps(
            {name: asdict(anchor) for name, anchor in DEFAULT_ANCHORS.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    positive_count = sum(item["class"] == CLASS_NAME for item in manifest)
    background_count = len(manifest) - positive_count
    split_counts = {
        split: sum(item["split"] == split for item in manifest)
        for split in ("train", "val")
    }
    summary: dict[str, object] = {
        "profile": "traffic_housing_v2",
        "traffic_source": str(traffic_bag_root.resolve()),
        "driving_background_source": (
            None
            if driving_background_bag is None
            else str(driving_background_bag.resolve())
        ),
        "output": str(output.resolve()),
        "traffic_stride": traffic_stride,
        "driving_background_stride": driving_background_stride,
        "images": len(manifest),
        "positive_images": positive_count,
        "background_images": background_count,
        "split_counts": split_counts,
        "confirmed_tail_backgrounds": CONFIRMED_TAIL_BACKGROUNDS,
        "items": manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the single-class traffic-light housing YOLO dataset"
    )
    parser.add_argument(
        "traffic_bag_root", type=Path, help="Folder containing traffic_* bags"
    )
    parser.add_argument("output", type=Path, help="New v2 dataset output folder")
    parser.add_argument(
        "--driving-background-bag",
        type=Path,
        help="Reviewed bag containing no traffic signal",
    )
    parser.add_argument(
        "--driving-background-topic", default="/image_raw/compressed"
    )
    parser.add_argument("--traffic-stride", type=int, default=6)
    parser.add_argument("--driving-background-stride", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = prepare_housing_dataset(
        args.traffic_bag_root,
        args.output,
        driving_background_bag=args.driving_background_bag,
        driving_background_topic=args.driving_background_topic,
        traffic_stride=args.traffic_stride,
        driving_background_stride=args.driving_background_stride,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
