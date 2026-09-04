from __future__ import annotations

import numpy as np

import final_drive_vision.prepare_traffic_dataset as dataset


def test_default_anchors_have_two_ordered_checked_frames() -> None:
    assert len(dataset.DEFAULT_ANCHORS) == 15
    for anchor in dataset.DEFAULT_ANCHORS.values():
        assert anchor.class_name in dataset.CLASS_NAMES
        assert 0 <= anchor.first_labeled <= anchor.last_labeled
        assert 0 <= anchor.frame_index < anchor.near_frame
        assert all(value >= 0 for value in anchor.bbox)
        assert all(value >= 0 for value in anchor.near_bbox)


def test_normalized_box_uses_center_width_height_order() -> None:
    result = dataset._normalized_yolo_box((64, 48, 128, 96), (480, 640, 3))
    assert result == (0.2, 0.2, 0.2, 0.2)


def test_two_anchor_blend_uses_each_checked_endpoint(monkeypatch) -> None:
    middle = [np.asarray((index, 0, 10, 4), np.float32) for index in range(7)]
    near = [np.asarray((100 + index, 20, 20, 8), np.float32) for index in range(7)]
    calls = []

    def fake_propagate(_frames, index, _box):
        calls.append(index)
        return middle if index == 2 else near

    monkeypatch.setattr(dataset, "propagate_boxes", fake_propagate)
    anchor = dataset.SequenceAnchor(
        "red", 2, (2, 0, 10, 4), 5, (105, 20, 20, 8), 1, 6
    )
    boxes = dataset.propagate_two_anchor_boxes([np.zeros((2, 2), np.uint8)] * 7, anchor)

    assert calls == [2, 5]
    assert np.array_equal(boxes[2], middle[2])
    assert np.array_equal(boxes[5], near[5])
    assert np.array_equal(boxes[6], near[6])
    assert middle[3][0] < boxes[3][0] < near[3][0]
