import cv2
import numpy as np

from final_drive_vision.detectors import (
    CheckerboardDetector,
    CheckerboardDetection,
    DetectionRoi,
    TemporalBoolFilter,
    TemporalStateFilter,
    TrafficLightDetector,
    TrafficDetection,
    TRAFFIC_GREEN,
    TRAFFIC_NONE,
    TRAFFIC_RED,
    TRAFFIC_YELLOW,
    TRAFFIC_GREEN_LEFT,
)
from final_drive_vision.housing_classifier import TrafficHousingStateClassifier
from final_drive_vision.traffic_housing_detector import TrafficHousingDetector


def test_blank_frame_has_no_detection():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert TrafficLightDetector().detect(frame).state == TRAFFIC_NONE
    assert not CheckerboardDetector().detect(frame).detected


def test_detection_roi_keeps_center_top_and_restores_coordinates():
    roi = DetectionRoi()
    assert roi.bounds((480, 640, 3)) == (96, 0, 544, 408)

    traffic = roi.restore_traffic(
        TrafficDetection(state=TRAFFIC_GREEN, bbox=(10, 20, 30, 40), slot_centers=((12.0, 24.0),)),
        96,
        0,
    )
    assert traffic.bbox == (106, 20, 30, 40)
    assert traffic.slot_centers == ((108.0, 24.0),)

    checker = roi.restore_checker(
        CheckerboardDetection(detected=True, bbox=(4, 8, 12, 16)), 96, 0
    )
    assert checker.bbox == (100, 8, 12, 16)


def test_traffic_detector_uses_y_zero_through_239():
    detector = TrafficLightDetector(roi=DetectionRoi(), max_y_fraction=0.50)
    assert detector.active_bounds((480, 640, 3)) == (96, 0, 544, 240)


def test_traffic_detector_rejects_invalid_max_y_fraction():
    for value in (0.0, 1.01):
        try:
            TrafficLightDetector(max_y_fraction=value)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid fraction to fail: {value}")


def test_traffic_temporal_filter_requires_consensus_and_clears():
    state_filter = TemporalStateFilter(window_size=5, required_hits=3, clear_hits=2)
    assert state_filter.update(TRAFFIC_GREEN) == TRAFFIC_NONE
    assert state_filter.update(TRAFFIC_GREEN) == TRAFFIC_NONE
    assert state_filter.update(TRAFFIC_GREEN) == TRAFFIC_GREEN
    assert state_filter.update(TRAFFIC_NONE) == TRAFFIC_GREEN
    assert state_filter.update(TRAFFIC_NONE) == TRAFFIC_NONE


def test_boolean_filter_debounces_both_edges():
    state_filter = TemporalBoolFilter(on_hits=2, off_hits=2)
    assert not state_filter.update(True)
    assert state_filter.update(True)
    assert state_filter.update(False)
    assert not state_filter.update(False)


def _four_lamp_frame(active_slots=()):
    frame = np.full((120, 320, 3), 110, dtype=np.uint8)
    bbox = (60, 40, 200, 50)
    for index in range(4):
        center = (85 + 50 * index, 65)
        cv2.circle(frame, center, 17, (55, 55, 55), -1)
        if index in active_slots:
            cv2.circle(frame, center, 12, (255, 255, 255), -1)
    return frame, bbox


def test_housing_classifier_maps_fixed_lamp_slots():
    classifier = TrafficHousingStateClassifier()
    expected = {
        (0,): TRAFFIC_RED,
        (1,): TRAFFIC_YELLOW,
        (3,): TRAFFIC_GREEN,
        (2, 3): TRAFFIC_GREEN_LEFT,
        (): TRAFFIC_NONE,
    }
    for active_slots, expected_state in expected.items():
        frame, bbox = _four_lamp_frame(active_slots)
        result = classifier.classify(frame, bbox)
        assert result.state == expected_state
        assert tuple(slot.label for slot in result.slots) == ("R", "Y", "L", "G")


def test_yolo_output_accepts_both_opencv_layouts():
    channel_first = np.zeros((1, 5, 8400), dtype=np.float32)
    rows = TrafficHousingDetector._prediction_rows(channel_first)
    assert rows.shape == (8400, 5)

    anchor_first = np.zeros((1, 8400, 5), dtype=np.float32)
    rows = TrafficHousingDetector._prediction_rows(anchor_first)
    assert rows.shape == (8400, 5)
