from __future__ import annotations

import final_drive_vision.prepare_traffic_housing_dataset as housing
from final_drive_vision.prepare_traffic_dataset import DEFAULT_ANCHORS


def test_confirmed_backgrounds_start_after_positive_review() -> None:
    for bag_name, (first_background, last_background) in (
        housing.CONFIRMED_TAIL_BACKGROUNDS.items()
    ):
        anchor = DEFAULT_ANCHORS[bag_name]
        assert anchor.last_labeled < first_background <= last_background


def test_housing_yaml_is_single_class_and_quotes_the_name(tmp_path) -> None:
    housing._write_data_yaml(tmp_path)
    assert (tmp_path / "data.yaml").read_text(encoding="utf-8") == (
        'train: images/train\n'
        'val: images/val\n'
        'names:\n'
        '  0: "traffic_light"\n'
    )
