from pathlib import Path

import pytest

from drivex26.instructions import load_annotations, maneuver_tags

ANNOTATIONS = Path(__file__).resolve().parents[1] / "third_party" / "doScenes" / "Annotations"


@pytest.mark.parametrize("text, expected", [
    ("turn left at the intersection", ["turn_left"]),
    ("Make a right turn and go straight", ["turn_right", "straight"]),
    ("Take a left and merge onto the far right lane", ["turn_left", "lane_change"]),
    ("Change one lane to the right", ["lane_change"]),
    ("move into the left lane", ["lane_change"]),
    ("Stop behind the gray car at the stoplight", ["stop", "follow"]),
    ("When the light turns green continue straight", ["straight"]),
    ("swerve around the parked car", ["nudge"]),
    ("Slow down to allow the pedestrian to cross", ["slow"]),
    ("follow the black car in front of you", ["follow"]),
    ("make a U-turn", ["u_turn"]),
    ("keep right and take the first right", ["turn_right"]),
    ("", []),
])
def test_maneuver_tags(text, expected):
    assert maneuver_tags(text) == expected


def test_turn_green_is_not_a_turn():
    assert not {"turn_left", "turn_right"} & set(maneuver_tags("wait until the light turns green"))


@pytest.mark.skipif(not ANNOTATIONS.is_dir(), reason="third_party/doScenes submodule missing")
def test_load_annotations_matches_spec_counts():
    df = load_annotations(ANNOTATIONS)
    # 2,450 non-blank instructions over 732 scenes, as the official loader yields
    assert len(df) == 2450
    assert df["scene_name"].nunique() == 732
    assert df["scene_name"].str.fullmatch(r"scene-\d{4}").all()
