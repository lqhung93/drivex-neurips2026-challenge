import math

import numpy as np
import pytest
from nuscenes.utils.splits import create_splits_scenes

from drivex26.data import assert_split_allowed, build_cache, build_windows, load_cache, save_cache, to_anchor_frame

SPLITS = create_splits_scenes()


# --- to_anchor_frame: hand computed examples ---

def test_point_ahead_maps_to_positive_x():
    out = to_anchor_frame(np.array([[100.0, 205.0]]), np.array([100.0, 200.0]), math.pi / 2)
    np.testing.assert_allclose(out, [[5.0, 0.0]], atol=1e-9)


def test_point_on_left_maps_to_positive_y():
    out = to_anchor_frame(np.array([[95.0, 200.0]]), np.array([100.0, 200.0]), math.pi / 2)
    np.testing.assert_allclose(out, [[0.0, 5.0]], atol=1e-9)


# --- leakage guard ---

@pytest.mark.parametrize("purpose", ["train", "val", "tuning", "debug"])
def test_test_split_rejected_outside_submission(purpose):
    with pytest.raises(PermissionError):
        assert_split_allowed("test", purpose)


def test_test_split_allowed_for_submission():
    assert_split_allowed("test", "submission")


def test_unknown_split_rejected():
    with pytest.raises(ValueError):
        assert_split_allowed("trainval", "train")


# --- build_cache through the official doScenes loader, on a fake nuScenes ---

class FakeNuScenes:
    """Minimal stand-in for nuscenes.NuScenes: only what the official loader touches."""

    def __init__(self, dataroot):
        self.dataroot = str(dataroot)
        self.scene = []
        self._tables = {"scene": {}, "sample": {}, "sample_data": {}, "ego_pose": {}}

    def get(self, table, token):
        return self._tables[table][token]

    def add_scene(self, name, n_samples=20, start=(100.0, 200.0), yaw=math.pi / 2, speed=5.0):
        """Ego drives straight at constant speed along `yaw`, one sample every 0.5 s."""
        tokens = [f"{name}-s{i}" for i in range(n_samples)]
        w, z = math.cos(yaw / 2), math.sin(yaw / 2)
        for i, tok in enumerate(tokens):
            d = speed * 0.5 * i
            self._tables["ego_pose"][f"{tok}-ep"] = {
                "translation": [start[0] + d * math.cos(yaw), start[1] + d * math.sin(yaw), 0.0],
                "rotation": [w, 0.0, 0.0, z],
            }
            self._tables["sample_data"][f"{tok}-lidar"] = {"ego_pose_token": f"{tok}-ep", "filename": f"{tok}.bin"}
            self._tables["sample"][tok] = {
                "token": tok,
                "timestamp": 1_000_000 + 500_000 * i,
                "next": tokens[i + 1] if i + 1 < n_samples else "",
                "data": {"LIDAR_TOP": f"{tok}-lidar"},
            }
        scene = {"name": name, "token": f"{name}-tok", "nbr_samples": n_samples, "first_sample_token": tokens[0]}
        self.scene.append(scene)
        self._tables["scene"][scene["token"]] = scene


def write_annotations(dirpath, rows):
    dirpath.mkdir()
    lines = ["Scene Number,Instruction Type,Instruction"]
    lines += [f'{int(name.split("-")[1])},{typ},"{text}"' for name, typ, text in rows]
    (dirpath / "Annotator 1.csv").write_text("\n".join(lines) + "\n")
    return dirpath


@pytest.fixture
def trainval(tmp_path):
    nusc = FakeNuScenes(tmp_path)
    train_scene, val_scene = SPLITS["train"][0], SPLITS["val"][0]
    nusc.add_scene(train_scene)
    nusc.add_scene(val_scene)
    ann = write_annotations(tmp_path / "ann", [
        (train_scene, "s", "follow the road"),
        (train_scene, "", "go straight"),
        (val_scene, "d", "follow the car ahead"),
    ])
    return nusc, ann, train_scene, val_scene


def test_straight_drive_is_on_x_axis_in_anchor_frame(trainval):
    nusc, ann, train_scene, _ = trainval
    cache = build_cache(nusc, ann, "train", purpose="train")
    np.testing.assert_allclose(cache["hist_xy"][0], [[-10, 0], [-7.5, 0], [-5, 0], [-2.5, 0]], atol=1e-3)
    np.testing.assert_allclose(cache["fut_xy"][0, :, 0], 2.5 * np.arange(1, 13), atol=1e-3)
    np.testing.assert_allclose(cache["fut_xy"][0, :, 1], 0.0, atol=1e-3)
    np.testing.assert_allclose(cache["hist_t"][0], [-2.0, -1.5, -1.0, -0.5])


def test_cache_keeps_only_requested_split(trainval):
    nusc, ann, train_scene, val_scene = trainval
    train = build_cache(nusc, ann, "train", purpose="train")
    val = build_cache(nusc, ann, "val", purpose="val")
    assert set(train["scene_name"]) == {train_scene} and len(train["scene_name"]) == 2
    assert list(val["scene_name"]) == [val_scene]
    assert val["has_dynamic"].tolist() == [True] and val["has_static"].tolist() == [False]


def test_cache_round_trips_through_disk(trainval, tmp_path):
    nusc, ann, *_ = trainval
    cache = build_cache(nusc, ann, "train", purpose="train")
    save_cache(cache, tmp_path / "train.npz")
    loaded = load_cache(tmp_path / "train.npz")
    assert loaded.keys() == cache.keys()
    for k in cache:
        np.testing.assert_array_equal(loaded[k], cache[k])


@pytest.fixture
def test_nusc(tmp_path):
    nusc = FakeNuScenes(tmp_path)
    scene = SPLITS["test"][0]
    nusc.add_scene(scene)
    return nusc, write_annotations(tmp_path / "ann", [(scene, "s", "turn left")])


def test_test_cache_requires_submission_purpose(test_nusc):
    with pytest.raises(PermissionError):
        build_cache(*test_nusc, "test", purpose="val")


def test_test_cache_never_stores_future(test_nusc):
    cache = build_cache(*test_nusc, "test", purpose="submission")
    assert "fut_xy" not in cache
    assert cache["hist_xy"].shape == (1, 4, 2)


def test_cache_reading_does_not_need_nuscenes_devkit(tmp_path):
    # Training jobs only read the cache; installing the devkit on some hosts downgrades numpy.
    import subprocess
    import sys
    path = tmp_path / "c.npz"
    np.savez_compressed(path, hist_xy=np.zeros((1, 4, 2), dtype=np.float32))
    code = (
        "import sys; sys.modules['nuscenes'] = None\n"
        "from drivex26.data import assert_split_allowed, load_cache, to_anchor_frame\n"
        f"assert load_cache({str(path)!r})['hist_xy'].shape == (1, 4, 2)\n"
        "assert_split_allowed('val', 'eval')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-500:]


# --- P2: sliding windows over official train scenes, no instructions ---

def test_windows_cover_every_17_sample_span_of_train_scenes(trainval):
    nusc, _, train_scene, _ = trainval  # 20 samples per scene -> anchors 4..7
    w = build_windows(nusc, purpose="pretrain")
    assert set(w["scene_name"]) == {train_scene}
    assert w["anchor_index"].tolist() == [4, 5, 6, 7]
    assert build_windows(nusc, purpose="pretrain", stride=2)["anchor_index"].tolist() == [4, 6]


def test_windows_never_include_non_train_scenes(trainval):
    nusc, _, _, val_scene = trainval
    assert val_scene not in set(build_windows(nusc, purpose="pretrain")["scene_name"])


def test_windows_carry_no_instruction(trainval):
    w = build_windows(trainval[0], purpose="pretrain")
    assert not {"instruction", "instruction_type", "has_static", "has_dynamic"} & set(w)
    assert w["hist_xy"].shape == (4, 4, 2) and w["fut_xy"].shape == (4, 12, 2) and w["hist_t"].shape == (4, 4)


def test_later_window_is_in_its_own_anchor_frame(trainval):
    w = build_windows(trainval[0], purpose="pretrain")
    i = w["anchor_index"].tolist().index(6)
    np.testing.assert_allclose(w["hist_xy"][i], [[-10, 0], [-7.5, 0], [-5, 0], [-2.5, 0]], atol=1e-3)
    np.testing.assert_allclose(w["fut_xy"][i, :, 0], 2.5 * np.arange(1, 13), atol=1e-3)
    np.testing.assert_allclose(w["hist_t"][i], [-2.0, -1.5, -1.0, -0.5])


def test_first_window_matches_the_official_loader_cache(trainval):
    nusc, ann, train_scene, _ = trainval
    nusc.add_scene(SPLITS["train"][1], yaw=0.7, speed=3.0)  # turned, slower scene
    cache = build_cache(nusc, ann, "train", purpose="train")
    w = build_windows(nusc, purpose="pretrain")
    first = (w["scene_name"] == train_scene) & (w["anchor_index"] == 4)
    row = list(cache["scene_name"]).index(train_scene)
    for k in ("hist_xy", "hist_t", "fut_xy", "anchor_xy", "anchor_yaw"):
        np.testing.assert_allclose(w[k][first][0], cache[k][row], atol=1e-5)


# --- map encoder v1 on a fake map: one straight lane north, then a left and a right branch ---

class FakeMap:
    def __init__(self):
        n = np.arange(0, 20.5, 0.5)
        self.lanes = {
            "main": [(100.0, 200.0 + d, math.pi / 2) for d in n],
            "left": [(100.0 - d, 220.0 + d, 3 * math.pi / 4) for d in np.arange(0, 30, 0.5) / math.sqrt(2)],
            "right": [(100.0 + d, 220.0 + d, math.pi / 4) for d in np.arange(0, 30, 0.5) / math.sqrt(2)],
            "oncoming": [(103.0, 220.0 - d, -math.pi / 2) for d in n],
        }

    def get_records_in_radius(self, x, y, radius, layers):
        return {"lane": ["main", "oncoming"], "lane_connector": []}

    def discretize_lanes(self, tokens, resolution):
        return {t: self.lanes[t] for t in tokens}

    def get_outgoing_lane_ids(self, token):
        return ["left", "right"] if token == "main" else []


def test_map_features_follow_the_ego_lane_and_its_branches():
    from drivex26.map_features import NAMES, SCALE, lane_features
    f = dict(zip(NAMES, lane_features(FakeMap(), np.array([100.5, 205.0]), math.pi / 2)))
    assert f["has_lane"] == 1.0 and f["n_branches"] == 0.5  # two branches / 4
    assert abs(f["lateral_offset"] * SCALE - 0.5) < 1e-6  # lane is 0.5 m to the left of the ego
    assert abs(f["main_x10"] * SCALE - 10.0) < 1e-6 and abs(f["main_y10"] * SCALE - 0.5) < 1e-6
    assert f["left_y30"] > 0 > f["right_y30"]  # y left in the anchor frame
    assert abs(f["dist_to_branch"] * SCALE - 15.0) < 1e-6  # main lane ends 15 m ahead
