"""doScenes data: leakage guard, anchor-frame transform, and a compact cache built via the official loader.

The cache is one .npz per split, one row per (instruction, scene) pair, so training and
evaluation never need nuScenes itself. Coordinates are metres in the ego frame at the
anchor (x forward, y left), as the submission format requires. `build_windows` adds the
instruction-free sliding windows over official train scenes for motion pretraining (P2).
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
OFFICIAL_LOADER = Path(__file__).resolve().parents[2] / "third_party" / "doScenes" / "dataloader.py"
HISTORY_LEN, FUTURE_LEN = 4, 12  # same layout as the official loader: 4 history + anchor + 12 future


def assert_split_allowed(split: str, purpose: str) -> None:
    """nuScenes v1.0-test may only be read to produce the submission file."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    if split == "test" and purpose != "submission":
        raise PermissionError(f"test split is only allowed for purpose='submission', got {purpose!r}")


def to_anchor_frame(xy: np.ndarray, anchor_xy: np.ndarray, anchor_yaw: float) -> np.ndarray:
    """Global (N, 2) -> ego frame at anchor: x forward, y left, metres."""
    d = np.asarray(xy, dtype=np.float64) - np.asarray(anchor_xy, dtype=np.float64)
    c, s = np.cos(anchor_yaw), np.sin(anchor_yaw)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=-1)


def _official_dataset_cls():
    spec = importlib.util.spec_from_file_location("doscenes_dataloader", OFFICIAL_LOADER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses in the loader look their module up here
    spec.loader.exec_module(module)
    return module.DoScenesNuScenesDataset


def build_cache(nusc, annotations, split: str, purpose: str) -> dict[str, np.ndarray]:
    """Run the official loader on `split` scenes of `nusc` and return column arrays.

    For the test split the ground-truth future is dropped here and never stored.
    """
    from nuscenes.utils.splits import create_splits_scenes  # only building needs the devkit

    assert_split_allowed(split, purpose)
    scenes = set(create_splits_scenes()[split])
    # The official loader treats an empty allow-list as "no filter", so never pass one.
    assert scenes, f"no scenes for split {split!r}"
    dataset = _official_dataset_cls()(nusc, annotations, camera_channels=(), allowed_scene_names=scenes)

    cols: dict[str, list] = {k: [] for k in (
        "scene_name", "scene_token", "instruction", "instruction_type", "annotator_file",
        "has_static", "has_dynamic", "anchor_xy", "anchor_yaw", "hist_xy", "hist_t", "fut_xy")}
    for item in dataset:
        anchor_xy = item["anchor_xy"].numpy().astype(np.float64)
        anchor_yaw = float(item["anchor_yaw"])
        if not np.isfinite(anchor_yaw):
            raise ValueError(f"missing anchor yaw for {item['scene_name']} (is pyquaternion installed?)")
        t0 = int(item["anchor_timestamp_us"])
        cols["scene_name"].append(item["scene_name"])
        cols["scene_token"].append(item["scene_token"])
        cols["instruction"].append(item["instruction"])
        cols["instruction_type"].append(item["instruction_type"])
        cols["annotator_file"].append(item["annotator_file"])
        cols["has_static"].append(item["has_static_reference"])
        cols["has_dynamic"].append(item["has_dynamic_reference"])
        cols["anchor_xy"].append(anchor_xy)
        cols["anchor_yaw"].append(anchor_yaw)
        cols["hist_xy"].append(to_anchor_frame(item["history_xy"].numpy(), anchor_xy, anchor_yaw))
        cols["hist_t"].append((item["history_timestamps_us"].numpy() - t0) / 1e6)
        cols["fut_xy"].append(to_anchor_frame(item["future_xy"].numpy(), anchor_xy, anchor_yaw))

    if split == "test":
        del cols["fut_xy"]
    out = {k: np.asarray(v) for k, v in cols.items()}
    for k in ("anchor_xy", "anchor_yaw", "hist_xy", "hist_t", "fut_xy"):
        if k in out:
            out[k] = out[k].astype(np.float32)
    return out


def _yaw_from_wxyz(rotation) -> float:
    """Yaw of a (w, x, y, z) quaternion, the formula of the official loader."""
    w, x, y, z = rotation
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_windows(nusc, purpose: str, stride: int = 1) -> dict[str, np.ndarray]:
    """Every 17-sample window of every official train scene, without instructions (P2 pretraining).

    Only `create_splits_scenes()["train"]` is read, so val and test scenes never enter. Rows keep
    `scene_name` and `anchor_index` (sample index of the anchor, 4 for the first segment) so
    training can drop the scenes of the fold used for validation. Ego poses come from LIDAR_TOP
    and are rounded to float32 like the official loader, so the window with anchor_index 4 of a
    doScenes scene equals its `build_cache` row.
    """
    from nuscenes.utils.splits import create_splits_scenes  # only building needs the devkit

    assert_split_allowed("train", purpose)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    train = set(create_splits_scenes()["train"])
    cols: dict[str, list] = {k: [] for k in (
        "scene_name", "scene_token", "anchor_index", "anchor_xy", "anchor_yaw", "hist_xy", "hist_t", "fut_xy")}
    for scene in nusc.scene:
        if scene["name"] not in train:
            continue
        xy, yaw, ts = [], [], []
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            pose = nusc.get("ego_pose", nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
            xy.append(pose["translation"][:2])
            yaw.append(_yaw_from_wxyz(pose["rotation"]))
            ts.append(int(sample["timestamp"]))
            token = sample["next"]
        xy = np.asarray(xy, dtype=np.float32).astype(np.float64)
        yaw = np.asarray(yaw, dtype=np.float32)
        ts = np.asarray(ts, dtype=np.int64)
        for a in range(HISTORY_LEN, len(xy) - FUTURE_LEN, stride):
            cols["scene_name"].append(scene["name"])
            cols["scene_token"].append(scene["token"])
            cols["anchor_index"].append(a)
            cols["anchor_xy"].append(xy[a])
            cols["anchor_yaw"].append(float(yaw[a]))
            cols["hist_xy"].append(to_anchor_frame(xy[a - HISTORY_LEN:a], xy[a], float(yaw[a])))
            cols["hist_t"].append((ts[a - HISTORY_LEN:a] - ts[a]) / 1e6)
            cols["fut_xy"].append(to_anchor_frame(xy[a + 1:a + 1 + FUTURE_LEN], xy[a], float(yaw[a])))
    out = {k: np.asarray(v) for k, v in cols.items()}
    for k in ("anchor_xy", "anchor_yaw", "hist_xy", "hist_t", "fut_xy"):
        out[k] = out[k].astype(np.float32)
    return out


def save_cache(cache: dict[str, np.ndarray], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)


def load_cache(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
