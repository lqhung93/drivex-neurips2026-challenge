"""Compute map encoder v1 features once per split, stored like the text features.

  python scripts/build_map_features.py --nusc-root data/nuscenes --splits train val

Needs the nuScenes map expansion v1.3 under <nusc-root>/maps/expansion. Writes
data/features/map/map_<split>.npy (+ .json with the instruction sha256), one row per cache row,
so run_l1.py can use it with --encoder map --feature-dir data/features/map. The test split needs
--purpose submission (the test split is read only to write the submission).
"""

import argparse
import time
from pathlib import Path

import numpy as np

from drivex26.data import assert_split_allowed, load_cache
from drivex26.map_features import DIM, NAMES, lane_features
from drivex26.text import save_features

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nusc-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    p.add_argument("--feature-dir", type=Path, default=ROOT / "data" / "features" / "map")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--purpose", default="eval")
    args = p.parse_args()
    for split in args.splits:
        assert_split_allowed(split, args.purpose)

    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.nuscenes import NuScenes

    maps = {}
    for split in args.splits:
        version = "v1.0-test" if split == "test" else "v1.0-trainval"
        nusc = NuScenes(version=version, dataroot=str(args.nusc_root), verbose=False)
        cache = load_cache(args.cache_dir / f"{split}.npz")
        t0 = time.perf_counter()
        per_scene, out = {}, np.zeros((len(cache["scene_token"]), DIM), dtype=np.float32)
        for row, token in enumerate(cache["scene_token"]):
            if token not in per_scene:
                location = nusc.get("log", nusc.get("scene", str(token))["log_token"])["location"]
                if location not in maps:
                    maps[location] = NuScenesMap(dataroot=str(args.nusc_root), map_name=location)
                per_scene[token] = lane_features(maps[location], cache["anchor_xy"][row].astype(np.float64),
                                                 float(cache["anchor_yaw"][row]))
            out[row] = per_scene[token]
        info = {"hf_id": "nuscenes-map-expansion-v1.3, drivex26.map_features v1", "dim": DIM, "names": list(NAMES),
                "has_lane": float(out[:, 0].mean()), "sec": round(time.perf_counter() - t0, 1)}
        texts = [str(t) for t in cache["instruction"]]
        print(save_features(args.feature_dir, "map", split, out, texts, info), out.shape, info["has_lane"])


if __name__ == "__main__":
    main()
