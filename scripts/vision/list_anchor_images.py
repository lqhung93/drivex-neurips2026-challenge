"""List the CAM_FRONT keyframe at the anchor (sample 4 of the first segment) of every cache scene.

  python scripts/vision/list_anchor_images.py --nusc-root data/nuscenes --splits train val
  python scripts/vision/list_anchor_images.py --nusc-root data/nuscenes --splits test --purpose submission

Writes data/features/vision/anchor_images_<split>.json: {scene_token: "samples/CAM_FRONT/<file>.jpg"}, the
member names to extract from the nuScenes keyframe archives. The test split needs --purpose submission.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from drivex26.data import assert_split_allowed, load_cache

ROOT = Path(__file__).resolve().parents[2]
ANCHOR = 4  # history = samples 0..3, anchor = sample 4 (official loader)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nusc-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "features" / "vision")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--purpose", default="eval")
    args = p.parse_args()
    for split in args.splits:
        assert_split_allowed(split, args.purpose)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for split in args.splits:
        version = "v1.0-test" if split == "test" else "v1.0-trainval"
        if version not in tables:
            load = lambda name: json.loads((args.nusc_root / version / f"{name}.json").read_text())
            sample = {r["token"]: r for r in load("sample")}
            front = {r["sample_token"]: r["filename"] for r in load("sample_data")
                     if r["is_key_frame"] and r["filename"].startswith("samples/CAM_FRONT/")}
            tables[version] = ({r["token"]: r for r in load("scene")}, sample, front)
        scenes, sample, front = tables[version]
        out = {}
        for token in np.unique(load_cache(args.cache_dir / f"{split}.npz")["scene_token"]):
            s = scenes[str(token)]["first_sample_token"]
            for _ in range(ANCHOR):
                s = sample[s]["next"]
            out[str(token)] = front[s]
        path = args.out_dir / f"anchor_images_{split}.json"
        path.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(path, len(out))


if __name__ == "__main__":
    main()
