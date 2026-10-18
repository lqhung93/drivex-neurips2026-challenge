"""Build data/cache/<split>.npz from nuScenes + doScenes annotations via the official loader.

  python scripts/build_cache.py --nusc-root /data/nuscenes --split train
  python scripts/build_cache.py --nusc-root /data/nuscenes --split test --purpose submission
"""

import argparse
from pathlib import Path

from nuscenes.nuscenes import NuScenes

from drivex26.data import SPLITS, build_cache, save_cache

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nusc-root", required=True)
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--purpose", help="defaults to the split name; the test split requires 'submission'")
    p.add_argument("--version", help="defaults to v1.0-test for test, else v1.0-trainval (v1.0-mini also works)")
    p.add_argument("--annotations", default=ROOT / "third_party" / "doScenes" / "Annotations")
    p.add_argument("--out", help="defaults to data/cache/<split>.npz")
    args = p.parse_args()

    version = args.version or ("v1.0-test" if args.split == "test" else "v1.0-trainval")
    nusc = NuScenes(version=version, dataroot=args.nusc_root, verbose=False)
    cache = build_cache(nusc, args.annotations, args.split, purpose=args.purpose or args.split)
    out = Path(args.out or ROOT / "data" / "cache" / f"{args.split}.npz")
    save_cache(cache, out)
    print(f"{args.split}: {len(cache['scene_name'])} pairs, {len(set(cache['scene_name']))} scenes -> {out}")


if __name__ == "__main__":
    main()
