"""Build data/cache/windows_train.npz: every 17-sample window of the official train scenes (P2 pretraining).

  python scripts/build_windows.py --nusc-root data/nuscenes [--stride 1]

No instructions; val and test scenes never enter. Rows carry scene_name and anchor_index so
training can drop the scenes of the validation fold.
"""

import argparse
from pathlib import Path

from nuscenes.nuscenes import NuScenes

from drivex26.data import build_windows, save_cache

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nusc-root", required=True)
    p.add_argument("--version", default="v1.0-trainval", help="v1.0-mini also works")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", help="defaults to data/cache/windows_train.npz")
    args = p.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
    windows = build_windows(nusc, purpose="pretrain", stride=args.stride)
    out = Path(args.out or ROOT / "data" / "cache" / "windows_train.npz")
    save_cache(windows, out)
    print(f"{len(windows['scene_name'])} windows from {len(set(windows['scene_name']))} train scenes -> {out}")


if __name__ == "__main__":
    main()
