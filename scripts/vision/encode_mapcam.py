"""Camera input: map encoder v1 features plus frozen CLIP features of the CAM_FRONT anchor image ("mapcam").

  python scripts/vision/encode_mapcam.py --clip-dir <vision kernel output> --splits train val
  python scripts/vision/encode_mapcam.py --clip-dir <vision kernel output> --splits test --purpose submission

Per cache row: the 17 map columns of that row, then K PCA components of the CLIP embedding of the image and K of
the same image flipped left to right (columns img_left_k and img_right_k, which drivex26.mirror swaps). The PCA is
fitted on the train images only (both views) and stored, so val and test use the train projection.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from drivex26.data import assert_split_allowed, load_cache
from drivex26.text import load_features, save_features
from drivex26.vision import K, MAPCAM_NAMES

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    p.add_argument("--map-dir", type=Path, default=ROOT / "data" / "features" / "map")
    p.add_argument("--feature-dir", type=Path, default=ROOT / "data" / "features" / "map")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--purpose", default="eval")
    args = p.parse_args()
    for split in args.splits:
        assert_split_allowed(split, args.purpose)
    pca_path = args.feature_dir / "mapcam_pca.npz"
    if not pca_path.exists():  # fit on train images, both views
        tr = np.load(args.clip_dir / "clip_train.npz")
        x = np.concatenate([tr["emb"], tr["emb_flip"]])
        mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        np.savez(pca_path, mean=mean, components=vt[:K], explained=(s[:K] ** 2).sum() / (s ** 2).sum())
    pca = np.load(pca_path)
    for split in args.splits:
        cache = load_cache(args.cache_dir / f"{split}.npz")
        texts = [str(t) for t in cache["instruction"]]
        clip = np.load(args.clip_dir / f"clip_{split}.npz")
        row = {str(t): i for i, t in enumerate(clip["tokens"])}
        index = np.array([row[str(t)] for t in cache["scene_token"]])  # every scene must have its image
        project = lambda e: (e[index] - pca["mean"]) @ pca["components"].T
        mapf = load_features(args.map_dir, "map", split, cache["instruction"])
        feats = np.concatenate([mapf, project(clip["emb"]), project(clip["emb_flip"])], axis=1).astype(np.float32)
        assert feats.shape[1] == len(MAPCAM_NAMES)
        info = {"hf_id": "map v1 + CLIP ViT-B/32 anchor CAM_FRONT, PCA %d (train)" % K, "dim": feats.shape[1],
                "pca_explained": float(pca["explained"])}
        print(save_features(args.feature_dir, "mapcam", split, feats, texts, info), feats.shape)


if __name__ == "__main__":
    main()
