"""Encode cache instructions once with a frozen text encoder, CPU.

  python scripts/language/encode_text.py --encoders minilm --splits train val

Writes data/features/text/<encoder>_<split>.npy (+ .json with the instruction sha256), one row
per cache row.
The test split needs --purpose submission (it is read only to write the submission).
"""

import argparse
from pathlib import Path

from drivex26.data import assert_split_allowed, load_cache
from drivex26.text import ENCODERS, encode, save_features

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    p.add_argument("--feature-dir", type=Path, default=ROOT / "data" / "features" / "text")
    p.add_argument("--encoders", nargs="+", default=["minilm"], choices=list(ENCODERS))
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--purpose", default="eval")
    args = p.parse_args()

    for split in args.splits:
        assert_split_allowed(split, args.purpose)
    for name in args.encoders:
        for split in args.splits:
            texts = [str(t) for t in load_cache(args.cache_dir / f"{split}.npz")["instruction"]]
            emb, info = encode(name, texts)
            print(save_features(args.feature_dir, name, split, emb, texts, info), emb.shape)


if __name__ == "__main__":
    main()
