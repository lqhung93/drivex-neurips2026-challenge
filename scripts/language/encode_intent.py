"""L3 input: one-hot vectors from structured LLM intents, stored like text features.

  python scripts/language/encode_intent.py results/runs/20261018-llm-intent-full --splits train val

Reads intent_<split>.json (one entry per cache row, checked against the cache's instruction
sha256) and writes data/features/text/intent_<split>.npy: maneuver (10), next maneuver (10 + none),
speed (4), wait first (1) = 26 columns. Rows the LLM could not parse are all zero, like a blank
instruction. run_l1.py reads them with --encoder intent. Test needs --purpose submission.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from drivex26.data import assert_split_allowed, load_cache
from drivex26.text import save_features, texts_sha256

ROOT = Path(__file__).resolve().parents[2]
MANEUVERS = ["go_straight", "turn_left", "turn_right", "lane_change_left", "lane_change_right", "stop",
             "slow_down", "follow", "u_turn", "other"]
SPEEDS = ["stop", "slow", "keep", "faster"]
DIM = len(MANEUVERS) + len(MANEUVERS) + 1 + len(SPEEDS) + 1


def one_hot(intent: dict | None) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    if intent is None:
        return v
    v[MANEUVERS.index(intent["maneuver"])] = 1
    then = intent.get("then")
    v[len(MANEUVERS) + (MANEUVERS.index(then) if then in MANEUVERS else len(MANEUVERS))] = 1
    v[2 * len(MANEUVERS) + 1 + SPEEDS.index(intent["speed"])] = 1
    v[-1] = float(bool(intent.get("wait_first")))
    return v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("intent_run", type=Path)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    p.add_argument("--feature-dir", type=Path, default=ROOT / "data" / "features" / "text")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--purpose", default="eval")
    p.add_argument("--name", default="intent", help="feature name, e.g. intentpara for the intents of paraphrases")
    args = p.parse_args()
    for split in args.splits:
        assert_split_allowed(split, args.purpose)
        texts = [str(t) for t in load_cache(args.cache_dir / f"{split}.npz")["instruction"]]
        intents = json.loads((args.intent_run / f"intent_{split}.json").read_text())
        if intents["texts_sha256"] != texts_sha256(texts) or len(intents["rows"]) != len(texts):
            raise ValueError(f"{split}: intents were built from different instructions than the cache")
        emb = np.stack([one_hot(r) for r in intents["rows"]])
        info = {"hf_id": f"{intents['model']} intents, one-hot (scripts/language/encode_intent.py)", "dim": DIM,
                "parsed": float(np.mean([r is not None for r in intents["rows"]]))}
        print(save_features(args.feature_dir, args.name, split, emb, texts, info), emb.shape, round(info["parsed"], 3))


if __name__ == "__main__":
    main()
