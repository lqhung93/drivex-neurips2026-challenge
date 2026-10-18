"""Evaluate a kinematic baseline (H0a cv / H0b ca) on a cached split and log a run.

  python scripts/run_baseline.py --model cv --split val
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from drivex26.baselines import constant_acceleration, constant_velocity
from drivex26.data import assert_split_allowed, load_cache
from drivex26.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"cv": constant_velocity, "ca": constant_acceleration}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--cache-dir", default=ROOT / "data" / "cache")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = p.parse_args()

    assert_split_allowed(args.split, "eval")
    cache = load_cache(Path(args.cache_dir) / f"{args.split}.npz")
    pred = MODELS[args.model](cache["hist_xy"], cache["hist_t"])
    metrics = evaluate(pred, cache["fut_xy"], cache["scene_name"], cache["has_static"], cache["has_dynamic"],
                       seed=args.seed)

    run_id = f"{datetime.now():%Y%m%d-%H%M}-{args.model}-{args.split}"
    run_dir = ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True)
    record = {
        "run_id": run_id,
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "config": {"model": args.model, "split": args.split, "seed": args.seed},
        "device": "cpu",
        "gpu_hours": 0.0,
        "metrics": metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
