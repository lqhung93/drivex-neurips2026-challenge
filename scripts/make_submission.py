"""Write the submission CSV for a kinematic baseline from the test cache, then validate it.

  python scripts/make_submission.py --model cv --team <TEAM> --version 0

The file is named as the upload form requires, <TEAM>_CSV_version<N>.csv;
<N> is 0 for the first upload and increases with every resubmission.

Rows are collapsed to one per scene with the first instruction of the scene;
--policy mean averages the trajectories of all its instructions instead.
Keep a record of every leaderboard upload.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from drivex26.baselines import constant_acceleration, constant_velocity
from drivex26.data import assert_split_allowed, load_cache
from drivex26.submission import (form_csv_name, first_per_scene, mean_per_scene, validate_submission,
                                  write_submission)

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"cv": constant_velocity, "ca": constant_acceleration}
POLICIES = {"first": first_per_scene, "mean": mean_per_scene}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--test-cache", default=ROOT / "data" / "cache" / "test.npz")
    p.add_argument("--team", required=True, help="team name exactly as entered in the upload form")
    p.add_argument("--version", type=int, required=True, help="0 for the first upload, +1 per resubmission")
    p.add_argument("--policy", choices=POLICIES, default="first", help="one row per scene")
    p.add_argument("--out-dir", help="defaults to data/submissions/<timestamp>-<model>/")
    args = p.parse_args()

    assert_split_allowed("test", "submission")
    cache = load_cache(args.test_cache)
    pred = MODELS[args.model](cache["hist_xy"], cache["hist_t"])
    tokens, instructions, pred = POLICIES[args.policy](cache["scene_token"], cache["instruction"], pred)

    out_dir = Path(args.out_dir or ROOT / "data" / "submissions" / f"{datetime.now():%Y%m%d-%H%M}-{args.model}")
    try:
        out = out_dir / form_csv_name(args.team, args.version)
    except ValueError as e:
        p.error(str(e))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(out, tokens, instructions, pred)
    errors, warnings = validate_submission(out, expected_tokens=set(np.unique(cache["scene_token"])), team=args.team)
    for msg in [f"ERROR: {e}" for e in errors] + [f"WARNING (heuristic): {w}" for w in warnings]:
        print(msg)
    print(f"{len(tokens)} rows -> {out}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
