"""Check a submission CSV against the doScenes format and the upload form file name; exits 1 on any error.

  python scripts/validate_submission.py path/to/<TEAM>_CSV_version<N>.csv [--team <TEAM>] [--test-cache data/cache/test.npz]
"""

import argparse
import sys

import numpy as np

from drivex26.data import load_cache
from drivex26.submission import validate_submission


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--test-cache", help="test cache .npz; enables the missing/unexpected token check")
    p.add_argument("--team", help="require exactly this team name in the file name")
    args = p.parse_args()

    expected = set(np.unique(load_cache(args.test_cache)["scene_token"])) if args.test_cache else None
    errors, warnings = validate_submission(args.path, expected_tokens=expected, team=args.team)
    for e in errors:
        print(f"ERROR: {e}")
    for w in warnings:
        print(f"WARNING (heuristic): {w}")
    print("FAIL" if errors else "PASS")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
