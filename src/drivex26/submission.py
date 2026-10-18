"""Submission CSV writer and validator for the doScenes challenge format and the upload form file name.

The spec gives two headers (with / without `instruction`); we write the one with it and
accept both. One row per sample_token (= nuScenes scene_token), 12 (x, y) in metres,
anchor frame. File name follows the upload form, <TEAM>_CSV_version<N>.csv: a wrongly named
upload is not a valid entry, so nothing else is accepted, except the matched history only
baseline of the same entry, <TEAM>_CSV_version<N>_historyonly.csv.
"""

import csv
import math
import re
from pathlib import Path

import numpy as np

COORD_COLUMNS = [f"{axis}{i}" for i in range(1, 13) for axis in "xy"]
HEADERS = (["sample_token", "instruction", *COORD_COLUMNS], ["sample_token", *COORD_COLUMNS])
MAX_STEP_M = 25.0  # 0.5 s step; 50 m/s is far above any nuScenes ego speed
FILE_NAME = re.compile(r"([A-Za-z0-9][A-Za-z0-9_-]*)_CSV_version(\d+)(_historyonly)?\.csv")


def form_csv_name(team: str, version: int, history_only: bool = False) -> str:
    """Upload form file name, e.g. form_csv_name("drivex26", 0) == "drivex26_CSV_version0.csv";
    with history_only, the matched baseline of the same entry: "drivex26_CSV_version0_historyonly.csv"."""
    name = f"{team}_CSV_version{version}{'_historyonly' if history_only else ''}.csv"
    if version < 0 or not FILE_NAME.fullmatch(name):
        raise ValueError(f"invalid team or version for the form file name: {team!r}, {version!r}")
    return name


def first_per_scene(scene_tokens: np.ndarray, instructions: np.ndarray, preds: np.ndarray):
    """Collapse (instruction, scene) rows to one row per scene: the first instruction in cache order
    and its trajectory. Default policy, as earlier entries submitted.
    """
    tokens, first = np.unique(scene_tokens, return_index=True)
    return tokens, instructions[first], preds[first]


def mean_per_scene(scene_tokens: np.ndarray, instructions: np.ndarray, preds: np.ndarray):
    """Collapse (instruction, scene) rows to one row per scene: mean trajectory, first instruction.

    Alternative to first_per_scene, compared on CV.
    """
    tokens, first, inverse = np.unique(scene_tokens, return_index=True, return_inverse=True)
    sums = np.zeros((len(tokens), *preds.shape[1:]))
    np.add.at(sums, inverse, preds)
    return tokens, instructions[first], sums / np.bincount(inverse)[:, None, None]


def write_submission(path, sample_tokens, instructions, preds: np.ndarray) -> None:
    if len(set(sample_tokens)) != len(sample_tokens):
        raise ValueError("sample_tokens must be unique; aggregate per scene first")
    if preds.shape != (len(sample_tokens), 12, 2) or not np.isfinite(preds).all():
        raise ValueError(f"preds must be finite with shape ({len(sample_tokens)}, 12, 2), got {preds.shape}")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADERS[0])
        for token, instruction, traj in zip(sample_tokens, instructions, preds):
            w.writerow([token, instruction, *(float(v) for v in traj.reshape(-1))])


def validate_submission(path, expected_tokens=None, team: str | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Warnings are heuristics, not format rules.

    With `team`, the file name must carry exactly that team name.
    """
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    name = FILE_NAME.fullmatch(path.name)
    if not name:
        errors.append(f"file name must be <TEAM>_CSV_version<N>.csv (upload form rule) or its _historyonly baseline, got {path.name}")
    elif team is not None and name.group(1) != team:
        errors.append(f"file name team {name.group(1)!r} does not match {team!r}")

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] not in HEADERS:
        return errors + [f"header must be one of {[','.join(h[:3]) + ',...' for h in HEADERS]}"], warnings
    n_meta = len(rows[0]) - len(COORD_COLUMNS)

    tokens, trajs = [], []
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != len(rows[0]):
            errors.append(f"line {i}: expected {len(rows[0])} fields, got {len(row)}")
            continue
        try:
            values = [float(v) for v in row[n_meta:]]
        except ValueError:
            values = [math.nan]
        if not all(math.isfinite(v) for v in values):
            errors.append(f"line {i}: non-numeric or non-finite coordinate")
            continue
        tokens.append(row[0])
        trajs.append(np.array(values).reshape(12, 2))

    seen = set()
    for token in tokens:
        if token in seen:
            errors.append(f"duplicate sample_token {token}")
        seen.add(token)
    if expected_tokens is not None:
        missing, extra = set(expected_tokens) - seen, seen - set(expected_tokens)
        if missing:
            errors.append(f"{len(missing)} missing sample_tokens, e.g. {sorted(missing)[:3]}")
        if extra:
            errors.append(f"{len(extra)} unexpected sample_tokens, e.g. {sorted(extra)[:3]}")

    if trajs:
        trajs = np.stack(trajs)
        if np.median(trajs[:, -1, 0]) <= 0:
            warnings.append("median final x <= 0: most trajectories do not go forward (frame bug?)")
        steps = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(trajs), 1, 2)), trajs], axis=1), axis=1), axis=-1)
        n_jumps = int((steps.max(axis=1) > MAX_STEP_M).sum())
        if n_jumps:
            warnings.append(f"{n_jumps} rows have a step > {MAX_STEP_M} m in 0.5 s")
    return errors, warnings
