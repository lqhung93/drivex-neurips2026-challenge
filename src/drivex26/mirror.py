"""Mirror augmentation (left <-> right): y -> -y in the anchor frame, for every input a model sees.

Each input block is described by column names. A column whose name has a left/right twin takes
that twin's value, and a lateral column (a y coordinate, a lateral or heading offset) changes sign.
Targets and the CA base only change the sign of y.
"""

from __future__ import annotations

import re

import numpy as np

from drivex26.map_features import NAMES, NAMES2

HISTORY_NAMES = (*(f"{a}{k}" for k in range(4) for a in ("x", "y")), *(f"t{k}" for k in range(4)))
# Column layout of scripts/language/encode_intent.py (L3 one-hot, 26 columns)
MANEUVERS = ("go_straight", "turn_left", "turn_right", "lane_change_left", "lane_change_right", "stop",
             "slow_down", "follow", "u_turn", "other")
INTENT_NAMES = (*(f"m_{m}" for m in MANEUVERS), *(f"then_{m}" for m in MANEUVERS), "then_none",
                *(f"speed_{s}" for s in ("stop", "slow", "keep", "faster")), "wait_first")
COLUMNS = {"intent": INTENT_NAMES, "map": NAMES, "map2": NAMES2}
LATERAL = re.compile(r"(.*_)?y\d+|.*_offset")


def column_names(encoder: str) -> tuple[str, ...]:
    if encoder not in COLUMNS:
        raise ValueError(f"cannot mirror {encoder!r} features (column meaning unknown)")
    return COLUMNS[encoder]


def mirror_columns(names) -> tuple[np.ndarray, np.ndarray]:
    """(perm, sign) such that mirrored = features[:, perm] * sign."""
    names = list(names)
    index = {n: i for i, n in enumerate(names)}
    twin = lambda n: n.replace("left", "\0").replace("right", "left").replace("\0", "right")
    perm = np.array([index.get(twin(n), i) for i, n in enumerate(names)])
    sign = np.array([-1.0 if LATERAL.fullmatch(n) else 1.0 for n in names], dtype=np.float32)
    return perm, sign


def mirror_specs(text: str, extra: str | None = None) -> tuple[tuple, tuple]:
    """Mirror of the baseline input (history, extra) and of the language model input (history, extra, text)."""
    base = HISTORY_NAMES + (column_names(extra) if extra else ())
    return mirror_columns(base), mirror_columns(base + column_names(text))
