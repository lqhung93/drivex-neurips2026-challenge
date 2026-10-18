"""ADE/FDE as in third_party/doScenes/metrics.py (ade_6s, fde), plus scene-level bootstrap CI,
scene grouped folds and paired ΔADE (protocol P1).

All trajectories are (N, T, 2) in the anchor frame.
"""

import hashlib

import numpy as np


def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-trajectory mean L2 over all steps, shape (N,)."""
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def fde(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-trajectory L2 at the last step, shape (N,)."""
    return np.linalg.norm(pred[:, -1] - gt[:, -1], axis=-1)


def _scene_means(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(groups, return_inverse=True)
    return np.bincount(inverse, weights=values) / np.bincount(inverse)


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """CI of the mean of per-group (scene) means, resampling groups with replacement."""
    means = _scene_means(values, groups)
    rng = np.random.default_rng(seed)
    boots = means[rng.integers(0, len(means), size=(n, len(means)))].mean(axis=1)
    alpha = (1 - level) / 2 * 100
    lo, hi = np.percentile(boots, [alpha, 100 - alpha])
    return float(lo), float(hi)


def instruction_category(has_static: np.ndarray, has_dynamic: np.ndarray) -> np.ndarray:
    """"s", "d", "sd" or "none" per row, from the official static/dynamic reference flags."""
    has_static, has_dynamic = np.asarray(has_static, bool), np.asarray(has_dynamic, bool)
    return np.where(has_static & has_dynamic, "sd",
                    np.where(has_static, "s", np.where(has_dynamic, "d", "none")))


def evaluate(pred: np.ndarray, gt: np.ndarray, scene_names: np.ndarray,
             has_static: np.ndarray, has_dynamic: np.ndarray, seed: int = 0) -> dict:
    """Pair-level and scene-level ADE/FDE, 95% scene bootstrap CI, per instruction category."""
    a, f = ade(pred, gt), fde(pred, gt)
    category = instruction_category(has_static, has_dynamic)
    return {
        "ade": float(a.mean()),
        "fde": float(f.mean()),
        "ade_ci95": bootstrap_ci(a, scene_names, seed=seed),
        "ade_scene": float(_scene_means(a, scene_names).mean()),
        "n_pairs": int(len(a)),
        "n_scenes": int(len(np.unique(scene_names))),
        "per_category": {c: {"n": int((category == c).sum()), "ade": float(a[category == c].mean())}
                         for c in ("s", "d", "sd", "none") if (category == c).any()},
    }


def scene_folds(scene_names: np.ndarray, k: int = 5) -> np.ndarray:
    """Fold id in [0, k) per row from the md5 of its scene name.

    Every instruction of a scene lands in the same fold, and the assignment does not depend on
    row order, machine or Python run (unlike the salted built-in hash()).
    """
    if k < 2:
        raise ValueError(f"need at least 2 folds, got {k}")
    names, inverse = np.unique(scene_names, return_inverse=True)
    fold_of = np.array([int(hashlib.md5(str(n).encode()).hexdigest(), 16) % k for n in names])
    return fold_of[inverse]


def paired_delta(ade_baseline: np.ndarray, ade_instruction: np.ndarray, groups: np.ndarray,
                 n: int = 1000, seed: int = 0) -> dict:
    """ΔADE = ADE_baseline - ADE_instruction on the same rows (positive: language helps).

    `delta` averages over pairs (like `evaluate()["ade"]`), `delta_scene` over scenes; the 95% CI
    resamples scenes (paired: both models see the same resampled scenes), centred on `delta_scene`.
    Average per-row ADE over seeds before calling to report a multi-seed ΔADE.
    """
    base, instr, groups = np.asarray(ade_baseline, float), np.asarray(ade_instruction, float), np.asarray(groups)
    if not (base.shape == instr.shape == groups.shape) or base.ndim != 1:
        raise ValueError(f"need equal 1-D shapes, got {base.shape}, {instr.shape}, {groups.shape}")
    diff = base - instr
    return {
        "delta": float(diff.mean()),
        "delta_scene": float(_scene_means(diff, groups).mean()),
        "ci95": bootstrap_ci(diff, groups, n=n, seed=seed),
        "n_pairs": int(len(diff)),
        "n_scenes": int(len(np.unique(groups))),
    }
