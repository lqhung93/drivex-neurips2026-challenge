"""Deterministic H1 training, with optional train-fold learning curves.

P1 selection is handled by scripts/run_h1_cv.py. Fixed-epoch fitting takes no
validation cache, so official val labels cannot select an epoch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from drivex26.h1 import HistoryMLP, predict, prepare
from drivex26.metrics import ade, fde


@dataclass(frozen=True)
class H1Config:
    hidden: int = 128
    lr: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.hidden < 1 or self.batch_size < 1:
            raise ValueError("hidden and batch_size must be positive")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


def split_fold(cache: dict[str, np.ndarray], folds: np.ndarray,
               fold: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split all rows of a scene together; fail on overlap or empty partitions."""
    if folds.shape != (len(cache["scene_name"]),):
        raise ValueError("fold assignment must match cache rows")
    held = folds == fold
    if not held.any() or held.all():
        raise ValueError("empty training or held-out fold")
    train = {key: value[~held] for key, value in cache.items()}
    valid = {key: value[held] for key, value in cache.items()}
    for key in ("scene_name", "scene_token"):
        if set(train[key]) & set(valid[key]):
            raise ValueError(f"scene leakage across fold: {key}")
    return train, valid


def fit(train: dict[str, np.ndarray], config: H1Config, seed: int, epochs: int,
        *, held_out: dict[str, np.ndarray] | None = None,
        eval_every: int = 10) -> tuple[HistoryMLP, dict[int, np.ndarray], list[dict]]:
    """Fit on CPU; predictions/curve, if requested, concern a train CV fold.

    Always runs the full budget, without early stopping. A shared epoch is
    selected across folds/seeds later, rather than picking a different one per
    fold. Without held_out, performs no evaluation at all.
    """
    if epochs < 0 or eval_every < 1:
        raise ValueError("epochs must be nonnegative and eval_every positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    features, base, target = prepare(train)
    if not len(features):
        raise ValueError("empty training cache")
    model = HistoryMLP(config.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    predictions: dict[int, np.ndarray] = {}
    curve: list[dict] = []
    if held_out is not None:
        eval_x, eval_base, _ = prepare(held_out)

    def record(epoch: int) -> None:
        if held_out is None:
            return
        pred = predict(model, eval_x, eval_base)
        if not np.isfinite(pred).all():
            raise ValueError("nonfinite H1 prediction")
        predictions[epoch] = pred
        names, inverse = np.unique(held_out["scene_name"], return_inverse=True)
        errors = ade(pred, held_out["fut_xy"])
        means = np.bincount(inverse, weights=errors) / np.bincount(inverse)
        curve.append({"epoch": epoch, "ade": float(errors.mean()),
                      "ade_scene": float(means.mean()),
                      "fde": float(fde(pred, held_out["fut_xy"]).mean()),
                      "n_scenes": len(names)})

    record(0)
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(features), generator=generator).split(config.batch_size):
            pred = model(features[indices], base[indices])
            loss = torch.linalg.vector_norm(pred - target[indices], dim=-1).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite H1 training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % eval_every == 0 or epoch == epochs:
            record(epoch)
    return model, predictions, curve


def select_epoch(errors: np.ndarray, epochs: list[int],
                 scene_names: np.ndarray) -> tuple[int, list[dict]]:
    """Choose one epoch by equal-scene OOF ADE averaged over all seeds.

    errors has shape (seeds, epochs, rows), in original cache order. Earliest
    epoch wins ties. Per-pair ADE is also logged, without changing the selector.
    """
    if errors.ndim != 3 or errors.shape[1:] != (len(epochs), len(scene_names)):
        raise ValueError("expected errors of shape (seeds, epochs, cache rows)")
    if not errors.shape[0] or not np.isfinite(errors).all():
        raise ValueError("OOF errors must be nonempty and finite")
    if epochs != sorted(set(epochs)) or not epochs:
        raise ValueError("epochs must be sorted and unique")
    _, inverse = np.unique(scene_names, return_inverse=True)
    curve = []
    for index, epoch in enumerate(epochs):
        per_seed = errors[:, index]
        scenes = np.stack([np.bincount(inverse, weights=row) / np.bincount(inverse)
                           for row in per_seed])
        curve.append({"epoch": epoch, "ade": float(per_seed.mean()),
                      "ade_scene": float(scenes.mean()),
                      "ade_by_seed": per_seed.mean(axis=1).tolist(),
                      "ade_scene_by_seed": scenes.mean(axis=1).tolist()})
    index = int(np.argmin([row["ade_scene"] for row in curve]))
    return epochs[index], curve
