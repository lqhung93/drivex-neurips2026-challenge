"""Small history-only residual motion model for the doScenes first segment.

The kinematic CA path is a fixed starting point. The MLP learns a 12-step
correction from only the four ego poses and their timestamps.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from drivex26.baselines import constant_acceleration


def prepare_inputs(cache: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare history inputs without requiring future labels or language."""
    hist = cache["hist_xy"].astype(np.float32)
    times = cache["hist_t"].astype(np.float32)
    if hist.ndim != 3 or hist.shape[1:] != (4, 2) or times.shape != (len(hist), 4):
        raise ValueError("invalid history cache shapes")
    features = np.concatenate((hist.reshape(-1, 8) / 20.0, times / 2.0), axis=1)
    base = constant_acceleration(hist, times).astype(np.float32)
    if not (np.isfinite(features).all() and np.isfinite(base).all()):
        raise ValueError("nonfinite motion data")
    return torch.from_numpy(features), torch.from_numpy(base)


def prepare(cache: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return history inputs and future labels for training/evaluation."""
    features, base = prepare_inputs(cache)
    target = cache["fut_xy"].astype(np.float32)
    if target.shape != tuple(base.shape):
        raise ValueError("invalid train/val cache shapes")
    if not np.isfinite(target).all():
        raise ValueError("nonfinite motion data")
    return features, base, torch.from_numpy(target)


class HistoryMLP(nn.Module):
    """Predict a residual in anchor-frame metres; language is never an input."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 24),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return base + self.net(features).reshape(-1, 12, 2)


@torch.no_grad()
def predict(model: HistoryMLP, features: torch.Tensor, base: torch.Tensor,
            batch_size: int = 512) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(features), batch_size):
        chunks.append(model(features[start:start + batch_size].to(device),
                            base[start:start + batch_size].to(device)).cpu().numpy())
    return np.concatenate(chunks)
