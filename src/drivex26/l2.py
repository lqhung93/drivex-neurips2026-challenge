"""L2: H1 whose last hidden layer is modulated by the instruction (FiLM).

The history network is exactly H1 (same RNG draws first, so the same seed gives the same
weights); a small text head predicts a per unit scale and shift for the second hidden layer.
Its last layer starts at zero, so L2 starts as H1 and the instruction can only learn a
correction on top of the history prediction.
"""

from __future__ import annotations

import torch
from torch import nn

from drivex26.h1 import HistoryMLP
from drivex26.l1 import HISTORY_DIM


class FiLMMLP(nn.Module):
    """Residual on CA from history features, FiLM conditioned on a text embedding."""

    def __init__(self, text_dim: int, hidden: int = 128, text_hidden: int = 32) -> None:
        super().__init__()
        if text_dim < 1:
            raise ValueError("text_dim must be positive")
        self.text_dim = text_dim
        self.history = HistoryMLP(hidden)  # drawn first: same weights as H1 under the same seed
        self.film = nn.Sequential(nn.Linear(text_dim, text_hidden), nn.ReLU(), nn.Linear(text_hidden, 2 * hidden))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def load_history(self, state: dict) -> None:
        """Copy a (pretrained) HistoryMLP state into the history network."""
        self.history.load_state_dict(state)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        history, text = features[:, :HISTORY_DIM], features[:, HISTORY_DIM:]
        net = self.history.net
        h = net[3](net[2](net[1](net[0](history))))
        gamma, beta = self.film(text).chunk(2, dim=-1)
        return base + net[4](h * (1 + gamma) + beta).reshape(-1, 12, 2)
