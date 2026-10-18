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

    def __init__(self, text_dim: int, hidden: int = 128, text_hidden: int = 32, extra_dim: int = 0) -> None:
        super().__init__()
        if text_dim < 1 or extra_dim < 0:
            raise ValueError("text_dim must be positive and extra_dim nonnegative")
        self.text_dim, self.extra_dim = text_dim, extra_dim
        self.history = HistoryMLP(hidden)  # drawn first: same weights as H1 under the same seed
        if extra_dim:  # non language inputs (e.g. the map) as extra first layer columns starting at zero
            first, widened = self.history.net[0], nn.Linear(HISTORY_DIM + extra_dim, hidden)
            with torch.no_grad():
                widened.weight.zero_()
                widened.weight[:, :HISTORY_DIM] = first.weight
                widened.bias.copy_(first.bias)
            self.history.net[0] = widened
        self.film = nn.Sequential(nn.Linear(text_dim, text_hidden), nn.ReLU(), nn.Linear(text_hidden, 2 * hidden))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def load_history(self, state: dict) -> None:
        """Copy a (pretrained) HistoryMLP state into the history network; extra columns stay as they are."""
        with torch.no_grad():
            net = self.history.net
            net[0].weight[:, :HISTORY_DIM] = state["net.0.weight"]
            net[0].bias.copy_(state["net.0.bias"])
            for i in (2, 4):
                net[i].weight.copy_(state[f"net.{i}.weight"])
                net[i].bias.copy_(state[f"net.{i}.bias"])

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        split = HISTORY_DIM + self.extra_dim
        history, text = features[:, :split], features[:, split:]
        net = self.history.net
        h = net[3](net[2](net[1](net[0](history))))
        gamma, beta = self.film(text).chunk(2, dim=-1)
        return base + net[4](h * (1 + gamma) + beta).reshape(-1, 12, 2)
