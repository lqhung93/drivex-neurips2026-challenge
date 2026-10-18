"""L1: H1 with a frozen instruction embedding concatenated to the history features.

Same MLP, same residual on CA, same history input as H1 (matched history only baseline). The
only change is a wider first layer whose text columns start at zero, so under the same seed L1
starts as exactly the H1 network and any difference after training comes from language.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from drivex26.h1 import HistoryMLP

HISTORY_DIM = 12


class InstructionMLP(nn.Module):
    """Residual in anchor-frame metres from [history features, text embedding]."""

    def __init__(self, text_dim: int, hidden: int = 128) -> None:
        super().__init__()
        if text_dim < 1:
            raise ValueError("text_dim must be positive")
        self.text_dim = text_dim
        history = HistoryMLP(hidden)  # same RNG draws as H1, so the same seed gives the same start
        self.net = history.net
        first = self.net[0]
        widened = nn.Linear(HISTORY_DIM + text_dim, hidden)
        with torch.no_grad():
            widened.weight.zero_()
            widened.weight[:, :HISTORY_DIM] = first.weight
            widened.bias.copy_(first.bias)
        self.net[0] = widened

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return base + self.net(features).reshape(-1, 12, 2)

    def load_history(self, state: dict) -> None:
        """Copy a (pretrained) HistoryMLP state into the history part; text columns stay as they are."""
        with torch.no_grad():
            self.net[0].weight[:, :HISTORY_DIM] = state["net.0.weight"]
            self.net[0].bias.copy_(state["net.0.bias"])
            for i in (2, 4):
                self.net[i].weight.copy_(state[f"net.{i}.weight"])
                self.net[i].bias.copy_(state[f"net.{i}.bias"])


def with_text(features: torch.Tensor, text: np.ndarray) -> torch.Tensor:
    """Concatenate H1 history features (N, 12) with embeddings (N, D)."""
    text = torch.as_tensor(np.asarray(text, dtype=np.float32))
    if features.shape[1] != HISTORY_DIM or text.ndim != 2 or len(text) != len(features):
        raise ValueError(f"cannot join features {tuple(features.shape)} with text {tuple(text.shape)}")
    return torch.cat((features, text), dim=1)
