from pathlib import Path

import numpy as np
import torch

from drivex26.baselines import constant_acceleration
from drivex26.h1 import HistoryMLP, predict, prepare_inputs


def history_cache() -> dict[str, np.ndarray]:
    times = np.tile([-2.0, -1.5, -1.0, -0.5], (3, 1)).astype(np.float32)
    return {"hist_t": times, "hist_xy": np.stack((5 * times, np.zeros_like(times)), axis=-1)}


def test_initial_prediction_matches_ca_without_future_or_instruction() -> None:
    cache = history_cache()
    features, base = prepare_inputs(cache)
    pred = predict(HistoryMLP(), features, base, batch_size=2)
    assert pred.shape == (3, 12, 2)
    np.testing.assert_allclose(pred, constant_acceleration(cache["hist_xy"], cache["hist_t"]))


def test_cpu_learning_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    features, base = prepare_inputs(history_cache())
    model = HistoryMLP(hidden=16)
    target = base + 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial_loss = torch.linalg.vector_norm(model(features, base) - target, dim=-1).mean().item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.linalg.vector_norm(model(features, base) - target, dim=-1).mean()
        loss.backward()
        optimizer.step()
    pred = predict(model, features, base)
    assert np.linalg.norm(pred - target.numpy(), axis=-1).mean() < initial_loss * 0.1
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = HistoryMLP(hidden=16)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    np.testing.assert_array_equal(predict(restored, features, base), pred)
