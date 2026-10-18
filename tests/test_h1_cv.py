"""P1 leakage, shared-epoch selection, and frozen final evaluation checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from drivex26.data import save_cache
from drivex26.h1 import HistoryMLP
from drivex26.h1_training import H1Config, fit, select_epoch, split_fold
from drivex26.metrics import scene_folds


spec = importlib.util.spec_from_file_location("run_h1_cv", Path(__file__).resolve().parents[1] / "scripts/run_h1_cv.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_cache(names: list[str]) -> dict[str, np.ndarray]:
    n = len(names)
    times = np.tile([-2.0, -1.5, -1.0, -0.5], (n, 1)).astype(np.float32)
    future = np.tile(np.arange(1, 13, dtype=np.float32) * 2.5, (n, 1))
    return {"scene_name": np.array(names), "scene_token": np.array([f"token-{s}" for s in names]),
            "hist_t": times, "hist_xy": np.stack((5 * times, np.zeros_like(times)), axis=-1),
            "fut_xy": np.stack((future, np.zeros_like(future)), axis=-1) + 1,
            "has_static": np.ones(n, dtype=bool), "has_dynamic": np.zeros(n, dtype=bool)}


def plan() -> dict:
    return {"protocol": "P1", "folds": 5, "seeds": [0, 1, 2], "max_epochs": 310,
            "eval_every": 10, "selection_metric": "ade_scene",
            "candidates": [{"name": "tiny", **runner.asdict(H1Config(hidden=8))}]}


def prov() -> dict:
    return {"git_sha": "a" * 40, "git_dirty": False, "source_sha256": {"file": "abc"},
            "environment": {"torch": "test"}, "device": "cpu", "gpu_hours": 0.0}


def selection(train_hash: str) -> dict:
    p = plan()
    return {**prov(), "protocol": "P1", "plan": p, "cv_run_id": "cv",
            "train_cache_sha256": train_hash, "selection_metric": "ade_scene",
            "selected_candidate": "tiny", "config": {k: v for k, v in p["candidates"][0].items() if k != "name"},
            "epochs": 10, "seeds": p["seeds"], "selection_at_cap": False}


def test_scene_split_rejects_cross_fold_duplicate_token() -> None:
    cache = make_cache(["a", "a", "b"])
    train, held = split_fold(cache, np.array([0, 0, 1]), 0)
    assert set(train["scene_name"]) == {"b"} and set(held["scene_name"]) == {"a"}
    with pytest.raises(ValueError, match="leakage"):
        split_fold(cache, np.array([0, 1, 1]), 0)
    cache["scene_token"][:] = "same-token"
    with pytest.raises(ValueError, match="scene_token"):
        split_fold(cache, np.array([0, 0, 1]), 0)


def test_shared_epoch_uses_equal_scene_weights_and_all_seeds() -> None:
    # Three duplicate rows of a must not outweigh the one row of b.
    errors = np.array([[[1, 1, 1, 8], [4, 4, 4, 3]],
                       [[1, 1, 1, 6], [4, 4, 4, 3]],
                       [[1, 1, 1, 8], [4, 4, 4, 3]]], dtype=float)
    epoch, curve = select_epoch(errors, [0, 10], np.array(["a", "a", "a", "b"]))
    assert epoch == 10
    assert curve[0]["ade"] < curve[1]["ade"]
    assert curve[0]["ade_scene"] > curve[1]["ade_scene"]
    with pytest.raises(ValueError, match="finite"):
        select_epoch(errors * np.nan, [0, 10], np.array(["a", "a", "a", "b"]))


def test_fixed_fit_reproducible_and_without_validation() -> None:
    cache = make_cache(["a", "b"])
    config = H1Config(hidden=8)
    model1, preds, curve = fit(cache, config, seed=2, epochs=10)
    model2, _, _ = fit(cache, config, seed=2, epochs=10)
    assert preds == {} and curve == []
    for key, value in model1.state_dict().items():
        torch.testing.assert_close(value, model2.state_dict()[key], rtol=0, atol=0)


def test_cv_only_opens_train_and_covers_every_row(tmp_path: Path, monkeypatch) -> None:
    cache = make_cache([f"scene-{i:04d}" for i in range(30)] * 2)
    save_cache(cache, tmp_path / "train.npz")
    (tmp_path / "val.npz").write_text("must never be opened")
    (tmp_path / "test.npz").write_text("must never be opened")
    out = tmp_path / "cv"
    out.mkdir()
    calls = []

    def fake_fit(train, config, seed, epochs, *, held_out, eval_every):
        assert not set(train["scene_name"]) & set(held_out["scene_name"])
        calls.append((seed, len(held_out["scene_name"])))
        # Best epoch is 10, with deliberate fold/seed dependence for row alignment.
        offset = seed * 0.01 + scene_folds(held_out["scene_name"])[0] * 0.1
        preds = {e: held_out["fut_xy"] + abs(e - 10) + offset
                 for e in range(0, epochs + 1, eval_every)}
        return HistoryMLP(8), preds, []

    monkeypatch.setattr(runner, "fit", fake_fit)
    record = runner.cross_validate(tmp_path, plan(), out, prov())
    assert len(calls) == 15
    assert record["official_val_read"] is False and record["test_read"] is False
    assert record["selection"]["epochs"] == 10
    predictions = np.load(out / "preds_oof.npy")
    for index, seed in enumerate([0, 1, 2]):
        expected = cache["fut_xy"] + (seed * 0.01 + scene_folds(cache["scene_name"]) * 0.1)[:, None, None]
        np.testing.assert_allclose(predictions[index], expected, atol=3e-6)


@pytest.mark.parametrize("field,value,match", [
    ("git_dirty", True, "clean source"), ("git_sha", "wrong", "clean source"),
    ("source_sha256", {}, "source differs"), ("train_cache_sha256", "wrong", "train cache"),
    ("selection_at_cap", True, "hits cap"), ("epochs", 310, "hits cap"),
    ("seeds", [0, 1], "seeds differ"), ("environment", {}, "environment differs"),
])
def test_final_rejects_invalid_selection_before_val(field, value, match) -> None:
    chosen = selection("train")
    chosen[field] = value
    with pytest.raises(ValueError, match=match):
        runner.validate_selection(chosen, prov(), "train")


def test_final_freezes_all_models_before_opening_val_and_prevents_repeat(tmp_path: Path, monkeypatch) -> None:
    train = make_cache(["a", "b"])
    val = make_cache(["c", "d"])
    val["fut_xy"] += 100  # cannot influence fitted models or selected epoch
    save_cache(train, tmp_path / "train.npz")
    save_cache(val, tmp_path / "val.npz")
    cv = tmp_path / "cv"
    cv.mkdir()
    chosen = selection(runner.sha256(tmp_path / "train.npz"))
    path = cv / "selection.json"
    path.write_text(json.dumps(chosen))
    out = tmp_path / "final"
    out.mkdir()
    events = []
    original_load, original_fit = runner.load_cache, runner.fit

    def tracked_load(path):
        events.append(path.name)
        return original_load(path)

    def tracked_fit(*args, **kwargs):
        assert "held_out" not in kwargs
        events.append("fit")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(runner, "load_cache", tracked_load)
    monkeypatch.setattr(runner, "fit", tracked_fit)
    record = runner.final_evaluation(tmp_path, path, out, prov())
    assert events == ["train.npz", "fit", "fit", "fit", "val.npz"]
    assert record["official_val_evaluations_per_seed"] == 1
    expected, _, _ = fit(train, H1Config(**chosen["config"]), 0, epochs=10)
    saved = torch.load(out / "s0/model.pt", weights_only=True)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="already reserved"):
        runner.final_evaluation(tmp_path, path, out, prov())
