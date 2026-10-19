import pytest
import numpy as np
import torch

from drivex26.baselines import constant_acceleration
from drivex26.h1 import HistoryMLP, predict, prepare_inputs
from drivex26.l1 import InstructionMLP, with_text
from drivex26.text import load_features, save_features, shuffle_across_scenes


def history_cache(n: int = 6) -> dict[str, np.ndarray]:
    times = np.tile([-2.0, -1.5, -1.0, -0.5], (n, 1)).astype(np.float32)
    speed = np.arange(1, n + 1, dtype=np.float32)[:, None]
    return {"hist_t": times, "hist_xy": np.stack((speed * times, np.zeros_like(times)), axis=-1)}


def test_l1_starts_as_h1_under_the_same_seed() -> None:
    features, base = prepare_inputs(history_cache())
    text = np.random.default_rng(0).normal(size=(len(features), 8)).astype(np.float32)
    torch.manual_seed(3)
    h1 = HistoryMLP(hidden=16)
    torch.manual_seed(3)
    l1 = InstructionMLP(8, hidden=16)
    np.testing.assert_array_equal(l1.net[0].weight[:, :12].detach(), h1.net[0].weight.detach())
    assert not l1.net[0].weight[:, 12:].any()
    # identical hidden layers too, so L1 = H1 for any instruction until it trains
    np.testing.assert_array_equal(l1.net[2].weight.detach(), h1.net[2].weight.detach())
    pred = predict(l1, with_text(features, text), base)
    np.testing.assert_array_equal(pred, predict(h1, features, base))
    cache = history_cache()
    np.testing.assert_allclose(pred, constant_acceleration(cache["hist_xy"], cache["hist_t"]))


def test_text_columns_learn() -> None:
    torch.manual_seed(0)
    features, base = prepare_inputs(history_cache())
    text = torch.eye(len(features))  # one-hot text, the only thing that separates the targets
    target = base + torch.arange(len(features), dtype=torch.float32)[:, None, None]
    model = InstructionMLP(len(features), hidden=16)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(200):
        loss = torch.linalg.vector_norm(model(with_text(features, text), base) - target, dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert model.net[0].weight[:, 12:].abs().sum() > 0
    assert loss.item() < 1.0


def test_shuffle_never_keeps_the_scene() -> None:
    scenes = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 2)
    for seed in range(20):
        perm = shuffle_across_scenes(scenes, seed=seed)
        assert sorted(perm) == list(range(len(scenes)))
        assert not (scenes[perm] == scenes).any()


def test_features_must_match_cache_text(tmp_path) -> None:
    texts = ["turn left", "stop"]
    emb = np.ones((2, 4), np.float32)
    save_features(tmp_path, "toy", "train", emb, texts, {})
    np.testing.assert_array_equal(load_features(tmp_path, "toy", "train", texts), emb)
    try:
        load_features(tmp_path, "toy", "train", ["stop", "turn left"])
    except ValueError:
        pass
    else:
        raise AssertionError("reordered instructions were accepted")


def test_l2_starts_as_h1_under_the_same_seed() -> None:
    from drivex26.l2 import FiLMMLP
    features, base = prepare_inputs(history_cache())
    text = np.random.default_rng(0).normal(size=(len(features), 8)).astype(np.float32)
    torch.manual_seed(3)
    h1 = HistoryMLP(hidden=16)
    torch.manual_seed(3)
    l2 = FiLMMLP(8, hidden=16)
    np.testing.assert_array_equal(l2.history.net[0].weight.detach(), h1.net[0].weight.detach())
    # once H1's last layer has learned something, the zero FiLM head still leaves the output equal
    with torch.no_grad():
        h1.net[4].weight.normal_()
    l2.load_history(h1.state_dict())
    np.testing.assert_allclose(predict(l2, with_text(features, text), base), predict(h1, features, base), atol=1e-6)


def test_load_history_copies_pretrained_weights_into_l1() -> None:
    features, base = prepare_inputs(history_cache())
    text = np.random.default_rng(1).normal(size=(len(features), 8)).astype(np.float32)
    torch.manual_seed(5)
    pretrained = HistoryMLP(hidden=16)
    with torch.no_grad():
        pretrained.net[4].weight.normal_()
    torch.manual_seed(0)
    l1 = InstructionMLP(8, hidden=16)
    l1.load_history(pretrained.state_dict())
    assert not l1.net[0].weight[:, 12:].any()
    np.testing.assert_allclose(predict(l1, with_text(features, text), base),
                               predict(pretrained, features, base), atol=1e-6)


def _run_l1():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("run_l1", Path(__file__).resolve().parents[1] / "scripts/run_l1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pretraining_windows_drop_the_held_out_scenes() -> None:
    runner = _run_l1()
    windows = {"scene_name": np.array(["a", "a", "b", "c", "c"]), "anchor_index": np.arange(5)}
    kept = runner.windows_without(windows, np.array(["c", "a", "c"]))
    assert kept["scene_name"].tolist() == ["b"] and kept["anchor_index"].tolist() == [2]


def test_text_dropout_keeps_the_batch_order_of_h1() -> None:
    runner = _run_l1()
    features, base = prepare_inputs(history_cache(8))
    text = torch.ones(len(features), 4)
    target = base + 1.0
    cfg = {"lr": 0.01, "weight_decay": 0.0, "batch_size": 3}
    seen = []

    class Spy(HistoryMLP):
        def forward(self, x, b):
            seen.append(x[:, 12:].sum(dim=1).tolist())
            return super().forward(x[:, :12], b)

    torch.manual_seed(0)
    runner.train_eval(Spy(8), torch.cat((features, text), 1), base, target, {}, cfg, 0, [2], text_dropout=0.5)
    dropped, kept = seen, []
    seen = []
    torch.manual_seed(0)
    runner.train_eval(Spy(8), torch.cat((features, text), 1), base, target, {}, cfg, 0, [2])
    kept = seen
    assert [len(b) for b in dropped] == [len(b) for b in kept]
    flat = [v for b in dropped for v in b]
    assert 0.0 in flat and 4.0 in flat and all(v == 4.0 for b in kept for v in b)


def test_test_stage_writes_two_valid_submission_files(tmp_path) -> None:
    import json
    import types

    from drivex26.data import save_cache
    from drivex26.submission import validate_submission

    runner = _run_l1()
    rng = np.random.default_rng(0)

    def cache(n, scenes, future=True):
        times = np.tile([-2.0, -1.5, -1.0, -0.5], (n, 1)).astype(np.float32)
        c = {"scene_name": np.array([f"scene-{i % scenes:04d}" for i in range(n)]),
             "scene_token": np.array([f"tok{i % scenes}" for i in range(n)]),
             "instruction": np.array([f"go {i}" for i in range(n)]),
             "hist_t": times, "hist_xy": rng.normal(size=(n, 4, 2)).astype(np.float32)}
        if future:
            c["fut_xy"] = rng.normal(size=(n, 12, 2)).astype(np.float32)
        return c

    caches, feats, runs = tmp_path / "cache", tmp_path / "feat", tmp_path / "runs"
    train, test = cache(12, 6), cache(7, 3, future=False)
    save_cache(train, caches / "train.npz")
    save_cache(test, caches / "test.npz")
    for split, c in (("train", train), ("test", test)):
        save_features(feats, "minilm", split, rng.normal(size=(len(c["instruction"]), 4)).astype(np.float32),
                      [str(t) for t in c["instruction"]], {})
    (runs / "cv").mkdir(parents=True)
    (runs / "cv" / "metrics.json").write_text(json.dumps({
        "config": {"stage": "cv", "model": "l2-film-frozen-text", "encoder": "minilm", "seeds": [0, 1], "fusion": "film",
                   "pretrain_epochs": 0, "text_dropout": 0.3, "hidden": 8, "lr": 0.01, "weight_decay": 0.0, "batch_size": 4},
        "selected_epoch": {"h1": 2, "l1": 3}, "epoch_at_cap": {"h1": False, "l1": False}, "selection": "x"}))
    args = types.SimpleNamespace(output_dir=runs, from_cv="cv", cache_dir=caches, feature_dir=feats,
                                 windows=tmp_path / "none.npz", team="DHN", version=0, run_id="sub", out_dir=tmp_path / "out")
    runner.run_test(args)
    for name in ("DHN_CSV_version0.csv", "DHN_CSV_version0_historyonly.csv"):
        path = tmp_path / "out" / name
        assert validate_submission(path, expected_tokens={"tok0", "tok1", "tok2"}, team="DHN")[0] == []
        assert len(path.read_text().splitlines()) == 4  # header + one row per test scene
    baseline = (tmp_path / "out" / "DHN_CSV_version0_historyonly.csv").read_text().splitlines()[1]
    assert baseline.split(",")[1] == ""  # empty instruction column in the history only file
    record = json.loads((runs / "sub" / "metrics.json").read_text())
    assert record["config"]["purpose"] == "submission" and record["uploaded"] is False

    test["fut_xy"] = np.zeros((7, 12, 2), np.float32)  # a test cache with futures is refused
    save_cache(test, caches / "test.npz")
    args.run_id = "sub2"
    with pytest.raises(ValueError):
        runner.run_test(args)


def test_intent_one_hot_layout() -> None:
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "encode_intent", Path(__file__).resolve().parents[1] / "scripts/language/encode_intent.py")
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    v = enc.one_hot({"maneuver": "turn_left", "then": None, "speed": "slow", "wait_first": True})
    assert v.shape == (26,) and v.sum() == 4
    assert v[1] == 1 and v[10 + 10] == 1 and v[21 + 1] == 1 and v[25] == 1  # turn_left, then none, slow, wait
    assert not enc.one_hot(None).any()  # unparsed intent = blank instruction


def test_intent_parser_matches_one_hot_vocabulary() -> None:
    import importlib.util
    from pathlib import Path
    mods = {}
    for name in ("extract_intent", "encode_intent"):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[1] / f"scripts/language/{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[name])
    ext, enc = mods["extract_intent"], mods["encode_intent"]
    assert ext.MANEUVERS == enc.MANEUVERS and ext.SPEEDS == enc.SPEEDS
    out = ext.parse('Sure: {"maneuver": "stop", "then": "turn_right", "speed": "stop", "wait_first": false}')
    assert out["then"] == "turn_right" and enc.one_hot(out).sum() == 3
    assert ext.parse('{"maneuver": "fly", "speed": "keep"}') is None
    assert ext.parse("no json") is None


def test_film_with_extra_inputs_starts_as_its_matched_baseline() -> None:
    runner = _run_l1()
    features, base = prepare_inputs(history_cache())
    rng = np.random.default_rng(2)
    extra, text = rng.normal(size=(len(features), 5)), rng.normal(size=(len(features), 4))
    torch.manual_seed(7)
    pretrained = HistoryMLP(hidden=16)
    with torch.no_grad():
        pretrained.net[4].weight.normal_()
    h2, lm = runner.make_pair(4, {"hidden": 16}, 0, "film", 5, pretrained.state_dict())
    x = with_text(features, extra)
    xt = torch.cat((x, torch.as_tensor(text, dtype=torch.float32)), 1)
    np.testing.assert_allclose(predict(lm, xt, base), predict(h2, x, base), atol=1e-6)
    np.testing.assert_allclose(predict(h2, x, base), predict(pretrained, features, base), atol=1e-6)


def test_text_dropout_keeps_extra_columns() -> None:
    runner = _run_l1()
    features, base = prepare_inputs(history_cache(8))
    x = torch.cat((features, torch.full((8, 3), 2.0), torch.ones(8, 4)), 1)
    seen = []

    class Spy(HistoryMLP):
        def forward(self, xb, b):
            seen.append((xb[:, 12:15].clone(), xb[:, 15:].sum(dim=1).tolist()))
            return super().forward(xb[:, :12], b)

    runner.train_eval(Spy(8), x, base, base + 1.0, {}, {"lr": 0.01, "weight_decay": 0.0, "batch_size": 4}, 0, [3],
                      text_dropout=0.5, text_start=15)
    assert all((extra == 2.0).all() for extra, _ in seen)  # map columns never dropped
    flat = [v for _, t in seen for v in t]
    assert 0.0 in flat and 4.0 in flat
