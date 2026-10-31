import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from drivex26.h1 import prepare_inputs
from drivex26.map_features import NAMES, NAMES2, lane_features, lane_features_v2
from drivex26.mirror import HISTORY_NAMES, INTENT_NAMES, mirror_columns, mirror_specs
from test_data import FakeMap


def flip(features, names):
    perm, sign = mirror_columns(names)
    return np.asarray(features)[..., perm] * sign


def test_history_mirror_matches_a_mirrored_history() -> None:
    rng = np.random.default_rng(0)
    cache = {"hist_t": np.tile([-2.0, -1.5, -1.0, -0.5], (5, 1)), "hist_xy": rng.normal(size=(5, 4, 2))}
    mirrored = {**cache, "hist_xy": cache["hist_xy"] * [1, -1]}
    (x, base), (mx, mbase) = prepare_inputs(cache), prepare_inputs(mirrored)
    np.testing.assert_allclose(flip(x.numpy(), HISTORY_NAMES), mx.numpy(), atol=1e-6)
    np.testing.assert_allclose(base.numpy() * [1, -1], mbase.numpy(), atol=1e-5)  # CA is linear in the poses
    np.testing.assert_allclose(flip(flip(x.numpy(), HISTORY_NAMES), HISTORY_NAMES), x.numpy())


def test_intent_mirror_swaps_left_and_right() -> None:
    spec = importlib.util.spec_from_file_location(
        "encode_intent", Path(__file__).resolve().parents[1] / "scripts/language/encode_intent.py")
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    assert len(INTENT_NAMES) == enc.DIM and [n[2:] for n in INTENT_NAMES[:10]] == enc.MANEUVERS
    hot = lambda m, then: enc.one_hot({"maneuver": m, "then": then, "speed": "slow", "wait_first": True})
    np.testing.assert_array_equal(flip(hot("turn_left", "lane_change_right"), INTENT_NAMES),
                                  hot("turn_right", "lane_change_left"))
    np.testing.assert_array_equal(flip(hot("stop", None), INTENT_NAMES), hot("stop", None))


class MirroredMap(FakeMap):
    """FakeMap reflected about x = 100 (a left / right flip for an ego heading north), with a lane on each side."""

    def __init__(self, mirror: bool):
        super().__init__()
        n = np.arange(0, 60.5, 0.5)
        self.lanes |= {"west": [(97.0, 195.0 + d, math.pi / 2) for d in n],
                       "east": [(104.0, 195.0 + d, math.pi / 2) for d in n]}
        if mirror:
            self.lanes = {t: [(200.0 - x, y, math.pi - h) for x, y, h in p] for t, p in self.lanes.items()}

    def get_records_in_radius(self, x, y, radius, layers):
        return {"lane": ["main", "oncoming", "west", "east"], "lane_connector": []}


def test_map_features_of_a_mirrored_map_are_the_mirrored_features() -> None:
    ego, yaw = np.array([100.5, 205.0]), math.pi / 2
    for features, names in ((lane_features, NAMES), (lane_features_v2, NAMES2)):
        f = features(MirroredMap(False), ego, yaw)
        g = features(MirroredMap(True), ego * [-1, 1] + [200, 0], yaw)
        np.testing.assert_allclose(flip(f, names), g, atol=1e-5)
    f = dict(zip(NAMES2, lane_features_v2(MirroredMap(False), ego, yaw)))
    assert f["nb_left_valid"] == f["nb_right_valid"] == 1.0  # west lane 3.5 m left, east lane 3.5 m right
    assert f["nb_left_y75"] > 0 > f["nb_right_y75"] and f["ego_left_y600"] > 0 > f["ego_right_y600"]


def test_mirror_flips_whole_rows_of_input_base_and_target() -> None:
    import test_l1
    runner = test_l1._run_l1()
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.normal(size=(16, 12 + 17 + 26)), dtype=torch.float32)
    base, y = (torch.as_tensor(rng.normal(size=(16, 12, 2)), dtype=torch.float32) for _ in range(2))
    seen = []

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, xb, bb):
            seen.append((xb.clone(), bb.clone()))
            return bb + self.w

    _, spec = mirror_specs("intent", "map")
    runner.train_eval(Spy(), x, base, y, {}, {"lr": 0.0, "weight_decay": 0.0, "batch_size": 16}, 0, [1], mirror=spec)
    xb, bb = seen[0]
    order = torch.randperm(16, generator=torch.Generator().manual_seed(0))  # batch order unchanged
    flipped = ~torch.isclose(bb[:, 0, 1], base[order][:, 0, 1])
    assert 0 < flipped.sum() < 16
    names = HISTORY_NAMES + NAMES + INTENT_NAMES
    np.testing.assert_allclose(xb[flipped].numpy(), flip(x[order][flipped].numpy(), names), atol=1e-6)
    np.testing.assert_allclose(xb[~flipped].numpy(), x[order][~flipped].numpy())
