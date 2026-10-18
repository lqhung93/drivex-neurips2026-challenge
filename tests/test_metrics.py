import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import pytest

from drivex26.metrics import ade, bootstrap_ci, evaluate, fde, paired_delta, scene_folds

OFFICIAL_METRICS = Path(__file__).resolve().parents[1] / "third_party" / "doScenes" / "metrics.py"


def test_ade_fde_hand_example():
    # per-step errors 0, 1, 2
    gt = np.array([[[1, 0], [2, 0], [3, 0]]], dtype=float)
    pred = np.array([[[1, 0], [2, 1], [3, 2]]], dtype=float)
    np.testing.assert_allclose(ade(pred, gt), [1.0])
    np.testing.assert_allclose(fde(pred, gt), [2.0])


class _NoMap:
    """Map stub so the official compute_ego_metrics runs; map metrics are not compared."""

    def record_on_point(self, x, y, layer):
        return ""

    def get_closest_lane(self, x, y, radius):
        return ""


def test_matches_official_metrics_py():
    spec = importlib.util.spec_from_file_location("doscenes_metrics", OFFICIAL_METRICS)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    rng = np.random.default_rng(0)
    pred, gt = rng.normal(size=(20, 12, 2)) * 5, rng.normal(size=(20, 12, 2)) * 5
    for i in range(len(pred)):
        ref = official.compute_ego_metrics(pred[i], gt[i], np.zeros(2), 0.0, _NoMap())
        np.testing.assert_allclose(ade(pred[i:i + 1], gt[i:i + 1])[0], ref["ade_6s"])
        np.testing.assert_allclose(fde(pred[i:i + 1], gt[i:i + 1])[0], ref["fde"])


def test_bootstrap_ci_resamples_scenes_not_pairs():
    # scene "a" has 3 pairs, scene "b" has 1; per-scene means are 0 and 10.
    values = np.array([0.0, 0.0, 0.0, 10.0])
    groups = np.array(["a", "a", "a", "b"])
    lo, hi = bootstrap_ci(values, groups, n=2000, seed=0)
    assert lo == 0.0 and hi == 10.0  # only possible scene-level resample means: 0, 5, 10


def test_bootstrap_ci_constant_is_degenerate():
    lo, hi = bootstrap_ci(np.full(10, 3.0), np.arange(10), n=100, seed=0)
    assert lo == hi == 3.0


def test_evaluate_reports_per_category_counts():
    gt = np.zeros((4, 12, 2))
    pred = gt.copy()
    pred[:, :, 0] = np.array([1.0, 2.0, 3.0, 4.0])[:, None]  # constant offset -> ADE = offset
    res = evaluate(pred, gt, np.array(["a", "a", "b", "c"]),
                   has_static=np.array([True, True, False, False]),
                   has_dynamic=np.array([False, True, True, False]))
    assert res["ade"] == 2.5 and res["fde"] == 2.5
    assert res["n_pairs"] == 4 and res["n_scenes"] == 3
    assert res["ade_scene"] == (1.5 + 3.0 + 4.0) / 3
    assert res["per_category"] == {
        "s": {"n": 1, "ade": 1.0}, "sd": {"n": 1, "ade": 2.0},
        "d": {"n": 1, "ade": 3.0}, "none": {"n": 1, "ade": 4.0},
    }


# --- P1: scene grouped folds and paired ΔADE ---

def test_scene_folds_keep_every_scene_in_one_fold():
    names = np.array(["scene-0001", "scene-0002", "scene-0001", "scene-0003", "scene-0002"])
    folds = scene_folds(names, k=5)
    for n in np.unique(names):
        assert len(set(folds[names == n])) == 1


def test_scene_folds_do_not_depend_on_row_order_or_process():
    names = np.array([f"scene-{i:04d}" for i in range(1, 601)])
    folds = scene_folds(names, k=5)
    perm = np.random.default_rng(0).permutation(len(names))
    np.testing.assert_array_equal(scene_folds(names[perm], k=5), folds[perm])
    # md5 based, so the same in every Python process (built-in hash() is salted per process)
    code = "import numpy as np; from drivex26.metrics import scene_folds; " \
           "print(''.join(map(str, scene_folds(np.array([f'scene-{i:04d}' for i in range(1, 601)]), k=5))))"
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip() for seed in ("1", "2")}
    assert runs == {"".join(map(str, folds))}


def test_scene_folds_are_roughly_balanced():
    names = np.array([f"scene-{i:04d}" for i in range(1, 492)])  # 491 train scenes
    sizes = np.bincount(scene_folds(names, k=5), minlength=5)
    assert sizes.min() > 0.7 * 491 / 5 and sizes.max() < 1.3 * 491 / 5


def test_scene_folds_reject_k_below_two():
    with pytest.raises(ValueError):
        scene_folds(np.array(["scene-0001"]), k=1)


def test_paired_delta_sign_and_degenerate_ci():
    groups = np.array(["a", "a", "b", "c"])
    base = np.array([3.0, 3.0, 4.0, 5.0])
    res = paired_delta(base, base - 0.5, groups, n=200)
    assert res["delta"] == pytest.approx(0.5) and res["delta_scene"] == pytest.approx(0.5)
    assert res["ci95"] == pytest.approx((0.5, 0.5))
    assert res["n_pairs"] == 4 and res["n_scenes"] == 3
    worse = paired_delta(base, base + 0.2, groups, n=200)
    assert worse["delta"] == pytest.approx(-0.2)  # positive ΔADE means language helps


def test_paired_delta_resamples_scenes_not_pairs():
    # scene a: 3 pairs with delta 0; scene b: 1 pair with delta 1
    groups = np.array(["a", "a", "a", "b"])
    base = np.array([1.0, 1.0, 1.0, 2.0])
    instr = np.array([1.0, 1.0, 1.0, 1.0])
    res = paired_delta(base, instr, groups, n=2000, seed=0)
    assert res["delta"] == pytest.approx(0.25)
    assert res["delta_scene"] == pytest.approx(0.5)
    assert res["ci95"] == (0.0, 1.0)


def test_paired_delta_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_delta(np.zeros(3), np.zeros(4), np.array(["a", "b", "c"]))


def test_instruction_category_from_flags():
    from drivex26.metrics import instruction_category
    cats = instruction_category(np.array([True, True, False, False]), np.array([False, True, True, False]))
    assert cats.tolist() == ["s", "sd", "d", "none"]
