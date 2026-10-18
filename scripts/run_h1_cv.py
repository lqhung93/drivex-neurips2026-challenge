"""Select the H1 config and epoch by train CV (P1), then evaluate the frozen models on val.

  python scripts/run_h1_cv.py cv --run-id h1-p1-cv
  python scripts/run_h1_cv.py final --selection results/runs/h1-p1-cv/selection.json --run-id h1-p1-final

CPU only. Neither stage opens test. CV never opens official val. Run from a
clean checkout, with outputs outside that checkout, to preserve provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from drivex26.data import assert_split_allowed, load_cache
from drivex26.h1 import predict, prepare_inputs
from drivex26.h1_training import H1Config, fit, select_epoch, split_fold
from drivex26.metrics import ade, bootstrap_ci, evaluate, fde, paired_delta, scene_folds


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/run_h1_cv.py", "src/drivex26/h1_training.py",
           "src/drivex26/h1.py", "src/drivex26/baselines.py",
           "src/drivex26/data.py", "src/drivex26/metrics.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def provenance() -> dict:
    for name in ("drivex26.h1", "drivex26.h1_training", "drivex26.baselines",
                 "drivex26.data", "drivex26.metrics"):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(ROOT / "src"):
            raise RuntimeError(f"{name} was imported from a different checkout; set PYTHONPATH to this src/")
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain"):
        raise RuntimeError("P1 runs require a clean checkout; use a clean worktree and external output-dir")
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": False,
            "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "platform": platform.platform()},
            "device": "cpu", "gpu_hours": 0.0}


def validate_plan(plan: dict) -> None:
    if plan["protocol"] != "P1" or plan["folds"] != 5:
        raise ValueError("P1 requires five scene folds")
    seeds = plan["seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) for s in seeds):
        raise ValueError("P1 requires at least three distinct integer seeds")
    if (plan["max_epochs"] <= 300 or plan["eval_every"] < 1
            or plan["max_epochs"] % plan["eval_every"]):
        raise ValueError("need max_epochs > 300, divisible by eval_every")
    if plan["selection_metric"] != "ade_scene":
        raise ValueError("this runner selects equal-scene ADE")
    names = [row["name"] for row in plan["candidates"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("candidate names must be nonempty and unique")
    for row in plan["candidates"]:
        if not row["name"] or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in row["name"]):
            raise ValueError("invalid candidate name")
        H1Config(**{key: value for key, value in row.items() if key != "name"})


def reserve(output_dir: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a directory name")
    path = output_dir / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_run(path: Path, record: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "metrics.json", record)
    write_json(path / "config.yaml", record["config"])
    (path / "git_sha.txt").write_text(record["git_sha"] + "\n")
    seeds = record["config"].get("seeds", [record["config"].get("seed")])
    (path / "seed.txt").write_text(",".join(map(str, seeds)) + "\n")


def write_predictions(path: Path, cache: dict[str, np.ndarray], pred: np.ndarray) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_index", "scene_name", "scene_token"] +
                        [f"{axis}{step}" for step in range(1, 13) for axis in ("x", "y")])
        for index, (name, token, xy) in enumerate(zip(cache["scene_name"], cache["scene_token"], pred)):
            writer.writerow([index, name, token, *xy.reshape(-1).tolist()])


def summarize(predictions: np.ndarray, cache: dict[str, np.ndarray], seed: int = 0) -> dict:
    """Mean errors over seeds, not an ensemble prediction; bootstrap scenes."""
    errors = np.stack([ade(pred, cache["fut_xy"]) for pred in predictions])
    final = np.stack([fde(pred, cache["fut_xy"]) for pred in predictions])
    metrics = [evaluate(pred, cache["fut_xy"], cache["scene_name"],
                        cache["has_static"], cache["has_dynamic"], seed=seed)
               for pred in predictions]
    _, base = prepare_inputs(cache)
    base_errors = ade(base.numpy(), cache["fut_xy"])
    result = {
        "ade_mean": float(errors.mean()), "ade_std": float(errors.mean(axis=1).std(ddof=1)),
        "fde_mean": float(final.mean()), "fde_std": float(final.mean(axis=1).std(ddof=1)),
        "ade_scene_mean": float(np.mean([row["ade_scene"] for row in metrics])),
        "ade_scene_std": float(np.std([row["ade_scene"] for row in metrics], ddof=1)),
        "ade_scene_ci95": bootstrap_ci(errors.mean(axis=0), cache["scene_name"], seed=seed),
        "n_pairs": len(cache["scene_name"]), "n_scenes": len(np.unique(cache["scene_name"])),
        "ca_ade": float(base_errors.mean()),
        # Motion-model improvement, NOT the language ΔADE from the challenge.
        "ca_minus_h1_ade": paired_delta(base_errors, errors.mean(axis=0), cache["scene_name"], seed=seed),
        "per_category": {category: {"n": metrics[0]["per_category"][category]["n"],
                                    "ade_mean": float(np.mean([m["per_category"][category]["ade"] for m in metrics]))}
                         for category in metrics[0]["per_category"]},
        "by_seed": metrics,
    }
    return result


def cross_validate(cache_dir: Path, plan: dict, run_dir: Path, prov: dict) -> dict:
    """Only train.npz is ever loaded here; all predictions are out of fold."""
    validate_plan(plan)
    assert_split_allowed("train", "train")
    cache = load_cache(cache_dir / "train.npz")
    cache_hash = sha256(cache_dir / "train.npz")
    folds = scene_folds(cache["scene_name"], k=5)
    epochs = list(range(0, plan["max_epochs"] + 1, plan["eval_every"]))
    count = len(cache["scene_name"])
    np.save(run_dir / "fold_ids.npy", folds)
    with (run_dir / "folds.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_name", "scene_token", "fold"])
        _, indices = np.unique(cache["scene_name"], return_index=True)
        for index in indices:
            writer.writerow([cache["scene_name"][index], cache["scene_token"][index], folds[index]])
    start = time.monotonic()
    candidates = []
    winner: dict | None = None
    winner_pred: np.ndarray | None = None
    for candidate in plan["candidates"]:
        config = H1Config(**{key: value for key, value in candidate.items() if key != "name"})
        oof = np.empty((len(plan["seeds"]), len(epochs), count, 12, 2), dtype=np.float32)
        coverage = np.zeros((len(plan["seeds"]), count), dtype=np.int32)
        fold_records = []
        for seed_index, seed in enumerate(plan["seeds"]):
            for fold in range(5):
                train, held = split_fold(cache, folds, fold)
                begin = time.monotonic()
                _, predictions, curve = fit(train, config, seed, plan["max_epochs"],
                                            held_out=held, eval_every=plan["eval_every"])
                for epoch_index, epoch in enumerate(epochs):
                    oof[seed_index, epoch_index, folds == fold] = predictions[epoch]
                coverage[seed_index, folds == fold] += 1
                fold_records.append((seed, fold, held, {
                    **prov, "run_id": f"{run_dir.name}/{candidate['name']}-s{seed}-f{fold}",
                    "config": {"model": "h1-mlp-ca-residual", **asdict(config), "seed": seed,
                               "fold": fold, "split": "train-cv", "epochs": plan["max_epochs"]},
                    "cache_sha256": {"train": cache_hash}, "train_seconds": time.monotonic() - begin,
                    "epochs_completed": plan["max_epochs"], "learning_curve": curve,
                    "train_scenes": sorted(set(train["scene_name"].tolist())),
                    "held_out_scenes": sorted(set(held["scene_name"].tolist())),
                    "n_train_pairs": len(train["scene_name"]), "n_held_out_pairs": len(held["scene_name"]),
                }))
                print(f"{candidate['name']} seed={seed} fold={fold} complete", flush=True)
        if not np.all(coverage == 1):
            raise RuntimeError("each row must receive exactly one held-out prediction per seed")
        errors = np.stack([np.stack([ade(pred, cache["fut_xy"]) for pred in rows]) for rows in oof])
        chosen_epoch, curve = select_epoch(errors, epochs, cache["scene_name"])
        chosen_pred = oof[:, epochs.index(chosen_epoch)].copy()
        candidate_record = {"name": candidate["name"], "config": asdict(config),
                            "selected_epoch": chosen_epoch, "learning_curve": curve,
                            "metrics": summarize(chosen_pred, cache),
                            "selection_at_cap": chosen_epoch == plan["max_epochs"]}
        candidates.append(candidate_record)
        for seed, fold, held, record in fold_records:
            pred = chosen_pred[plan["seeds"].index(seed), folds == fold]
            record["selected_epoch"] = chosen_epoch
            record["metrics"] = evaluate(pred, held["fut_xy"], held["scene_name"],
                                         held["has_static"], held["has_dynamic"], seed=seed)
            path = run_dir / f"{candidate['name']}-s{seed}-f{fold}"
            write_run(path, record)
            write_predictions(path / "preds_cv.csv", held, pred)
        if winner is None or candidate_record["metrics"]["ade_scene_mean"] < winner["metrics"]["ade_scene_mean"]:
            winner, winner_pred = candidate_record, chosen_pred
    assert winner is not None and winner_pred is not None
    selection = {**prov, "protocol": "P1", "cv_run_id": run_dir.name, "plan": plan,
                 "train_cache_sha256": cache_hash, "selection_metric": "ade_scene",
                 "selected_candidate": winner["name"], "config": winner["config"],
                 "epochs": winner["selected_epoch"], "seeds": plan["seeds"],
                 "selection_at_cap": winner["selection_at_cap"]}
    # No final val is permitted when the chosen epoch hits the new cap.
    write_json(run_dir / "selection.json", selection)
    np.save(run_dir / "preds_oof.npy", winner_pred)
    record = {**prov, "run_id": run_dir.name, "config": plan,
              "cache_sha256": {"train": cache_hash}, "train_seconds": time.monotonic() - start,
              "candidates": candidates, "selection": selection, "metrics": winner["metrics"],
              "official_val_read": False, "test_read": False}
    write_run(run_dir, record)
    return record


def validate_selection(selection: dict, prov: dict, train_hash: str) -> None:
    validate_plan(selection["plan"])
    if selection["protocol"] != "P1" or selection["selection_metric"] != "ade_scene":
        raise ValueError("invalid selection protocol")
    if selection["git_dirty"] or selection["git_sha"] != prov["git_sha"]:
        raise ValueError("final must use the same clean source commit as CV")
    if selection["source_sha256"] != prov["source_sha256"]:
        raise ValueError("source differs from CV selection")
    if selection["environment"] != prov["environment"]:
        raise ValueError("final environment differs from CV selection")
    if selection["train_cache_sha256"] != train_hash:
        raise ValueError("train cache differs from CV selection")
    if selection["selection_at_cap"] or selection["epochs"] >= selection["plan"]["max_epochs"]:
        raise ValueError("selected epoch hits cap; extend CV before opening val")
    epochs = selection["epochs"]
    if not isinstance(epochs, int) or epochs < 0 or epochs % selection["plan"]["eval_every"]:
        raise ValueError("invalid selected epoch")
    if selection["seeds"] != selection["plan"]["seeds"]:
        raise ValueError("final seeds differ from CV")
    matches = [c for c in selection["plan"]["candidates"] if c["name"] == selection["selected_candidate"]]
    if len(matches) != 1 or {k: v for k, v in matches[0].items() if k != "name"} != selection["config"]:
        raise ValueError("selected config is not a planned candidate")


def final_evaluation(cache_dir: Path, selection_path: Path, run_dir: Path, prov: dict) -> dict:
    receipt_path = selection_path.parent / "final_evaluation.json"
    if receipt_path.exists():
        raise RuntimeError("this CV selection already reserved its final val evaluation")
    selection = json.loads(selection_path.read_text())
    assert_split_allowed("train", "train")
    train_hash = sha256(cache_dir / "train.npz")
    validate_selection(selection, prov, train_hash)
    train = load_cache(cache_dir / "train.npz")
    config = H1Config(**selection["config"])
    write_json(run_dir / "selection.json", selection)
    models = []
    training_times = []
    start = time.monotonic()
    for seed in selection["seeds"]:
        begin = time.monotonic()
        model, _, _ = fit(train, config, seed, selection["epochs"])
        training_times.append(time.monotonic() - begin)
        models.append(model)
        path = run_dir / f"s{seed}"
        path.mkdir()
        torch.save(model.state_dict(), path / "model.pt")
        print(f"final seed={seed} fixed_epoch={selection['epochs']} fitted", flush=True)
    # All models/configs are frozen BEFORE opening official val once.
    with receipt_path.open("x") as stream:
        json.dump({"final_run_id": run_dir.name, "selection_sha256": sha256(selection_path),
                   "phase": "val-evaluation-reserved"}, stream, indent=2)
    assert_split_allowed("val", "eval")
    val = load_cache(cache_dir / "val.npz")
    for key in ("scene_name", "scene_token"):
        if set(train[key]) & set(val[key]):
            raise ValueError(f"train/official-val scene overlap: {key}")
    val_x, val_base = prepare_inputs(val)
    cache_hashes = {"train": train_hash, "val": sha256(cache_dir / "val.npz")}
    predictions = np.stack([predict(model, val_x, val_base) for model in models])
    summary = summarize(predictions, val)
    run_config = {"model": "h1-mlp-ca-residual", **asdict(config), "epochs": selection["epochs"],
                  "seeds": selection["seeds"], "split": "val", "protocol": "P1",
                  "selection_run_id": selection["cv_run_id"], "selection_sha256": sha256(selection_path),
                  "epoch_selection": "train-cv-only", "early_stopping": False}
    for index, seed in enumerate(selection["seeds"]):
        path = run_dir / f"s{seed}"
        record = {**prov, "run_id": f"{run_dir.name}/s{seed}",
                  "config": {**run_config, "seed": seed}, "cache_sha256": cache_hashes,
                  "epochs_completed": selection["epochs"], "train_seconds": training_times[index],
                  "ca_val_ade": summary["ca_ade"],
                  "beats_ca_val": summary["by_seed"][index]["ade"] < summary["ca_ade"],
                  "metrics": summary["by_seed"][index], "official_val_evaluations": 1}
        write_run(path, record)
        np.save(path / "preds_val.npy", predictions[index])
        write_predictions(path / "preds_val.csv", val, predictions[index])
    record = {**prov, "run_id": run_dir.name, "config": run_config,
              "cache_sha256": cache_hashes, "train_seconds": sum(training_times),
              "elapsed_seconds": time.monotonic() - start, "metrics": summary,
              "official_val_loads": 1, "official_val_evaluations_per_seed": 1, "test_read": False}
    write_run(run_dir, record)
    write_json(receipt_path, {"final_run_id": run_dir.name,
                              "selection_sha256": sha256(selection_path), "phase": "complete"})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cv", "final"))
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/h1-p1.json")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.stage == "final" and args.selection is None:
        parser.error("final requires --selection from train CV")
    prov = provenance()  # capture before creating outputs, never fake git_dirty
    run_dir = reserve(args.output_dir, args.run_id)
    if args.stage == "cv":
        plan = json.loads(args.plan.read_text())
        record = cross_validate(args.cache_dir, plan, run_dir, prov)
    else:
        record = final_evaluation(args.cache_dir, args.selection, run_dir, prov)
    print(json.dumps({"run_id": record["run_id"], "metrics": record["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
