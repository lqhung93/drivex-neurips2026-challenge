"""Paired H1 / language model runs: ΔADE of a frozen instruction embedding, plus the shuffle test.

H1 and the language model share the history network, input, optimiser, seeds and batch order;
the language model only adds the instruction embedding, by concatenation (L1,
src/drivex26/l1.py) or by FiLM on the last hidden layer (L2, src/drivex26/l2.py). Metrics
keys call the language model "l1" whatever the fusion; config.fusion says which. With
--pretrain-epochs, both start from one H1 network pretrained on the sliding windows (P2),
without the scenes of the held out fold. Two stages, following protocol P1 (5 fold cross validation by scene on train):

  # 1. 5-fold CV by scene on train: picks the epoch of each model, reports CV ΔADE and the
  #    shuffle test on out-of-fold rows. Never opens val.
  python scripts/run_l1.py cv --encoder minilm --seeds 0 1 2
  python scripts/run_l1.py cv --fusion film --pretrain-epochs 100 --text-dropout 0.3 \
      --baseline-run <cv run without pretraining>   # also ΔADE of this H1 against that H1
  # 2. Once, for the final config: retrain on all of train at the CV epochs, evaluate val.
  python scripts/run_l1.py val --from-cv <cv_run_id>
  # 3. Submission files (test read for that purpose only), from the same CV run.
  python scripts/run_l1.py test --from-cv <cv_run_id> --team DHN --version 0

Epochs are picked like H1 under P1: equal-scene out-of-fold ADE averaged over seeds,
same 3,000 epoch cap, so the paired H1 matches scripts/run_h1_cv.py. Also reported: ΔADE per instruction
category and per scene under the submission policies first / mean. Seed-wise
predictions are kept as preds_<split>.npz (gitignored) for error analysis.

Needs data/features/text/<encoder>_{train,val}.npy from scripts/language/encode_text.py.
ΔADE = ADE_H1 - ADE_L1 (positive: language helps). Shuffle cost = ADE_L1(instruction of another
scene) - ADE_L1 (positive: L1 uses the instruction). Only the test stage reads test, to write
the submission files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from drivex26.data import assert_split_allowed, load_cache
from drivex26.h1 import HistoryMLP, predict, prepare
from drivex26.h1_training import select_epoch
from drivex26.l1 import HISTORY_DIM, InstructionMLP, with_text
from drivex26.l2 import FiLMMLP
from drivex26.metrics import ade, evaluate, instruction_category, paired_delta, scene_folds
from drivex26.submission import first_per_scene, mean_per_scene
from drivex26.text import ENCODERS, load_features, shuffle_across_scenes

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/run_l1.py", "src/drivex26/l1.py", "src/drivex26/l2.py", "src/drivex26/h1.py", "src/drivex26/text.py",
           "src/drivex26/baselines.py", "src/drivex26/data.py", "src/drivex26/metrics.py",
           "src/drivex26/h1_training.py", "src/drivex26/submission.py")
VARIANTS = ("h1", "l1", "l1_shuffled")
POLICIES = {"first": first_per_scene, "mean": mean_per_scene}
FUSIONS = {"concat": InstructionMLP, "film": FiLMMLP}
MODEL_NAMES = {"concat": "l1-concat-frozen-text", "film": "l2-film-frozen-text"}
PRETRAIN_RULE = "windows of every official train scene except the held out fold (val: all windows)"


def git_sha() -> str:
    override = os.environ.get("DRIVEX_GIT_SHA")
    if override:
        return override
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_dirty() -> bool | None:
    """True or False in a git checkout; None (unknown) outside one, like git_sha() returns "unknown"."""
    if "DRIVEX_GIT_DIRTY" in os.environ:
        return os.environ["DRIVEX_GIT_DIRTY"].lower() == "true"
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            text=True, capture_output=True, check=False)
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_eval(model, x, base, y, evals: dict, cfg: dict, seed: int, checkpoints: list[int],
               text_dropout: float = 0.0) -> dict:
    """Train like scripts/run_h1.py (fixed budget, no early stopping); return {name: (C, N) preds}.

    `evals` maps a name to (features, base) evaluated at every epoch in `checkpoints` (0: before
    training). `text_dropout` zeroes the text columns of that share of training rows, drawn from
    its own generator so the batch order stays the one of H1.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    drop_gen = torch.Generator().manual_seed(seed + 10_000)
    out = {name: [] for name in evals}
    if 0 in checkpoints:
        for name, (ex, eb) in evals.items():
            out[name].append(predict(model, ex, eb))
    for epoch in range(1, checkpoints[-1] + 1):
        model.train()
        for indices in torch.randperm(len(x), generator=generator).split(cfg["batch_size"]):
            xb = x[indices]
            if text_dropout > 0:
                xb = xb.clone()
                xb[torch.rand(len(indices), generator=drop_gen) < text_dropout, HISTORY_DIM:] = 0
            loss = torch.linalg.vector_norm(model(xb, base[indices]) - y[indices], dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in checkpoints:
            for name, (ex, eb) in evals.items():
                out[name].append(predict(model, ex, eb))
    return {name: np.stack(preds) for name, preds in out.items()}


def pretrain(windows: dict, cfg: dict, seed: int, epochs: int) -> dict:
    """H1 trained on sliding windows without instructions (P2); returns its state_dict."""
    x, base, y = prepare(windows)
    seed_everything(seed)
    model = HistoryMLP(cfg["hidden"])
    train_eval(model, x, base, y, {}, cfg, seed, [epochs])
    return {k: v.clone() for k, v in model.state_dict().items()}


def fit_pair(train: dict, text_train: np.ndarray, test: dict, text_test: np.ndarray,
             cfg: dict, seed: int, checkpoints: list[int], shuffle_seed: int,
             fusion: str = "concat", history_state: dict | None = None, text_dropout: float = 0.0) -> dict:
    """H1 and the language model trained from the same start on `train`; predictions on `test`."""
    x, base, y = prepare(train)
    tx, tbase, _ = prepare(test)
    perm = shuffle_across_scenes(test["scene_name"], seed=shuffle_seed)
    seed_everything(seed)
    h1 = HistoryMLP(cfg["hidden"])
    seed_everything(seed)
    lm = FUSIONS[fusion](text_train.shape[1], cfg["hidden"])
    if history_state is not None:
        h1.load_state_dict(history_state)
        lm.load_history(history_state)
    preds = train_eval(h1, x, base, y, {"h1": (tx, tbase)}, cfg, seed, checkpoints)
    preds |= train_eval(lm, with_text(x, text_train), base, y, {
        "l1": (with_text(tx, text_test), tbase),
        "l1_shuffled": (with_text(tx, text_test[perm]), tbase),
    }, cfg, seed, checkpoints, text_dropout=text_dropout)
    return preds


def windows_without(windows: dict, scenes) -> dict:
    """Pretraining rows of every train scene except `scenes` (the held out fold)."""
    return subset(windows, ~np.isin(windows["scene_name"], np.asarray(sorted(set(scenes)))))


def vs_baseline(baseline_run: str | None, output_dir: Path, split: str, cache: dict, ades: dict,
                seeds: list, seed: int) -> dict | None:
    """ΔADE of a baseline run's H1 against this run's H1 and language model, on the same rows."""
    if not baseline_run:
        return None
    other = np.load(output_dir / baseline_run / f"preds_{split}.npz")["h1"]
    base_cfg = json.loads((output_dir / baseline_run / "metrics.json").read_text())["config"]
    if base_cfg["seeds"] != seeds or other.shape[1] != len(cache["fut_xy"]):
        raise ValueError(f"{baseline_run} does not cover the same seeds and rows")
    base_ade = ade(other, cache["fut_xy"][None]).mean(axis=0)
    return {"run_id": baseline_run, "ade_baseline_h1": float(base_ade.mean()),
            **{v: paired_delta(base_ade, ades[v], cache["scene_name"], seed=seed) for v in ("h1", "l1")}}


def subset(cache: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in cache.items()}


def scene_ades(cache: dict, preds: np.ndarray, policy: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-scene ADE, mean over seeds, of the one row per scene a submission would keep."""
    tokens, _, gt = POLICIES["first"](cache["scene_token"], cache["instruction"], cache["fut_xy"])
    _, _, gt_mean = mean_per_scene(cache["scene_token"], cache["instruction"], cache["fut_xy"])
    if not np.allclose(gt, gt_mean, atol=1e-4):
        raise ValueError("rows of a scene must share the same future")
    per_seed = [ade(POLICIES[policy](cache["scene_token"], cache["instruction"], p)[2], gt) for p in preds]
    return tokens, np.mean(per_seed, axis=0)


def report(ades: dict, cache: dict, preds: dict, seed: int) -> dict:
    """ΔADE and shuffle test on seed-averaged per-row ADE, per category and per submission policy."""
    groups = cache["scene_name"]
    category = instruction_category(cache["has_static"], cache["has_dynamic"])
    policies = {}
    for policy in POLICIES:
        scenes = {v: scene_ades(cache, preds[v], policy) for v in ("h1", "l1")}
        tokens = scenes["h1"][0]
        policies[policy] = {
            "ade": {v: float(a.mean()) for v, (_, a) in scenes.items()},
            "delta_ade": paired_delta(scenes["h1"][1], scenes["l1"][1], tokens, seed=seed),
        }
    return {
        "ade": {v: float(ades[v].mean()) for v in VARIANTS},
        "ade_scene": {v: float(np.mean([ades[v][groups == g].mean() for g in np.unique(groups)]))
                      for v in VARIANTS},
        "delta_ade": paired_delta(ades["h1"], ades["l1"], groups, seed=seed),
        "shuffle_cost": paired_delta(ades["l1_shuffled"], ades["l1"], groups, seed=seed),
        "per_seed": {v: [float(ade(p, cache["fut_xy"]).mean()) for p in preds[v]] for v in VARIANTS},
        "delta_ade_per_category": {
            c: {"ade_h1": float(ades["h1"][category == c].mean()), "ade_l1": float(ades["l1"][category == c].mean()),
                **paired_delta(ades["h1"][category == c], ades["l1"][category == c], groups[category == c], seed=seed)}
            for c in ("s", "d", "sd", "none") if (category == c).any()},
        "scene_policy": policies,
    }


def save_preds(run_dir: Path, split: str, preds: dict) -> None:
    """Seed-wise (S, N, 12, 2) predictions per variant, cache row order; gitignored."""
    np.savez_compressed(run_dir / f"preds_{split}.npz", **{v: p.astype(np.float32) for v, p in preds.items()})


def write_run(run_dir: Path, record: dict, config: dict, seeds: list[int]) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "git_sha.txt").write_text(record["git_sha"] + "\n")
    (run_dir / "seed.txt").write_text(" ".join(map(str, seeds)) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")


def provenance(args, cache_paths: list[Path], feature_paths: list[Path]) -> dict:
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "cache_sha256": {p.stem: sha256(p) for p in cache_paths},
        "feature_sha256": {p.name: sha256(p) for p in feature_paths},
        "device": "cpu",
        "gpu_hours": 0.0,
    }


def run_cv(args) -> None:
    assert_split_allowed("train", "train")
    train = load_cache(args.cache_dir / "train.npz")
    text = load_features(args.feature_dir, args.encoder, "train", train["instruction"])
    folds = scene_folds(train["scene_name"], k=args.folds)
    if len(np.unique(folds)) != args.folds:
        raise ValueError(f"some of the {args.folds} scene folds are empty")
    checkpoints = list(range(args.eval_every, args.epochs + 1, args.eval_every))
    if args.pretrain_epochs:
        checkpoints = [0] + checkpoints  # epoch 0: pretrained only, no fine tuning
        windows = load_cache(args.windows)
    cfg = {k: getattr(args, k) for k in ("hidden", "lr", "weight_decay", "batch_size")}

    start = time.monotonic()
    # oof[variant][seed] -> (C, N, 12, 2), filled fold by fold
    oof = {v: np.zeros((len(args.seeds), len(checkpoints), *train["fut_xy"].shape), np.float32) for v in VARIANTS}
    for si, seed in enumerate(args.seeds):
        for fold in range(args.folds):
            held = folds == fold
            state = (pretrain(windows_without(windows, train["scene_name"][held]), cfg, seed, args.pretrain_epochs)
                     if args.pretrain_epochs else None)
            preds = fit_pair(subset(train, ~held), text[~held], subset(train, held), text[held],
                             cfg, seed, checkpoints, shuffle_seed=seed * 100 + fold, fusion=args.fusion,
                             history_state=state, text_dropout=args.text_dropout)
            for v in VARIANTS:
                oof[v][si][:, held] = preds[v]
    elapsed = time.monotonic() - start

    gt = train["fut_xy"]
    curve, epochs = {}, {}
    for v in ("h1", "l1"):  # (S, C, N) errors; equal-scene OOF ADE over seeds, earliest epoch on ties
        epochs[v], curve[v] = select_epoch(ade(oof[v], gt[None, None]), checkpoints, train["scene_name"])
    best = {v: checkpoints.index(e) for v, e in epochs.items()}
    # H1 uses its own CV epoch; L1 and its shuffle test share L1's epoch
    picked = {"h1": oof["h1"][:, best["h1"]]} | {v: oof[v][:, best["l1"]] for v in VARIANTS[1:]}
    ades = {v: ade(p, gt[None]).mean(axis=0) for v, p in picked.items()}

    name = ("l1" if args.fusion == "concat" else "l2") + ("-p2" if args.pretrain_epochs else "")
    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M}-{name}-{args.encoder}-cv"
    config = {"stage": "cv", "model": MODEL_NAMES[args.fusion], "fusion": args.fusion,
              "baseline": "h1-mlp-ca-residual", "encoder": args.encoder, "encoder_hf_id": ENCODERS[args.encoder],
              "split": "train", "folds": args.folds, "fold_rule": "scene_folds (md5 of scene name)",
              "seeds": args.seeds, "epochs_max": args.epochs, "eval_every": args.eval_every,
              "pretrain_epochs": args.pretrain_epochs, "pretrain_rule": PRETRAIN_RULE if args.pretrain_epochs else None,
              "text_dropout": args.text_dropout, **cfg}
    caches = [args.cache_dir / "train.npz"] + ([args.windows] if args.pretrain_epochs else [])
    record = {
        "run_id": run_id, **provenance(args, caches, [args.feature_dir / f"{args.encoder}_train.npy"]),
        "config": config,
        "train_seconds": elapsed,
        "n_pairs": int(len(gt)), "n_scenes": int(len(np.unique(train["scene_name"]))),
        "selection": "equal-scene out-of-fold ADE, mean over seeds (h1_training.select_epoch)",
        "selected_epoch": epochs,
        "epoch_at_cap": {v: e == checkpoints[-1] for v, e in epochs.items()},
        "cv_curve": {"epochs": checkpoints,
                     **{v: {k: [row[k] for row in c] for k in ("ade", "ade_scene")} for v, c in curve.items()}},
        "ca_ade": float(ade(prepare(train)[1].numpy(), gt).mean()),
        "metrics": report(ades, train, picked, args.seeds[0]),
        "vs_baseline": vs_baseline(args.baseline_run, args.output_dir, "oof", train, ades, args.seeds, args.seeds[0]),
    }
    write_run(args.output_dir / run_id, record, config, args.seeds)
    save_preds(args.output_dir / run_id, "oof", picked)
    print(json.dumps({k: record[k] for k in ("run_id", "selected_epoch", "epoch_at_cap")}, indent=2))
    print(json.dumps({k: record["metrics"][k] for k in ("ade", "delta_ade", "shuffle_cost")},
                     indent=2))


def load_cv(args) -> tuple:
    """Config, epochs and pretraining windows that a finished CV run picked."""
    cv = json.loads((args.output_dir / args.from_cv / "metrics.json").read_text())
    if cv["config"].get("stage") != "cv":
        raise ValueError(f"{args.from_cv} is not a CV run")
    if any(cv["epoch_at_cap"].values()) or "selection" not in cv:
        raise ValueError(f"{args.from_cv}: epoch at the CV cap or old selection rule; rerun CV first")
    cfg = {k: cv["config"][k] for k in ("hidden", "lr", "weight_decay", "batch_size")}
    pretrain_epochs = cv["config"].get("pretrain_epochs", 0)
    return (cv, cfg, cv["config"]["encoder"], cv["config"]["seeds"], cv["selected_epoch"],
            cv["config"].get("fusion", "concat"), pretrain_epochs, cv["config"].get("text_dropout", 0.0),
            load_cache(args.windows) if pretrain_epochs else None)


def run_val(args) -> None:
    cv, cfg, encoder, seeds, epochs, fusion, pretrain_epochs, text_dropout, windows = load_cv(args)

    assert_split_allowed("train", "train")
    assert_split_allowed("val", "eval")
    train = load_cache(args.cache_dir / "train.npz")
    val = load_cache(args.cache_dir / "val.npz")
    text_train = load_features(args.feature_dir, encoder, "train", train["instruction"])
    text_val = load_features(args.feature_dir, encoder, "val", val["instruction"])

    start = time.monotonic()
    preds = {v: [] for v in VARIANTS}
    for seed in seeds:
        checkpoints = sorted(set(epochs.values()))
        state = pretrain(windows, cfg, seed, pretrain_epochs) if pretrain_epochs else None
        out = fit_pair(train, text_train, val, text_val, cfg, seed, checkpoints, shuffle_seed=seed,
                       fusion=fusion, history_state=state, text_dropout=text_dropout)
        for v in VARIANTS:
            epoch = epochs["h1" if v == "h1" else "l1"]
            preds[v].append(out[v][checkpoints.index(epoch)])
    elapsed = time.monotonic() - start
    preds = {v: np.stack(p) for v, p in preds.items()}
    ades = {v: ade(p, val["fut_xy"][None]).mean(axis=0) for v, p in preds.items()}

    name = ("l1" if fusion == "concat" else "l2") + ("-p2" if pretrain_epochs else "")
    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M}-{name}-{encoder}-val"
    config = {"stage": "val", "from_cv": args.from_cv, "model": cv["config"]["model"], "fusion": fusion,
              "baseline": cv["config"]["baseline"], "encoder": encoder,
              "encoder_hf_id": ENCODERS[encoder], "split": "val", "seeds": seeds, "epochs": epochs,
              "pretrain_epochs": pretrain_epochs, "text_dropout": text_dropout, **cfg}
    caches = [args.cache_dir / f"{s}.npz" for s in ("train", "val")] + ([args.windows] if pretrain_epochs else [])
    record = {
        "run_id": run_id, **provenance(args, caches,
                                       [args.feature_dir / f"{encoder}_{s}.npy" for s in ("train", "val")]),
        "config": config,
        "train_seconds": elapsed,
        "ca_val_ade": float(ade(prepare(val)[1].numpy(), val["fut_xy"]).mean()),
        "metrics": report(ades, val, preds, seeds[0]),
        "vs_baseline": vs_baseline(args.baseline_run, args.output_dir, "val", val, ades, seeds, seeds[0]),
        "evaluate": {v: evaluate(preds[v].mean(axis=0), val["fut_xy"], val["scene_name"],
                                 val["has_static"], val["has_dynamic"], seed=seeds[0]) for v in ("h1", "l1")},
    }
    run_dir = args.output_dir / run_id
    write_run(run_dir, record, config, seeds)
    save_preds(run_dir, "val", preds)
    header = ["scene_name", "scene_token"] + [f"{axis}{step}" for step in range(1, 13) for axis in ("x", "y")]
    with (run_dir / "preds_val.csv").open("w", newline="") as stream:  # L1, mean over seeds
        writer = csv.writer(stream)
        writer.writerow(header)
        for scene, token, path in zip(val["scene_name"], val["scene_token"], preds["l1"].mean(axis=0)):
            writer.writerow([scene, token, *path.reshape(-1).tolist()])
    print(json.dumps({k: record["metrics"][k] for k in ("ade", "delta_ade", "shuffle_cost")},
                     indent=2))


def run_test(args) -> None:
    """Submission files: retrain the CV picked pair on all of train, predict test (the only stage that reads test).

    Writes <out-dir>/<language|history_only>/<TEAM>_CSV_version<N>.csv, one row per test scene from
    its first instruction, trajectory averaged over the seeds, each checked by the validator,
    and a run record with the sha256 of both files. Nothing is uploaded.
    """
    from drivex26.h1 import prepare_inputs
    from drivex26.submission import form_csv_name, validate_submission, write_submission

    cv, cfg, encoder, seeds, epochs, fusion, pretrain_epochs, text_dropout, windows = load_cv(args)
    assert_split_allowed("train", "train")
    assert_split_allowed("test", "submission")
    train = load_cache(args.cache_dir / "train.npz")
    test = load_cache(args.cache_dir / "test.npz")
    if "fut_xy" in test:
        raise ValueError("the test cache must not hold future poses")
    text_train = load_features(args.feature_dir, encoder, "train", train["instruction"])
    text_test = load_features(args.feature_dir, encoder, "test", test["instruction"])
    x, base, y = prepare(train)
    tx, tbase = prepare_inputs(test)

    start = time.monotonic()
    preds = {"h1": [], "l1": []}
    for seed in seeds:
        state = pretrain(windows, cfg, seed, pretrain_epochs) if pretrain_epochs else None
        seed_everything(seed)
        h1 = HistoryMLP(cfg["hidden"])
        seed_everything(seed)
        lm = FUSIONS[fusion](text_train.shape[1], cfg["hidden"])
        if state is not None:
            h1.load_state_dict(state)
            lm.load_history(state)
        preds["h1"].append(train_eval(h1, x, base, y, {"t": (tx, tbase)}, cfg, seed, [epochs["h1"]])["t"][0])
        preds["l1"].append(train_eval(lm, with_text(x, text_train), base, y, {"t": (with_text(tx, text_test), tbase)},
                                      cfg, seed, [epochs["l1"]], text_dropout=text_dropout)["t"][0])
    elapsed = time.monotonic() - start

    name = form_csv_name(args.team, args.version)
    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M}-submission-v{args.version}"
    out_dir = Path(args.out_dir or ROOT / "data" / "submissions" / run_id)
    expected = set(np.unique(test["scene_token"]))
    files = {}
    for variant, folder in (("l1", "language"), ("h1", "history_only")):
        tokens, instructions, traj = first_per_scene(test["scene_token"], test["instruction"],
                                                     np.mean(preds[variant], axis=0))
        path = out_dir / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_submission(path, tokens, instructions, traj)
        errors, warnings = validate_submission(path, expected_tokens=expected, team=args.team)
        if errors:
            raise ValueError(f"{path}: {errors}")
        files[folder] = {"file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                         "sha256": sha256(path), "rows": len(tokens), "warnings": warnings}
        print(f"{folder}: {len(tokens)} rows -> {path}" + (f" (warnings: {warnings})" if warnings else ""))
    config = {"stage": "test", "purpose": "submission", "from_cv": args.from_cv, "model": cv["config"]["model"],
              "fusion": fusion, "encoder": encoder, "seeds": seeds, "epochs": epochs, "pretrain_epochs": pretrain_epochs,
              "text_dropout": text_dropout, "policy": "first instruction per scene, mean trajectory over seeds",
              "team": args.team, "version": args.version, **cfg}
    caches = [args.cache_dir / f"{s}.npz" for s in ("train", "test")] + ([args.windows] if pretrain_epochs else [])
    record = {"run_id": run_id, **provenance(args, caches, [args.feature_dir / f"{encoder}_{s}.npy" for s in ("train", "test")]),
              "config": config, "train_seconds": elapsed, "files": files, "uploaded": False}
    write_run(args.output_dir / run_id, record, config, seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "data" / "features" / "text")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--windows", type=Path, default=ROOT / "data" / "cache" / "windows_train.npz")
    parser.add_argument("--baseline-run", default=None, help="run of the same stage whose H1 this one is compared to")
    sub = parser.add_subparsers(dest="stage", required=True)
    cv = sub.add_parser("cv", help="5-fold CV by scene on train")
    cv.add_argument("--encoder", choices=list(ENCODERS), default="minilm")
    cv.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--epochs", type=int, default=3000)  # H1 P1 cap (configs/h1-p1.json)
    cv.add_argument("--eval-every", type=int, default=10)
    cv.add_argument("--batch-size", type=int, default=128)
    cv.add_argument("--hidden", type=int, default=128)
    cv.add_argument("--lr", type=float, default=0.001)
    cv.add_argument("--weight-decay", type=float, default=0.0001)
    cv.add_argument("--fusion", choices=list(FUSIONS), default="concat")
    cv.add_argument("--pretrain-epochs", type=int, default=0, help="P2 pretraining on sliding windows; 0: none")
    cv.add_argument("--text-dropout", type=float, default=0.0)
    val = sub.add_parser("val", help="one-time val evaluation at the epochs a CV run picked")
    val.add_argument("--from-cv", required=True)
    test = sub.add_parser("test", help="submission files from the pair a CV run picked (test read for submission only)")
    test.add_argument("--from-cv", required=True)
    test.add_argument("--team", required=True, help="team name exactly as in the upload form")
    test.add_argument("--version", type=int, required=True, help="0 for the first upload, +1 per resubmission")
    test.add_argument("--out-dir", default=None, help="defaults to data/submissions/<run_id>/")
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if args.stage == "cv":
        if args.epochs < args.eval_every or args.eval_every < 1 or args.batch_size < 1:
            parser.error("need epochs >= eval-every >= 1 and a positive batch size")
        run_cv(args)
    elif args.stage == "val":
        run_val(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
