# drivex-neurips2026-challenge

Code of team DHN for the **doScenes Instructed Driving Challenge**, Track 3 of the
[DriveX Grand Challenge @ NeurIPS 2026](https://drivex-workshop.github.io/neurips2026/)
([challenge page](https://mi3-lab.github.io/doScenes_challenge)).

Task: for each nuScenes scene, given 2 s of ego history and a natural language instruction,
predict one 6 s ego trajectory (12 points at 2 Hz) in the ego frame at the anchor (metres,
x forward, y left). Entries report the instruction conditioned ADE, the ADE of a history only
baseline with the same non language input, and their difference ΔADE = ADE_baseline − ADE_instruction.

## Setup

Python 3.10 or newer. Clone with git (not as a zip): every run records the commit it ran from.

```bash
scripts/setup_third_party.sh                 # official doScenes loader, metrics.py, annotations (pinned commit)
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -q                          # unit tests, no nuScenes data needed
uv pip install --python .venv/bin/python -e ".[dev,lang]"   # frozen text encoders, for the language models
```

## Data

Nothing from nuScenes or doScenes is included in this repository.

- **nuScenes** (CC BY-NC-SA 4.0 and the nuScenes Terms of Use, Motional): only the metadata is
  needed, from the official open data bucket (no AWS account required):
  ```bash
  mkdir -p data/nuscenes/_downloads
  aws s3 cp --no-sign-request --region ap-northeast-1 s3://motional-nuscenes/public/v1.0/v1.0-trainval_meta.tgz data/nuscenes/_downloads/
  tar xzf data/nuscenes/_downloads/v1.0-trainval_meta.tgz -C data/nuscenes
  ```
  `v1.0-test_meta.tgz` is needed only to write the submission file.
- **doScenes annotations**: fetched by `scripts/setup_third_party.sh` from
  [rossgreer/doScenes](https://github.com/rossgreer/doScenes); they belong to their authors.

## Reproduce

Two protocol names appear in the code. **P1**: every model choice (config, epoch) is made by 5 fold
cross validation grouped by scene on the training split, and the official validation split is
evaluated once per final configuration. **P2**: the motion model is pretrained on every 17 sample
window of the 700 nuScenes training scenes (no instructions), leaving out the scenes of the held
out fold.

The test split is read only to write the submission file: every code path that opens it goes
through `assert_split_allowed(split, purpose)` and requires `--purpose submission`, and the test
cache never stores future poses.

```bash
# caches: one row per (instruction, scene) pair, anchor frame
python scripts/build_cache.py --nusc-root data/nuscenes --split train
python scripts/build_cache.py --nusc-root data/nuscenes --split val
python scripts/build_windows.py --nusc-root data/nuscenes          # sliding windows for motion pretraining

# kinematic baselines on the official validation split
python scripts/run_baseline.py --model cv --split val
python scripts/run_baseline.py --model ca --split val

# H1 (history only): config and epoch chosen by 5 fold cross validation by scene on train,
# then the official validation split is evaluated once
python scripts/run_h1_cv.py cv --run-id h1-cv
python scripts/run_h1_cv.py final --selection results/runs/h1-cv/selection.json --run-id h1-final

# language models paired with their history only model (same network, rows, batches, seeds):
# ΔADE and shuffle test; L1 concat and L2 FiLM read a frozen MiniLM sentence embedding
python scripts/language/encode_text.py --encoders minilm --splits train val
python scripts/run_l1.py --run-id l1-cv cv --encoder minilm --seeds 0 1 2                                        # L1 baseline run
python scripts/run_l1.py --run-id l2-p2-cv --baseline-run l1-cv cv --fusion film --text-dropout 0.3 --pretrain-epochs 300

# submitted config: L3 + map. Qwen2.5-3B-Instruct turns each instruction into a structured intent
# (one GPU, fp16), a 26 dim one-hot vector enters H1 through FiLM; the 17 lane map features enter
# both models, so the history only model is H2 = history + map. Pretraining 1000 epochs (P2).
# Map features need the nuScenes map expansion v1.3 under data/nuscenes/maps/expansion; they are
# written next to the text features, where run_l1.py looks for every input by default.
python scripts/build_map_features.py --nusc-root data/nuscenes --splits train val --feature-dir data/features/text
python scripts/language/extract_intent.py --splits train val --out-dir results/runs/intent-3b
python scripts/language/encode_intent.py results/runs/intent-3b --splits train val
python scripts/run_l1.py --run-id l3map-cv --baseline-run l1-cv cv --encoder intent --fusion film --extra-encoder map \
    --text-dropout 0.3 --pretrain-epochs 1000 --epochs 600
python scripts/run_l1.py --run-id l3map-val val --from-cv l3map-cv                                               # once

# submission file, named as the upload form requires, validated on write
python scripts/build_cache.py --nusc-root data/nuscenes --split test --purpose submission
python scripts/build_map_features.py --nusc-root data/nuscenes --splits test --purpose submission --feature-dir data/features/text
python scripts/language/extract_intent.py --splits test --purpose submission --out-dir results/runs/intent-3b
python scripts/language/encode_intent.py results/runs/intent-3b --splits test --purpose submission
python scripts/run_l1.py test --from-cv l3map-cv --team DHN --version 0   # DHN_CSV_version0.csv + DHN_CSV_version0_historyonly.csv
python scripts/validate_submission.py data/submissions/<run_id>/DHN_CSV_version0.csv --team DHN --test-cache data/cache/test.npz
# kinematic baselines only: python scripts/make_submission.py --model ca --team DHN --version 0
```

Every run writes `results/runs/<run_id>/metrics.json` with the git commit, config, seeds and
the sha256 of its inputs.

## Results

Submission version 0 is the first row: both files are uploaded in one Language + History entry,
`DHN_CSV_version0.csv` from the language model and `DHN_CSV_version0_historyonly.csv` (empty
instruction column) from its history only model with the same network and non language input.
Test scores are computed by the organisers.

<!-- results:start -->
Official nuScenes validation split (339 instruction, scene pairs, 112 scenes), evaluated once per
configuration chosen by 5 fold cross validation on train; mean error over seeds. ΔADE = ADE of the
history only model (same network and non language input) minus ADE with the instruction; positive
means the instruction helps. 95% interval: bootstrap over scenes.

| Model | ADE (instruction) | ADE (history only) | ΔADE [95% CI] | ΔADE, one trajectory per scene | Seeds |
|---|---|---|---|---|---|
| l2-film-frozen-text, intent + map (both models), P2 pretraining 1000 epochs | 2.853 | 2.826 | -0.027 [-0.12, +0.03] | -0.053 | 3 |
| l2-film-frozen-text, minilm, P2 pretraining 300 epochs | 2.725 | 2.699 | -0.026 [-0.15, +0.10] | -0.045 | 3 |
| l1-concat-frozen-text, minilm | 3.810 | 3.069 | -0.740 [-1.16, -0.47] | -0.794 | 3 |
<!-- results:end -->

## License

The code in this repository is released under the MIT License (`LICENSE`). It does not cover
any data: nuScenes is licensed CC BY-NC-SA 4.0 under the nuScenes Terms of Use (Motional), and
the doScenes code and annotations belong to their authors; this repository contains neither, and
`scripts/setup_third_party.sh` fetches doScenes from its original repository.

## Citation

```bibtex
@inproceedings{roy2025doscenes,
  title={doScenes: An Autonomous Driving Dataset with Natural Language Instruction for Human Interaction and Vision-Language Navigation},
  author={Parthib Roy and Srinivasa Perisetla and Shashank Shriram and Harsha Krishnaswamy and Aryan Keskar and Ross Greer},
  booktitle={2025 IEEE 28th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={1651--1658},
  year={2025}
}
@inproceedings{caesar2020nuscenes,
  title={nuScenes: A Multimodal Dataset for Autonomous Driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and Giancarlo Baldan and Oscar Beijbom},
  booktitle={CVPR},
  year={2020}
}
```
