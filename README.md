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

# language models paired with their H1 (same network, rows, batches, seeds): ΔADE and shuffle test.
# Submitted config: L2 (FiLM on the last hidden layer, 30% blank instructions) on an H1 pretrained
# for 300 epochs on the sliding windows (P2), epochs picked by 5 fold cross validation by scene
python scripts/language/encode_text.py --encoders minilm --splits train val
python scripts/run_l1.py --run-id l1-cv cv --encoder minilm --seeds 0 1 2                                        # L1 baseline run
python scripts/run_l1.py --run-id l2-p2-cv --baseline-run l1-cv cv --fusion film --text-dropout 0.3 --pretrain-epochs 300
python scripts/run_l1.py val --from-cv l2-p2-cv                                                                  # once

# submission file, named as the upload form requires, validated on write
python scripts/build_cache.py --nusc-root data/nuscenes --split test --purpose submission
python scripts/language/encode_text.py --encoders minilm --splits test --purpose submission
python scripts/run_l1.py test --from-cv l2-p2-cv --team DHN --version 0   # language/ and history_only/ CSVs
python scripts/validate_submission.py data/submissions/<run_id>/language/DHN_CSV_version0.csv --team DHN --test-cache data/cache/test.npz
# kinematic baselines only: python scripts/make_submission.py --model ca --team DHN --version 0
```

Every run writes `results/runs/<run_id>/metrics.json` with the git commit, config, seeds and
the sha256 of its inputs.

## Results

Submission version 0 is the first row: the language model in the Language + History track and its
history only model (same network and non language input) in the Ablation track. Test scores are
computed by the organisers.

<!-- results:start -->
Official nuScenes validation split (339 instruction, scene pairs, 112 scenes), evaluated once per
configuration chosen by 5 fold cross validation on train; mean error over seeds. ΔADE = ADE of the
history only model (same network and non language input) minus ADE with the instruction; positive
means the instruction helps. 95% interval: bootstrap over scenes.

| Model | ADE (instruction) | ADE (history only) | ΔADE [95% CI] | ΔADE, one trajectory per scene | Seeds |
|---|---|---|---|---|---|
| l2-film-frozen-text, P2 pretraining 300 epochs | 2.725 | 2.699 | -0.026 [-0.15, +0.10] | -0.045 | 3 |
| l1-concat-frozen-text | 3.810 | 3.069 | -0.740 [-1.16, -0.47] | -0.794 | 3 |
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
