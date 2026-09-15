# Architecture

## 1. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11–3.12 | Wheels are available for everything in the stack (PyTorch, ONNX Runtime, OpenCV, albumentations). Kaggle is the reference environment for every run that produces numbers: it uses Kaggle's preinstalled torch and timm via `PYTHONPATH=src` and never pip-installs the project, which could replace Kaggle's CUDA-matched torch. Each run record's `environment` block is the source of truth for versions; the local pins exist for tests. `pyproject.toml` still enforces `>=3.11,<3.12` until that range is widened (`PROGRESS.md`) |
| Environment | uv + `pyproject.toml` + `uv.lock` | One tool for the interpreter, virtualenv and a lockfile. It is fast enough to recreate environments on ephemeral Kaggle/Colab machines |
| Deep learning | PyTorch + torchvision | The standard research framework, with first-class MPS support on Apple silicon for local smoke tests and CUDA on Kaggle/Colab |
| Backbones | timm | Pretrained ImageNet backbones behind one API, so swapping the backbone is a config change instead of a code change |
| Augmentation | albumentations | Fast, composable image augmentations, including the photometric, blur and compression transforms that matter for spoof cues |
| Image I/O | Pillow, torchvision | Training decodes images with Pillow and resizes and normalizes them with torchvision transforms. The resize interpolation and the normalization mean/std come from the backbone's timm pretrained config. Face cropping is not implemented. Pillow decodes uploads in the API. `opencv-python-headless` is a declared dependency, but no code uses it yet |
| Tabular data | pandas, numpy | Building and validating the manifest, and aggregating metrics by attribute |
| Metrics helpers | scikit-learn | ROC/DET curve primitives and calibration utilities. The PAD metrics themselves are implemented and tested in `antispoof.eval` |
| Plots | matplotlib | Static figures for evaluation reports that can be committed to the repo |
| Config | PyYAML | Every hyperparameter lives in versioned YAML files (see `RULES.md`) |
| Progress | tqdm | Progress bars for long manifest builds and evaluation loops |
| Experiment tracking | Weights & Biases | Works the same from a laptop, Kaggle and Colab, and keeps metrics and artifacts in one place. `EXPERIMENTS.md` is the human-readable ledger |
| Export / runtime | ONNX + ONNX Runtime | A framework-independent model format with a fast CPU runtime and built-in quantization tooling, and no PyTorch dependency at serve time |
| Serving | FastAPI + uvicorn | Typed request/response models (pydantic), automatic OpenAPI docs and async file uploads |
| Quality | ruff, mypy, pytest, pytest-cov | Lint, format, type check and tests, all configured in `pyproject.toml` |

## 2. Repository layout

```
.
├── .claude/settings.json    # Repo-level assistant settings (commit/PR attribution disabled)
├── .githooks/commit-msg     # Strips disallowed trailers from commit messages
├── configs/                 # YAML configs (data.yaml, experiments) and splits/split_assignment.csv
├── data/                    # Derived manifests, regenerated on Kaggle. Gitignored, never committed
├── docs/                    # PRD, architecture, design, schemas, rules, progress, experiment ledger
├── notebooks/               # EDA only. No logic that training or serving depends on
├── reports/                 # Generated figures, metric tables, evaluation outputs
├── scripts/                 # Thin CLI entry points that call into src/antispoof
├── src/antispoof/
│   ├── data/                # Dataset class, manifest builders, subject-disjoint splits, transforms
│   ├── models/              # Backbone wrappers and classification heads
│   ├── training/            # Training loop, callbacks, checkpointing
│   ├── eval/                # ISO/IEC 30107-3 metrics, threshold selection, error analysis
│   └── serving/             # FastAPI app, ONNX Runtime inference, face/quality gating
├── tests/                   # pytest suite. Uses synthetic fixtures, never data/
├── pyproject.toml           # Dependencies and tool configuration
├── .gitignore
└── README.md
```

## 3. End-to-end data flow

```
CelebA-Spoof mirror on Kaggle (/kaggle/input/..., read in place, never downloaded; ADR-008)
        │  scripts/build_manifest.py  → antispoof.data   (conflict policy; seeded subject-level val)
        ▼
Manifests manifest_{train,val,test}.csv (data/manifests/, gitignored; SCHEMA.md §1)
  + split assignment configs/splits/split_assignment.csv (subject_id → split, committed; §2)
        │  invariant check: validate_splits at build time and at load time (ADR-009)
        ▼
Dataset + transforms (face crop → resize → augment [train only] → normalize)
        │  scripts/train.py --data-config configs/data.yaml --config configs/<exp>.yaml
        │                   --manifest-dir <dir> --output-dir <dir>   → antispoof.training
        ▼
Training on Kaggle/Colab GPU (seeded; W&B logging; run record; SCHEMA.md §3)
        │
        ▼
Checkpoint (*.pt, gitignored; stored as a W&B artifact or Kaggle output)
        │  scripts/evaluate.py        → antispoof.eval   (val: threshold selection; test: once)
        ▼
Evaluation report (reports/<run_id>/; DESIGN.md §3)
        │  scripts/export_onnx.py     → quantize → *.onnx (gitignored)
        ▼
Quantized ONNX model + FP32-vs-quantized parity check + CPU latency benchmark
        │
        ▼
FastAPI service (antispoof.serving): decode → face detect → quality gate → crop → ONNX Runtime
        │
        ▼
JSON verdict (SCHEMA.md §4) → demo page / calling KYC backend
```

`scripts/build_manifest.py` and `scripts/train.py` exist. `scripts/evaluate.py` and
`scripts/export_onnx.py` describe where those entry points will live; they do not exist yet.

## 4. Compute split: local vs Kaggle/Colab

| Task | Where | Notes |
|---|---|---|
| Manifest build, split generation, schema validation | Kaggle (CPU session) | The data stays on Kaggle (ADR-008). CPU and I/O only; no GPU needed |
| EDA notebooks | Kaggle | The data stays on Kaggle (ADR-008). Use a sampled subset if the full manifest is too large for comfortable interactive work |
| Unit tests, lint, type check | Local | Synthetic fixtures only |
| Training smoke test (a few batches, overfit check) | Local (CPU or MPS) on synthetic images, or Kaggle on a tiny subset | Checks that the pipeline runs end to end; not used for metrics |
| Full training runs | Kaggle / Colab GPU | Configs are identical; only paths and device differ, set through config |
| Full validation and test evaluation | Kaggle / Colab GPU | The test set is evaluated once, for the final model |
| ONNX export, quantization, parity check | Export and quantization: local or Kaggle. Parity check: Kaggle | Parity needs evaluation data, which stays on Kaggle (ADR-008). It is checked on the same evaluation subset for FP32 and quantized models |
| CPU latency benchmark | Local | The serving target is CPU; hardware is recorded with every number |
| API and demo development | Local | uvicorn on localhost |

Open items:

- **Getting the dataset onto Kaggle/Colab:** resolved by ADR-008. A hosted mirror on Kaggle is used
  in place. Whether the dataset license permits using that hosted copy is still open (see
  `PROGRESS.md`).
- **Checkpoint handoff** between Kaggle/Colab and local (W&B artifacts vs downloaded output).
  - Checkpoints are never committed. A run meant to be kept is saved as a Kaggle notebook version,
    so its `/kaggle/working` output, including the checkpoint, persists (`SCHEMA.md` §3).
  - Handoff to local for ONNX export: TBD — decide before Week 6 ONNX export.

## 5. Technical decisions (ADR log)

Each ADR records context, decision and consequence. Superseded decisions are marked, not deleted.

### ADR-001: Use uv for environment and dependency management
- **Context:** The project runs on a local Mac and on ephemeral cloud GPU machines. Environments must
  be quick to recreate and resolve the same way everywhere.
- **Decision:** uv, with `pyproject.toml` as the single dependency declaration and a committed
  `uv.lock`.
- **Consequence:** One command (`uv sync`) sets up the environment. Kaggle/Colab come with their own
  CUDA PyTorch build, so the cloud setup may install the project on top of the platform's torch
  instead of from the lockfile. Any version drift must be recorded in the run record. *(Superseded
  in part, 2026-09-16: on Kaggle the project is never installed. Runs use Kaggle's preinstalled torch
  and timm with `PYTHONPATH=src`, and each run record's `environment` block is the source of truth
  for versions. `uv sync` and `uv.lock` set up the local environment used for tests. See the
  Language row in §1.)*

### ADR-002: All hyperparameters live in YAML configs
- **Context:** Results that cannot be traced back to exact settings are worthless for comparison.
- **Decision:** Every tunable value (paths, model, optimizer, schedule, augmentation, seed,
  thresholds) lives in a YAML file under `configs/`. Training code reads the config and contains no
  hardcoded hyperparameters. Each run records a hash of the fully resolved config.
- **Consequence:** A little more plumbing up front. In return, every run in `EXPERIMENTS.md` points
  to one config file and one hash.

### ADR-003: Subject-disjoint train/val/test splits
- **Context:** CelebA-Spoof has multiple images per subject. If a subject appears in more than one
  split, the model can learn identity and background shortcuts, and the metrics become optimistic.
- **Decision:** Splits are assigned at the subject level. The subject sets of train, val and test
  must be pairwise disjoint. This is enforced by a unit test and by an assertion when data is loaded.
- **Consequence:** Split sizes cannot be set exactly at the image level. Whether to reuse the official
  CelebA-Spoof split or build a custom one is TBD — decide before Week 1 manifest, after checking
  whether the official split is subject-disjoint. *(Resolved by ADR-009: the official split is not
  subject-disjoint. Test is kept unchanged, the shared subjects are removed from train, and val is
  carved out of train by subject.)*

### ADR-004: ISO/IEC 30107-3 metrics are the primary evaluation
- **Context:** Accuracy and AUC hide the asymmetric costs of accepting an attack and rejecting a real
  user.
- **Decision:** Reports lead with APCER (maximum over PAI species), BPCER, and BPCER at APCER = 1%.
  ACER is reported alongside them for comparison with the literature. Thresholds are chosen on
  validation. The test set is used once, for the final model.
- **Consequence:** Metric functions need careful, tested implementations. Model selection cannot
  look at test metrics.

### ADR-005: Serve with ONNX Runtime on CPU
- **Context:** KYC liveness often runs where GPUs are unavailable or expensive, and the serving image
  should stay small.
- **Decision:** Export the final model to ONNX, apply post-training quantization and serve with ONNX
  Runtime. PyTorch is not needed at inference time.
- **Consequence:** Any accuracy lost to quantization must be measured and reported. Which
  quantization method to use (dynamic vs static, and the calibration data) is TBD — decide before
  Week 6 quantization.

### ADR-006: Weights & Biases for tracking, `EXPERIMENTS.md` as the ledger
- **Context:** Runs happen on three different machines, and a reviewer needs a quick summary without
  logging in anywhere.
- **Decision:** W&B stores curves, system metrics and artifacts. `EXPERIMENTS.md` records one row per
  run: hypothesis, change, config, SHA, headline metrics and conclusion.
- **Consequence:** Each run has to be logged in both places. The session-end command enforces the
  ledger update.

### ADR-007: `src/` layout, with scripts as thin entry points
- **Context:** Notebooks and scripts tend to accumulate logic that cannot be tested.
- **Decision:** All reusable logic lives in `src/antispoof`. `scripts/` only parses arguments and
  calls library functions. Notebooks are for EDA only.
- **Consequence:** Tests import the same code that training and serving run.

### ADR-008: Kaggle-first data workflow
- **Context:**
  - CelebA-Spoof is available as a hosted mirror on Kaggle, where the GPU training runs.
  - The local machine only needs the code.
  - The dataset license terms are not yet checked, including whether images may be redistributed.
- **Decision:**
  - The data never leaves Kaggle. Images and label files are read in place from the mirror at
    `/kaggle/input/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/CelebA_Spoof_/CelebA_Spoof`.
    They are never downloaded to a local machine or committed.
  - The repository holds code, configs and the small `configs/splits/split_assignment.csv`.
  - Manifests are regenerated on Kaggle by `scripts/build_manifest.py` into `data/manifests/`
    (gitignored). They are never committed.
  - Local tests run only on synthetic fixtures (`tests/conftest.py`).
- **Consequence:**
  - Anything that needs real images or labels runs on Kaggle: manifest build, EDA, evaluation, and
    the quantization parity check.
  - The split can be reproduced from the committed assignment file and `configs/data.yaml`, without
    the manifests.
  - Counts measured on Kaggle enter the docs with a provenance line and stay marked as not yet
    reproduced in-repo until an in-repo run prints them (`RULES.md` §6 item 7).

### ADR-009: Official-split defects and how they are handled
- **Context:** checking the mirror's `intra_test` protocol found two defects. Counts are in
  `SCHEMA.md` §1.2, reproduced in-repo by `scripts/build_manifest.py` on 2026-09-15.
  1. **Not subject-disjoint.** Subjects `5028`, `7332` and `9735` appear in both the official train
     and test splits, in the label files and on disk. No image path appears in both.
  2. **Label/path conflicts.** 2,022 images stored under `train/*/live/` carry index 43 == 1
     (spoof). The test split has no such conflicts.
- **Decision:**
  1. **The official test split is never modified**, so results stay comparable to published work.
     This is a hard invariant: `ensure_test_untouched` refuses any build that would drop or relabel
     a test row.
  2. **The 3 shared subjects are removed from train only** (`data.excluded_subjects`). Validation is
     carved out of the remaining train subjects by subject:
     - 10% of subjects (`split.val_fraction`), seeded;
     - stratified by spoof-image fraction, so the live/spoof ratio stays close across train and val.

     `validate_splits` enforces the disjointness invariant everywhere a split is produced or loaded.
  3. **Conflicting rows** are handled by `data.conflict_policy`: `exclude`, `trust_label` or
     `trust_path`. The default is **`exclude`**.
     - Reason: visual inspection of a random sample of the conflicting images was inconclusive, so
       neither the annotation nor the folder name could be established as authoritative.
     - Exclusion is the conservative choice and costs 1.2% of train live images.
     - Every manifest build logs the number of affected rows.
- **Consequence:**
  - Train loses the conflicting rows and the shared subjects' train images. The resulting train, val
    and test counts are in `SCHEMA.md` §1.2 ("Produced split").
  - These images are called *conflicting*, because which side is wrong was not established.
  - Changing the policy is a config change, recorded in the resolved config.
  - Supersedes the "official vs custom split" pending decision.

### Pending decisions

| Decision | Options under consideration | Decide before |
|---|---|---|
| Backbone | TBD | Week 2 baseline |
| Input resolution and crop margin | TBD | Week 2 baseline |
| ~~Official vs custom subject-disjoint split~~ | Resolved by ADR-009 | — |
| Auxiliary attribute heads (spoof type, illumination, environment) | Binary head only / multi-task heads | Week 4 iteration |
| Threshold selection rule | Fixed APCER on val / fixed BPCER on val / cost-weighted | Week 5 freeze |
| Confidence calibration | TBD | Week 5 freeze |
| Quantization method | Dynamic / static with calibration set | Week 6 quantization |
| Inference-time face detector | An OpenCV built-in detector / other | Week 7 serving |
