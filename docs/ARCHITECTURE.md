# Architecture

## 1. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Wheels are available for everything in the stack (PyTorch, ONNX Runtime, OpenCV, albumentations). Compatibility with the Kaggle/Colab runtime Python version is TBD — verify before Week 2 baseline |
| Environment | uv + `pyproject.toml` + `uv.lock` | One tool for the interpreter, virtualenv and a lockfile. It is fast enough to recreate environments on ephemeral Kaggle/Colab machines |
| Deep learning | PyTorch + torchvision | The standard research framework, with first-class MPS support on Apple silicon for local smoke tests and CUDA on Kaggle/Colab |
| Backbones | timm | Pretrained ImageNet backbones behind one API, so swapping the backbone is a config change instead of a code change |
| Augmentation | albumentations | Fast, composable image augmentations, including the photometric, blur and compression transforms that matter for spoof cues |
| Image I/O | opencv-python-headless, Pillow | OpenCV handles decoding, resizing and face cropping. The headless build avoids GUI dependencies on servers. Pillow decodes uploads in the API |
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
├── configs/                 # YAML experiment configs; the single source of hyperparameters
├── data/                    # Raw dataset and derived manifests. Gitignored, never committed
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
CelebA-Spoof raw files (data/raw/, gitignored)
        │  scripts/build_manifest.py  → antispoof.data
        ▼
Image manifest CSV (one row per image; SCHEMA.md §1)
        │  scripts/make_split.py      → antispoof.data   (seeded, subject-level)
        ▼
Split file (subject_id → train/val/test; SCHEMA.md §2)
        │  invariant check: subject sets are pairwise disjoint (test + load-time assert)
        ▼
Dataset + transforms (face crop → resize → augment [train only] → normalize)
        │  scripts/train.py --config configs/<exp>.yaml   → antispoof.training
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

The script names above describe where each entry point will live. None of them exist yet.

## 4. Compute split: local vs Kaggle/Colab

| Task | Where | Notes |
|---|---|---|
| Manifest build, split generation, schema validation | Local (M5, 16 GB) | CPU and I/O only; no GPU needed |
| EDA notebooks | Local | Use a sampled subset if the full manifest is too large for comfortable interactive work |
| Unit tests, lint, type check | Local | Synthetic fixtures only |
| Training smoke test (a few batches, tiny subset, overfit check) | Local (CPU or MPS) | Checks that the pipeline runs end to end; not used for metrics |
| Full training runs | Kaggle / Colab GPU | Configs are identical; only paths and device differ, set through config |
| Full validation and test evaluation | Kaggle / Colab GPU | The test set is evaluated once, for the final model |
| ONNX export, quantization, parity check | Local or Kaggle | Parity is checked on the same evaluation subset for FP32 and quantized models |
| CPU latency benchmark | Local | The serving target is CPU; hardware is recorded with every number |
| API and demo development | Local | uvicorn on localhost |

Open items:

- **Getting the dataset onto Kaggle/Colab** (official download vs an existing hosted copy, subject to
  license terms). TBD — decide before Week 1 manifest.
- **Checkpoint handoff** between Kaggle/Colab and local (W&B artifacts vs downloaded output). TBD —
  decide before Week 2 baseline.

## 5. Technical decisions (ADR log)

Each ADR records context, decision and consequence. Superseded decisions are marked, not deleted.

### ADR-001: Use uv for environment and dependency management
- **Context:** The project runs on a local Mac and on ephemeral cloud GPU machines. Environments must
  be quick to recreate and resolve the same way everywhere.
- **Decision:** uv, with `pyproject.toml` as the single dependency declaration and a committed
  `uv.lock`.
- **Consequence:** One command (`uv sync`) sets up the environment. Kaggle/Colab come with their own
  CUDA PyTorch build, so the cloud setup may install the project on top of the platform's torch
  instead of from the lockfile. Any version drift must be recorded in the run record.

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
  whether the official split is subject-disjoint.

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

### Pending decisions

| Decision | Options under consideration | Decide before |
|---|---|---|
| Backbone | TBD | Week 2 baseline |
| Input resolution and crop margin | TBD | Week 2 baseline |
| Official vs custom subject-disjoint split | Official split (if disjoint) / seeded custom split | Week 1 manifest |
| Auxiliary attribute heads (spoof type, illumination, environment) | Binary head only / multi-task heads | Week 4 iteration |
| Threshold selection rule | Fixed APCER on val / fixed BPCER on val / cost-weighted | Week 5 freeze |
| Confidence calibration | TBD | Week 5 freeze |
| Quantization method | Dynamic / static with calibration set | Week 6 quantization |
| Inference-time face detector | An OpenCV built-in detector / other | Week 7 serving |
