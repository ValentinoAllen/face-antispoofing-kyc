# Progress

## Current status

The baseline training pipeline is committed, with Kaggle smoke runs on 4k/2k subsets whose numbers
are not a model-quality estimate. A header-metadata probe on the same subsets,
`20260916-075616-probe_metadata`, meets the pre-registered rule for a strong header-level shortcut.
The EDA notebook is still an open Week 1 item.
- **Code:** `antispoof.data` holds the verified annotation layout and code convention, the config
  loader, the manifest builder, the subject-level split with its invariants and the build
  orchestration. It also holds `ManifestDataset`, `make_subset`, `read_manifest` and the baseline
  transform (resize and normalize with the timm pretrained stats; no face crop, no augmentation).
  `antispoof.models.factory` builds a single-logit timm model, and `antispoof.eval.pad_metrics`
  computes pooled APCER, BPCER and ACER. `antispoof.training` holds the experiment config loader,
  seeding and provenance helpers, the training loop and the run orchestration behind
  `scripts/train.py` with `configs/baseline.yaml`.
  - `antispoof.eval.metadata_probe`, `scripts/probe_metadata.py` and `configs/probe_metadata.yaml`
    read image headers only (no pixel decode) for the baseline's subsets. They report per-class
    summaries, a descriptive single-feature val ROC AUC ranking, pooled PAD metrics of a
    HistGradientBoosting (primary) and a logistic regression (secondary) classifier, and an optional
    comparison with a baseline `predictions.csv`.
  - Run records now carry `manifest_sha256`, `environment.pillow` and `environment.sklearn`.
- **Checks:** ruff, mypy (20 source files) and `uv run pytest` (182 tests) passed before commit
  `0f03586`, locally with Pillow 12.3.0 and scikit-learn 1.9.1.
- **Dataset:** the owner's third Kaggle build (2026-09-15) printed counts equal to every count in
  `docs/SCHEMA.md` §1.2 and §1.1.1. Split: train 442,859 rows / 7,370 subjects, val 49,308 / 819,
  test 67,170 / 1,004. The committed `configs/splits/split_assignment.csv` is the one the runs used:
  its SHA-256 equals the records' `split_sha256`. The two 2026-09-16 runs read manifests with
  identical `manifest_sha256`.
- **Runs:** `20260915-153606-baseline`, `20260916-075448-baseline` and
  `20260916-075616-probe_metadata` are in `docs/EXPERIMENTS.md`.
  - Baseline: 1 epoch on a 4,000-image train subset, evaluated on a 2,000-image val subset (654 live,
    1,346 spoof) at a fixed threshold of 0.5. Pooled APCER 1.86% (25/1,346), BPCER 1.53% (10/654),
    pooled ACER 1.69%.
  - `20260915-153137-baseline`, `20260915-153606-baseline` (both at `6f368c0`) and
    `20260916-075448-baseline` (at `9d75207`) have identical metrics, counts and mean train loss;
    only epoch wall time and images/s differ.
  - `20260915-153137-baseline` was launched first (by `created_at`). Its epoch took 55.8 s (71.7
    train images/s) against 41.0 s for `20260915-153606-baseline`. This is consistent with a cold
    file cache on the first pass over the images; it was not measured. 71.7 images/s is therefore
    the more realistic throughput for a first pass over unseen images.
  - Probe: the primary classifier has pooled APCER 0.00% (0/1,346), BPCER 0.00% (0/654) and pooled
    ACER 0.00% on the val subset: a strong header-level shortcut under the pre-registered rule. The
    secondary classifier also has 0/1,346 and 0/654. The CNN-vs-metadata comparison is
    uninformative, because the metadata classifier made no errors and its score is constant within
    each class.
  - Header facts (`probe_summary.json`): in both subsets every live and spoof image is `.jpg`, JPEG,
    RGB, 4:2:0, non-progressive, with no EXIF and no ICC profile. The mean of the JPEG luminance
    quantization table is constant within each class on train and val (29.03125 live, 5.765625
    spoof). Spoof pixel count has median 270,000 with IQR 0 on train and val. The descriptive
    single-feature val ROC AUC is 1.0 for `jpeg_luma_quant_mean`, between 0.95 and 1.0 for the
    six size and dimension features, and 0.5 for the seven categorical and flag features.
  - Records are in `reports/runs/`; checkpoints, `predictions.csv` and `features.csv` stayed on
    Kaggle.
- **Run records:** training writes `<output-dir>/<run_id>/` on Kaggle. The owner copies
  `record.json` and `resolved_config.json` into `reports/runs/<run_id>/` and commits them with the
  ledger row; for the probe, `probe_summary.json` is committed too (owner decision), which
  `docs/SCHEMA.md` §3 and `docs/EXPERIMENTS.md` do not yet describe. Checkpoints and predictions are
  not committed; a run to be kept is saved as a Kaggle notebook version. The split's identity is
  `split_sha256`; there are no sidecar files.
- **Environment:** supported Python is 3.11–3.12. Kaggle, with its preinstalled torch and timm via
  `PYTHONPATH=src`, is the reference environment for every run that produces numbers, and each
  record's `environment` block is the source of truth for versions. The 2026-09-16 Kaggle runs used
  Pillow 11.3.0 and scikit-learn 1.6.1. `pyproject.toml` still enforces Python <3.12, and `uv.lock`
  pins different torch and timm versions than Kaggle's.
- **Not implemented:** BPCER@APCER=1%, per-species APCER, a threshold fitted on val, face crop,
  augmentation, any test evaluation, and `validate_attack_codes` violation counts in the build.
- **Unverified:**
  - Whether the CNN uses the header-level traces: the probe shows they are available, not that they
    are used.
  - Whether the class-level differences in JPEG quantization tables and image dimensions come from
    the original CelebA-Spoof release or from this Kaggle mirror.
  - Pixel-level shortcuts are untested.
  - The SCHEMA §1.2 directory-listing checks are still externally measured.
  - Codes on the 2,022 conflicting train rows and on live rows are unmeasured, and val's code
    distribution is not printed.
- **Known risks:** covariate shift between the train and test attack populations (`docs/PRD.md`
  §8). Val and test ACER are not directly comparable, and per-condition test cells are small.
- **Next step:** test whether the CNN uses the header-level traces, and measure the JPEG
  quantization tables. Then design the Week 2 evaluation: a val-fitted threshold, BPCER@APCER=1%,
  per-species APCER and the PRD §8 consequences (a)–(c).

## Milestones

- [ ] **Week 1 (2026-09-14 → 2026-09-20):** repo scaffold, dataset access, manifest builder,
  subject-disjoint split + test, EDA notebook
  - [x] Repo scaffold (2026-09-14)
  - [x] Dataset access (2026-09-14; CelebA-Spoof mirror on Kaggle checked by the owner, ADR-008)
  - [x] Manifest builder (2026-09-15; ran on the Kaggle mirror and reproduced SCHEMA §1.2)
  - [x] Subject-disjoint split + invariant test (2026-09-15; `split_assignment.csv` committed and
    tested)
  - [ ] EDA notebook
- [ ] **Week 2 (2026-09-21 → 2026-09-27):** PRD targets set, PAD metrics + tests, transforms,
  dataset class, config loader, baseline training run
  - [ ] PRD targets set
  - [ ] PAD metrics + tests (pooled APCER/BPCER/ACER with tests, 2026-09-15; BPCER@APCER=1% and
    per-species APCER not yet)
  - [ ] Transforms (baseline resize + normalize, 2026-09-15; no face crop, no augmentation)
  - [x] Dataset class (2026-09-15; `ManifestDataset`, `make_subset`)
  - [x] Config loader (2026-09-15; `antispoof.training.config`)
  - [ ] Baseline training run (smoke run `20260915-153606-baseline` on 4k/2k subsets, 2026-09-15)
- [ ] **Week 3 (2026-09-28 → 2026-10-04):** evaluation report generator, error analysis v1
- [ ] **Week 4 (2026-10-05 → 2026-10-11):** hypothesis-driven model iteration
- [ ] **Week 5 (2026-10-12 → 2026-10-18):** model freeze, threshold selection, confidence definition
- [ ] **Week 6 (2026-10-19 → 2026-10-25):** quantization, ONNX export, parity check, CPU latency
  benchmark
- [ ] **Week 7 (2026-10-26 → 2026-11-01):** FastAPI service, face/quality gating, demo page, API tests
- [ ] **Week 8 (2026-11-02 → 2026-11-08):** one-shot test evaluation, final report, README results,
  write-up

## Session log

### 2026-09-16: Metadata probe for the capture-source shortcut

**Done**
- `dce18e9` (docs):
  - `docs/PRD.md` §6: the target deadlines are "before the first full-train run".
  - `docs/SCHEMA.md` §1.1: states the `validate_attack_codes` decision.
  - `docs/ARCHITECTURE.md` §4: checkpoints are never committed, and kept runs persist as Kaggle
    notebook versions. Handoff to local for ONNX export is "TBD — decide before Week 6 ONNX export";
    that deadline was proposed in this session and approved by the owner with the plan.
  - `README.md`: the Python line matches the ARCHITECTURE Language row.
- `b210394` (provenance):
  - `src/antispoof/training/run.py`: `manifest_sha256` in the run record, with `manifest_paths` as
    the single source of the files read and hashed. Shared record helpers `RecordHeader`,
    `build_record`, `fill_pooled_metrics`, `close_failed_record` and `write_json`. The
    `new_record` docstring no longer mentions a split sidecar.
  - `docs/SCHEMA.md` §3 row; `tests/test_training.py` asserts the hashes.
- `9d75207` (probe):
  - `src/antispoof/eval/metadata_probe.py`, `configs/probe_metadata.yaml` (hypothesis, what_changed
    and notes verbatim from the owner), `scripts/probe_metadata.py` and
    `tests/test_metadata_probe.py`.
  - All 14 requested features are read from header parsing in the installed Pillow 12.3.0; none was
    dropped.
  - `environment_info` adds `pillow` and `sklearn`; `parse_section` in `antispoof.training.config`
    is public.
  - `docs/SCHEMA.md` §3: `environment.pillow`, `environment.sklearn`, and `training_epochs` nullable
    for runs that do not train.
  - Tests on synthetic JPEG and PNG files: extracted values, PNG nulls and missing indicators, a spy
    showing no pixel `load` call (one header and a full `run_probe`), unreadable headers naming the
    path, the exact-join check, n/a handling, and the committed config's verbatim fields and
    threshold.
- Checks: ruff format, ruff check, mypy and `uv run pytest` passed on the content of each commit,
  run just before committing: 165 tests for `dce18e9` and `b210394`, 182 for `9d75207` and
  `0f03586`.
- Recorded the owner's Kaggle runs at `9d75207` (2026-09-16, run by the owner outside this session)
  and committed their record folders with two ledger rows as `0f03586`:
  - `20260916-075448-baseline`, compared key by key with both 2026-09-15 baseline records: metrics,
    counts, mean train loss and data subsets are identical, and `config_hash` equals
    `20260915-153606-baseline`'s. Apart from run identity (run_id, created_at, git_sha, checkpoint
    path) and the keys added since, only epoch wall time and images/s differ.
  - `20260916-075616-probe_metadata`: pooled val ACER 0.00% (0/1,346 attacks accepted, 0/654 bona
    fide rejected); under the pre-registered rule, a strong header-level shortcut. Its
    `predictions_path` is the rerun's `predictions.csv`.
  - For both records: `git_dirty: false`; `config_hash` recomputed from `resolved_config.json`
    matches; `split_sha256` equals the committed file's SHA-256; `manifest_sha256` is identical.
  - `docs/EXPERIMENTS.md`: both rows start from `format_ledger_row` on the committed records; the
    notes add the comparison with the earlier baselines and the decision-rule outcome.
- `docs/PROGRESS.md` open questions: the capture-source question gains the header-level result, and
  three questions are added (origin of the differences, whether the CNN uses the traces, pixel-level
  shortcuts).

**Broke / not verified**
- The Kaggle console output (manifest build, baseline rerun, probe) was not pasted into this
  session. Everything recorded comes from the committed record files; the 2026-09-16 manifest
  build's printed counts were not seen.
- The CNN-vs-metadata comparison is uninformative: the metadata classifier made no errors and its
  score is constant within each class.
- Whether the CNN uses the header-level traces is untested; the probe shows they are available, not
  that they are used.
- Whether the class-level differences come from the original CelebA-Spoof release or from this
  Kaggle mirror is unknown. Pixel-level shortcuts are untested.
- The no-pixel-decode spy test ran only locally (Pillow 12.3.0, scikit-learn 1.9.1). The Kaggle probe
  ran with Pillow 11.3.0 and scikit-learn 1.6.1.
- One local pytest run took 61.6 s right after `ruff format`; the next two took 4.21 s and 4.46 s,
  with no test slower than 0.61 s. The slow run did not reproduce.
- `docs/SCHEMA.md` §3 and `docs/EXPERIMENTS.md` still describe two committed files per run; the
  probe folder also has `probe_summary.json`.

**Next**
- Test whether the CNN uses the header-level traces, and measure the JPEG quantization tables.
- Find out whether the quantization-table and dimension differences come from the original
  CelebA-Spoof release or from this Kaggle mirror.
- Pixel-level shortcuts remain untested.
- Describe `probe_summary.json` in the SCHEMA §3 and EXPERIMENTS run-record workflow.
- Week 2 evaluation design: a val-fitted threshold, BPCER@APCER=1%, per-species APCER and the PRD
  §8 consequences (a)–(c).
- Code session: make the manifest build report `validate_attack_codes` violation counts.
- Widen `requires-python` in `pyproject.toml` to include 3.12.
- Set the PRD targets before the first full-train run. Write the EDA notebook.

### 2026-09-16: Seven doc gaps closed from owner decisions

**Done**
- Docs-only session, committed as `d6e2c3c`. No code, config or run record changed, and nothing
  under `data/` was read.
- `docs/SCHEMA.md`:
  - §3 path: records are written to `<output-dir>/<run_id>/` on Kaggle. The owner copies
    `record.json` and `resolved_config.json` into `reports/runs/<run_id>/` and commits them with the
    ledger row. `checkpoint.pt` and `predictions.csv` are not committed; a run to be kept is saved
    as a Kaggle notebook version.
  - §3 table: added `what_changed`, `data_subsets`, `training_epochs`, `error` (failed or aborted
    runs only), `metrics.apcer_pooled`, `metrics.acer_pooled`, `metrics.n_attack_accepted`,
    `metrics.n_bona_fide_rejected`, and one row per `environment` key, including `timm` and
    `deterministic_algorithms`. Types and nullability were taken from `antispoof.training.run`,
    `antispoof.training.reproducibility`, `EpochStats`, `SplitSummary` and the committed records.
  - `split_sha256` is documented as the SHA-256 of `configs/splits/split_assignment.csv`.
  - Owner decision: both sidecars (`split_assignment.meta.yaml` in §2, `manifest.meta.json` in §1.3)
    were removed; a file's SHA-256 is its identity.
- `docs/EXPERIMENTS.md`: "How to log a run" describes the copy-and-commit workflow. A note above
  the table says the APCER and ACER columns are the ISO/IEC 30107-3 worst case over PAI species and
  that pooled values need the "(pooled)" suffix. Existing rows were not edited.
- `docs/ARCHITECTURE.md`:
  - §1 Image I/O: Pillow decodes, and torchvision resizes and normalizes with the interpolation and
    mean/std from timm's pretrained config. OpenCV is declared but unused.
  - §1 Language row and a "superseded in part" note on ADR-001: the Python/Kaggle decision,
    including that `pyproject.toml` still enforces <3.12.
  - §3: the real `scripts/train.py` command, and the list of scripts that exist.
- `docs/RULES.md` §2: machine-specific paths live in `configs/data.yaml` and the `scripts/train.py`
  path flags. The per-environment config was dropped.
- `docs/PROGRESS.md` open questions:
  - `validate_attack_codes` decided: every build reports violation counts, and the check becomes a
    hard assertion after a Kaggle build shows zero violations.
  - Sidecar keys closed.
  - Bounding-box deadline is now "before the first run with face crop".
  - PRD targets deadline is now "before the first full-train run", because the smoke numbers are
    not interpretable until the capture-source shortcut is checked and targets must not be fitted to
    them.
  - Python/Kaggle decided.
- Recorded the owner's reading of the epoch times in Current status: the first run's slower epoch
  is consistent with a cold file cache.
- Checks on the docs:
  - A Python script over both `reports/runs/*/record.json` files found every key, plus `error`, in
    SCHEMA §3.
  - `git grep` finds the sidecar files named only in the struck open question, older session-log
    text and the `run.py` docstring.
  - `git diff` of `docs/EXPERIMENTS.md` removed no table row.

**Broke / not verified**
- ruff, mypy and pytest were not run (docs only).
- The cold-cache explanation was not measured.
- The copy-and-commit workflow and the `PYTHONPATH=src` setup were not exercised on Kaggle this
  session. Whether the 2026-09-15 runs used `PYTHONPATH=src` is not recorded.
- Left untouched, because the owner's answer on knock-on edits named only the ARCHITECTURE Python
  row:
  - `docs/PRD.md` §6 target cells still say "decide before Week 2 baseline";
  - `docs/SCHEMA.md` §1.1 still has asserting `validate_attack_codes` as TBD before the Week 2
    baseline;
  - `docs/ARCHITECTURE.md` §4 checkpoint handoff is still TBD before the Week 2 baseline.
- Also still stale: the `new_record` docstring in `src/antispoof/training/run.py` says the split
  sidecar does not exist yet, and `README.md` says Python 3.11.

**Next**
- Check for the capture-source shortcut; decide how first.
- Week 2 evaluation design: a val-fitted threshold, BPCER@APCER=1%, per-species APCER and the PRD
  §8 consequences (a)–(c).
- Code session: make the manifest build report `validate_attack_codes` violation counts.
- Widen `requires-python` in `pyproject.toml` to include 3.12.
- Set the PRD targets before the first full-train run. Write the EDA notebook.

### 2026-09-15: Baseline training pipeline and first Kaggle smoke run

**Done**
- Built the baseline pipeline, committed as `6f368c0`:
  - `src/antispoof/data/dataset.py` (`ManifestDataset`, `make_subset`), `read_manifest` in
    `src/antispoof/data/manifest.py`, and `src/antispoof/data/transforms.py`;
  - `src/antispoof/models/factory.py` and `src/antispoof/eval/pad_metrics.py`;
  - `src/antispoof/training/config.py`, `reproducibility.py`, `loop.py` and `run.py`;
  - `configs/baseline.yaml` and `scripts/train.py`;
  - tests: `tests/test_dataset.py`, `test_pad_metrics.py`, `test_train_config.py`,
    `test_reproducibility.py` and `test_training.py`, plus a `write_images` fixture in
    `tests/conftest.py`.
- Owner decision: the run record stores pooled rates under the extra keys `metrics.apcer_pooled`
  and `metrics.acer_pooled`. `apcer_max`, `acer` and `apcer_per_species` stay null.
- Checks before `6f368c0`: `ruff format --check` (40 files), `ruff check`, `mypy` (no issues in 19
  source files) and `uv run pytest` (165 passed).
- Overfit test:
  - Checked its criterion in a scratch script on seeds 0–9. All passed, with final losses at most
    0.0091.
  - Added an absolute bound to the test (final loss < 0.05).
- Ran `scripts/train.py` end to end on synthetic images in a throwaway git repo, with 2 DataLoader
  workers. The status was completed and all four run files were written. The metrics are synthetic
  and meaningless.
- Two attempts to run the Kaggle command on the local Mac failed with
  `ModuleNotFoundError: No module named 'antispoof'` (plain `python`, outside the project
  environment). Nothing was written.
- Recorded the owner's Kaggle work (2026-09-15, outside this session):
  - A manifest rebuild. Its printed counts equal every count in `docs/SCHEMA.md` §1.2 and §1.1.1,
    compared in this session against the pasted output. The owner's own check printed
    `split_assignment identical to committed: True`.
  - Two training runs, `20260915-153137-baseline` and `20260915-153606-baseline`, both at
    `6f368c0` with `git_dirty: false` on CUDA. The owner copied their `record.json` and
    `resolved_config.json` into `reports/runs/`; no checkpoints were copied.
- Diffed the two records key by key:
  - Exactly five fields differ: `run_id`, `created_at`, `artifacts.checkpoint`,
    `training_epochs[0].wall_time_s` (55.77 s vs 41.00 s) and `training_epochs[0].images_per_s`
    (71.72 vs 97.56).
  - All metrics and the mean train loss are identical.
  - The two resolved configs are byte-identical.
- Checked `20260915-153606-baseline`:
  - `config_hash`, recomputed from its `resolved_config.json` and again from the committed configs
    with the manifest-dir override, equals the record's.
  - `split_sha256` equals the SHA-256 of `configs/splits/split_assignment.csv`.
  - `git_sha` is `6f368c0`.
  - `format_ledger_row` on the copied record reproduces the row printed on Kaggle.
- `docs/EXPERIMENTS.md`: one row for `20260915-153606-baseline`.
  - Metrics: pooled APCER 1.86%, BPCER 1.53%, pooled ACER 1.69%, BPCER@APCER=1% —.
  - The notes say the repeat run matched exactly and that the row is not a model-quality estimate.
  - Committed with both record folders as `5500cad`.
  - On the owner's instruction, `20260915-153137-baseline` has no row of its own; the note covers
    it.

**Broke / not verified**
- A capture-source shortcut is not ruled out, so the smoke-run metrics are not a model-quality
  estimate.
- Environment drift on Kaggle: Python 3.12.13, torch 2.10.0+cu128 and timm 1.0.26. By contrast,
  `pyproject.toml` requires Python <3.12 and `uv.lock` pins torch 2.14.0 and timm 1.0.29. How the
  package was made importable on Kaggle is not recorded.
- Why two Kaggle runs exist is not recorded. Their identical metrics are one observation under
  `warn_only` determinism, not a guarantee.
- Checkpoints exist only on Kaggle (`/kaggle/working/runs/...`). The checkpoint handoff is still
  undecided (`docs/ARCHITECTURE.md` §4).
- Not implemented: BPCER@APCER=1%, per-species APCER, a val-fitted threshold, face crop and
  augmentation. The W&B-enabled path in `antispoof.training.run` is untested.
- Seven doc gaps found this session are not fixed; the owner deferred them:
  1. SCHEMA §3 and EXPERIMENTS.md step 3 put records at `reports/runs/<run_id>/`. The script
     writes to `<output-dir>/<run_id>/`, and the owner copies them over.
  2. SCHEMA §3 lacks the extra record keys: `what_changed`, `data_subsets`, `training_epochs`,
     `error`, `metrics.apcer_pooled`, `metrics.acer_pooled`, `metrics.n_attack_accepted`,
     `metrics.n_bona_fide_rejected`, `environment.timm` and `environment.deterministic_algorithms`.
  3. SCHEMA §3 says `split_sha256` matches the split sidecar, which does not exist. The code
     hashes `split_assignment.csv` itself.
  4. The EXPERIMENTS.md APCER column does not say whether it is pooled or the maximum over species.
  5. ARCHITECTURE §1 says OpenCV resizes, but the baseline uses PIL and torchvision. ARCHITECTURE
     §3 shows an outdated `train.py` command and says only `build_manifest.py` exists.
  6. RULES §2 describes a per-environment config included by the experiment config; none exists.
  7. Open questions in this file marked "before the Week 2 baseline" are still open.
- ruff, mypy and pytest were not rerun after `6f368c0`, because only run records and docs changed.
- In one full pytest run, `test_unreadable_image_raises_naming_the_path` took 3.51 s. It did not
  reproduce (0.01 s on rerun).

**Next**
- Fix the seven doc gaps above.
- Decide how to check for the capture-source shortcut.
- Week 2 evaluation design:
  - a val-fitted threshold;
  - BPCER@APCER=1%;
  - per-species APCER;
  - the PRD §8 consequences (a)–(c).
- Decide how to handle the Kaggle Python 3.12 runtime against `requires-python` and the lockfile.
- Set the PRD targets. Write the EDA notebook.

### 2026-09-15: Second Kaggle build recorded; covariate shift replaces prior-shift risk

**Done**
- Recorded the owner's second Kaggle run of `scripts/build_manifest.py` (2026-09-15, run by the
  owner outside this session). As reported by the owner, all figures were printed by repo code,
  and the official-train index 40–42 counts matched SCHEMA §1.1.1. The run also printed the test
  counts for the first time:
  - test `spoof/` rows = 47,247
  - `spoof_type`: 1=3,600 2=5,421 3=6,083 4=4,287 5=6,097 6=3,530 7=6,477 8=3,659 9=4,483 10=3,610
  - `illumination`: 1=35,119 2=5,971 3=2,461 4=3,696
  - `environment`: 1=40,722 2=6,525
- Checked in this session with a Python one-liner over the recorded counts: each test distribution
  sums to 47,247. The owner's train-vs-test shares match the counts (code 7: 8.81% vs 13.71%;
  code 10: 12.31% vs 7.64%; illumination 1: 58.98% vs 74.33%; illumination 3: 10.64% vs 5.21%;
  environment 1: 76.29% vs 86.19%).
- Read `src/antispoof/data/build.py` to confirm what the report counts. `count_attack_codes` covers
  `spoof/` rows of official train after the conflict policy (before val is carved out) and of test.
  A code `0` would be listed if present. Live rows and val are not counted.
- Docs (no code changes; nothing under `data/` read or written):
  - `docs/SCHEMA.md` §1.1: validator note. No `spoof/` row in official train or test has code `0`.
    The conflicting train rows and live rows remain unmeasured.
  - `docs/SCHEMA.md` §1.1.1: the external mark is replaced by "reproduced in-repo" provenance. Adds
    the test table, the train-vs-test covariate shift, the small-cell note, and the fact that val's
    distribution is not printed.
  - `docs/SCHEMA.md` §1.2: the live-fraction note is corrected. APCER/BPCER are class-conditional,
    and the risk is the attack-population difference.
  - `docs/PRD.md` §8: the skew entry now carries the test share. The prior-shift entry is replaced
    by a covariate-shift entry with the Week 2 consequences (a)–(c) and the correction.
  - `docs/PROGRESS.md`: current status and open questions.

**Broke / not verified**
- ruff, mypy and `uv run pytest` were not run this session (docs only).
- Only the test counts from the second run were seen in this session. The match of the train
  counts rests on the owner's report.
- Codes on the 2,022 conflicting train rows and on live rows are unmeasured. Val's code
  distribution is not printed by the build, so consequence (c) cannot yet be checked against val.
- The manifests written by the second run were not inspected (e.g. the renamed columns).
- The SCHEMA §1.2 directory-listing checks are still externally measured.

**Next**
- EDA notebook (the remaining Week 1 item).
- Week 2 evaluation design: report val and test ACER with the split named, confidence intervals on
  per-condition test breakdowns, and a check of any val/test gap against the distribution
  difference.
- Decide whether the build report should also print val's code distribution, which (c) needs.
- Decide whether the build asserts `validate_attack_codes`.

### 2026-09-15: Verified indices 41/42, first Kaggle build reconciled, split committed

**Done**
- Recorded the owner's first Kaggle run of `scripts/build_manifest.py` (2026-09-15, run by the owner
  outside this session). As reported by the owner, it reproduced every count in `docs/SCHEMA.md` §1.2
  exactly and produced:
  - train: rows=442,859 subjects=7,370 live=146,280 spoof=296,579
  - val: rows=49,308 subjects=819 live=16,123 spoof=33,185
  - test: rows=67,170 subjects=1,004 live=19,923 spoof=47,247
  - config: `conflict_policy=exclude`, `val_fraction=0.1`, `seed=42`, `stratify_bins=10`
  - official train after the conflict policy, before subject exclusion: 492,383 rows / 8,192
    subjects; the 3 excluded subjects account for the 216-row difference
- Checked the downloaded `configs/splits/split_assignment.csv` in this session: train 7,370, val
  819, test 1,004 subjects (9,193 total), no duplicate ids, sorted. `load_split_assignment` with the
  configured excluded subjects passed, and `5028`, `7332`, `9735` are listed only as `test`.
  Committed the file.
- Recorded the owner's Kaggle measurement of indices 40–42 over the 329,921 official-train `spoof/`
  images: index 40 has 10 codes, 41 (illumination) 4, 42 (environment) 2; codes are 1-indexed and
  `0` means not applicable (live only). No committed code printed these, so they carry the
  "externally measured" mark (RULES §6 item 7).
- Code:
  - `src/antispoof/data/labels.py`: `INDEX_ILLUMINATION`, `INDEX_ENVIRONMENT`,
    `get_illumination`, `get_environment`, `ATTACK_CODE_INDICES`, `CODE_NOT_APPLICABLE`, and
    `validate_attack_codes`. The validator rejects `0` on a spoof vector and non-zero on a live
    vector. It is not yet called by the build.
  - `src/antispoof/data/manifest.py`: columns `attr_41`/`attr_42` renamed to
    `illumination`/`environment`; `ATTACK_CODE_COLUMNS`.
  - `src/antispoof/data/build.py`: `count_attack_codes` and `CodeCounts`; the build report prints
    per-code counts on `spoof/` rows for train and test.
- Tests:
  - `tests/test_manifest.py`: validator tests in both directions for each of indices 40–42.
  - `tests/test_splits.py`: code-count tests, and a test that loads the committed split assignment
    and checks pairwise disjointness and that the excluded subjects are only in test.
  - `tests/conftest.py`: synthetic spoof vectors now carry illumination/environment code 1.
- Checks: `ruff format` left 26 files unchanged, `ruff check` passed, `mypy` found no issues in 11
  source files, and `uv run pytest` passed 104 tests.
- Docs:
  - `docs/SCHEMA.md`:
    - indices 41/42 verified, with the code convention;
    - §1.1.1 code distributions with provenance and the illumination skew;
    - §1.2 marked reproduced in-repo, except the directory-listing checks;
    - the produced split, the renamed columns, and §2 marking the split assignment committed.
  - `docs/PRD.md` §3 and §8: counts reproduced; new risks for illumination skew and the val/test
    live-fraction shift. The shift wording was corrected with the owner: APCER/BPCER are
    class-conditional, so the prior alone does not move them at a fixed threshold. Prior-dependent
    rules and the test split's make-up can.
  - `docs/ARCHITECTURE.md` ADR-009, `README.md`, `scripts/build_manifest.py` docstring: dropped the
    "not yet reproduced" qualifier.
- `/session-end` (`.claude/commands/session-end.md`) now ends with a plain `git push` (never forced;
  a failure is reported as local only). `docs/RULES.md` §6 item 5 was amended to allow exactly that
  push.

**Broke / not verified**
- The new code-count output of the build report has not run on real data. The index 40–42
  distributions are not yet reproduced in-repo.
- Manifests previously built on Kaggle use the old `attr_41`/`attr_42` column names and must be
  regenerated. The split assignment itself does not depend on those columns.
- `validate_attack_codes` has not been run on real vectors. The codes on the 2,022 conflicting train
  rows and on the test split are unmeasured.
- The SCHEMA §1.2 directory-listing checks are still externally measured.
- The new `/session-end` push step has not run. On the owner's instruction it was skipped this
  session, so this commit is local only.

**Next**
- Rerun `scripts/build_manifest.py` on Kaggle.
  - Regenerate the manifests under the new column names.
  - Reconcile the printed code counts with SCHEMA §1.1.1 and remove the external mark if they match.
- Decide whether the build should assert `validate_attack_codes`, after measuring codes on the
  conflicting rows and the test split.
- EDA notebook (the remaining Week 1 item).

### 2026-09-15: CelebA-Spoof mirror findings, manifest builder and subject-level split

**Done**
- Recorded the owner's check of the CelebA-Spoof Kaggle mirror (`intra_test` protocol), measured on
  Kaggle on 2026-09-14 outside this repository:
  - The annotation vector has 44 integers. Index 40 (spoof type) and index 43 (label, `0` = live,
    `1` = spoof) are verified. The meaning of indices 41 and 42 is provisional.
  - Defect 1: subjects `5028`, `7332` and `9735` are in both official train and test. Decision:
    remove them from train only; test is never modified.
  - Defect 2: 2,022 train images under `live/` carry index 43 == 1. Decision: `conflict_policy:
    exclude` by default, because visual inspection of a sample was inconclusive.
- Added the data code:
  - `src/antispoof/data/labels.py`, `config.py`, `manifest.py`, `splits.py` and `build.py`;
  - `scripts/build_manifest.py`;
  - `configs/data.yaml` and `configs/splits/.gitkeep`.
- Added tests on synthetic fixtures only: `tests/conftest.py`, `tests/test_manifest.py` and
  `tests/test_splits.py`.
- Added `pandas.*` and `yaml.*` to mypy's `ignore_missing_imports` override in `pyproject.toml`.
- Checks:
  - `ruff format` reformatted 5 new files. After that, `ruff format --check` and `ruff check` pass.
  - `mypy` passes (11 source files).
  - `uv run pytest`: 88 passed.
  - Ran `scripts/build_manifest.py` on a synthetic dataset in a temporary directory. It detected the
    planted conflict and the planted shared subject, and wrote all four CSVs.
- Docs:
  - `docs/SCHEMA.md` §1–2 rewritten: verified index mapping, externally measured counts with
    provenance, the manifest v1 and `split_assignment.csv` contracts, and the set-equation
    invariant. Bounding boxes are deferred to manifest v2.
  - `docs/ARCHITECTURE.md`: ADR-008 (Kaggle-first data workflow) and ADR-009 (the two defects).
    §2–§4 and the pending-decisions table were updated to match.
  - `docs/PRD.md`: a data-quality risk entry, and the paper's figures now separated from the
    mirror's.
  - `docs/RULES.md`: new §6 item 7, provenance for figures measured outside this repository.
  - `README.md`: a dataset paragraph.

**Broke / not verified**
- `scripts/build_manifest.py` has not run on the real dataset. Nothing under `data/` was created.
- The counts in `docs/SCHEMA.md` §1.2 are externally measured and not yet reproduced in-repo.
- `configs/splits/split_assignment.csv` does not exist yet.
- Still unverified: the meaning of indices 41 and 42, the spoof-type code names, and the
  bounding-box format.

**Next**
- Run `scripts/build_manifest.py` on Kaggle.
  - Reconcile its printed counts against `docs/SCHEMA.md` §1.2 and record the outcome (RULES §6
    item 7).
  - Commit the generated `configs/splits/split_assignment.csv`.
- Confirm indices 41 and 42 by unique-value counts, and verify the `{image_id}_BB.txt`
  bounding-box format.
- EDA notebook (the remaining Week 1 item).

### 2026-09-14: Repository scaffolding

**Done**
- Created the folder structure: `configs/`, `docs/`, `notebooks/`, `reports/`, `scripts/`,
  `src/antispoof/{data,models,training,eval,serving}`, `tests/`.
- Wrote `pyproject.toml`:
  - 16 runtime dependencies, unpinned, and a dev group (pytest, pytest-cov, ruff, mypy).
  - The `uv_build` backend.
  - Configuration for ruff (line length 100), pytest and mypy.
- Added `.python-version` (3.11) and a `.gitignore` covering `data/`, W&B output, model weights,
  virtualenvs, caches and credential files.
- Added `.githooks/commit-msg`, which strips AI attribution trailers. Checked it against a sample
  message: the disallowed lines were removed and other lines were kept. Added `.claude/settings.json`
  with attribution disabled.
- Added the import-only smoke test `tests/test_smoke.py`.
- Wrote `docs/`: PRD, ARCHITECTURE, DESIGN, SCHEMA, RULES, PROGRESS, EXPERIMENTS.
- Added the `/session-end` command, README.md and CLAUDE.md.

**Broke / not verified**
- Dependencies are not installed (by design this session), so `uv sync` and `uv run pytest` have not
  been run.

**Next**
- Run `uv sync` and `uv run pytest`.
- Check dataset license terms and how to get CelebA-Spoof onto Kaggle/Colab.
- Inspect the raw annotation format (label codes, spoof-type names, bbox convention).
- Decide official vs custom split.
- Implement the manifest builder and the subject-disjoint split with tests.

## Open questions

- ~~Is the official CelebA-Spoof train/test split subject-disjoint? This decides whether to reuse it
  or build a custom split. Decide before the Week 1 manifest.~~
  **Answered:** no.
  - Subjects `5028`, `7332` and `9735` are in both train and test (reproduced in-repo in the label
    files, SCHEMA.md §1.2).
  - Decision: test is kept unchanged, the shared subjects are removed from train only, and val is
    carved out of train by subject (ADR-009).
- ~~Exact raw annotation encoding: label codes, spoof-type and illumination/environment names, and
  the bbox coordinate convention. Verify before the Week 1 manifest.~~
  **Answered for the label vector:** 44 integers.
  - Index 40 (spoof type) is verified.
  - Index 43 (label, `0` = live, `1` = spoof) is verified.
  - Layout in SCHEMA.md §1.1.
  - The unverified parts are split into the questions below.
- ~~Meaning of annotation indices 41 and 42. They are provisionally illumination and environment, by
  documentation convention only. Confirm by unique-value counts on the mirror before the Week 3
  error analysis.~~
  **Answered:** index 41 is illumination (4 codes) and index 42 is environment (2 codes).
  - Measured by the owner on 2026-09-15 over the 329,921 official-train `spoof/` images.
  - Codes are 1-indexed. `0` means not applicable and occurs only on live rows.
  - Distributions are in SCHEMA.md §1.1.1, reproduced in-repo by the second Kaggle build on
    2026-09-15.
- Names for the index-40 spoof type codes. There are 10 codes (SCHEMA.md §1.1.1), but what any of
  them means is not known. The illumination and environment code names are not recorded either.
  Decide before the Week 3 error analysis.
- ~~Reconcile the externally measured index 40–42 distributions (SCHEMA.md §1.1.1) with the build
  report.~~
  **Answered:** the second Kaggle run of `scripts/build_manifest.py` on 2026-09-15 reproduced them
  in-repo, as reported by the owner.
- ~~Codes at indices 40–42 on the test split.~~
  **Answered:** printed by the same run (SCHEMA.md §1.1.1).
  - 47,247 test `spoof/` rows carry 10 / 4 / 2 distinct codes, and none has code `0`.
  - The distributions differ materially from train (covariate shift, PRD.md §8).
- ~~Should the build assert `validate_attack_codes`? Codes on the 2,022 conflicting train rows are
  still unmeasured. The build report counts only `spoof/` rows, so codes on live rows are unmeasured
  too. Decide before the Week 2 baseline.~~
  **Decided (owner, 2026-09-16):** every manifest build reports `validate_attack_codes` violation
  counts. The check becomes a hard assertion once a Kaggle build shows zero violations.
  - Not implemented: the build does not call the validator yet. The code change is a later session.
- ~~Live/spoof prior shift between val and test: the live fraction is 33.0% in train and 32.7% in
  val, but 29.7% in test (SCHEMA.md §1.2). A threshold calibrated on val is not guaranteed to keep
  its operating point on test.~~
  **Corrected:** APCER and BPCER are class-conditional and are not moved by class mix alone at a
  fixed threshold. The real risk is the covariate shift below.
- Covariate shift between the train and test attack populations (SCHEMA.md §1.1.1, PRD.md §8). The
  Week 2 evaluation design must:
  - (a) report val and test ACER always together, each with its split named;
  - (b) report confidence intervals, not bare point estimates, for per-condition test breakdowns
    (illumination code 3 has only 2,461 test spoof images);
  - (c) check any val/test ACER gap against the distribution difference before attributing it to
    the model.

  Decide the design before the Week 2 baseline.
- Bounding-box format.
  - One observed clue: a file named `004046_BB.txt` at the dataset root, which suggests per-image
    `{image_id}_BB.txt` sidecar files. Unverified.
  - Bounding boxes go into manifest v2 once confirmed. Verify before the first run with face crop.
- ~~Reconcile the externally measured counts in SCHEMA.md §1.2 with the output of the first Kaggle run
  of `scripts/build_manifest.py`. Do this before the Week 2 baseline.~~
  **Answered:** reproduced exactly by `scripts/build_manifest.py` on 2026-09-15. The only exception
  is the directory-listing checks, which the script does not perform and which remain externally
  measured.
- ~~Keys of the `manifest.meta.json` and `split_assignment.meta.yaml` sidecar files. Decide before the
  Week 2 baseline.~~
  **Closed (owner, 2026-09-16):** there are no sidecar files; a file's SHA-256 is its identity.
  - The split is identified by `split_sha256`, the SHA-256 of `configs/splits/split_assignment.csv`,
    in each run record (SCHEMA.md §2–3).
  - Both sidecars were removed from SCHEMA.md. Each run record stores the SHA-256 of every manifest
    it reads as `manifest_sha256` (implemented in `b210394`).
- Dataset license terms: can failure-gallery images appear in public reports? Can a hosted copy be
  used on Kaggle? Check before the Week 1 manifest.
- PRD success-metric targets (APCER, BPCER, ACER, BPCER@APCER=1%, model size, CPU latency). Decide
  before the first full-train run.
  - Reason for this deadline: the smoke-run numbers are not interpretable until the capture-source
    shortcut is checked, and targets must not be fitted to them.
- Backbone and input resolution. Decide before the Week 2 baseline.
  - The smoke run used `mobilenetv3_large_100` at 224 px (`configs/baseline.yaml`) as a starting
    default, because ARCHITECTURE.md lists the backbone as TBD. This is not a decision.
- ~~Python version compatibility with the Kaggle/Colab runtimes. Verify before the Week 2 baseline.~~
  **Decided (owner, 2026-09-16):**
  - Supported Python is 3.11–3.12.
  - Kaggle is the reference environment for every run that produces numbers. It uses Kaggle's
    preinstalled torch and timm via `PYTHONPATH=src` and never pip-installs the project, since that
    could replace Kaggle's CUDA-matched torch.
  - Each run record's `environment` block is the source of truth for versions. The local pins exist
    for tests.
  - The Kaggle runs on 2026-09-15 used Python 3.12.13, torch 2.10.0+cu128 and timm 1.0.26 (run
    records). `pyproject.toml` still requires Python >=3.11,<3.12, and `uv.lock` pins torch 2.14.0
    and timm 1.0.29, until `requires-python` is widened (a Next item).
- Capture-source shortcut: not yet ruled out (owner's note on the `20260915-153606-baseline` ledger
  row). Until it is, no baseline metric is a model-quality estimate.
  - **Header level (2026-09-16):** from header metadata alone, `20260916-075616-probe_metadata`
    reached pooled val ACER 0.00% (0/1,346, 0/654): a strong header-level shortcut under the
    pre-registered rule. The three questions below follow from it.
- Do the class-level differences in JPEG quantization tables and image dimensions
  (`reports/runs/20260916-075616-probe_metadata/probe_summary.json`) come from the original
  CelebA-Spoof release or from this Kaggle mirror?
- Does the CNN use these header-level traces? The probe shows they are available, not that they are
  used. The next session tests this.
- Pixel-level shortcuts remain untested.

## Blocked on

- Nothing currently blocked.
