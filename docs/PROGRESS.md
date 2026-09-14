# Progress

## Current status

The Week 1 data pipeline code exists and passes its local checks, but it has not yet run on real
data.
- **Code:** `antispoof.data` now contains:
  - the annotation-vector layout (`labels.py`);
  - the data config loader for `configs/data.yaml` (`config.py`);
  - the manifest builder with the `exclude`/`trust_label`/`trust_path` conflict policies
    (`manifest.py`);
  - the subject-level train/val split, the disjointness invariant and the test-split tripwire
    (`splits.py`);
  - the orchestration behind `scripts/build_manifest.py` (`build.py`).
- **Checks:** the synthetic-fixture tests, ruff and mypy pass locally. The build script has run end
  to end on a synthetic dataset only.
- **Dataset:** the owner has checked the CelebA-Spoof mirror on Kaggle. Its two defects are recorded
  in `docs/SCHEMA.md` §1.2 and ADR-009: 3 subjects shared between official train and test, and 2,022
  train images under `live/` carrying a spoof label. These counts are externally measured and not
  yet reproduced in-repo.
- **Missing:** `configs/splits/split_assignment.csv` does not exist yet.
- **Next step:** run `scripts/build_manifest.py` on Kaggle, reconcile its printed counts against
  `docs/SCHEMA.md` §1.2, and commit the resulting split assignment.

## Milestones

- [ ] **Week 1 (2026-09-14 → 2026-09-20):** repo scaffold, dataset access, manifest builder,
  subject-disjoint split + test, EDA notebook
  - [x] Repo scaffold (2026-09-14)
  - [x] Dataset access (2026-09-14; CelebA-Spoof mirror on Kaggle checked by the owner, ADR-008)
  - [ ] Manifest builder
  - [ ] Subject-disjoint split + invariant test
  - [ ] EDA notebook
- [ ] **Week 2 (2026-09-21 → 2026-09-27):** PRD targets set, PAD metrics + tests, transforms,
  dataset class, config loader, baseline training run
- [ ] **Week 3 (2026-09-28 → 2026-10-04):** evaluation report generator, error analysis v1
- [ ] **Week 4 (2026-10-05 → 2026-10-11):** hypothesis-driven model iteration
- [ ] **Week 5 (2026-10-12 → 2026-10-18):** model freeze, threshold selection, confidence definition
- [ ] **Week 6 (2026-10-19 → 2026-10-25):** quantization, ONNX export, parity check, CPU latency
  benchmark
- [ ] **Week 7 (2026-10-26 → 2026-11-01):** FastAPI service, face/quality gating, demo page, API tests
- [ ] **Week 8 (2026-11-02 → 2026-11-08):** one-shot test evaluation, final report, README results,
  write-up

## Session log

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
  - Subjects `5028`, `7332` and `9735` are in both train and test (externally measured, SCHEMA.md
    §1.2).
  - Decision: test is kept unchanged, the shared subjects are removed from train only, and val is
    carved out of train by subject (ADR-009).
- ~~Exact raw annotation encoding: label codes, spoof-type and illumination/environment names, and
  the bbox coordinate convention. Verify before the Week 1 manifest.~~
  **Answered for the label vector:** 44 integers.
  - Index 40 (spoof type) is verified.
  - Index 43 (label, `0` = live, `1` = spoof) is verified.
  - Layout in SCHEMA.md §1.1.
  - The unverified parts are split into the questions below.
- Meaning of annotation indices 41 and 42. They are provisionally illumination and environment, by
  documentation convention only. Confirm by unique-value counts on the mirror before the Week 3
  error analysis.
- Names for the index-40 spoof type codes. Decide before the Week 3 error analysis.
- Bounding-box format.
  - One observed clue: a file named `004046_BB.txt` at the dataset root, which suggests per-image
    `{image_id}_BB.txt` sidecar files. Unverified.
  - Bounding boxes go into manifest v2 once confirmed. Verify before the Week 2 baseline.
- Reconcile the externally measured counts in SCHEMA.md §1.2 with the output of the first Kaggle run
  of `scripts/build_manifest.py`. Do this before the Week 2 baseline.
- Keys of the `manifest.meta.json` and `split_assignment.meta.yaml` sidecar files. Decide before the
  Week 2 baseline.
- Dataset license terms: can failure-gallery images appear in public reports? Can a hosted copy be
  used on Kaggle? Check before the Week 1 manifest.
- PRD success-metric targets (APCER, BPCER, ACER, BPCER@APCER=1%, model size, CPU latency). Decide
  before the Week 2 baseline.
- Backbone and input resolution. Decide before the Week 2 baseline.
- Python version compatibility with the Kaggle/Colab runtimes. Verify before the Week 2 baseline.

## Blocked on

- Nothing currently blocked.
