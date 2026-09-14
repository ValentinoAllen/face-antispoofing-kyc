# Progress

## Current status

The Week 1 manifest builder and subject-disjoint split have run on the real Kaggle mirror, and the
resulting split assignment is committed. The EDA notebook is the remaining Week 1 item.
- **Code:** `antispoof.data` holds the verified annotation layout (indices 40 spoof type, 41
  illumination, 42 environment, 43 label; codes at 40–42 are 1-indexed with `0` = not applicable),
  the validator `validate_attack_codes`, the config loader, the manifest builder, the subject-level
  split with its invariants, and the build orchestration. The build report now also prints per-code
  counts for indices 40–42.
- **Checks:** ruff, mypy and `uv run pytest` (104 tests) pass locally.
- **Dataset:** `scripts/build_manifest.py` reproduced every count in `docs/SCHEMA.md` §1.2 on Kaggle
  on 2026-09-15 and produced the split: train 442,859 rows / 7,370 subjects, val 49,308 / 819, test
  67,170 / 1,004. `configs/splits/split_assignment.csv` is committed and tested.
- **Unverified:** the index 40–42 distributions are externally measured, and the build report that
  prints them has not yet run on Kaggle. The directory-listing checks in SCHEMA §1.2 are still
  externally measured. Manifests on Kaggle carry the old `attr_41`/`attr_42` column names.
- **Known risks:** illumination code 1 is 59% of official-train spoof images, and the live fraction
  is 29.7% in test vs 33.0% train / 32.7% val (`docs/PRD.md` §8).
- **Next step:** rerun `scripts/build_manifest.py` on Kaggle to regenerate the manifests under the new
  column names and reconcile the printed code counts with SCHEMA §1.1.1; then the EDA notebook.

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
- [ ] **Week 3 (2026-09-28 → 2026-10-04):** evaluation report generator, error analysis v1
- [ ] **Week 4 (2026-10-05 → 2026-10-11):** hypothesis-driven model iteration
- [ ] **Week 5 (2026-10-12 → 2026-10-18):** model freeze, threshold selection, confidence definition
- [ ] **Week 6 (2026-10-19 → 2026-10-25):** quantization, ONNX export, parity check, CPU latency
  benchmark
- [ ] **Week 7 (2026-10-26 → 2026-11-01):** FastAPI service, face/quality gating, demo page, API tests
- [ ] **Week 8 (2026-11-02 → 2026-11-08):** one-shot test evaluation, final report, README results,
  write-up

## Session log

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
  - Distributions are in SCHEMA.md §1.1.1: externally measured, to be reconciled on the next build.
- Names for the index-40 spoof type codes. There are 10 codes (SCHEMA.md §1.1.1), but what any of
  them means is not known. The illumination and environment code names are not recorded either.
  Decide before the Week 3 error analysis.
- Should the build assert `validate_attack_codes`? Codes on the 2,022 conflicting train rows and on
  the test split have not been measured. Decide before the Week 2 baseline.
- Live/spoof prior shift between val and test: the live fraction is 33.0% in train and 32.7% in val,
  but 29.7% in test (SCHEMA.md §1.2).
  - The shift comes from the official test split, which is not modified.
  - A threshold calibrated on val is not guaranteed to keep its operating point on test (PRD.md §8).
  - Eval must report the operating point it was calibrated at and the split it was calibrated on.
  - How to handle it is part of the Week 2 evaluation design; no fix is chosen.
- Bounding-box format.
  - One observed clue: a file named `004046_BB.txt` at the dataset root, which suggests per-image
    `{image_id}_BB.txt` sidecar files. Unverified.
  - Bounding boxes go into manifest v2 once confirmed. Verify before the Week 2 baseline.
- ~~Reconcile the externally measured counts in SCHEMA.md §1.2 with the output of the first Kaggle run
  of `scripts/build_manifest.py`. Do this before the Week 2 baseline.~~
  **Answered:** reproduced exactly by `scripts/build_manifest.py` on 2026-09-15. The only exception
  is the directory-listing checks, which the script does not perform and which remain externally
  measured.
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
