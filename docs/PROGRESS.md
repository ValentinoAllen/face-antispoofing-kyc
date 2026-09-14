# Progress

## Current status

The repository is scaffolded:

- Folder structure, `pyproject.toml` (uv, Python 3.11, ruff, pytest, mypy config), a commit-msg hook
  and an import-only smoke test are in place.
- The PRD, architecture, design, schema, rules and experiment-ledger docs are written.
- There is no data, model or training code yet.
- Dependencies are not installed yet, so the smoke test has not been run.
- The next step is Week 1 data work: dataset access, manifest builder, and the subject-disjoint split
  with its invariant test.

## Milestones

- [ ] **Week 1 (2026-09-14 → 2026-09-20):** repo scaffold, dataset access, manifest builder,
  subject-disjoint split + test, EDA notebook
  - [x] Repo scaffold (2026-09-14)
  - [ ] Dataset access
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

- Is the official CelebA-Spoof train/test split subject-disjoint? This decides whether to reuse it or
  build a custom split. Decide before the Week 1 manifest.
- Exact raw annotation encoding: label codes, spoof-type and illumination/environment names, and the
  bbox coordinate convention. Verify before the Week 1 manifest.
- Dataset license terms: can failure-gallery images appear in public reports? Can a hosted copy be
  used on Kaggle? Check before the Week 1 manifest.
- PRD success-metric targets (APCER, BPCER, ACER, BPCER@APCER=1%, model size, CPU latency). Decide
  before the Week 2 baseline.
- Backbone and input resolution. Decide before the Week 2 baseline.
- Python version compatibility with the Kaggle/Colab runtimes. Verify before the Week 2 baseline.

## Blocked on

- Nothing currently blocked.
