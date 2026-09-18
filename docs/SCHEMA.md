# Data Contracts

These are the contracts between pipeline stages. Any change to a contract must update this file in
the same commit as the code change. Code that produces or consumes these artifacts validates them
against this file.

Conventions: dtypes are pandas/NumPy dtypes for CSV artifacts and JSON types for JSON artifacts.
"Nullable = no" means an empty value is a validation error.

## 1. Image manifest (manifest v1)

- **Paths:** `data/manifests/manifest_train.csv`, `manifest_val.csv` and `manifest_test.csv`. They
  are gitignored, never committed, and regenerated on Kaggle (`ARCHITECTURE.md` ADR-008).
- **Format:** UTF-8 CSV with a header row and one row per image, sorted by `image_path`.
- **Produced by:** `scripts/build_manifest.py` → `antispoof.data.build.run`, configured by
  `configs/data.yaml`.

### 1.1 Source: CelebA-Spoof label files

- **Mirror:**
  `/kaggle/input/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/CelebA_Spoof_/CelebA_Spoof`
- **Label files used:** `metas/intra_test/train_label.json` and `metas/intra_test/test_label.json`.
  - The `train_label.txt` and `test_label.txt` files in the same directory are not used.
  - Other protocols available under `metas/`: `protocol1/` and
    `protocol2/{test_on_high_quality_device,test_on_middle_quality_device,test_on_low_quality_device}/`.
    They are not used for now.
- **Structure:** each label file is a JSON object.
  - Keys are relative image paths of the form `Data/{train,test}/{subject_id}/{live,spoof}/{filename}`.
    The builder parses them by locating the `train`/`test` component, not by a fixed index.
  - Values are lists of exactly **44** integers. The builder raises on any other length.
  - The paper's attribute count suggests 43, but the vector has a 44th entry, which is the label.

The index layout is defined in code only in `antispoof.data.labels`.

| Index | Meaning | Status | Notes | Manifest column |
|---|---|---|---|---|
| 0–39 | 40 CelebA face attributes | Verified | Populated only for live images; all zero for spoof images | not in v1 |
| 40 | Spoof type code | Verified | 1-indexed code, 10 distinct values on spoof images (§1.1.1). `0` = not applicable, live images only. Names for the codes are not recorded yet | `spoof_type` |
| 41 | Illumination condition | Verified | 1-indexed code, 4 distinct values on spoof images (§1.1.1). `0` = not applicable, live images only. Names for the codes are not recorded | `illumination` |
| 42 | Environment | Verified | 1-indexed code, 2 distinct values on spoof images (§1.1.1). `0` = not applicable, live images only. Names for the codes are not recorded | `environment` |
| 43 | Live/spoof label: `0` = live, `1` = spoof. The training target | Verified | On the test split it agrees with the `live/` vs `spoof/` path segment for all 67,170 entries | `label` |

**Code convention for indices 40–42.**

- The codes are **1-indexed** categories.
- `0` is not a category. It means "not applicable" and occurs only on live images
  (`labels.CODE_NOT_APPLICABLE`).
- A raw code is never a zero-based class index. Any class index must be derived from the code
  explicitly.
- `antispoof.data.labels.validate_attack_codes` checks one vector. On a spoof vector (index 43 ==
  1) it rejects `0` at any of indices 40–42. On a live vector it rejects any non-zero value there.
- The builder does not call the validator yet. The build report (§1.1.1) shows that no `spoof/` row
  in official train (after the conflict policy) or in test has code `0` at indices 40–42. Two
  populations are still unmeasured:
  - the 2,022 conflicting train rows (§1.2), which `exclude` drops before the codes are counted;
  - live rows in either split, because the report counts only `spoof/` rows.

  Asserting the convention at build time could therefore still abort a build on rows it was never
  checked against. **Decision (owner, 2026-09-16):** every manifest build reports the violation
  counts, and the check becomes a hard assertion once a Kaggle build shows zero violations. Neither
  is implemented yet (`PROGRESS.md`).

### 1.1.1 Measured code distributions (indices 40–42)

> **Reproduced in-repo on 2026-09-15 by `scripts/build_manifest.py`.**
> - The official-train counts were first measured by the owner on Kaggle on 2026-09-15, against the
>   mirror and the `intra_test` `train_label.json` above.
> - The owner's second Kaggle run of `python scripts/build_manifest.py --config configs/data.yaml`,
>   on 2026-09-15, printed these counts under "Attack codes on rows under spoof/, after conflict
>   policy". As reported by the owner, its official-train counts matched every count previously
>   recorded here. The same run printed the test counts for the first time (`RULES.md` §6 item 7).
> - Population: rows whose path contains `spoof/`, after `conflict_policy: exclude`, before val is
>   carved out. For train this is the same 329,921 images as the original measurement, because all
>   2,022 train conflicts are stored under `live/` (§1.2).

**Official train** (329,921 `spoof/` rows):

| Index | Column | Distinct codes | Images per code |
|---|---|---|---|
| 40 | `spoof_type` | 10 | 1: 35,547 · 2: 31,221 · 3: 31,776 · 4: 33,647 · 5: 30,167 · 6: 33,285 · 7: 29,050 · 8: 33,085 · 9: 31,527 · 10: 40,616 |
| 41 | `illumination` | 4 | 1: 194,589 · 2: 66,342 · 3: 35,101 · 4: 33,889 |
| 42 | `environment` | 2 | 1: 251,685 · 2: 78,236 |

**Official test** (47,247 `spoof/` rows):

| Index | Column | Distinct codes | Images per code |
|---|---|---|---|
| 40 | `spoof_type` | 10 | 1: 3,600 · 2: 5,421 · 3: 6,083 · 4: 4,287 · 5: 6,097 · 6: 3,530 · 7: 6,477 · 8: 3,659 · 9: 4,483 · 10: 3,610 |
| 41 | `illumination` | 4 | 1: 35,119 · 2: 5,971 · 3: 2,461 · 4: 3,696 |
| 42 | `environment` | 2 | 1: 40,722 · 2: 6,525 |

- Each train distribution sums to 329,921 and each test distribution to 47,247. No `spoof/` row in
  either population has code `0`.
- **Covariate shift between train and test.** Shares of `spoof/` rows, computed from the counts
  above:
  - `spoof_type` is near-uniform in train but not in test: code 7 is 8.8% of train vs 13.7% of
    test; code 10 is 12.3% vs 7.6%.
  - `illumination` is more concentrated in test: code 1 is 59.0% vs 74.3%; code 3 is 10.6% vs 5.2%.
  - `environment` likewise: code 1 is 76.3% vs 86.2%.
- **Small cells on test:** illumination code 3 has only 2,461 test spoof images. Per-condition test
  results need confidence intervals.
- Consequences for evaluation are recorded in `PRD.md` §8.
- Val is carved from official train by subject (§2). The build does not print val's own code
  distribution.

### 1.2 Measured counts (`intra_test`, this mirror)

> **Reproduced in-repo on 2026-09-15 by `scripts/build_manifest.py`.**
> - Originally measured by the owner on Kaggle, 2026-09-14, against the mirror and `intra_test`
>   label files above, including the `Data/train` and `Data/test` directory listings.
> - The owner ran `python scripts/build_manifest.py --config configs/data.yaml` on Kaggle on
>   2026-09-15. Its printed counts matched every count previously recorded in this section exactly
>   (`RULES.md` §6 item 7).
> - Exception: the script reads only the label files. The two checks against the directory
>   listings (shared subjects on disk; label files vs directories) remain **externally measured,
>   not yet reproduced in-repo**.

Official splits, counted by the `live/` vs `spoof/` **path segment**:

| split | images  | subjects | live    | spoof   |
|-------|---------|----------|---------|---------|
| train | 494,405 | 8,192    | 164,484 | 329,921 |
| test  |  67,170 | 1,004    |  19,923 |  47,247 |

The same splits, counted by **index 43**:

| split | images  | live    | spoof   |
|-------|---------|---------|---------|
| train | 494,405 | 162,462 | 331,943 |
| test  |  67,170 |  19,923 |  47,247 |

- **Conflicts** (path segment says live, index 43 says spoof): 2,022 in train, zero in test.
- **Train after `conflict_policy: exclude`:** 492,383 rows (live 162,462, spoof 329,921).
- **Shared subjects:** train and test share subjects `5028`, `7332` and `9735`.
  - Confirmed in the label files (reproduced in-repo) and on disk (externally measured, not yet
    reproduced in-repo).
  - No image path appears in both splits.
- **Label files vs directories** (externally measured, not yet reproduced in-repo): the listings
  under `Data/train` and `Data/test` match the label files exactly. No subject is in a label file
  without a folder, and no folder lacks label entries.
- **Observed discrepancy:**
  - This mirror contains 561,575 images across 9,193 unique subjects.
  - The CelebA-Spoof paper reports 625,537 images and 10,177 subjects.
  - This is recorded as an observation about this mirror, not as a corrected figure. `intra_test`
    may not span the full dataset.

**Produced split.** Printed by the same 2026-09-15 run of `scripts/build_manifest.py`, with
`conflict_policy=exclude`, `val_fraction=0.1`, `seed=42` and `stratify_bins=10`:

| split | rows    | subjects | live    | spoof   | live fraction |
|-------|---------|----------|---------|---------|---------------|
| train | 442,859 | 7,370    | 146,280 | 296,579 | 33.0%         |
| val   |  49,308 |   819    |  16,123 |  33,185 | 32.7%         |
| test  |  67,170 | 1,004    |  19,923 |  47,247 | 29.7%         |

- Official train after the conflict policy, before subject exclusion: 492,383 rows across 8,192
  subjects. The 3 excluded subjects account for the 216-row difference between that figure and
  train + val.
- The live fraction of train and val agrees, so the stratification works. The lower live fraction in
  test is a property of the official test split, which is not modified. APCER and BPCER are
  class-conditional, so this class-mix difference alone does not move them at a fixed threshold.
  The evaluation risk is the difference in attack population between train and test (§1.1.1,
  `PRD.md` §8).
- The subject assignment is committed as `configs/splits/split_assignment.csv` (§2).

### 1.3 Columns

| Column | dtype | Nullable | Constraint |
|---|---|---|---|
| `image_path` | string | no | The label-file key: POSIX path relative to the dataset root. Unique across the three manifests |
| `subject_id` | string | no | Path component after `train`/`test`. Stored as a string so leading zeros survive |
| `split` | category (string) | no | `train`, `val` or `test`. Test rows keep the official split; val is assigned per subject (§2) |
| `label` | int8 | no | `0` = live (bona fide), `1` = spoof (attack). From index 43, or from `path_kind` under `trust_path` |
| `spoof_type` | int32 | no | Raw index-40 code, 1-indexed; `0` (not applicable) only on live rows (§1.1). Names for the codes are TBD — decide before Week 3 error analysis |
| `illumination` | int32 | no | Raw index-41 code, 1-indexed; `0` (not applicable) only on live rows (§1.1) |
| `environment` | int32 | no | Raw index-42 code, 1-indexed; `0` (not applicable) only on live rows (§1.1) |
| `path_kind` | category (string) | no | `live` or `spoof`: the path component after `subject_id` |
| `conflict` | bool | no | `True` when `path_kind` disagrees with index 43, in either direction |

**Conflict policy** (`data.conflict_policy`, `ARCHITECTURE.md` ADR-009):

| Value | Effect on conflicting rows |
|---|---|
| `exclude` (default) | Dropped |
| `trust_label` | Kept; `label` comes from index 43 |
| `trust_path` | Kept; `label` comes from `path_kind` |

- Every build logs how many conflicting rows it found, dropped and relabelled.
- The build refuses (`ProtectedTestSplitError`) any policy that would drop or relabel a test row.
- These rows are described as **conflicting**, because which side is wrong has not been established.

Row-level invariants:

- `conflict` is computed from index 43 before the policy is applied. Under `trust_path`, a row can
  therefore have `conflict == True` while `label` matches `path_kind`.
- Under `exclude`, no row has `conflict == True`.
- Where `conflict == False`: `label == 0` ⇔ `path_kind == "live"`.
- Expected from the §1.1 code convention, but not asserted by the builder: on rows with
  `conflict == False`, `label == 0` ⇒ `spoof_type == illumination == environment == 0`, and
  `label == 1` ⇒ all three are ≥ 1.
- The 40 CelebA face attributes are **not** in manifest v1. If they are needed for slicing, they go
  in a separate table keyed by `image_path`.

Manifest-level invariants:

- `image_path` is unique across the three files. This holds by construction: label-file keys are
  unique, and train and test paths differ in their split component.
- Every `subject_id` maps to exactly one `split` across the three files. `validate_splits` checks
  this on the stacked manifests at build time (§2).

### 1.4 Bounding boxes (not in manifest v1)

- Manifest v1 has **no** bounding-box columns, not even empty placeholders.
- The bounding-box format is an open question. One observed clue: a file named `004046_BB.txt` at
  the dataset root, which suggests per-image `{image_id}_BB.txt` sidecar files. This is unverified.
- Bounding-box columns will be added in manifest v2, once the format is confirmed.

### 1.5 Normalized image cache

- **Path:** a cache root outside the repository, e.g. `/kaggle/working/cache`. Like the manifests it
  is large, gitignored and rebuilt on Kaggle (ADR-008).
- **Produced by:** `scripts/build_cache.py` → `antispoof.data.cache_build.build_cache`, configured by
  `configs/cache_v1.yaml`.
- **Read by:** `antispoof.data.cache`, which is the contract below and holds no encoding logic.
- **Purpose:** every row, live or spoof, is stored at one size, one JPEG quality and one
  subsampling, so the header-level class signature measured in `20260916-075616-probe_metadata` is
  not in the pixels a cached run reads.

Cached images mirror the source `image_path` layout (§1.3), with the **manifest** split as the first
component, so a live and a spoof file that share a basename cannot collide:

```
<cache-root>/<split>/<subject_id>/<live|spoof>/<filename>
```

The file always holds JPEG bytes, under the source filename. The build counts any cached file whose
source suffix is not `.jpg` or `.jpeg` and reports the count.

**`cache_manifest_<split>.csv`** — UTF-8 CSV with a header row, one row per successfully cached
image, in manifest order. Its first nine columns are manifest v1 (§1.3) copied unchanged, then:

| Column | dtype | Nullable | Constraint |
|---|---|---|---|
| `cached_path` | string | no | POSIX path of the cached image, relative to the cache root, so a cache can be moved without rewriting its manifests |
| `source_sha256` | string | no | SHA-256 hex of the source image's bytes. A rerun skips a row whose cached file exists and whose source still hashes to this value, which is what makes the build resumable |

A row whose source cannot be read is counted and listed in `cache_summary.json` and has no row here,
so a cache manifest can be shorter than its source manifest.

**`cache_summary.json`** — one JSON object at the cache root.

| Field | JSON type | Nullable | Constraint |
|---|---|---|---|
| `name` | string | no | `cache.name`; the cache's identity, stored in every run record that reads it |
| `created_at` | string | no | ISO 8601 UTC |
| `config_path` | string | no | Repo-relative path under `configs/` |
| `config_hash` | string | no | SHA-256 hex of the resolved `{experiment, data}` config, as in §3 |
| `git_sha`, `git_dirty` | string, boolean | no | Git state of the build |
| `environment.python`, `environment.platform`, `environment.pillow` | string | no | Pillow decodes and re-encodes every image; no model is built, so there is no torch entry |
| `settings` | object | no | The resolved `cache:` section: `name`, `target_size`, `quality`, `subsampling`, `resample_filter`, `face_crop` |
| `layout` | string | no | States that the cached path mirrors the source `image_path` |
| `cached_path` | string | no | States that `cached_path` is POSIX and relative to the cache root |
| `source_manifest_sha256` | object | no | Map from each source manifest's file name to the SHA-256 hex of its bytes. A run refuses a cache whose values differ from the manifests it reads |
| `limit` | integer | yes | `--limit`, or null. A cache built with a limit covers only part of each split, and a run that needs a missing row fails |
| `splits.<split>` | object | no | `{rows, cached, skipped, failed, bytes, non_jpeg_suffix, wall_time_s, cache_manifest, cache_manifest_sha256, verification}` |
| `splits.<split>.verification` | object | no | `{sample_rows, checked, seed}`: a seeded sample of cached files whose headers were checked against the target tables and size. A mismatch fails the build |
| `failures` | array[object] | no | One `{image_path, split, error}` per source image that could not be read |

## 2. Split assignment file

- **Path:** `configs/splits/split_assignment.csv`.
  - It holds subject identifiers only (no images or personal data), and it is committed so the split
    is reproducible.
  - **Committed.** Produced by the owner's Kaggle run of `scripts/build_manifest.py` on 2026-09-15:
    train 7,370, val 819 and test 1,004 subjects (§1.2).
- **Format:** UTF-8 CSV with a header row and one row per subject, sorted by `subject_id`.
- **Produced by:** `antispoof.data.build.run`.
- **Loaded with:** `antispoof.data.splits.load_split_assignment`, which validates the file.

| Column | dtype | Nullable | Constraint |
|---|---|---|---|
| `subject_id` | string | no | Unique within the file; present in exactly one manifest |
| `split` | category (string) | no | `train`, `val` or `test` |

How the split is built (ADR-009, settings in `configs/data.yaml`):

- **Test:** the official test split, unchanged.
- **Val:** a fraction `val_fraction` of the remaining official-train subjects.
  - Assigned per subject, seeded with `seed`.
  - Stratified by each subject's spoof-image fraction into `stratify_bins` equal-count strata.
- **Train:** the remaining subjects.
- **Excluded subjects:** listed only as `test`.

The file is the split's identity. It has no sidecar: each run record stores the file's SHA-256 as
`split_sha256` (§3).

### Invariant: subject disjointness

Definitions:

- `R`: subjects in the official train manifest after the conflict policy.
- `T`: subjects in the official test manifest.
- `E`: `data.excluded_subjects` = {`5028`, `7332`, `9735`}.
- `S_train`, `S_val`, `S_test`: the subjects assigned to each split.

```
S_test           = T
S_train ∪ S_val  = R \ E
S_train ∩ S_val  = ∅
S_train ∩ S_test = ∅
S_val   ∩ S_test = ∅
```

The official splits share exactly the subjects in `E` (§1.2). The last two equations therefore hold
only because `E` is removed from train and val, and never from test.

Enforcement:

1. `antispoof.data.splits.validate_splits` is the single source of truth for disjointness.
   - It asserts that the three pairwise intersections are empty and that no subject in `E` is in
     train or val.
   - On failure it raises `SplitLeakageError` listing the offending subject ids.
   - It is called at build time on the assignment and on the stacked manifests, and by
     `load_split_assignment`. Dataset loaders must load splits through `load_split_assignment`.
2. `validate_coverage` asserts the first two equations at build time.
3. `ensure_test_untouched` refuses any build whose conflict policy would drop or relabel a test row.
4. `tests/test_splits.py` covers these functions on synthetic fixtures. It includes subjects with a
   single image, tiny datasets, shared subjects, determinism for a fixed seed, and stratification
   tolerance. It also loads the committed `configs/splits/split_assignment.csv` and asserts that its
   three subject sets are pairwise disjoint and that the excluded subjects are only in test.
5. Each run record stores `split_sha256`, so the split behind every reported number can be identified.

## 3. Experiment record

- **Path:** `<output-dir>/<run_id>/record.json`, one file per run. `scripts/train.py` (through
  `antispoof.training.run`) writes it on Kaggle, next to `resolved_config.json`, `checkpoint.pt` and
  `predictions.csv`.
  - The record is first written with `status: running` and rewritten when the run ends.
  - **Committed copy:** the owner copies `record.json` and `resolved_config.json` into
    `reports/runs/<run_id>/` and commits them together with the run's `EXPERIMENTS.md` row. A run
    that writes an experiment-specific summary commits that file too: `probe_summary.json`
    (`scripts/probe_metadata.py`), `counterfactual_summary.json` (`scripts/counterfactual_eval.py`)
    or `audit_summary.json` (`scripts/audit_jpeg_tables.py`).
  - `checkpoint.pt` and `predictions.csv` are never committed. A run meant to be kept is saved as a
    Kaggle notebook version, so `/kaggle/working` persists.
  - When `wandb.enabled` is true, the resolved config and the non-null metrics are also logged to
    W&B.
- **Format:** JSON object.
- A key marked "present only when …" is absent otherwise. Every other key is always present.

| Field | JSON type | Nullable | Constraint |
|---|---|---|---|
| `run_id` | string | no | `YYYYMMDD-HHMMSS-<slug>` in UTC; unique |
| `created_at` | string | no | ISO 8601 UTC |
| `hypothesis` | string | no | One sentence; copied to `EXPERIMENTS.md` |
| `what_changed` | string | no | From `run.what_changed`; copied to the "what changed" column of `EXPERIMENTS.md` |
| `config_path` | string | no | Repo-relative path under `configs/` |
| `config_hash` | string | no | SHA-256 hex of the fully resolved config serialized with sorted keys |
| `git_sha` | string | no | 40-character hex of `HEAD` at launch |
| `git_dirty` | boolean | no | `true` if the working tree had uncommitted changes. Such runs cannot be cited as results |
| `seed` | integer | no | ≥ 0; the same value as in the config |
| `split_name` | string | no | Matches a file in `configs/splits/` |
| `split_sha256` | string | no | SHA-256 hex of the split assignment file `configs/splits/split_assignment.csv` (`data.split_assignment_path`). That file is the split's identity (§2) |
| `manifest_sha256` | object | no | Map from the file name of each manifest the run reads to the SHA-256 hex of its bytes. A run that trains or evaluates a model reads `manifest_train.csv` and `manifest_val.csv`; the full-split JPEG audit (`scripts/audit_jpeg_tables.py`) also reads `manifest_test.csv`, headers only, and hashes all three. Catches a changed manifest even when `git_sha` and `split_sha256` are unchanged |
| `environment.python` | string | no | Python version |
| `environment.torch` | string | no | `torch.__version__` |
| `environment.cuda` | string | yes | CUDA version torch was built with; null on CPU-only torch builds |
| `environment.device` | string | no | Device the run used |
| `environment.platform` | string | no | `platform.platform()` |
| `environment.timm` | string | no | `timm.__version__` |
| `environment.pillow` | string | no | `PIL.__version__`. Pillow decodes training images and reads the headers in the metadata probe |
| `environment.sklearn` | string | no | `sklearn.__version__`. scikit-learn fits the metadata probe's classifiers |
| `environment.deterministic_algorithms` | string | no | Description of the determinism settings applied by `antispoof.training.reproducibility.seed_everything` |
| `status` | string | no | `running`, `completed`, `failed` or `aborted`. `running` is written at launch; the other values when the run ends |
| `error` | string | no | Present only when `status` is `failed` or `aborted`. `<ExceptionType>: <message>` of the exception that ended the run |
| `data_subsets` | object | no | One key per split the run read, each `{rows, subjects, live, spoof}` as integers: counts of the subset the run actually used. The full-split JPEG audit draws no subset, so its values are the full manifest counts of each split it read |
| `training_epochs` | array[object] | yes | Present only when `status` is `completed`; null for runs that do not train (the metadata probe, the counterfactual evaluation, the JPEG audit). One item per epoch: `{steps: integer, images: integer, mean_loss: number, wall_time_s: number, images_per_s: number}`. Wall time includes data loading |
| `metrics_arm` | string | no | Present only when a run evaluates another run's checkpoint under several arms (`scripts/counterfactual_eval.py`). Names the arm whose pooled metrics fill `metrics` (`reencode_live_quality`); every arm's metrics are in `counterfactual_summary.json` |
| `source_run.run_id` | string | no | Present only when a run evaluates another run's checkpoint (`scripts/counterfactual_eval.py`). `run_id` of the evaluated run (`--run-dir`) |
| `source_run.config_path` | string | no | Present only with `source_run`. The evaluated run's `config_path`; must be the committed baseline config |
| `source_run.config_hash` | string | no | Present only with `source_run`. The evaluated run's `config_hash`; equals the hash of its `resolved_config.json` and of the committed baseline config resolved with this run's data paths |
| `source_run.checkpoint` | string | no | Present only with `source_run`. Path of the checkpoint that was read. This run writes no checkpoint, so `artifacts.checkpoint` is null |
| `source_run.checkpoint_sha256` | string | no | Present only with `source_run`. SHA-256 hex of the checkpoint file's bytes |
| `cache.name` | string | no | Present only when the experiment config sets `cache` (`configs/baseline_cache.yaml`). The cache's identity, which equalled the `name` in its `cache_summary.json` (§1.5). A directory alone does not identify a cache |
| `cache.dir` | string | no | Present only with `cache`. The cache root the run read, after the `scripts/train.py --cache-dir` override |
| `cache.settings` | object | no | Present only with `cache`. The `settings` block of that cache's `cache_summary.json`, copied verbatim: how the images the run trained on were built |
| `cache.summary_sha256` | string | no | Present only with `cache`. SHA-256 hex of `cache_summary.json` |
| `cache.cache_manifest_sha256` | object | no | Present only with `cache`. Map from each cache manifest's file name to the SHA-256 hex of its bytes. The run refuses a cache whose `source_manifest_sha256` differs from its own `manifest_sha256`, or that does not cover every subset row |
| `eval_split` | string | yes | `val` or `test`; null until evaluated |
| `threshold` | number | yes | Operating threshold used for the metrics, in [0, 1] |
| `threshold_rule` | string | yes | How the threshold was chosen (e.g. `apcer_on_val`) |
| `metrics.apcer_max` | number | yes | In [0, 1]; maximum over PAI species |
| `metrics.apcer_per_species` | object | yes | map species → number in [0, 1] |
| `metrics.bpcer` | number | yes | In [0, 1] |
| `metrics.acer` | number | yes | In [0, 1]; equals `(apcer_max + bpcer) / 2` |
| `metrics.bpcer_at_apcer_1pct` | number | yes | In [0, 1] |
| `metrics.n_bona_fide` | integer | yes | Count of bona fide samples evaluated |
| `metrics.n_attack` | integer | yes | Count of attack samples evaluated |
| `metrics.apcer_pooled` | number | yes | In [0, 1]; `n_attack_accepted / n_attack`, pooled over all PAI species. Not the ISO/IEC 30107-3 APCER, which is `apcer_max` |
| `metrics.acer_pooled` | number | yes | In [0, 1]; equals `(apcer_pooled + bpcer) / 2` |
| `metrics.n_attack_accepted` | integer | yes | Attack samples classified as bona fide (score < `threshold`) |
| `metrics.n_bona_fide_rejected` | integer | yes | Bona fide samples classified as attacks (score >= `threshold`) |
| `artifacts.checkpoint` | string | yes | Path where the run wrote the checkpoint (`<output-dir>/<run_id>/checkpoint.pt`), or a W&B artifact reference. Never committed |
| `artifacts.onnx` | string | yes | Path or W&B artifact reference |
| `artifacts.report_dir` | string | yes | `reports/<run_id>/` |
| `artifacts.wandb_url` | string | yes | URL of the W&B run |
| `notes` | string | yes | Free text |

Metrics are stored as fractions in [0, 1]. Docs and reports display them as percentages.

A run whose experiment config has no `cache` section resolves, and therefore hashes, exactly as it
did before that section existed: `cache` is left out of `resolved_config.json` entirely, not written
as null. Every record in `reports/runs/` predates the section and stays reproducible.

A run that measures the data rather than a model leaves every `metrics` value null and gets no
`EXPERIMENTS.md` row. `scripts/audit_jpeg_tables.py` is the only such run today; it prints that fact
as the last line of its report instead of a ledger row.

The baseline computes pooled rates only. It fills `bpcer`, the four counts, `apcer_pooled` and
`acer_pooled`, and leaves `apcer_max`, `apcer_per_species`, `acer` and `bpcer_at_apcer_1pct` null.

## 4. API JSON schemas

### 4.1 Request: `POST /v1/predict`

`multipart/form-data`:

| Part | Type | Nullable | Constraint |
|---|---|---|---|
| `image` | binary file | no | `image/jpeg` or `image/png`; maximum size TBD — decide before Week 7 serving |
| `include_explanation` | boolean (form string `true`/`false`) | no (has default) | Default `true` |

### 4.2 Success response: `200 OK`

| Field | JSON type | Nullable | Constraint |
|---|---|---|---|
| `request_id` | string | no | UUID v4 |
| `model_version` | string | no | Model identifier that includes the source `run_id` |
| `verdict` | string | no | `live` or `spoof` |
| `spoof_score` | number | no | In [0, 1] |
| `threshold` | number | no | In [0, 1]; `verdict == "spoof"` iff `spoof_score >= threshold` |
| `confidence` | number | no | In [0, 1]; definition TBD — decide before Week 5 freeze (see `DESIGN.md` §1) |
| `face.bbox` | array[integer] | no | `[x, y, w, h]` in input-image pixels; `w, h > 0` |
| `face.detector_score` | number | no | In [0, 1] |
| `explanation` | object | yes | Null when `include_explanation=false` |
| `explanation.spoof_type` | object | yes | `{predicted: string, probabilities: map[string, number]}`; null without an auxiliary head |
| `explanation.illumination` | object | yes | Same shape as `spoof_type` |
| `explanation.environment` | object | yes | Same shape as `spoof_type` |
| `explanation.quality_checks` | array[object] | no (when `explanation` present) | Items: `{name: string, passed: boolean, value: number}` |
| `timing_ms.preprocess` | number | no | ≥ 0 |
| `timing_ms.inference` | number | no | ≥ 0 |
| `timing_ms.total` | number | no | ≥ `preprocess + inference` |

### 4.3 Error response: `4xx` / `5xx`

| Field | JSON type | Nullable | Constraint |
|---|---|---|---|
| `request_id` | string | no | UUID v4 |
| `error.code` | string | no | One of the codes in `DESIGN.md` §4 |
| `error.message` | string | no | Human-readable; no stack traces |
| `error.details` | object | yes | Code-specific, e.g. `{"reason": "blur"}` or `{"face_count": <int>}` |

### 4.4 JSON Schema (draft 2020-12): success response

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "PredictResponse",
  "type": "object",
  "required": ["request_id", "model_version", "verdict", "spoof_score", "threshold",
               "confidence", "face", "explanation", "timing_ms"],
  "additionalProperties": false,
  "properties": {
    "request_id": { "type": "string", "format": "uuid" },
    "model_version": { "type": "string", "minLength": 1 },
    "verdict": { "enum": ["live", "spoof"] },
    "spoof_score": { "type": "number", "minimum": 0, "maximum": 1 },
    "threshold": { "type": "number", "minimum": 0, "maximum": 1 },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "face": {
      "type": "object",
      "required": ["bbox", "detector_score"],
      "properties": {
        "bbox": { "type": "array", "items": { "type": "integer", "minimum": 0 },
                  "minItems": 4, "maxItems": 4 },
        "detector_score": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },
    "explanation": {
      "type": ["object", "null"],
      "required": ["spoof_type", "illumination", "environment", "quality_checks"],
      "properties": {
        "spoof_type": { "$ref": "#/$defs/attributePrediction" },
        "illumination": { "$ref": "#/$defs/attributePrediction" },
        "environment": { "$ref": "#/$defs/attributePrediction" },
        "quality_checks": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "passed", "value"],
            "properties": {
              "name": { "type": "string" },
              "passed": { "type": "boolean" },
              "value": { "type": "number" }
            }
          }
        }
      }
    },
    "timing_ms": {
      "type": "object",
      "required": ["preprocess", "inference", "total"],
      "properties": {
        "preprocess": { "type": "number", "minimum": 0 },
        "inference": { "type": "number", "minimum": 0 },
        "total": { "type": "number", "minimum": 0 }
      }
    }
  },
  "$defs": {
    "attributePrediction": {
      "type": ["object", "null"],
      "required": ["predicted", "probabilities"],
      "properties": {
        "predicted": { "type": "string" },
        "probabilities": {
          "type": "object",
          "additionalProperties": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      }
    }
  }
}
```
