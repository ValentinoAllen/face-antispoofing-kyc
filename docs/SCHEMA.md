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
| 40 | Spoof type code | Verified | Zero for live images. Names for the codes are not recorded yet | `spoof_type` |
| 41 | **Provisional.** Illumination condition by documentation convention; not verified against this mirror | Provisional | Zero for live images (measured) | `attr_41` |
| 42 | **Provisional.** Environment by documentation convention; not verified against this mirror | Provisional | Zero for live images (measured) | `attr_42` |
| 43 | Live/spoof label: `0` = live, `1` = spoof. The training target | Verified | On the test split it agrees with the `live/` vs `spoof/` path segment for all 67,170 entries | `label` |

The meaning of indices 41 and 42 is an open question. It will be confirmed by unique-value counts on
the mirror (`PROGRESS.md`). Until then, code and manifest columns name them only by index.

### 1.2 Measured counts (`intra_test`, this mirror)

> **Externally measured, not yet reproduced in-repo.**
> - Provenance: measured by the owner on Kaggle, 2026-09-14, against the mirror and `intra_test`
>   label files above, including the `Data/train` and `Data/test` directory listings.
> - To be reproduced by `python scripts/build_manifest.py --config configs/data.yaml` on its first
>   Kaggle run (`RULES.md` §6 item 7).

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
  - Confirmed in the label files and on disk.
  - No image path appears in both splits.
- **Label files vs directories:** the listings under `Data/train` and `Data/test` match the label
  files exactly. No subject is in a label file without a folder, and no folder lacks label entries.
- **Not known yet:** row counts after removing the 3 excluded subjects, and the train/val sizes. The
  first Kaggle run of `scripts/build_manifest.py` produces them.
- **Observed discrepancy:**
  - This mirror contains 561,575 images across 9,193 unique subjects.
  - The CelebA-Spoof paper reports 625,537 images and 10,177 subjects.
  - This is recorded as an observation about this mirror, not as a corrected figure. `intra_test`
    may not span the full dataset.

### 1.3 Columns

| Column | dtype | Nullable | Constraint |
|---|---|---|---|
| `image_path` | string | no | The label-file key: POSIX path relative to the dataset root. Unique across the three manifests |
| `subject_id` | string | no | Path component after `train`/`test`. Stored as a string so leading zeros survive |
| `split` | category (string) | no | `train`, `val` or `test`. Test rows keep the official split; val is assigned per subject (§2) |
| `label` | int8 | no | `0` = live (bona fide), `1` = spoof (attack). From index 43, or from `path_kind` under `trust_path` |
| `spoof_type` | int32 | no | Raw index-40 code. Names for the codes are TBD — decide before Week 3 error analysis |
| `attr_41` | int32 | no | Raw index-41 code. Meaning provisional (§1.1) |
| `attr_42` | int32 | no | Raw index-42 code. Meaning provisional (§1.1) |
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
- Expected from §1.1, but not asserted by the builder: `label == 0` ⇒ `spoof_type == attr_41 ==
  attr_42 == 0` on rows with `conflict == False`.
- The 40 CelebA face attributes are **not** in manifest v1. If they are needed for slicing, they go
  in a separate table keyed by `image_path`.

Manifest-level invariants:

- `image_path` is unique across the three files. This holds by construction: label-file keys are
  unique, and train and test paths differ in their split component.
- Every `subject_id` maps to exactly one `split` across the three files. `validate_splits` checks
  this on the stacked manifests at build time (§2).
- Sidecar `manifest.meta.json` (`sha256` of each CSV, `row_count`, `created_at`, `git_sha`): not yet
  produced. TBD — decide before Week 2 baseline.

### 1.4 Bounding boxes (not in manifest v1)

- Manifest v1 has **no** bounding-box columns, not even empty placeholders.
- The bounding-box format is an open question. One observed clue: a file named `004046_BB.txt` at
  the dataset root, which suggests per-image `{image_id}_BB.txt` sidecar files. This is unverified.
- Bounding-box columns will be added in manifest v2, once the format is confirmed.

## 2. Split assignment file

- **Path:** `configs/splits/split_assignment.csv`.
  - It holds subject identifiers only (no images or personal data), and it is committed so the split
    is reproducible.
  - **Not yet generated.** The first Kaggle run of `scripts/build_manifest.py` produces it, and it is
    committed afterwards.
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

Sidecar `configs/splits/split_assignment.meta.yaml`:
- Not yet produced.
- Run records need it to cite `split_sha256` (§3).
- Its keys are TBD — decide before Week 2 baseline.

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
   tolerance.
5. Each run record stores `split_sha256`, so the split behind every reported number can be identified.

## 3. Experiment record

- **Path:** `reports/runs/<run_id>/record.json`, one file per run. It is also logged to W&B as run
  config/summary.
- **Format:** JSON object.

| Field | JSON type | Nullable | Constraint |
|---|---|---|---|
| `run_id` | string | no | `YYYYMMDD-HHMMSS-<slug>` in UTC; unique |
| `created_at` | string | no | ISO 8601 UTC |
| `hypothesis` | string | no | One sentence; copied to `EXPERIMENTS.md` |
| `config_path` | string | no | Repo-relative path under `configs/` |
| `config_hash` | string | no | SHA-256 hex of the fully resolved config serialized with sorted keys |
| `git_sha` | string | no | 40-character hex of `HEAD` at launch |
| `git_dirty` | boolean | no | `true` if the working tree had uncommitted changes. Such runs cannot be cited as results |
| `seed` | integer | no | ≥ 0; the same value as in the config |
| `split_name` | string | no | Matches a file in `configs/splits/` |
| `split_sha256` | string | no | Matches the split sidecar |
| `environment` | object | no | `{python, torch, cuda, device, platform}`; strings, `cuda` nullable |
| `status` | string | no | `running`, `completed`, `failed` or `aborted` |
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
| `artifacts.checkpoint` | string | yes | Path or W&B artifact reference |
| `artifacts.onnx` | string | yes | Path or W&B artifact reference |
| `artifacts.report_dir` | string | yes | `reports/<run_id>/` |
| `artifacts.wandb_url` | string | yes | URL of the W&B run |
| `notes` | string | yes | Free text |

Metrics are stored as fractions in [0, 1]. Docs and reports display them as percentages.

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
