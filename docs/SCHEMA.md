# Data Contracts

These are the contracts between pipeline stages. Any change to a contract must update this file in
the same commit as the code change. Code that produces or consumes these artifacts validates them
against this file.

Conventions: dtypes are pandas/NumPy dtypes for CSV artifacts and JSON types for JSON artifacts.
"Nullable = no" means an empty value is a validation error.

## 1. Image manifest CSV

- **Path:** `data/manifests/manifest.csv` (gitignored).
- **Format:** UTF-8 CSV with a header row and one row per image.
- **Produced by:** the manifest builder in `antispoof.data`.

| Column | dtype | Nullable | Constraint |
|---|---|---|---|
| `image_path` | string | no | POSIX path relative to the dataset root; unique across the manifest; the file must exist when the manifest is built |
| `subject_id` | string | no | Subject identifier from the dataset directory structure; stored as a string to avoid losing leading zeros |
| `label` | int8 | no | `0` = bona fide (live), `1` = attack (spoof) |
| `spoof_type` | category (string) | no | `live` for bona fide rows; otherwise one of the dataset's PAI species names. The exact name list and the mapping from raw annotation codes are TBD — verify against the CelebA-Spoof annotation docs before Week 1 manifest |
| `illumination` | category (string) | yes | Illumination condition name from the dataset annotations. Whether bona fide rows carry a value or null is TBD — verify before Week 1 manifest |
| `environment` | category (string) | yes | Environment condition name from the dataset annotations. Same verification note as `illumination` |
| `bbox_x` | int32 | yes | Left edge in **original image pixel coordinates**, ≥ 0 |
| `bbox_y` | int32 | yes | Top edge in original image pixel coordinates, ≥ 0 |
| `bbox_w` | int32 | yes | > 0; `bbox_x + bbox_w` ≤ image width |
| `bbox_h` | int32 | yes | > 0; `bbox_y + bbox_h` ≤ image height |
| `source_split` | category (string) | no | The dataset's own split for this image (`train` / `test`), kept for traceability |
| `split` | category (string) | no | `train`, `val` or `test`. Joined from the split file (§2); it is never set independently |

Row-level invariants:

- `label == 0` ⇔ `spoof_type == "live"`.
- The four `bbox_*` columns are either all null or all non-null.
- The raw bounding-box coordinate convention (absolute pixels vs a rescaled frame) is TBD — verify
  before Week 1 manifest. The manifest always stores original-image pixels.
- The 40 CelebA face attributes are **not** in manifest v1. If they are needed for slicing, they go in
  a separate table keyed by `image_path`.

Manifest-level invariants:

- Every `subject_id` maps to exactly one `split` value.
- The manifest is accompanied by `manifest.meta.json` with `sha256` (of the CSV), `row_count`,
  `created_at` and `git_sha`. The counts are computed, never typed in.

## 2. Split file

- **Path:** `configs/splits/<split_name>.csv`. It contains no images or personal data, only subject
  identifiers, and it is committed so splits are reproducible.
- **Format:** UTF-8 CSV with a header row and one row per subject.

| Column | dtype | Nullable | Constraint |
|---|---|---|---|
| `subject_id` | string | no | Unique within the file; must exist in the manifest |
| `split` | category (string) | no | `train`, `val` or `test` |

Sidecar `configs/splits/<split_name>.meta.yaml`:

| Key | Type | Nullable | Constraint |
|---|---|---|---|
| `split_name` | string | no | Matches the file name |
| `strategy` | string | no | `official` or `custom_subject_disjoint` |
| `seed` | int | yes | Required when `strategy == custom_subject_disjoint` |
| `ratios` | map[str, float] | yes | Requested subject-level fractions for `train`/`val`/`test`; must sum to 1.0. Required for custom splits |
| `manifest_sha256` | string | no | Hash of the manifest the split was built from |
| `split_sha256` | string | no | Hash of the split CSV |
| `created_at` | string (ISO 8601 UTC) | no | |
| `git_sha` | string | no | 40-character hex |

### Invariant: subject disjointness

Let `S_train`, `S_val` and `S_test` be the sets of `subject_id` assigned to each split. Then:

```
S_train ∩ S_val  = ∅
S_train ∩ S_test = ∅
S_val   ∩ S_test = ∅
S_train ∪ S_val ∪ S_test = set(manifest.subject_id)
```

Enforcement:

1. A unit test in `tests/` checks the split function on a synthetic manifest, including adversarial
   cases such as subjects with a single image, or subjects that appear in both source splits.
2. The dataset loader asserts disjointness at start-up and refuses to train if it fails.
3. Each run record stores `split_sha256`, so the split behind every reported number can be identified.

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
