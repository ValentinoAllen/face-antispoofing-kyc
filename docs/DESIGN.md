# Design

This document covers system behaviour: the API contract, the demo flow, how evaluation reports are
laid out, and error handling. It does not cover visual styling. Exact JSON field types live in
`SCHEMA.md` §4.

## 1. FastAPI request/response contract

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/predict` | Classify one selfie image as bona fide (live) or attack (spoof) |
| `GET` | `/health` | Liveness/readiness of the service and whether a model is loaded |
| `GET` | `/v1/model` | Model metadata: version, input size, operating threshold, and the run it came from |
| `GET` | `/demo` | Static demo page (see §2) |

The API is versioned in the path (`/v1`). Any breaking change to a response shape requires `/v2`.

### `POST /v1/predict`

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | yes | JPEG or PNG. Maximum size TBD — decide before Week 7 serving |
| `include_explanation` | boolean | no | Defaults to `true`. When `false`, `explanation` is `null`, which saves work for backend callers |

**Processing pipeline**

1. Validate the content type and size, then decode the image.
2. Run face detection. Exactly one face is required.
3. Run quality gates on the face crop (resolution, blur, exposure; thresholds TBD — decide before
   Week 7 serving).
4. Crop, resize and normalize using the **same** preprocessing as training, read from the model's
   metadata rather than duplicated in code.
5. Run ONNX Runtime inference.
6. Apply the operating threshold from the model metadata to produce the verdict.
7. Build the explanation (if requested) and return.

**Success response (`200`):**

- `request_id`: a UUID, echoed in logs.
- `model_version`: identifies the ONNX model and its source `run_id`.
- `verdict`: `"live"` or `"spoof"`.
- `spoof_score`: the model's attack score in [0, 1]. Higher means more likely an attack.
- `threshold`: the operating threshold used. `verdict = "spoof"` iff `spoof_score >= threshold`.
- `confidence`: in [0, 1]. How it is computed is TBD — decide before Week 5 freeze. It will be either
  a calibrated probability of the returned verdict or a normalized margin from the threshold, and
  the chosen definition will be documented here.
- `face`: the detected bounding box and detector score.
- `explanation`: see below, or `null`.
- `timing_ms`: preprocessing, inference and total time on the server.

**Explanation object:** its purpose is to tell an analyst *why* a result may be wrong, or what kind
of attack is suspected.

- `spoof_type`: predicted PAI species with per-class probabilities. `null` if the model has no
  auxiliary spoof-type head (see the ADR pending decisions).
- `illumination`, `environment`: predicted capture conditions with probabilities. `null` if there
  are no auxiliary heads.
- `quality_checks`: a list of `{name, passed, value}` for every quality gate that ran, including the
  ones that passed.

The explanation describes attributes the model predicts. It is **not** a saliency map or a causal
explanation, and the demo labels it that way.

**Privacy:** images are processed in memory and never written to disk or logs. Logs contain
`request_id`, verdict, score, timings and error codes only.

## 2. Demo page UX

A single static page served by FastAPI at `/demo`, using plain HTML and JS with no build step.

**Flow**

1. **Notice.** A short statement at the top: this is a demo of a research model, not a KYC decision,
   and images are not stored.
2. **Capture.** Two tabs:
   - *Upload*: a file picker that accepts JPEG/PNG and shows a preview.
   - *Webcam*: a live preview with a capture button. It asks for camera permission only when this
     tab is opened.
3. **Submit.** A single "Check liveness" button that is disabled until an image exists, with a
   spinner while the request is in flight.
4. **Verdict.** A large badge reading **LIVE** or **SPOOF**. Error states from §4 show **RETAKE
   NEEDED** instead, never a verdict.
5. **Confidence.** A horizontal bar with the numeric value. `spoof_score` and the threshold are shown
   next to it so the verdict can be traced.
6. **Explanation panel.**
   - Predicted spoof type (top classes with probabilities).
   - Predicted illumination and environment.
   - Quality checks as a pass/fail list.
   - The detected face box drawn on the preview.
   - When a field is `null`, the panel says it is not available for this model version instead of
     hiding the field.
7. **Retake guidance.** For error responses, a specific instruction mapped from the error code
   (e.g. "Move closer so your face fills the frame"), followed by a return to step 2.

## 3. Evaluation report layout

Each evaluation produces `reports/<run_id>/` containing `summary.md`, `metrics.json` and `figures/`.
The report is generated from the run record; numbers are never typed in by hand.

| # | Section | Content | What a reader should be able to conclude |
|---|---|---|---|
| 1 | Header | run_id, date, git SHA, config hash, split name/hash, evaluated split (val/test), threshold and how it was chosen | Exactly what was evaluated, and that it can be reproduced |
| 2 | Headline metrics table | APCER (max over species), BPCER, ACER, BPCER@APCER=1%, at the operating threshold, next to the PRD targets | Whether this model meets the PRD targets |
| 3 | DET curve | APCER vs BPCER on normal-deviate axes, operating point marked, previous best run overlaid | How much user friction it costs to lower fraud acceptance, and whether this run beats the previous best over the whole curve |
| 4 | Score distributions | Histograms of `spoof_score` for bona fide vs each attack species, with the threshold line | How separable the classes are, and how sensitive the metrics are to small threshold shifts |
| 5 | APCER by PAI species | Table and bar chart with sample counts per species | Which attack types are the weakest point |
| 6 | Errors by capture condition | BPCER and APCER by illumination and by environment, with counts | Whether real users in poor lighting or unusual settings get rejected, and whether attacks slip through under specific conditions |
| 7 | Confusion matrix | At the operating threshold, with counts and rates | Error volumes in absolute terms |
| 8 | Threshold table | APCER and BPCER at thresholds selected for fixed APCER levels, and at the operating threshold | Where to move the threshold if the business risk appetite changes |
| 9 | Failure analysis | Hardest bona fide images (highest score) and hardest attacks (lowest score), with their attributes. Images are embedded only if dataset license terms allow (TBD — check license before Week 3 report) | Recurring patterns in failures that point to the next experiment |
| 10 | FP32 vs quantized parity | Metric deltas, score agreement rate, model size, CPU latency p50/p95 with hardware spec | Whether quantization is safe to ship |
| 11 | Conclusions | Written by the author from sections 2–10; each claim references a section | The decision: ship / iterate / reject, and the next hypothesis |

Every rate in the report is shown with its sample count. Small per-attribute groups are flagged, and
their rates are not interpreted as reliable.

## 4. Error handling

All errors use the envelope defined in `SCHEMA.md` §4.3. Input problems return `4xx` with a
machine-readable `code` and never a verdict. **The service never falls back to guessing a verdict
when the input is unusable.**

| Condition | HTTP | `code` | Behaviour / message to the caller | Demo retake guidance |
|---|---|---|---|---|
| Missing `image` field | 400 | `MISSING_IMAGE` | The request is rejected before decoding | "Choose or capture a photo first." |
| Unsupported content type | 415 | `UNSUPPORTED_MEDIA_TYPE` | Lists the accepted types | "Use a JPEG or PNG image." |
| File too large | 413 | `IMAGE_TOO_LARGE` | Includes the size limit in `details` | "Use a smaller image." |
| Corrupt or undecodable file | 400 | `INVALID_IMAGE` | Decoding failed | "That file couldn't be read. Try another photo." |
| No face detected | 422 | `NO_FACE_DETECTED` | The detector found no face above its confidence threshold | "Make sure your whole face is visible and well lit." |
| More than one face | 422 | `MULTIPLE_FACES` | `details.face_count` is included. The service does not pick one face | "Make sure only your face is in the frame." |
| Face too small | 422 | `LOW_QUALITY_INPUT` | `details.reason = "face_too_small"` | "Move closer so your face fills the frame." |
| Too blurry | 422 | `LOW_QUALITY_INPUT` | `details.reason = "blur"` | "Hold still and make sure the camera is focused." |
| Under- or over-exposed | 422 | `LOW_QUALITY_INPUT` | `details.reason = "exposure"` | "Find more even lighting." |
| Model not loaded | 503 | `MODEL_UNAVAILABLE` | `/health` also reports `model_loaded: false` | "Service is starting up. Try again shortly." |
| Unexpected server error | 500 | `INTERNAL_ERROR` | Generic message; the stack trace goes to logs only, keyed by `request_id` | "Something went wrong. Try again." |

Design notes:

- **Why reject multiple faces:** in KYC, a second face in the frame can itself be a presentation
  attack (a photo held next to the user). Guessing which face is "the user" hides that.
- **Why the quality gates run before the model:** the model was not trained to be reliable on
  unusable crops. A confident verdict on a blurry thumbnail would be misleading.
- **Why the quality thresholds are TBD:** they need to be set from the EDA of face-crop sizes and
  quality in the dataset (Week 1), and checked against the serving detector (Week 7). They must not
  be guessed.
