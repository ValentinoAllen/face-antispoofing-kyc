# Product Requirements: Face Anti-Spoofing for KYC Liveness

| | |
|---|---|
| Owner | Valentino Allen Prasetyo |
| Status | Draft v0.1 |
| Last updated | 2026-09-15 |

## 1. Problem statement

Remote onboarding (KYC) usually asks a user to submit an identity document and a selfie. A
face-matching step then checks that the selfie and the document photo show the same person. Face
matching answers *"is this the same person?"*, but it does not answer *"is a real, physically present
person in front of the camera?"*

An attacker who has a victim's photo (for example from social media) can present it to the camera as a
print, as a replay on a phone or laptop screen, as a paper cut-out, or as a mask. Without a liveness
control, that presentation can pass face matching. The business consequences are:

- **Fraud losses:** account takeover, mule accounts, and loans or credit opened in someone else's name.
- **Regulatory exposure:** KYC/AML obligations assume the customer was actually verified.
- **Operational cost:** every attack that is not caught automatically has to be caught later by
  manual review or chargeback processes.

**Presentation attack detection (PAD)** is the control that closes this gap. This project builds a
*passive*, single-image PAD model. It needs no challenge-response gestures from the user, so it adds
no friction to the onboarding flow.

The model can make two kinds of error, and they cost the business different things:

| Error | ISO/IEC 30107-3 term | Business cost |
|---|---|---|
| An attack is accepted as live | APCER (attack presentation classification error rate) | Fraud gets through |
| A real user is rejected as an attack | BPCER (bona fide presentation classification error rate) | Friction, retakes, drop-off, manual-review cost |

The two errors trade off against each other through the decision threshold. The product has to make
that trade-off visible and choose it deliberately.

## 2. Users of the system

| User | How they use it | What they need |
|---|---|---|
| KYC onboarding backend (primary) | Calls the inference API with a selfie, then routes the application (proceed / ask for retake / send to manual review) | A stable JSON contract, a verdict, a score, and explicit error codes for unusable input |
| Fraud / risk analyst | Reviews flagged cases | A human-readable explanation: likely attack type, capture conditions, and quality issues |
| ML reviewer | Decides whether a model version can ship | An evaluation report with ISO/IEC 30107-3 metrics broken down by attack type and capture condition |
| Demo visitor | Tries the demo page with an upload or a webcam | A clear verdict, a confidence value, and an explanation within a few seconds |

## 3. Scope

In scope:

- A binary classifier (bona fide vs attack) trained on CelebA-Spoof.
  - The paper reports 625,537 images, 10,177 subjects, and 43 attributes covering face,
    illumination, environment and spoof type.
  - The Kaggle mirror used here holds 561,575 images across 9,193 subjects in its `intra_test` label
    files. This figure was reproduced in-repo by `scripts/build_manifest.py` on 2026-09-15
    (`SCHEMA.md` §1.2).
- Subject-disjoint train/val/test splits, so no identity appears in more than one split. The
  official test split is kept unchanged (`ARCHITECTURE.md` ADR-009).
- Evaluation with ISO/IEC 30107-3 PAD metrics, broken down by spoof type, illumination, and
  environment.
- Post-training quantization, export to ONNX, and CPU inference through ONNX Runtime.
- A FastAPI inference service with a documented request/response contract and error handling for
  unusable input.
- A minimal demo page (upload or webcam → verdict → confidence → explanation).

## 4. Non-goals

These are out of scope for this project. They are recorded here so they are not assumed.

- **Face recognition or face matching.** The system does not verify identity.
- **Digital injection attacks and deepfakes.** Virtual cameras, injected video streams and
  synthetically generated faces are a different threat model, and CelebA-Spoof does not cover them.
- **Active liveness.** No blink, head-turn or challenge-response flows.
- **Video or multi-frame liveness.** Input is one still image.
- **Document verification.** ID card authenticity, OCR and MRZ parsing are not included.
- **Production operations.** No SLAs, autoscaling, auth/rate limiting beyond a basic demo, and no
  monitoring stack.
- **Certification claims.** Metrics follow ISO/IEC 30107-3 definitions, but this project does not
  claim conformance testing or certification.
- **Storage of biometric data.** The service does not persist uploaded images.

## 5. MVP definition

The MVP is done when **all** of the following hold:

1. A manifest and subject-disjoint split can be regenerated from the raw dataset with one command,
   and a test enforces subject disjointness.
2. At least one trained model has been evaluated on the held-out test set **once**, at a threshold
   chosen on the validation set, with a generated report (see `DESIGN.md` §3).
3. A quantized ONNX model has been produced and its metric deltas against the FP32 model have been
   reported.
4. `POST /v1/predict` returns the response defined in `SCHEMA.md` §4, and the no-face,
   multiple-face and low-quality cases return their documented error codes.
5. CPU latency p50/p95 has been measured on documented hardware and recorded.
6. The README results section is filled **only** from real run records.

## 6. Success metrics

All accuracy metrics are computed on the **subject-disjoint test set**. They are computed once for
the final model, at an operating threshold chosen on the validation set. Attack is the positive
class for APCER.

| Metric | Definition | Target |
|---|---|---|
| APCER | Proportion of attack presentations classified as bona fide. Computed per PAI species (spoof type); the headline number is the **maximum** over species | TBD — decide before Week 2 baseline |
| BPCER | Proportion of bona fide presentations classified as attacks | TBD — decide before Week 2 baseline |
| ACER | (APCER + BPCER) / 2. Reported for comparability with the anti-spoofing literature. ISO/IEC 30107-3 does not endorse averaging the two error types, so ACER is never reported alone | TBD — decide before Week 2 baseline |
| BPCER @ APCER = 1% (BPCER100) | BPCER at the threshold where APCER on the evaluated set equals 1%. This measures how much user friction it costs to hold fraud acceptance at 1% | TBD — decide before Week 2 baseline |
| Model size | On-disk size of the quantized `.onnx` file, in MB | TBD — decide before Week 2 baseline |
| CPU latency p50 / p95 | Wall-clock ms for one image at batch size 1: decode + face crop + preprocess + inference, excluding network. Measured after warm-up on documented CPU hardware | TBD — decide before Week 2 baseline |
| Quantization degradation | Change in APCER, BPCER and BPCER100 between the FP32 and quantized models on the same evaluation set | TBD — decide before Week 2 baseline |

## 7. Milestone plan (8 weeks)

| Week | Dates | Deliverables | Exit criterion |
|---|---|---|---|
| 1 | 2026-09-14 → 2026-09-20 | Repo scaffold; dataset access; manifest builder; subject-disjoint split with invariant test; EDA notebook | Manifest and split files regenerate from raw data; split test passes |
| 2 | 2026-09-21 → 2026-09-27 | PRD targets set; PAD metrics module with tests; transforms; dataset class; YAML config loader; baseline training on Kaggle/Colab | First baseline run logged in `EXPERIMENTS.md` with validation metrics |
| 3 | 2026-09-28 → 2026-10-04 | Evaluation report generator; error analysis v1 by spoof type, illumination, environment | Report for the baseline generated from a run record |
| 4 | 2026-10-05 → 2026-10-11 | Hypothesis-driven iteration (augmentation, backbone, auxiliary heads), each run logged | Each run has a hypothesis and a conclusion in the ledger |
| 5 | 2026-10-12 → 2026-10-18 | Freeze the model; choose the operating threshold on validation; decide how confidence is calibrated | Frozen config and threshold recorded |
| 6 | 2026-10-19 → 2026-10-25 | Quantization; ONNX export; FP32-vs-quantized parity; CPU latency benchmark | Parity table and latency numbers recorded from real measurements |
| 7 | 2026-10-26 → 2026-11-01 | FastAPI service; face detection and quality gating; error contract; demo page; API tests | API contract tests pass; demo works end to end locally |
| 8 | 2026-11-02 → 2026-11-08 | One-shot test-set evaluation; final report; README results; write-up; buffer | MVP definition (§5) fully met |

## 8. Risks and assumptions

- **Dataset-to-deployment gap.** CelebA-Spoof capture devices and conditions may not match a real
  KYC selfie flow. Test-set metrics describe in-distribution performance, not field performance.
- **Crop mismatch.** Training can use dataset-provided face boxes, but inference needs a face
  detector. If the crops differ, accuracy can silently degrade. The report must evaluate using the
  inference-time detector.
- **Compute limits.** Kaggle/Colab sessions are time-limited, so training must checkpoint and resume.
- **Data quality of the official split.** Checking the Kaggle mirror found two defects
  (`ARCHITECTURE.md` ADR-009, `SCHEMA.md` §1.2):
  - The official train/test split is not subject-disjoint: subjects `5028`, `7332` and `9735` are
    in both. They are removed from train only, and test is never modified.
  - 2,022 train images stored under `live/` carry a spoof label. These conflicting images are
    excluded by default, because neither the annotation nor the folder name could be shown to be
    authoritative.

  Residual risks:
  - Annotation errors that do not show up as a path conflict cannot be detected this way, in either
    split. They would bias both training and the reported metrics.
- **Skewed capture conditions.** Indices 41 (illumination) and 42 (environment) are verified, so
  evaluation can be broken down by them (`SCHEMA.md` §1.1).
  - Illumination is heavily skewed: code 1 covers 59.0% of the official-train spoof images and
    74.3% of the test spoof images (`SCHEMA.md` §1.1.1, reproduced in-repo).
  - Per-condition results for the other illumination codes rest on much smaller samples and are
    noisier, most of all on test (see the next entry).
- **Covariate shift between train and test.** The test split's attack population differs materially
  from train. Shares of `spoof/` images, from counts printed by `scripts/build_manifest.py` on
  2026-09-15 (`SCHEMA.md` §1.1.1):
  - `spoof_type` is near-uniform in train but not in test: code 7 is 8.8% of train vs 13.7% of
    test; code 10 is 12.3% vs 7.6%.
  - `illumination` is more concentrated in test: code 1 is 59.0% vs 74.3%; code 3 is 10.6% vs 5.2%.
  - `environment` likewise: code 1 is 76.3% vs 86.2%.
  - The difference is a property of the official test split, which is not modified. Val is carved
    from official train by subject; its own code distribution has not been printed.

  Consequences for the Week 2 evaluation design:
  - (a) Val and test ACER are not directly comparable. Both must always be reported, each with its
    split named.
  - (b) Per-condition breakdowns on test will have small cells: illumination code 3 has only 2,461
    test spoof images. Report confidence intervals, not bare point estimates.
  - (c) Any gap between val and test ACER must be checked against this distribution difference
    before it is attributed to the model.

  Correction to an earlier note: the live fraction also differs (33.0% train, 32.7% val, 29.7% test;
  `SCHEMA.md` §1.2) and was previously recorded here as a prior-shift risk. APCER and BPCER are
  class-conditional, so class mix alone does not move them at a fixed threshold. The real risk is
  the covariate shift above.
- **Licensing.** Dataset terms may restrict redistributing images, which affects failure galleries
  in public reports. TBD — check license terms before Week 1 manifest.
- **Demographic performance.** BPCER may differ across groups. CelebA face attributes allow coarse
  slicing, but they are annotated labels with known limitations and cannot support a full fairness
  claim.
