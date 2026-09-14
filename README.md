# Face Anti-Spoofing for KYC Liveness

A face presentation attack detection (PAD) model for selfie-based identity verification. It is
trained on CelebA-Spoof, evaluated with ISO/IEC 30107-3 metrics, quantized, and served on CPU behind
a FastAPI endpoint.

## The problem

Remote onboarding compares a selfie with an ID photo. Face matching checks that the faces belong to
the same person, but not that a real person is in front of the camera. A printed photo, a phone
screen replay or a mask can get through. This project builds a passive, single-image liveness check
that closes that gap. It measures both sides of the trade-off: attacks accepted (APCER) and genuine
users rejected (BPCER).

## Approach

- **Data:** CelebA-Spoof (625,537 images, 10,177 subjects, 43 attributes), using
  **subject-disjoint** train/val/test splits so identities never leak across splits.
- **Model:** a PyTorch classifier on a pretrained backbone (timm). Heavy training runs on
  Kaggle/Colab GPUs.
- **Evaluation:** APCER per attack type, BPCER, ACER, and BPCER at APCER = 1%, broken down by attack
  type, illumination and environment.
- **Deployment:** post-training quantization → ONNX Runtime → FastAPI, with explicit errors for
  no-face, multiple-face and low-quality input.

## Status

Scaffolding is complete; data and model work are in progress. Results will be added here only from
real evaluation runs. See [docs/PROGRESS.md](docs/PROGRESS.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
git clone <repo-url> && cd face-antispoofing-kyc
uv sync
git config core.hooksPath .githooks   # git hooks are not cloned; re-run after every fresh clone
```

## Smoke test

```bash
uv run pytest
```

## Documentation

| Doc | Contents |
|---|---|
| [PRD](docs/PRD.md) | Problem, scope, success metrics, 8-week plan |
| [Architecture](docs/ARCHITECTURE.md) | Stack, data flow, compute split, design decisions |
| [Design](docs/DESIGN.md) | API contract, demo UX, report layout, error handling |
| [Schema](docs/SCHEMA.md) | Manifest, split, experiment record and API data contracts |
| [Rules](docs/RULES.md) | Coding, config, reproducibility and testing rules |
| [Progress](docs/PROGRESS.md) | Current status and session log |
| [Experiments](docs/EXPERIMENTS.md) | Run-by-run experiment ledger |

## Author

Valentino Allen Prasetyo
