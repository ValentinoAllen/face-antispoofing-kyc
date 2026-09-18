# Experiment Ledger

This ledger is **append-only**. Add one row per completed run and never edit or delete earlier rows.
If a row turns out to be wrong, add a new row that references and corrects it.

## How to log a run

1. **Before launching,** write the hypothesis: one sentence that says what you expect to change and
   why.
2. **Launch from a clean working tree.** `git_dirty: true` runs cannot be cited as results.
3. **When the run finishes,** bring its record into the repository (`SCHEMA.md` §3):
   - On Kaggle, `scripts/train.py` writes `<output-dir>/<run_id>/` with `record.json`,
     `resolved_config.json`, `checkpoint.pt` and `predictions.csv`. The last line it prints is the
     ledger row.
   - If the run is meant to be kept, save the notebook as a Kaggle notebook version, so
     `/kaggle/working` persists.
   - Copy `record.json` and `resolved_config.json` into `reports/runs/<run_id>/`. `checkpoint.pt`
     and `predictions.csv` are not committed.
   - Copy these values from `reports/runs/<run_id>/record.json`:
     - `run_id`
     - date
     - config path
     - the first 7 characters of `git_sha`
     - the metrics
4. **Metrics** come from the validation split unless the notes say `test`. Write them as percentages
   with two decimals. Leave a cell as `—` if the metric was not computed. Never estimate a value.
5. **Write the conclusion:** was the hypothesis supported, and what is the next step?
6. **Add the row at the bottom** of the table. `/session-end` does this for runs completed during a
   session.
7. **Commit the row together with** `reports/runs/<run_id>/record.json` and
   `resolved_config.json`, in the same commit.

## Runs

**Metric columns.** `APCER` and `ACER` mean the ISO/IEC 30107-3 worst case over PAI species. APCER is
the maximum over species (`metrics.apcer_max`), and ACER is `(APCER_max + BPCER) / 2`
(`metrics.acer`). A pooled value, computed over all attacks together, may appear in these columns
only with an explicit "(pooled)" suffix, as in the `20260915-153606-baseline` row.

| run_id | date | hypothesis | what changed | config file | git SHA | APCER | BPCER | ACER | BPCER@APCER=1% | notes/conclusion |
|---|---|---|---|---|---|---|---|---|---|---|
| `EXAMPLE-format-only` | YYYY-MM-DD | *(example)* A pretrained backbone gives a usable baseline without augmentation | *(example)* First baseline; no augmentation | `configs/<experiment>.yaml` | `abc1234` | — | — | — | — | **FORMAT EXAMPLE ONLY: not a real run. Do not cite.** |
| `20260915-153606-baseline` | 2026-09-15 | An ImageNet-pretrained mobilenetv3_large_100 fine-tuned for one epoch on a 4k train subset trains end to end on Kaggle and gives finite pooled PAD metrics on a 2k val subset. | First baseline: binary head, no augmentation, no face crop | `configs/baseline.yaml` | `6f368c0` | 1.86% (pooled) | 1.53% | 1.69% (pooled) | — | smoke run: 1 epoch, 4k/2k subset, val subset only, no face crop, no augmentation; 20260915-153137-baseline ran with the identical config and SHA and its metrics matched exactly (its record.json differs only in run_id, created_at, artifacts.checkpoint and epoch wall_time_s/images_per_s; mean train loss also identical); not a model-quality estimate: capture-source shortcut not yet ruled out |
| `20260916-075448-baseline` | 2026-09-16 | An ImageNet-pretrained mobilenetv3_large_100 fine-tuned for one epoch on a 4k train subset trains end to end on Kaggle and gives finite pooled PAD metrics on a 2k val subset. | First baseline: binary head, no augmentation, no face crop | `configs/baseline.yaml` | `9d75207` | 1.86% (pooled) | 1.53% | 1.69% (pooled) | — | smoke run: 1 epoch, 4k/2k subset, val subset only, no face crop, no augmentation; reran 20260915-153606-baseline's config (same config_hash) under the new SHA to produce predictions.csv for 20260916-075616-probe_metadata; metrics, counts and mean train loss are identical to both 20260915-153606-baseline and 20260915-153137-baseline; only epoch wall_time_s and images_per_s differ, apart from run identity (run_id, created_at, git_sha, checkpoint path) and the record keys added since (manifest_sha256, environment.pillow, environment.sklearn) |
| `20260916-075616-probe_metadata` | 2026-09-16 | Header-level metadata alone (format, dimensions, file size, JPEG tables, EXIF/ICC presence; no pixels) separates live from spoof on the baseline's 4k/2k subsets. Decision rule fixed before the run, on the primary classifier's pooled val ACER: <= 5% strong header-level shortcut; >= 25% weak header-level shortcut, pixel-level cues untested; otherwise partial. | Metadata probe: no pixels, HistGradientBoosting on header features | `configs/probe_metadata.yaml` | `9d75207` | 0.00% (pooled) | 0.00% | 0.00% (pooled) | — | probe, not a model; same subsets and threshold as 20260915-153606-baseline; pre-registered rule on the primary classifier's pooled val ACER 0.00% (0/1346 attacks accepted, 0/654 bona fide rejected): strong header-level shortcut; the CNN-vs-metadata comparison (predictions from 20260916-075448-baseline) is uninformative because the metadata classifier made no errors and its score is constant within each class |
| `20260918-130622-baseline` | 2026-09-18 | An ImageNet-pretrained mobilenetv3_large_100 fine-tuned for one epoch on a 4k train subset trains end to end on Kaggle and gives finite pooled PAD metrics on a 2k val subset. | First baseline: binary head, no augmentation, no face crop | `configs/baseline.yaml` | `9924a87` | 1.86% (pooled) | 1.53% | 1.69% (pooled) | — | smoke run: 1 epoch, 4k/2k subset, val subset only, no face crop, no augmentation; reran configs/baseline.yaml (same config_hash as 20260915-153606-baseline) under the new SHA to produce the checkpoint for 20260918-131447-counterfactual_jpeg; metrics, counts and mean train loss are identical to 20260916-075448-baseline, 20260915-153606-baseline and 20260915-153137-baseline; against 20260916-075448-baseline only epoch wall_time_s and images_per_s differ, apart from run identity (run_id, created_at, git_sha, checkpoint path), and resolved_config.json is byte-identical |
| `20260918-131447-counterfactual_jpeg` | 2026-09-18 | The baseline CNN relies on the JPEG-encoding trace of attack images. Val subset, threshold 0.5, live images unchanged. Validity: reencode_same pooled APCER must be <= 5%, otherwise the test is inconclusive. Then on reencode_live_quality pooled APCER: >= 20% strong reliance; <= 5% no evidence of reliance; otherwise partial. reencode_live_quality_and_size is secondary, same bands. | Counterfactual eval: attack images re-encoded at evaluation time, no retraining | `configs/counterfactual_jpeg.yaml` | `9924a87` | 4.31% (pooled) | 1.53% | 2.92% (pooled) | — | same val subset and threshold as 20260915-153606-baseline; checkpoint from a rerun of configs/baseline.yaml; identity arm: 25/1346 attacks accepted and 10/654 bona fide rejected, equal to 20260918-130622-baseline; reencode_same pooled APCER 1.78% (24/1346); pre-registered rule: valid (reencode_same <= 5%); reencode_live_quality pooled APCER 4.31% (58/1346) <= 5%: no evidence of reliance; secondary reencode_live_quality_and_size pooled APCER 8.77% (118/1346): partial |
