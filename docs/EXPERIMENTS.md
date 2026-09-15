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
