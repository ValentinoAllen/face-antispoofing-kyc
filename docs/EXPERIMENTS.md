# Experiment Ledger

This ledger is **append-only**. Add one row per completed run and never edit or delete earlier rows.
If a row turns out to be wrong, add a new row that references and corrects it.

## How to log a run

1. **Before launching,** write the hypothesis: one sentence that says what you expect to change and
   why.
2. **Launch from a clean working tree.** `git_dirty: true` runs cannot be cited as results.
3. **When the run finishes,** copy these values from `reports/runs/<run_id>/record.json` (see
   `SCHEMA.md` §3):
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

## Runs

| run_id | date | hypothesis | what changed | config file | git SHA | APCER | BPCER | ACER | BPCER@APCER=1% | notes/conclusion |
|---|---|---|---|---|---|---|---|---|---|---|
| `EXAMPLE-format-only` | YYYY-MM-DD | *(example)* A pretrained backbone gives a usable baseline without augmentation | *(example)* First baseline; no augmentation | `configs/<experiment>.yaml` | `abc1234` | — | — | — | — | **FORMAT EXAMPLE ONLY: not a real run. Do not cite.** |
